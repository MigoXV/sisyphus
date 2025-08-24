from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------
# Causal FSMN block (stateless)
# ------------------------------
class CausalFSMNBlock(nn.Module):
    def __init__(
        self, hidden_dim: int, memory_order: int = 8, use_layernorm: bool = True
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.memory_order = memory_order
        self.proj = nn.Identity()
        self.taps = nn.Parameter(torch.zeros(hidden_dim, memory_order))
        self.ln = nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity()

    def _build_dw_kernel(self, dtype, device):
        # depthwise kernel: [H, 1, K+1] with current-frame weight fixed to 1
        H, K = self.hidden_dim, self.memory_order
        w = torch.zeros(H, 1, K + 1, dtype=dtype, device=device)
        w[:, 0, 0] = 1.0
        if K > 0:
            w[:, 0, 1:] = self.taps
        return w

    def forward_sequence(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B, T, H]; strict causal via left padding only
        B, T, H = h.shape
        assert H == self.hidden_dim
        K = self.memory_order
        x = self.proj(h).transpose(1, 2)  # [B, H, T]
        if K > 0:
            x = F.pad(x, (K, 0))  # left pad K for causality
        w = self._build_dw_kernel(dtype=x.dtype, device=x.device)
        y = F.conv1d(x, w, stride=1, padding=0, groups=H).transpose(1, 2)  # [B, T, H]
        y = self.ln(y)
        return y

    def forward_chunk(
        self, h_chunk: torch.Tensor, cache: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # h_chunk: [B, T, H]; cache: [B, H, K] or None (owned/managed by caller)
        B, T, H = h_chunk.shape
        assert H == self.hidden_dim
        K = self.memory_order
        x = self.proj(h_chunk).transpose(1, 2)  # [B, H, T]
        if K > 0:
            if cache is None:
                cache = torch.zeros(B, H, K, device=x.device, dtype=x.dtype)
            x_cat = torch.cat([cache, x], dim=2)  # prepend real history
            w = self._build_dw_kernel(dtype=x.dtype, device=x.device)
            y = F.conv1d(x_cat, w, stride=1, padding=0, groups=H)  # [B, H, T]
            new_cache = x_cat[:, :, -K:]
        else:
            w = self._build_dw_kernel(dtype=x.dtype, device=x.device)
            y = F.conv1d(x, w, stride=1, padding=0, groups=H)
            new_cache = x.new_zeros(B, H, 0)
        y = y.transpose(1, 2)  # [B, T, H]
        y = self.ln(y)
        return y, new_cache

    def forward_step(
        self, h_t: torch.Tensor, cache: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # h_t: [B, H]; single-step stateless call
        y, new_cache = self.forward_chunk(h_t.unsqueeze(1), cache)
        return y.squeeze(1), new_cache


# ------------------------------
# Policy head (stateless)
# ------------------------------
class FSMNPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int = 4,
        hidden_dim: int = 128,
        memory_order: int = 8,
        num_blocks: int = 2,
        action_dim: int = 2,
    ):
        super().__init__()
        self.obs_proj = nn.Linear(obs_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [CausalFSMNBlock(hidden_dim, memory_order) for _ in range(num_blocks)]
        )
        self.act_head = nn.Linear(hidden_dim, action_dim)
        self.val_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # obs_seq: [B, T, D] or [B, D]; strict-causal full-sequence outputs
        if obs_seq.dim() == 2:
            obs_seq = obs_seq.unsqueeze(1)
        x = F.relu(self.obs_proj(obs_seq))
        for blk in self.blocks:
            x = F.relu(blk.forward_sequence(x))
        logits = self.act_head(x)  # [B, T, A]
        value = self.val_head(x).squeeze(-1)  # [B, T]
        return logits, value

    def forward_chunk(
        self,
        obs_chunk: torch.Tensor,
        caches: Optional[List[torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        # obs_chunk: [B, T, D]; caches is a list of [B, H, K] (caller-owned)
        if obs_chunk.dim() == 2:
            obs_chunk = obs_chunk.unsqueeze(1)
        x = F.relu(self.obs_proj(obs_chunk))
        new_caches: List[torch.Tensor] = []
        if caches is None:
            caches = [None] * len(self.blocks)
        for blk, cache in zip(self.blocks, caches):
            x, cache = blk.forward_chunk(x, cache)
            x = F.relu(x)
            new_caches.append(cache)
        logits = self.act_head(x)
        value = self.val_head(x).squeeze(-1)
        return logits, value, new_caches

    def step(
        self,
        obs_t: torch.Tensor,
        caches: Optional[List[torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        # obs_t: [B, D]; single-step stateless call with external caches
        if obs_t.dim() != 2:
            obs_t = obs_t.view(obs_t.size(0), -1)
        x = F.relu(self.obs_proj(obs_t))
        new_caches: List[torch.Tensor] = []
        if caches is None:
            caches = [None] * len(self.blocks)
        for blk, cache in zip(self.blocks, caches):
            x, cache = blk.forward_step(x, cache)
            x = F.relu(x)
            new_caches.append(cache)
        logits = self.act_head(x)  # [B, A]
        value = self.val_head(x).squeeze(-1)  # [B]
        return logits, value, new_caches


# ------------------------------
# Helper functions (pure/functional; no module state)
# ------------------------------
@torch.no_grad()
def fsmn_allocate_caches(
    policy: FSMNPolicy,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> List[torch.Tensor]:
    # caller controls lifecycle; we do not attach to the module
    caches: List[torch.Tensor] = []
    H = policy.obs_proj.out_features
    for blk in policy.blocks:
        K = blk.memory_order
        if K > 0:
            caches.append(torch.zeros(batch_size, H, K, device=device, dtype=dtype))
        else:
            caches.append(torch.empty(batch_size, H, 0, device=device, dtype=dtype))
    return caches
