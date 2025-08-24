from __future__ import annotations
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------
# Causal FSMN block (stateless)
# ------------------------------
class CausalFSMNBlock(nn.Module):
    def __init__(self, hidden_dim: int, memory_order: int = 8, use_layernorm: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.memory_order = memory_order
        self.proj = nn.Linear(hidden_dim, hidden_dim)

        # taps 初始化（与新模型一致：小 std 正态）
        if memory_order > 0:
            self.taps = nn.Parameter(torch.zeros(hidden_dim, memory_order))
            nn.init.normal_(self.taps, mean=0.0, std=0.01)
        else:
            self.taps = nn.Parameter(torch.empty(hidden_dim, 0))

        self.ln = nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity()
        # 可选：Dropout
        # self.dropout = nn.Dropout(0.1)

    def _build_dw_kernel(self, dtype, device):
        # depthwise kernel: [H, 1, K+1]，当前帧权重固定为 1
        H, K = self.hidden_dim, self.memory_order
        w = torch.zeros(H, 1, K + 1, dtype=dtype, device=device)
        w[:, 0, 0] = 1.0
        if K > 0:
            w[:, 0, 1:] = self.taps
        return w

    def forward_sequence(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B, T, H]；严格因果：仅左侧 pad
        B, T, H = h.shape
        assert H == self.hidden_dim
        K = self.memory_order

        residual = h
        x = self.proj(h).transpose(1, 2)  # [B, H, T]
        if K > 0:
            x = F.pad(x, (K, 0))  # 左侧 pad K
        w = self._build_dw_kernel(dtype=x.dtype, device=x.device)
        y = F.conv1d(x, w, stride=1, padding=0, groups=H).transpose(1, 2)  # [B, T, H]
        y = self.ln(y + residual)
        # y = self.dropout(y)
        return y

    def forward_chunk(self, h_chunk: torch.Tensor, cache: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # h_chunk: [B, T, H]；cache: [B, H, K] 或 None（由调用方管理）
        B, T, H = h_chunk.shape
        assert H == self.hidden_dim
        K = self.memory_order

        residual = h_chunk
        x = self.proj(h_chunk).transpose(1, 2)  # [B, H, T]
        w = self._build_dw_kernel(dtype=x.dtype, device=x.device)

        if K > 0:
            if cache is None:
                cache = torch.zeros(B, H, K, device=x.device, dtype=x.dtype)
            x_cat = torch.cat([cache, x], dim=2)  # 真实历史放前面
            y = F.conv1d(x_cat, w, stride=1, padding=0, groups=H)  # [B, H, T]
            new_cache = x_cat[:, :, -K:]
        else:
            y = F.conv1d(x, w, stride=1, padding=0, groups=H)  # [B, H, T]
            new_cache = x.new_zeros(B, H, 0)

        y = y.transpose(1, 2)  # [B, T, H]
        y = self.ln(y + residual)
        return y, new_cache

    def forward_step(self, h_t: torch.Tensor, cache: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # h_t: [B, H]；单步；与 forward_chunk 语义一致
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
        num_blocks: int = 4,
        action_dim: int = 2,
    ):
        super().__init__()
        self.obs_proj = nn.Linear(obs_dim, hidden_dim)

        # 新的初始化（与您新版一致）
        nn.init.orthogonal_(self.obs_proj.weight)
        nn.init.zeros_(self.obs_proj.bias)

        self.blocks = nn.ModuleList([CausalFSMNBlock(hidden_dim, memory_order) for _ in range(num_blocks)])

        self.act_head = nn.Linear(hidden_dim, action_dim)
        self.val_head = nn.Linear(hidden_dim, 1)

        nn.init.orthogonal_(self.act_head.weight, gain=0.01)
        nn.init.zeros_(self.act_head.bias)
        nn.init.orthogonal_(self.val_head.weight)
        nn.init.zeros_(self.val_head.bias)

    def forward(self, obs_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # obs_seq: [B, T, D] 或 [B, D]；严格因果的全序列输出
        if obs_seq.dim() == 2:
            obs_seq = obs_seq.unsqueeze(1)
        x = torch.tanh(self.obs_proj(obs_seq))  # 新版：tanh
        for blk in self.blocks:
            x = blk.forward_sequence(x)  # 块内已含残差+LN
        logits = self.act_head(x)           # [B, T, A]
        value = self.val_head(x).squeeze(-1)  # [B, T]
        return logits, value

    def forward_chunk(
        self,
        obs_chunk: torch.Tensor,
        caches: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        # obs_chunk: [B, T, D] 或 [B, D]；caches: len == num_blocks，每个 [B, H, K]
        if obs_chunk.dim() == 2:
            obs_chunk = obs_chunk.unsqueeze(1)
        x = torch.tanh(self.obs_proj(obs_chunk))  # 与 forward 对齐
        new_caches: List[torch.Tensor] = []
        if caches is None:
            caches = [None] * len(self.blocks)
        for blk, cache in zip(self.blocks, caches):
            x, cache = blk.forward_chunk(x, cache)  # 块内已做残差+LN
            new_caches.append(cache)
        logits = self.act_head(x)
        value = self.val_head(x).squeeze(-1)
        return logits, value, new_caches

    def step(
        self,
        obs_t: torch.Tensor,
        caches: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        # obs_t: [B, D]；单步；与 forward 对齐
        if obs_t.dim() != 2:
            obs_t = obs_t.view(obs_t.size(0), -1)
        x = torch.tanh(self.obs_proj(obs_t))
        new_caches: List[torch.Tensor] = []
        if caches is None:
            caches = [None] * len(self.blocks)
        for blk, cache in zip(self.blocks, caches):
            x, cache = blk.forward_step(x, cache)  # 块内已做残差+LN
            new_caches.append(cache)
        logits = self.act_head(x)         # [B, A]
        value = self.val_head(x).squeeze(-1)  # [B]
        return logits, value, new_caches


# ------------------------------
# Helper (unchanged)
# ------------------------------
@torch.no_grad()
def fsmn_allocate_caches(
    policy: FSMNPolicy,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> List[torch.Tensor]:
    caches: List[torch.Tensor] = []
    H = policy.obs_proj.out_features
    for blk in policy.blocks:
        K = blk.memory_order
        if K > 0:
            caches.append(torch.zeros(batch_size, H, K, device=device, dtype=dtype))
        else:
            caches.append(torch.empty(batch_size, H, 0, device=device, dtype=dtype))
    return caches
