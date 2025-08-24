from __future__ import annotations

from collections import deque
from typing import Deque, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FSMNBlock(nn.Module):
    def __init__(
        self, hidden_dim: int, memory_order: int = 8, use_layernorm: bool = True
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.memory_order = memory_order

        # 线性前馈投影（可选），这里用恒等映射，留一个位置方便扩展
        self.proj = nn.Identity()

        # FSMN 记忆卷积的可学习系数（不包含当前时刻的权重）
        # 形状 [hidden_dim, memory_order]，每个通道一组系数
        self.taps = nn.Parameter(torch.zeros(hidden_dim, memory_order))

        # 稳定训练的层归一化
        self.ln = nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity()

    def forward_sequence(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B, T, H]
        B, T, H = h.shape
        assert H == self.hidden_dim

        # 准备深度可分离的 1D 卷积权重
        # W 的形状为 [H, 1, K+1]，第 0 个位置是恒等映射 1，后面是可学习的记忆系数
        K = self.memory_order
        w = torch.zeros(H, 1, K + 1, device=h.device, dtype=h.dtype)
        w[:, 0, 0] = 1.0  # 当前帧恒等映射
        if K > 0:
            w[:, 0, 1:] = self.taps

        # 只在左侧填充 K 个时间步，确保因果性
        x = self.proj(h)
        x = x.transpose(1, 2)  # [B, H, T]
        x = F.pad(x, (K, 0))  # 左填充

        # 分组卷积实现每个通道独立的记忆滤波
        y = F.conv1d(x, w, bias=None, stride=1, padding=0, groups=H)  # [B, H, T]
        y = y.transpose(1, 2)  # [B, T, H]
        y = self.ln(y)
        return y

    def forward_step(
        self, h_t: torch.Tensor, hist: Deque[torch.Tensor]
    ) -> Tuple[torch.Tensor, Deque[torch.Tensor]]:
        # h_t: [B, H]，单步；hist: 最近 K 个隐藏状态，最旧在左
        H = h_t.shape[-1]
        assert H == self.hidden_dim
        K = self.memory_order

        # 当前帧 + 记忆项的线性组合
        y_t = h_t  # 恒等映射
        if K > 0 and len(hist) > 0:
            # 记忆权重 taps: [H, K]；拼成 [K, H] 便于逐项相乘
            taps = self.taps.t()  # [K, H]
            # 从最近到最远对齐权重：hist[-1] 对应 i=1 的权重
            items: List[torch.Tensor] = list(hist)
            items = items[-K:]
            # 对齐方向：items 从最旧到最新，权重要反向匹配
            for i, prev in enumerate(reversed(items), start=1):
                # taps[i-1]: [H]，逐通道缩放
                y_t = y_t + prev * taps[i - 1]
        y_t = self.ln(y_t)

        # 更新历史队列：加入当前隐藏态，保持长度 K
        hist.append(h_t.detach())
        while len(hist) > K:
            hist.popleft()
        return y_t, hist


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
            [FSMNBlock(hidden_dim, memory_order) for _ in range(num_blocks)]
        )
        self.act_head = nn.Linear(hidden_dim, action_dim)
        self.val_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # obs_seq: [B, T, 4] 或 [B, 4]
        if obs_seq.dim() == 2:
            obs_seq = obs_seq.unsqueeze(1)  # [B, 1, 4]
        x = F.relu(self.obs_proj(obs_seq))  # [B, T, H]
        for blk in self.blocks:
            x = F.relu(blk.forward_sequence(x))
        logits = self.act_head(x)  # [B, T, 2]
        value = self.val_head(x).squeeze(-1)  # [B, T]
        return logits, value

    def step(
        self, obs_t: torch.Tensor, hists: List[Deque[torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Deque[torch.Tensor]]]:
        # obs_t: [B, 4] 单步输入；hists: 每个 Block 一份队列
        x = F.relu(self.obs_proj(obs_t))  # [B, H]
        new_hists: List[Deque[torch.Tensor]] = []
        for blk, hist in zip(self.blocks, hists):
            x, hist = blk.forward_step(x, hist)
            x = F.relu(x)
            new_hists.append(hist)
        logits = self.act_head(x)  # [B, 2]
        value = self.val_head(x).squeeze(-1)  # [B]
        return logits, value, new_hists

    def init_hists(
        self, device: torch.device | None = None
    ) -> List[Deque[torch.Tensor]]:
        # 为每个 Block 建立一个空的历史队列
        return [deque([], maxlen=blk.memory_order) for blk in self.blocks]
