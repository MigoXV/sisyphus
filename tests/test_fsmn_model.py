from __future__ import annotations

from typing import List, Tuple

import gymnasium as gym
import torch
import torch.nn.functional as F
from tqdm import tqdm, trange

from sisyphus.models.fsmn import FSMNPolicy, fsmn_allocate_caches

# ===== 采样工具 =====


def select_action_from_logits(
    logits: torch.Tensor, deterministic: bool = False
) -> torch.Tensor:
    # logits: [B, 2] 或 [B, T, 2]
    if logits.dim() == 3:
        logits = logits[:, -1, :]  # 仅取最后一步
    if deterministic:
        return torch.argmax(logits, dim=-1)
    probs = F.softmax(logits, dim=-1)
    dist = torch.distributions.Categorical(probs=probs)
    return dist.sample()


# ===== 无状态单环境 rollout（使用外部缓存） =====


def run_random_rollout(
    env_name: str = "CartPole-v1", episodes: int = 1, deterministic: bool = False
):
    env = gym.make(env_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FSMNPolicy().to(device)

    # 外部缓存（caller-owned）：每层一个 [B, H, K]，这里 B=1
    caches = fsmn_allocate_caches(
        model, batch_size=1, device=device, dtype=torch.float32
    )

    for ep in trange(episodes, desc="episodes", leave=False):
        obs, _ = env.reset()
        done = False
        total_r = 0.0
        steps = 0
        # 每回合开始时清空缓存
        for c in caches:
            c.zero_()
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(
                0
            )  # [1, 4]
            with torch.no_grad():
                logits, _, caches = model.step(obs_t, caches)
                action = select_action_from_logits(
                    logits, deterministic=deterministic
                ).item()
            obs, reward, terminated, truncated, _ = env.step(action)
            total_r += reward
            steps += 1
            done = terminated or truncated
        tqdm.write(f"ep={ep} return={total_r:.1f} steps={steps}")

    env.close()


# ===== 向量化 batch rollout（使用外部缓存） =====


class PolicyValueOutput:
    def __init__(self, logits: torch.Tensor, value: torch.Tensor):
        self.logits = logits  # [B, T, 2]
        self.value = value  # [B, T]


def rollout_batch(
    model: FSMNPolicy,
    env_name: str,
    batch_size: int = 8,
    horizon: int = 256,
    deterministic: bool = False,
) -> Tuple[dict, PolicyValueOutput]:
    device = next(model.parameters()).device
    envs = [gym.make(env_name) for _ in range(batch_size)]

    # 轨迹缓冲
    obs_buf = torch.zeros(batch_size, horizon, 4, device=device)
    act_buf = torch.zeros(batch_size, horizon, dtype=torch.long, device=device)
    rew_buf = torch.zeros(batch_size, horizon, device=device)
    done_buf = torch.zeros(batch_size, horizon, device=device)

    # 外部缓存（B=batch_size）
    caches = fsmn_allocate_caches(
        model, batch_size=batch_size, device=device, dtype=obs_buf.dtype
    )

    # 初始化观测
    obs_list = []
    for e in envs:
        o, _ = e.reset()
        obs_list.append(o)

    # 逐步交互
    for t in trange(horizon, desc="rollout", leave=False):
        obs_t = torch.tensor(obs_list, dtype=torch.float32, device=device)  # [B, 4]
        with torch.no_grad():
            logits, _, caches = model.step(obs_t, caches)  # logits: [B, 2]
            a_t = select_action_from_logits(logits, deterministic)

        next_obs_list: List = []
        rewards: List[float] = []
        dones: List[float] = []
        for i, e in enumerate(envs):
            obs, r, terminated, truncated, _ = e.step(int(a_t[i].item()))
            d = terminated or truncated
            if d:
                obs2, _ = e.reset()
                next_obs_list.append(obs2)
            else:
                next_obs_list.append(obs)
            rewards.append(r)
            dones.append(float(d))

        # 记录
        obs_buf[:, t] = obs_t
        act_buf[:, t] = a_t
        rew_buf[:, t] = torch.tensor(rewards, device=device)
        done_buf[:, t] = torch.tensor(dones, device=device)

        # 对于 done 的环境，清零其 batch 维对应的缓存行，避免跨回合泄漏
        if any(dones):
            done_mask = done_buf[:, t].bool()
            for k in range(len(caches)):
                if caches[k].numel() > 0:
                    caches[k][done_mask] = 0.0

        obs_list = next_obs_list

    for e in envs:
        e.close()

    # 全序列因果前向（用于价值、优势估计）
    with torch.no_grad():
        logits_seq, value_seq = model(obs_buf)  # [B, T, 2], [B, T]

    out = PolicyValueOutput(logits_seq, value_seq)
    batch = {
        "obs": obs_buf,  # [B, T, 4]
        "act": act_buf,  # [B, T]
        "rew": rew_buf,  # [B, T]
        "done": done_buf,  # [B, T]
    }
    return batch, out


if __name__ == "__main__":
    # 简单验证：随机权重 + 外部缓存
    run_random_rollout(env_name="CartPole-v1", episodes=3, deterministic=False)
