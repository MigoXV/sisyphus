from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Deque, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from sisyphus.models.fsmn import FSMNPolicy

# ===== 简易测试与示例推理 =====


def select_action_from_logits(
    logits: torch.Tensor, deterministic: bool = False
) -> torch.Tensor:
    # logits: [B, 2] 或 [B, T, 2]
    if logits.dim() == 3:
        logits = logits[:, -1, :]  # 取最后一个时间步
    if deterministic:
        return torch.argmax(logits, dim=-1)
    probs = F.softmax(logits, dim=-1)
    dist = torch.distributions.Categorical(probs=probs)
    return dist.sample()


def run_random_rollout(
    env_name: str = "CartPole-v1", episodes: int = 1, deterministic: bool = False
):
    import gymnasium as gym
    from tqdm import tqdm, trange

    # 初始化环境与模型
    env = gym.make(env_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FSMNPolicy().to(device)

    # 历史缓冲区
    hists = model.init_hists(device)

    for ep in trange(episodes, desc="episodes", leave=False):
        obs, _ = env.reset()
        done = False
        total_r = 0.0
        steps = 0
        while not done:
            # 准备张量
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                logits, _, hists = model.step(obs_t, hists)
                action = select_action_from_logits(
                    logits, deterministic=deterministic
                ).item()
            obs, reward, terminated, truncated, _ = env.step(action)
            total_r += reward
            steps += 1
            done = terminated or truncated
        tqdm.write(f"ep={ep} return={total_r:.1f} steps={steps}")

    env.close()


# ===== 训练接口（可选：REINFORCE/A2C 可在此处扩展） =====
# 下面仅提供一个占位的训练函数框架，便于后续接入你现有的 PPO/A2C 等训练器


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
    import gymnasium as gym
    from tqdm import trange

    device = next(model.parameters()).device
    envs = [gym.make(env_name) for _ in range(batch_size)]

    # 收集轨迹张量
    obs_buf = torch.zeros(batch_size, horizon, 4, device=device)
    act_buf = torch.zeros(batch_size, horizon, dtype=torch.long, device=device)
    rew_buf = torch.zeros(batch_size, horizon, device=device)
    done_buf = torch.zeros(batch_size, horizon, device=device)

    # 初始化
    obs_list = []
    for e in envs:
        o, _ = e.reset()
        obs_list.append(o)

    # 逐步采样
    for t in trange(horizon, desc="rollout", leave=False):
        obs_t = torch.tensor(obs_list, dtype=torch.float32, device=device)
        logits, value = model(obs_t)  # [B,1,2],[B,1]
        a_t = select_action_from_logits(logits, deterministic)

        # 与环境交互
        next_obs_list = []
        rewards = []
        dones = []
        for i, e in enumerate(envs):
            obs, r, terminated, truncated, _ = e.step(int(a_t[i].item()))
            d = terminated or truncated
            next_obs_list.append(obs if not d else e.reset()[0])
            rewards.append(r)
            dones.append(float(d))

        # 记录
        obs_buf[:, t] = obs_t
        act_buf[:, t] = a_t
        rew_buf[:, t] = torch.tensor(rewards, device=device)
        done_buf[:, t] = torch.tensor(dones, device=device)

        obs_list = next_obs_list

    # 关闭环境
    for e in envs:
        e.close()

    # 重新计算完整序列上的 logits/value（方便后续优势估计）
    logits_seq, value_seq = model(obs_buf)
    out = PolicyValueOutput(logits_seq, value_seq)
    batch = {
        "obs": obs_buf,  # [B,T,4]
        "act": act_buf,  # [B,T]
        "rew": rew_buf,  # [B,T]
        "done": done_buf,  # [B,T]
    }
    return batch, out


if __name__ == "__main__":
    # 简单跑几局，验证前向与 step 无误（随机权重，不会稳定控制）
    run_random_rollout(env_name="CartPole-v1", episodes=3, deterministic=False)
