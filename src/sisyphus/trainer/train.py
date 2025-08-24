import os
import time
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import torch
import torchmetrics
from lightning.fabric import Fabric
from lightning.pytorch.loggers import WandbLogger
from torch import Tensor
from torch.utils.data import BatchSampler, DistributedSampler, RandomSampler
from tqdm import trange
from typing import Dict

from sisyphus.tasks.fsmn_agent import PPOLightningAgent
from sisyphus.utils import linear_annealing, make_env
from  einops import rearrange


def flatten_data(
    obs,
    logprobs,
    actions,
    advantages,
    returns,
    values,
    context_len:int = 9
) -> Dict[str, Tensor]:
    obs = rearrange(obs, "t b a -> b t a")
    obs = obs.unfold(1, context_len, 1) # [b, t, a] -> [b, t, a, c]
    obs = rearrange(obs, "b t a c -> (b t) c a")
    
    logprobs = logprobs.T
    logprobs = logprobs.unfold(1, context_len, 1)
    logprobs = rearrange(logprobs, "b t c -> (b t) c")

    actions = actions.T
    actions = actions.unfold(1, context_len, 1)
    actions = rearrange(actions, "b t c -> (b t) c")

    values = values.T
    values = values.unfold(1, context_len, 1)
    values = rearrange(values, "b t c -> (b t) c")

    advantages = advantages.T
    advantages = advantages.unfold(1, context_len, 1)
    advantages = rearrange(advantages, "b t c -> (b t) c")

    returns = returns.T
    returns = returns.unfold(1, context_len, 1)
    returns = rearrange(returns, "b t c -> (b t) c")

    return {
        "obs": obs,
        "logprobs": logprobs,
        "actions": actions,
        "advantages": advantages,
        "returns": returns,
        "values": values,
    }


def train(
    fabric: Fabric,
    agent: PPOLightningAgent,
    optimizer: torch.optim.Optimizer,
    data: dict[str, Tensor],
    global_step: int,
    args,
):
    indexes = list(range(data["obs"].shape[0]))
    if args.share_data:
        sampler = DistributedSampler(
            indexes,
            num_replicas=fabric.world_size,
            rank=fabric.global_rank,
            shuffle=True,
            seed=args.seed,
        )
    else:
        sampler = RandomSampler(indexes)
    sampler = BatchSampler(
        sampler, batch_size=args.per_rank_batch_size, drop_last=False
    )

    for epoch in range(args.update_epochs):
        if args.share_data:
            sampler.sampler.set_epoch(epoch)
        for batch_idxes in sampler:
            loss = agent.training_step({k: v[batch_idxes] for k, v in data.items()})
            optimizer.zero_grad(set_to_none=True)
            fabric.backward(loss)
            fabric.clip_gradients(agent, optimizer, max_norm=args.max_grad_norm)
            optimizer.step()
        agent.on_train_epoch_end(global_step)


def main(args):
    run_name = f"{args.env_id}_{args.exp_name}_{args.seed}_{int(time.time())}"
    log_save_dir = Path(
        "outputs", "logs", "fabric_logs", datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    )
    log_save_dir.mkdir(parents=True, exist_ok=True)
    logger = WandbLogger(
        save_dir=log_save_dir,
        name=run_name,
        project="sisyphus"
    )

    # Initialize Fabric
    fabric = Fabric(loggers=logger)
    fabric.launch()
    rank = fabric.global_rank
    world_size = fabric.world_size
    device = fabric.device
    fabric.seed_everything(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Environment setup
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                args.env_id,
                args.seed + rank * args.num_envs + i,
                rank,
                args.capture_video,
                "outputs",
                "train",
            )
            for i in range(args.num_envs)
        ]
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"

    # Define the agent and the optimizer and setup them with Fabric
    agent: PPOLightningAgent = PPOLightningAgent(
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        clip_coef=args.clip_coef,
        clip_vloss=args.clip_vloss,
        normalize_advantages=args.normalize_advantages,
    )
    optimizer = agent.configure_optimizers(args.learning_rate)
    agent, optimizer = fabric.setup(agent, optimizer)

    # Player metrics
    rew_avg = torchmetrics.MeanMetric().to(device)
    ep_len_avg = torchmetrics.MeanMetric().to(device)

    # Local data
    obs = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape,
        device=device,
    )
    actions = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape, device=device
    )
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)

    # Global variables
    global_step = 0
    start_time = time.time()
    single_global_rollout = int(args.num_envs * args.num_steps * world_size)
    num_updates = args.total_timesteps // single_global_rollout

    # Get the first environment observation and start the optimization
    next_obs = torch.tensor(envs.reset(seed=args.seed)[0], device=device)
    next_done = torch.zeros(args.num_envs, device=device)
    episode_bar = trange(1, num_updates + 1, leave=False)
    for update in episode_bar:
        # Learning rate annealing
        if args.anneal_lr:
            linear_annealing(optimizer, update, num_updates, args.learning_rate)
        fabric.log("Info/learning_rate", optimizer.param_groups[0]["lr"], global_step)

        for step in trange(0, args.num_steps, leave=False):
            global_step += args.num_envs * world_size
            obs[step] = next_obs
            dones[step] = next_done

            # Sample an action given the observation received by the environment
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # Single environment step
            next_obs_np, reward_np, terminated_np, truncated_np, info = envs.step(
                action.cpu().numpy()
            )

            # Convert to tensors on the right device/dtype
            reward_t = torch.as_tensor(reward_np, device=device, dtype=torch.float32)
            term_t = torch.as_tensor(terminated_np, device=device, dtype=torch.bool)
            trunc_t = torch.as_tensor(truncated_np, device=device, dtype=torch.bool)
            done_t = term_t | trunc_t

            rewards[step] = reward_t.view(-1)
            next_obs = torch.as_tensor(next_obs_np, device=device)
            next_done = done_t

            # ---- Robust episode-statistics handling (vectorized info) ----
            if info:
                # Case 1: aggregated dict with per-env arrays (Gymnasium + RecordEpisodeStatistics)
                ep = info.get("episode", {})
                mask = info.get("_episode", ep.get("_l", ep.get("_r")))
                if mask is not None:
                    finished = (
                        torch.as_tensor(mask).nonzero(as_tuple=False).flatten().tolist()
                    )
                    for i in finished:
                        try:
                            r_i = float(ep.get("r", [0.0] * args.num_envs)[i])
                            l_i = int(ep.get("l", [0] * args.num_envs)[i])
                        except Exception:
                            # Fallback keys if a custom wrapper renamed them
                            r_i = float(ep.get("reward", [0.0] * args.num_envs)[i])
                            l_i = int(ep.get("length", [0] * args.num_envs)[i])
                        rew_avg(r_i)
                        ep_len_avg(l_i)

        # Sync the metrics
        rew_avg_reduced = rew_avg.compute()
        if not torch.isnan(rew_avg_reduced):
            fabric.log("Rewards/rew_avg", rew_avg_reduced, global_step)
            episode_bar.set_postfix(rew_avg=rew_avg_reduced.item())
        ep_len_avg_reduced = ep_len_avg.compute()
        if not torch.isnan(ep_len_avg_reduced):
            fabric.log("Game/ep_len_avg", ep_len_avg_reduced, global_step)
        rew_avg.reset()
        ep_len_avg.reset()

        # Estimate returns with GAE (https://arxiv.org/abs/1506.02438)
        returns, advantages = agent.estimate_returns_and_advantages(
            rewards,
            values,
            dones,
            next_obs,
            next_done,
            args.num_steps,
            args.gamma,
            args.gae_lambda,
        )

        # # Flatten the batch
        # local_data = {
        #     "obs": obs.reshape((-1,) + envs.single_observation_space.shape),
        #     "logprobs": logprobs.reshape(-1),
        #     "actions": actions.reshape((-1,) + envs.single_action_space.shape),
        #     "advantages": advantages.reshape(-1),
        #     "returns": returns.reshape(-1),
        #     "values": values.reshape(-1),
        # }
        # Flatten the batch
        # local_data = {
        #     "obs": obs.reshape((-1,) + envs.single_observation_space.shape),
        #     "logprobs": logprobs.reshape(-1),
        #     "actions": actions.reshape((-1,) + envs.single_action_space.shape),
        #     "advantages": advantages.reshape(-1),
        #     "returns": returns.reshape(-1),
        #     "values": values.reshape(-1),
        # }
        local_data = flatten_data(obs, logprobs, actions, advantages, returns, values)
        if args.share_data:
            # Gather all the tensors from all the world and reshape them
            gathered_data = fabric.all_gather(local_data)
            for k, v in gathered_data.items():
                if k == "obs":
                    gathered_data[k] = v.reshape(
                        (-1,) + envs.single_observation_space.shape
                    )
                elif k == "actions":
                    gathered_data[k] = v.reshape((-1,) + envs.single_action_space.shape)
                else:
                    gathered_data[k] = v.reshape(-1)
        else:
            gathered_data = local_data

        # Train the agent
        train(fabric, agent, optimizer, gathered_data, global_step, args)
        fabric.log(
            "Time/step_per_second",
            int(global_step / (time.time() - start_time)),
            global_step,
        )
    envs.close()
