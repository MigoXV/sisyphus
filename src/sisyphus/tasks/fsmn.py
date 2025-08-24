from __future__ import annotations

import gymnasium as gym
import lightning as L
import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyMemmapStorage
from torchrl.envs import GymEnv, ParallelEnv, TransformedEnv
from torchrl.envs.transforms import StepCounter, ToTensorImage, Compose
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.objectives import ClipPPOLoss
from typing import Any, Dict, List, Optional

from ..criterions.ppo import PPOLoss
from ..models.actor_critic import ActorCriticWrapper


class FSMNCartPolePPOModule(L.LightningModule):
    """
    PyTorch Lightning module for training CartPole-v1 with FSMN policy using PPO and TorchRL collector.
    """
    
    def __init__(
        self,
        # Model parameters
        obs_dim: int = 4,
        hidden_dim: int = 128,
        memory_order: int = 8,
        num_blocks: int = 2,
        action_dim: int = 2,
        
        # Environment parameters
        num_envs: int = 4,
        max_episode_steps: int = 500,
        
        # Training parameters
        frames_per_batch: int = 1000,
        rollout_length: int = 200,
        num_epochs: int = 10,
        mini_batch_size: int = 64,
        
        # PPO parameters
        clip_epsilon: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        gae_lambda: float = 0.95,
        gamma: float = 0.99,
        
        # Optimizer parameters
        lr: float = 3e-4,
        weight_decay: float = 1e-5,
        
        # Logging
        log_interval: int = 10,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Model
        self.actor_critic = ActorCriticWrapper(
            obs_dim=obs_dim,
            hidden_dim=hidden_dim,
            memory_order=memory_order,
            num_blocks=num_blocks,
            action_dim=action_dim,
        )
        
        # Loss function
        self.ppo_loss = PPOLoss(
            clip_epsilon=clip_epsilon,
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef,
            normalize_advantage=True,
        )
        
        # Store hyperparameters
        self.num_envs = num_envs
        self.frames_per_batch = frames_per_batch
        self.rollout_length = rollout_length
        self.num_epochs = num_epochs
        self.mini_batch_size = mini_batch_size
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.log_interval = log_interval
        
        # Environment and collector will be created in setup()
        self.env = None
        self.collector = None
        self.replay_buffer = None
        
        # Cache management
        self.env_caches = None
        
    def setup(self, stage: str) -> None:
        """Setup environments and data collector."""
        if stage == "fit":
            # Create environment
            def make_env():
                env = GymEnv("CartPole-v1")
                env = TransformedEnv(env, Compose(
                    StepCounter(max_steps=self.hparams.max_episode_steps),
                ))
                return env
            
            # Create parallel environment
            self.env = ParallelEnv(
                num_workers=self.num_envs,
                create_env_fn=make_env,
            )
            
            # Initialize environment caches
            self.env_caches = self.actor_critic.allocate_caches(
                batch_size=self.num_envs,
                device=self.device,
                dtype=torch.float32,
            )
            
            # Create collector
            policy = self._create_policy_module()
            self.collector = SyncDataCollector(
                env=self.env,
                policy=policy,
                frames_per_batch=self.frames_per_batch,
                max_frames_per_traj=self.rollout_length,
                total_frames=-1,  # Infinite collection
                device=self.device,
            )
            
            # Create replay buffer for mini-batch sampling
            self.replay_buffer = ReplayBuffer(
                storage=LazyMemmapStorage(self.frames_per_batch),
                sampler=SamplerWithoutReplacement(),
                batch_size=self.mini_batch_size,
            )
    
    def _create_policy_module(self):
        """Create policy module for TorchRL collector."""
        class PolicyModule(torch.nn.Module):
            def __init__(self, actor_critic, caches):
                super().__init__()
                self.actor_critic = actor_critic
                self.caches = caches
                
            def forward(self, tensordict):
                obs = tensordict["observation"]
                
                # Get action logits and values
                logits, values, new_caches = self.actor_critic.get_action_value(
                    obs, self.caches
                )
                self.caches = new_caches
                
                # Sample action
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                
                # Update tensordict
                tensordict.update({
                    "action": action,
                    "sample_log_prob": log_prob,
                    "state_value": values,
                    "logits": logits,
                })
                
                return tensordict
        
        return PolicyModule(self.actor_critic, self.env_caches)
    
    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Training step - not used as we handle training in on_train_epoch_start."""
        # Return a dummy loss to satisfy Lightning's requirements
        return torch.tensor(0.0, device=self.device, requires_grad=True)
    
    def on_train_epoch_start(self) -> None:
        """Collect data and train for one epoch."""
        # Set manual optimization
        self.automatic_optimization = False
        
        # Collect rollout data
        try:
            rollout_data = next(iter(self.collector))
        except Exception as e:
            self.log("train/collector_error", 1.0)
            print(f"Collector error: {e}")
            return
        
        # Compute advantages and returns
        rollout_data = self._compute_gae(rollout_data)
        
        # Clear and refill replay buffer
        if hasattr(self.replay_buffer._storage, '_storage'):
            self.replay_buffer._storage._storage.clear()
        self.replay_buffer.extend(rollout_data)
        
        # Train for multiple epochs on collected data
        total_loss = 0.0
        total_samples = 0
        optimizer = self.optimizers()
        
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            epoch_samples = 0
            mini_batch_idx = 0
            
            # Sample mini-batches and train
            for mini_batch in self.replay_buffer:
                try:
                    loss_dict = self._compute_loss(mini_batch)
                    loss = loss_dict['loss']
                    
                    # Zero gradients
                    optimizer.zero_grad()
                    
                    # Backward pass
                    self.manual_backward(loss)
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
                    
                    # Optimizer step
                    optimizer.step()
                    
                    # Accumulate metrics
                    batch_size = mini_batch.batch_size[0] if hasattr(mini_batch, 'batch_size') else len(mini_batch)
                    epoch_loss += loss.item() * batch_size
                    epoch_samples += batch_size
                    
                    # Log detailed metrics
                    if mini_batch_idx % self.log_interval == 0:
                        for key, value in loss_dict.items():
                            if isinstance(value, torch.Tensor):
                                self.log(f"train/{key}", value.item(), prog_bar=(key in ['loss', 'policy_loss']))
                            else:
                                self.log(f"train/{key}", value, prog_bar=(key in ['loss', 'policy_loss']))
                    
                    mini_batch_idx += 1
                    
                except Exception as e:
                    print(f"Training step error: {e}")
                    continue
            
            total_loss += epoch_loss
            total_samples += epoch_samples
        
        # Log epoch metrics
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        self.log("train/epoch_loss", avg_loss, prog_bar=True)
        
        # Log environment metrics
        self._log_env_metrics(rollout_data)
    
    def _log_env_metrics(self, rollout_data: TensorDict) -> None:
        """Log environment-related metrics."""
        # Log episode rewards if available
        if "episode_reward" in rollout_data.keys():
            episode_rewards = rollout_data["episode_reward"]
            if episode_rewards.numel() > 0:
                # Handle different tensor shapes
                if episode_rewards.dim() > 1:
                    episode_rewards = episode_rewards.flatten()
                
                # Filter out zero rewards (incomplete episodes)
                non_zero_rewards = episode_rewards[episode_rewards != 0]
                if len(non_zero_rewards) > 0:
                    self.log("train/episode_reward_mean", non_zero_rewards.float().mean(), prog_bar=True)
                    self.log("train/episode_reward_std", non_zero_rewards.float().std())
                    self.log("train/episode_reward_max", non_zero_rewards.float().max())
                    self.log("train/episode_reward_min", non_zero_rewards.float().min())
        
        # Log step rewards
        if "reward" in rollout_data.keys():
            step_rewards = rollout_data["reward"]
            if step_rewards.numel() > 0:
                self.log("train/step_reward_mean", step_rewards.float().mean())
                self.log("train/step_reward_sum", step_rewards.float().sum())
        
        # Log episode lengths if available
        if "step_count" in rollout_data.keys():
            step_counts = rollout_data["step_count"]
            if step_counts.numel() > 0:
                max_steps = step_counts.max()
                if max_steps > 0:
                    self.log("train/episode_length_max", max_steps.float())
                    self.log("train/episode_length_mean", step_counts.float().mean())
    
    def _compute_gae(self, rollout_data: TensorDict) -> TensorDict:
        """Compute Generalized Advantage Estimation."""
        rewards = rollout_data["reward"]
        values = rollout_data["state_value"]
        dones = rollout_data.get("done", torch.zeros_like(rewards))
        
        # Handle different tensor shapes
        if rewards.dim() == 1:
            # Flatten format: reshape to [num_envs, steps_per_env]
            steps_per_env = rewards.shape[0] // self.num_envs
            rewards = rewards.view(self.num_envs, steps_per_env)
            values = values.view(self.num_envs, steps_per_env)
            dones = dones.view(self.num_envs, steps_per_env)
        
        # Compute next values (for bootstrap)
        # For the last step, use current value if not done, 0 if done
        next_values = torch.zeros_like(values)
        next_values[..., :-1] = values[..., 1:]
        next_values[..., -1] = values[..., -1] * (1 - dones[..., -1])
        next_values = next_values * (1 - dones.float())
        
        # Compute TD errors
        td_errors = rewards + self.gamma * next_values - values
        
        # Compute GAE
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros(rewards.shape[0], device=rewards.device)  # [num_envs]
        
        for t in reversed(range(rewards.shape[-1])):
            # Reset GAE for terminated episodes
            gae = gae * (1 - dones[..., t])
            gae = td_errors[..., t] + self.gamma * self.gae_lambda * gae
            advantages[..., t] = gae
        
        # Compute returns
        returns = advantages + values
        
        # Flatten back if needed
        if rollout_data["reward"].dim() == 1:
            advantages = advantages.view(-1)
            returns = returns.view(-1)
        
        # Add to tensordict
        rollout_data.update({
            "advantage": advantages,
            "return": returns,
        })
        
        return rollout_data
    
    def _compute_loss(self, batch: TensorDict) -> Dict[str, torch.Tensor]:
        """Compute PPO loss."""
        # Get old policy outputs (detached)
        old_logits = batch["logits"].detach()
        old_values = batch["state_value"].detach()
        
        # Get current policy outputs
        obs = batch["observation"]
        current_logits, current_values = self.actor_critic.fsmn_policy(obs)
        
        # Get other required tensors
        actions = batch["action"]
        returns = batch["return"]
        advantages = batch["advantage"]
        
        # Compute loss
        loss_dict = self.ppo_loss(
            logits=current_logits,
            old_logits=old_logits,
            actions=actions,
            values=current_values,
            returns=returns,
            advantages=advantages,
        )
        
        return loss_dict
    
    def configure_optimizers(self):
        """Configure optimizer."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer
    
    def on_train_epoch_end(self) -> None:
        """Called at the end of training epoch."""
        # Clear replay buffer for next epoch
        if self.replay_buffer is not None:
            self.replay_buffer._storage._storage.clear()