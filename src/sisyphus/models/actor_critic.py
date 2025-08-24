from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from tensordict import TensorDict

from .fsmn import FSMNPolicy


class ActorCriticWrapper(nn.Module):
    """
    Actor-Critic wrapper for FSMN policy using composition.
    Separates actor and critic functionality while sharing the FSMN backbone.
    """

    def __init__(
        self,
        obs_dim: int = 4,
        hidden_dim: int = 128,
        memory_order: int = 8,
        num_blocks: int = 2,
        action_dim: int = 2,
    ):
        super().__init__()
        # Use composition: create FSMN policy instance
        self.fsmn_policy = FSMNPolicy(
            obs_dim=obs_dim,
            hidden_dim=hidden_dim,
            memory_order=memory_order,
            num_blocks=num_blocks,
            action_dim=action_dim,
        )

        # Store dimensions for convenience
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

    def forward(self, observation: torch.Tensor) -> TensorDict:
        """
        Forward pass for both actor and critic.

        Args:
            observation: Input observations [B, T, obs_dim] or [B, obs_dim]

        Returns:
            TensorDict with logits and state_value
        """
        logits, values = self.fsmn_policy(observation)

        return TensorDict(
            {
                "logits": logits,
                "state_value": values,
            },
            batch_size=observation.shape[:-1],
        )

    def get_action_value(
        self, observation: torch.Tensor, caches: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Get action logits and values for a single step with optional caching.

        Args:
            observation: Input observation [B, obs_dim]
            caches: Optional FSMN caches

        Returns:
            Tuple of (logits, values, new_caches)
        """
        if caches is not None:
            return self.fsmn_policy.step(observation, caches)
        else:
            # Use forward for batch processing
            logits, values = self.fsmn_policy(observation)
            return logits, values, None

    def get_value(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Get only the critic values.

        Args:
            observation: Input observations

        Returns:
            State values
        """
        _, values = self.fsmn_policy(observation)
        return values

    def get_action_distribution(
        self, observation: torch.Tensor
    ) -> torch.distributions.Categorical:
        """
        Get action distribution from actor.

        Args:
            observation: Input observations

        Returns:
            Categorical distribution over actions
        """
        logits, _ = self.fsmn_policy(observation)
        return torch.distributions.Categorical(logits=logits)

    def allocate_caches(
        self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32
    ) -> List[torch.Tensor]:
        """
        Allocate FSMN caches for stateful inference.

        Args:
            batch_size: Batch size
            device: Device to allocate on
            dtype: Data type

        Returns:
            List of cache tensors
        """
        from .fsmn import fsmn_allocate_caches

        return fsmn_allocate_caches(self.fsmn_policy, batch_size, device, dtype)
