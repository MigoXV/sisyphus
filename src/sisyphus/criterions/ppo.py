from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class PPOLoss(nn.Module):
    """PPO (Proximal Policy Optimization) loss implementation."""
    
    def __init__(
        self,
        clip_epsilon: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        normalize_advantage: bool = True,
    ):
        super().__init__()
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.normalize_advantage = normalize_advantage
    
    def forward(
        self,
        logits: torch.Tensor,  # [B, T, A] or [B, A]
        old_logits: torch.Tensor,  # [B, T, A] or [B, A]
        actions: torch.Tensor,  # [B, T] or [B]
        values: torch.Tensor,  # [B, T] or [B]
        returns: torch.Tensor,  # [B, T] or [B]
        advantages: torch.Tensor,  # [B, T] or [B]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute PPO loss components.
        
        Args:
            logits: Current policy logits
            old_logits: Old policy logits (detached)
            actions: Actions taken
            values: Estimated values
            returns: Computed returns
            advantages: Computed advantages
            
        Returns:
            Dictionary containing loss components
        """
        # Ensure consistent shapes
        if logits.dim() == 3 and actions.dim() == 2:
            # Sequence format: [B, T, ...]
            B, T = actions.shape
            logits = logits.view(-1, logits.size(-1))  # [B*T, A]
            old_logits = old_logits.view(-1, old_logits.size(-1))  # [B*T, A]
            actions = actions.view(-1)  # [B*T]
            values = values.view(-1)  # [B*T]
            returns = returns.view(-1)  # [B*T]
            advantages = advantages.view(-1)  # [B*T]
        
        # Normalize advantages
        if self.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert logits to log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        old_log_probs = F.log_softmax(old_logits, dim=-1)
        
        # Get log probabilities for taken actions
        log_prob_actions = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        old_log_prob_actions = old_log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        
        # Compute probability ratio
        ratio = torch.exp(log_prob_actions - old_log_prob_actions)
        
        # Clipped surrogate objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss (MSE)
        value_loss = F.mse_loss(values, returns)
        
        # Entropy bonus
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        entropy_loss = -self.entropy_coef * entropy
        
        # Total loss
        total_loss = policy_loss + self.value_loss_coef * value_loss + entropy_loss
        
        # Compute additional metrics
        with torch.no_grad():
            approx_kl = (old_log_prob_actions - log_prob_actions).mean()
            clipfrac = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean()
            explained_var = 1.0 - F.mse_loss(values, returns) / returns.var()
        
        return {
            'loss': total_loss,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy_loss': entropy_loss,
            'entropy': entropy,
            'approx_kl': approx_kl,
            'clipfrac': clipfrac,
            'explained_variance': explained_var,
        }
