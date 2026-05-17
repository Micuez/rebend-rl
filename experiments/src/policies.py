from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


class PolicyValueNet(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor):
        hidden = self.shared(x)
        return self.policy_head(hidden), self.value_head(hidden).squeeze(-1)


@dataclass
class RolloutBatch:
    states: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor


def select_action(model: PolicyValueNet, state: np.ndarray, device: torch.device):
    tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    logits, value = model(tensor)
    dist = Categorical(logits=logits)
    action = dist.sample()
    return (
        int(action.item()),
        float(dist.log_prob(action).item()),
        float(value.item()),
    )


def greedy_action(model: PolicyValueNet, state: np.ndarray, device: torch.device) -> int:
    tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    logits, _ = model(tensor)
    return int(torch.argmax(logits, dim=-1).item())


def compute_gae(
    rewards: Sequence[float],
    values: Sequence[float],
    dones: Sequence[bool],
    gamma: float,
    gae_lambda: float,
) -> List[float]:
    advantages = [0.0] * len(rewards)
    last_adv = 0.0
    next_value = 0.0
    for index in reversed(range(len(rewards))):
        mask = 0.0 if dones[index] else 1.0
        delta = rewards[index] + gamma * next_value * mask - values[index]
        last_adv = delta + gamma * gae_lambda * mask * last_adv
        advantages[index] = last_adv
        next_value = values[index]
    return advantages


def make_rollout_batch(transitions, gamma: float, gae_lambda: float, device: torch.device) -> RolloutBatch:
    rewards = [item.reward for item in transitions]
    values = [item.value for item in transitions]
    dones = [item.done for item in transitions]
    advantages = compute_gae(rewards, values, dones, gamma, gae_lambda)
    returns = [adv + val for adv, val in zip(advantages, values)]
    adv_tensor = torch.tensor(advantages, dtype=torch.float32, device=device)
    adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)
    return RolloutBatch(
        states=torch.tensor(np.array([item.state for item in transitions]), dtype=torch.float32, device=device),
        actions=torch.tensor([item.action for item in transitions], dtype=torch.int64, device=device),
        log_probs=torch.tensor([item.log_prob for item in transitions], dtype=torch.float32, device=device),
        returns=torch.tensor(returns, dtype=torch.float32, device=device),
        advantages=adv_tensor,
    )
