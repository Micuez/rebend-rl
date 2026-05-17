from __future__ import annotations

import random
from typing import Dict, List, Sequence, Tuple

import torch
from torch import nn
from torch.distributions import Categorical

from .environment import RepairEnvironment, Transition
from .policies import PolicyValueNet, make_rollout_batch, select_action


def train_ppo(
    model: PolicyValueNet,
    env_factory,
    device: torch.device,
    config: Dict,
) -> Dict[str, List[float]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["policy_lr"])
    history = {"episode_return": [], "final_objective": []}

    for _ in range(config["epochs"]):
        transitions: List[Transition] = []
        for episode in range(config["episodes_per_epoch"]):
            env: RepairEnvironment = env_factory()
            state = env.reset()
            done = False
            episode_return = 0.0
            while not done:
                action, log_prob, value = select_action(model, state, device)
                next_state, reward, done, _ = env.step(action)
                transitions.append(
                    Transition(
                        state=state,
                        action=action,
                        log_prob=log_prob,
                        reward=reward,
                        done=done,
                        value=value,
                    )
                )
                episode_return += reward
                state = next_state
            history["episode_return"].append(episode_return)
            history["final_objective"].append(env.metrics["objective"])

        batch = make_rollout_batch(
            transitions, config["gamma"], config["gae_lambda"], device
        )
        batch_size = batch.states.shape[0]
        indices = list(range(batch_size))
        for _ in range(config["ppo_epochs"]):
            random.shuffle(indices)
            for start in range(0, batch_size, config["minibatch_size"]):
                mb = indices[start : start + config["minibatch_size"]]
                states = batch.states[mb]
                actions = batch.actions[mb]
                old_log_probs = batch.log_probs[mb]
                returns = batch.returns[mb]
                advantages = batch.advantages[mb]
                logits, values = model(states)
                dist = Categorical(logits=logits)
                log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()
                ratio = torch.exp(log_probs - old_log_probs)
                unclipped = ratio * advantages
                clipped = torch.clamp(
                    ratio, 1 - config["clip_ratio"], 1 + config["clip_ratio"]
                ) * advantages
                policy_loss = -torch.min(unclipped, clipped).mean() - config["entropy_coef"] * entropy
                value_loss = nn.functional.mse_loss(values, returns) * config["value_coef"]
                loss = policy_loss + value_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
                optimizer.step()
    return history
