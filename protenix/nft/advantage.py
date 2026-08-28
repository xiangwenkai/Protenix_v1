# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reward normalization used by online NFT rollouts."""

from typing import Mapping, Optional

import torch


def compute_grouped_advantages(
    rewards: torch.Tensor,
    group_ids: Optional[torch.Tensor] = None,
    *,
    global_std: bool = True,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Normalize rewards after subtracting the mean of each condition group.

    Args:
        rewards: One scalar reward per generated structure, shape ``[N]``.
        group_ids: Integer condition identifiers, shape ``[N]``. If omitted,
            all rewards are treated as samples from one condition.
        global_std: Divide by the global standard deviation when true, or by
            the standard deviation within each condition otherwise.
        eps: Numerical stability constant.

    Returns:
        Normalized advantages with the same shape and dtype as ``rewards``.
    """
    if rewards.ndim != 1:
        raise ValueError(f"rewards must have shape [N], got {tuple(rewards.shape)}")
    if rewards.numel() == 0:
        raise ValueError("rewards must contain at least one value")

    if group_ids is None:
        group_ids = torch.zeros_like(rewards, dtype=torch.long)
    if group_ids.shape != rewards.shape:
        raise ValueError("group_ids must have the same shape as rewards")

    rewards_float = rewards.float()
    global_scale = rewards_float.std(unbiased=False).clamp_min(eps)
    advantages = torch.empty_like(rewards_float)
    for group_id in torch.unique(group_ids):
        group_mask = group_ids == group_id
        group_rewards = rewards_float[group_mask]
        scale = global_scale if global_std else group_rewards.std(unbiased=False).clamp_min(eps)
        advantages[group_mask] = (group_rewards - group_rewards.mean()) / scale
    return advantages.to(dtype=rewards.dtype)


def combine_reward_component_advantages(
    components: Mapping[str, torch.Tensor],
    component_weights: Mapping[str, float],
    *,
    component_scales: Mapping[str, torch.Tensor] | None = None,
    eps: float = 1e-4,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Center reward components independently and combine their normalized advantages."""
    if not components:
        raise ValueError("At least one reward component is required")
    if set(components) != set(component_weights):
        raise ValueError("component_weights must have exactly the same keys as components")
    shapes = {tuple(value.shape) for value in components.values()}
    if len(shapes) != 1:
        raise ValueError("All reward components must have the same shape")

    combined = torch.zeros_like(next(iter(components.values())))
    normalized = {}
    for name, values in components.items():
        if component_scales is None:
            scale = values.float().std(unbiased=False).clamp_min(eps)
        else:
            scale = component_scales[name].to(device=values.device).clamp_min(eps)
        advantage = (values - values.mean()) / scale
        normalized[name] = advantage
        combined = combined + float(component_weights[name]) * advantage
    return combined, normalized
