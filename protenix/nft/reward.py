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

"""Extensible experimental-signal reward API for Protenix NFT training."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

import torch

logger = logging.getLogger(__name__)


@dataclass
class RewardOutput:
    """Reward values and optional scalar diagnostics for a rollout."""

    rewards: torch.Tensor
    metrics: dict[str, torch.Tensor] = field(default_factory=dict)
    valid_mask: torch.Tensor | None = None
    components: dict[str, torch.Tensor] = field(default_factory=dict)
    component_weights: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ConditionCheck:
    """Cheap pre-rollout validation result for one conditioning example."""

    valid: bool
    reason: str = ""


class NFTReward(ABC):
    """Interface for scoring generated coordinate samples."""

    @abstractmethod
    def __call__(
        self,
        coordinates: torch.Tensor,
        input_feature_dict: Mapping[str, Any],
        label_dict: Mapping[str, Any],
    ) -> RewardOutput:
        """Score coordinates with shape ``[N_sample, N_atom, 3]``."""

    def check_condition(
        self,
        input_feature_dict: Mapping[str, Any],
        label_dict: Mapping[str, Any],
    ) -> ConditionCheck:
        """Validate labels and molecular content before expensive model rollout."""
        del input_feature_dict, label_dict
        return ConditionCheck(valid=True)


class NullReward(NFTReward):
    """Placeholder reward returning zero for every generated structure."""

    def __init__(self, **unused_kwargs: Any) -> None:
        del unused_kwargs
        self._warned = False

    @torch.no_grad()
    def __call__(
        self,
        coordinates: torch.Tensor,
        input_feature_dict: Mapping[str, Any],
        label_dict: Mapping[str, Any],
    ) -> RewardOutput:
        del input_feature_dict, label_dict
        if not self._warned:
            logger.warning(
                "NullReward is active: all advantages are zero and NFT training has no reward signal."
            )
            self._warned = True
        rewards = torch.zeros(
            coordinates.shape[-3], device=coordinates.device, dtype=torch.float32
        )
        return RewardOutput(
            rewards=rewards,
            metrics={"null_reward": rewards.mean()},
            valid_mask=torch.ones_like(rewards, dtype=torch.bool),
            components={"reward": rewards},
            component_weights={"reward": 1.0},
        )


REWARD_REGISTRY: dict[str, type[NFTReward]] = {"null": NullReward}


class WeightedReward(NFTReward):
    """Combine any number of registered experimental-signal rewards."""

    def __init__(self, components: Mapping[str, Mapping[str, Any]]) -> None:
        if not components:
            raise ValueError("WeightedReward requires at least one component")
        self.components = {}
        for component_name, specification in components.items():
            specification = dict(specification)
            reward_name = specification.pop("name", component_name)
            weight = float(specification.pop("weight", 1.0))
            if "kwargs" in specification:
                kwargs = specification.pop("kwargs")
            else:
                kwargs = specification
                specification = {}
            if specification:
                raise ValueError(
                    f"Unknown weighted reward fields for {component_name}: {sorted(specification)}"
                )
            self.components[component_name] = (weight, build_reward(reward_name, kwargs))

    def check_condition(
        self,
        input_feature_dict: Mapping[str, Any],
        label_dict: Mapping[str, Any],
    ) -> ConditionCheck:
        """Accept a condition when at least one nonzero weighted reward can score it."""
        checks = [
            reward.check_condition(input_feature_dict, label_dict)
            for weight, reward in self.components.values()
            if weight != 0.0
        ]
        if any(check.valid for check in checks):
            return ConditionCheck(valid=True)
        reasons = sorted({check.reason for check in checks if check.reason})
        return ConditionCheck(valid=False, reason=";".join(reasons) or "no_active_reward")

    @torch.no_grad()
    def __call__(
        self,
        coordinates: torch.Tensor,
        input_feature_dict: Mapping[str, Any],
        label_dict: Mapping[str, Any],
    ) -> RewardOutput:
        total = torch.zeros(coordinates.shape[-3], device=coordinates.device, dtype=torch.float32)
        valid_mask = torch.zeros_like(total, dtype=torch.bool)
        metrics = {}
        for component_name, (weight, reward) in self.components.items():
            output = reward(coordinates, input_feature_dict, label_dict)
            total = total + weight * output.rewards
            component_valid = output.valid_mask
            if component_valid is None:
                component_valid = torch.ones_like(output.rewards, dtype=torch.bool)
            if weight != 0.0:
                valid_mask |= component_valid.to(device=valid_mask.device, dtype=torch.bool)
            metrics.update(
                {f"{component_name}/{name}": value for name, value in output.metrics.items()}
            )
            metrics[f"{component_name}/weighted_reward"] = (weight * output.rewards).mean()
        metrics["weighted_reward"] = total.mean()
        return RewardOutput(
            total,
            metrics,
            valid_mask,
            components={"reward": total},
            component_weights={"reward": 1.0},
        )


def build_reward(name: str, kwargs: Mapping[str, Any] | None = None) -> NFTReward:
    """Build a registered reward without coupling it to the training runner."""
    if name == "weighted":
        return WeightedReward(**dict(kwargs or {}))
    if name == "eclip":
        from protenix.nft.eclip_reward import EclipPeakF1Reward

        return EclipPeakF1Reward(**dict(kwargs or {}))
    if name == "distillation_eclip":
        from protenix.nft.structure_reward import DistillationEclipReward

        return DistillationEclipReward(**dict(kwargs or {}))
    if name not in REWARD_REGISTRY:
        available = ", ".join(
            sorted({*REWARD_REGISTRY, "distillation_eclip", "eclip", "weighted"})
        )
        raise ValueError(f"Unknown NFT reward {name!r}. Available rewards: {available}")
    return REWARD_REGISTRY[name](**dict(kwargs or {}))


def register_reward(name: str, reward_type: type[NFTReward]) -> None:
    """Register one experimental signal reward implementation."""
    if name in REWARD_REGISTRY or name in {"distillation_eclip", "eclip", "weighted"}:
        raise ValueError(f"NFT reward {name!r} is already registered")
    REWARD_REGISTRY[name] = reward_type
