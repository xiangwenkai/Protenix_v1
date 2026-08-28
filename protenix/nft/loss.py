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

"""DiffusionNFT objective adapted to Protenix EDM denoised coordinates."""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass
class NFTLossOutput:
    """Total NFT loss and detached diagnostics."""

    loss: torch.Tensor
    metrics: dict[str, torch.Tensor]


class DiffusionNFTLoss(nn.Module):
    """Positive/negative NFT objective in denoised-coordinate space."""

    def __init__(
        self,
        interpolation_beta: float = 1.0,
        reference_weight: float = 1e-4,
        advantage_clip: float = 5.0,
        adaptive_weight_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if interpolation_beta <= 0:
            raise ValueError("interpolation_beta must be positive")
        if advantage_clip <= 0:
            raise ValueError("advantage_clip must be positive")
        if reference_weight < 0:
            raise ValueError("reference_weight cannot be negative")
        self.interpolation_beta = interpolation_beta
        self.reference_weight = reference_weight
        self.advantage_clip = advantage_clip
        self.adaptive_weight_eps = adaptive_weight_eps

    @staticmethod
    def _per_sample_mean(values: torch.Tensor, coordinate_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if coordinate_mask is None:
            return values.mean(dim=(-2, -1))
        mask = coordinate_mask.to(device=values.device, dtype=values.dtype)
        while mask.ndim < values.ndim - 1:
            mask = mask.unsqueeze(0)
        mask = mask.unsqueeze(-1)
        denominator = (mask.sum(dim=(-2, -1)) * values.shape[-1]).clamp_min(1.0)
        return (values * mask).sum(dim=(-2, -1)) / denominator

    def _adaptive_mse(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        coordinate_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        error = prediction - target
        weight = self._per_sample_mean(error.detach().abs(), coordinate_mask).clamp_min(
            self.adaptive_weight_eps
        )
        return self._per_sample_mean(error.square(), coordinate_mask) / weight

    def forward(
        self,
        current_denoised: torch.Tensor,
        old_denoised: torch.Tensor,
        reference_denoised: torch.Tensor,
        target_coordinates: torch.Tensor,
        advantages: torch.Tensor,
        coordinate_mask: Optional[torch.Tensor] = None,
        reference_output_scale: Optional[torch.Tensor] = None,
    ) -> NFTLossOutput:
        """Compute the NFT objective for one or more condition groups."""
        expected_shape = target_coordinates.shape
        for name, value in {
            "current_denoised": current_denoised,
            "old_denoised": old_denoised,
            "reference_denoised": reference_denoised,
        }.items():
            if value.shape != expected_shape:
                raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {tuple(expected_shape)}")
        if advantages.shape != target_coordinates.shape[:-2]:
            raise ValueError(
                "advantages must match the leading coordinate dimensions: "
                f"got {tuple(advantages.shape)} and {tuple(target_coordinates.shape[:-2])}"
            )

        beta = self.interpolation_beta
        old_denoised = old_denoised.detach()
        reference_denoised = reference_denoised.detach()
        positive_denoised = beta * current_denoised + (1.0 - beta) * old_denoised
        negative_denoised = (1.0 + beta) * old_denoised - beta * current_denoised

        positive_loss = self._adaptive_mse(positive_denoised, target_coordinates, coordinate_mask)
        negative_loss = self._adaptive_mse(negative_denoised, target_coordinates, coordinate_mask)
        clipped_advantages = advantages.clamp(-self.advantage_clip, self.advantage_clip)
        positive_weight = (clipped_advantages / self.advantage_clip / 2.0 + 0.5).clamp(0.0, 1.0)
        policy_per_sample = (
            positive_weight * positive_loss / beta
            + (1.0 - positive_weight) * negative_loss / beta
        )
        policy_loss = (policy_per_sample * self.advantage_clip).mean()

        reference_per_sample = self._per_sample_mean(
            (current_denoised - reference_denoised).square(), coordinate_mask
        )
        if reference_output_scale is not None:
            if reference_output_scale.shape != target_coordinates.shape[:-2]:
                raise ValueError(
                    "reference_output_scale must match the leading coordinate dimensions: "
                    f"got {tuple(reference_output_scale.shape)} and "
                    f"{tuple(target_coordinates.shape[:-2])}"
                )
            reference_per_sample = reference_per_sample / reference_output_scale.float().square().clamp_min(
                self.adaptive_weight_eps**2
            )
        reference_loss = reference_per_sample.mean()
        total_loss = policy_loss + self.reference_weight * reference_loss
        metrics = {
            "loss": total_loss.detach(),
            "policy_loss": policy_loss.detach(),
            "reference_loss": reference_loss.detach(),
            "positive_loss": positive_loss.mean().detach(),
            "negative_loss": negative_loss.mean().detach(),
            "advantage_mean": advantages.mean().detach(),
            "advantage_abs_mean": advantages.abs().mean().detach(),
            "current_old_mse": self._per_sample_mean(
                (current_denoised.detach() - old_denoised).square(), coordinate_mask
            ).mean(),
        }
        return NFTLossOutput(loss=total_loss, metrics=metrics)
