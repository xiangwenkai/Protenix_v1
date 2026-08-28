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

"""Old and reference policy state management for NFT."""

import copy

import torch
from torch import nn


def scheduled_old_policy_decay(step: int, decay_type: int, constant_decay: float) -> float:
    """Match the old-policy schedules used by the reference DiffusionNFT implementation."""
    if decay_type == -1:
        return float(constant_decay)
    if decay_type == 0:
        return 0.0
    if decay_type == 1:
        return min(max(step, 0) * 0.001, 0.5)
    if decay_type == 2:
        return 0.0 if step < 75 else min((step - 75) * 0.0075, 0.999)
    raise ValueError("old_policy_decay_type must be one of -1, 0, 1, or 2")


def freeze_module(module: nn.Module) -> nn.Module:
    """Put a module in evaluation mode and disable its gradients."""
    module.eval()
    module.requires_grad_(False)
    return module


class PolicyCopies:
    """Frozen old-policy and base-reference copies of a trainable denoiser."""

    def __init__(self, current_policy: nn.Module) -> None:
        self.old = freeze_module(copy.deepcopy(current_policy))
        self.reference = freeze_module(copy.deepcopy(current_policy))

    @torch.no_grad()
    def update_old(self, current_policy: nn.Module, decay: float) -> None:
        """EMA-update parameters and copy buffers from the current policy."""
        if not 0.0 <= decay <= 1.0:
            raise ValueError("old-policy decay must be in [0, 1]")
        current = current_policy.module if hasattr(current_policy, "module") else current_policy
        for old_parameter, current_parameter in zip(
            self.old.parameters(), current.parameters(), strict=True
        ):
            old_parameter.lerp_(current_parameter.detach(), 1.0 - decay)
        for old_buffer, current_buffer in zip(self.old.buffers(), current.buffers(), strict=True):
            old_buffer.copy_(current_buffer.detach())

    def state_dict(self) -> dict[str, dict[str, torch.Tensor]]:
        """Return serializable old and reference policy states."""
        return {"old": self.old.state_dict(), "reference": self.reference.state_dict()}

    def load_state_dict(self, state_dict: dict[str, dict[str, torch.Tensor]]) -> None:
        """Restore old and reference policy states."""
        self.old.load_state_dict(state_dict["old"])
        self.reference.load_state_dict(state_dict["reference"])
