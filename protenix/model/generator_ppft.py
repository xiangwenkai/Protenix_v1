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

"""Short differentiable rollout for eCLIP PPFT training."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Callable, Optional

import torch

from protenix.model.utils import centre_random_augmentation


def sample_diffusion_ppft(
    denoise_net: Callable,
    input_feature_dict: dict[str, Any],
    s_inputs: torch.Tensor,
    s_trunk: torch.Tensor,
    z_trunk: torch.Tensor,
    pair_z: torch.Tensor,
    p_lm: torch.Tensor,
    c_l: torch.Tensor,
    noise_schedule: torch.Tensor,
    *,
    N_sample: int = 1,
    record_grad_steps: set[int] | list[int] | tuple[int, ...] | None = None,
    detach_unrecorded_steps: bool = True,
    gamma0: float = 0.8,
    gamma_min: float = 1.0,
    noise_scale_lambda: float = 1.003,
    step_scale_eta: float = 1.5,
    inplace_safe: bool = False,
    attn_chunk_size: Optional[int] = None,
    enable_efficient_fusion: bool = False,
) -> torch.Tensor:
    """PPFT rollout with optional gradient-recorded denoising steps.

    Step indices are 1-based over denoising steps. The last step is always
    recorded with gradients to keep the property loss connected to model
    parameters.
    """

    if N_sample < 1:
        raise ValueError(f"N_sample must be >= 1, got {N_sample}.")
    num_steps = int(noise_schedule.numel()) - 1
    if num_steps <= 0:
        raise ValueError("noise_schedule must contain at least two time points.")

    if record_grad_steps is None:
        grad_steps = set(range(1, num_steps + 1))
    else:
        grad_steps = {int(step) for step in record_grad_steps}
    grad_steps.add(num_steps)

    N_atom = input_feature_dict["atom_to_token_idx"].size(-1)
    batch_shape = s_inputs.shape[:-2]
    device = s_inputs.device
    dtype = s_inputs.dtype

    x_l = noise_schedule[0] * torch.randn(
        size=(*batch_shape, N_sample, N_atom, 3), device=device, dtype=dtype
    )

    for step_idx, (c_tau_last, c_tau) in enumerate(
        zip(noise_schedule[:-1], noise_schedule[1:]), start=1
    ):
        enable_grad = step_idx in grad_steps
        ctx = nullcontext() if enable_grad else torch.no_grad()
        with ctx:
            x_l = (
                centre_random_augmentation(x_input_coords=x_l, N_sample=1)
                .squeeze(dim=-3)
                .to(dtype)
            )
            gamma = float(gamma0) if c_tau > gamma_min else 0
            t_hat_scalar = c_tau_last * (gamma + 1)
            delta_noise_level = torch.sqrt(t_hat_scalar**2 - c_tau_last**2)
            x_noisy = x_l + noise_scale_lambda * delta_noise_level * torch.randn(
                size=x_l.shape, device=device, dtype=dtype
            )
            t_hat = (
                t_hat_scalar.reshape((1,) * (len(batch_shape) + 1))
                .expand(*batch_shape, N_sample)
                .to(dtype)
            )
            x_denoised = denoise_net(
                x_noisy=x_noisy,
                t_hat_noise_level=t_hat,
                input_feature_dict=input_feature_dict,
                s_inputs=s_inputs,
                s_trunk=s_trunk,
                z_trunk=z_trunk,
                pair_z=pair_z,
                p_lm=p_lm,
                c_l=c_l,
                chunk_size=attn_chunk_size,
                inplace_safe=inplace_safe,
                enable_efficient_fusion=enable_efficient_fusion,
            )
            delta = (x_noisy - x_denoised) / t_hat[..., None, None]
            dt = c_tau - t_hat
            x_l = x_noisy + step_scale_eta * dt[..., None, None] * delta

        if detach_unrecorded_steps and not enable_grad:
            x_l = x_l.detach()

    return x_l
