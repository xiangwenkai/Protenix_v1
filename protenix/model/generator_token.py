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

"""
Token-level diffusion sampling functions
"""

from typing import Any, Callable, Optional

import torch


def centre_random_augmentation_token(
    x_input_coords: torch.Tensor, N_sample: int = 1
) -> torch.Tensor:
    """
    Token-level version: Apply random rotation and translation to token coordinates

    Args:
        x_input_coords: [..., N_sample, N_token, 3]
        N_sample: number of samples

    Returns:
        augmented coordinates [..., N_sample, N_token, 3]
    """
    # Centre coordinates
    centred = x_input_coords - x_input_coords.mean(dim=-2, keepdim=True)

    if N_sample == 1:
        return centred

    # Apply random rotation
    # Generate random rotation matrix
    batch_shape = centred.shape[:-3]
    device = centred.device
    dtype = centred.dtype

    # Random rotation using Gram-Schmidt
    random_matrix = torch.randn(
        *batch_shape, N_sample, 3, 3, device=device, dtype=dtype
    )
    q, _ = torch.linalg.qr(random_matrix)

    # Apply rotation: [N_sample, N_token, 3] @ [N_sample, 3, 3]
    rotated = torch.einsum("...ni,...ij->...nj", centred, q)

    return rotated


def sample_token_diffusion(
    denoise_net: Callable,
    input_feature_dict: dict[str, Any],
    s_inputs: torch.Tensor,
    s_trunk: torch.Tensor,
    z_trunk: torch.Tensor,
    pair_z: torch.Tensor,
    noise_schedule: torch.Tensor,
    N_sample: int = 1,
    gamma0: float = 0.8,
    gamma_min: float = 1.0,
    noise_scale_lambda: float = 1.003,
    step_scale_eta: float = 1.5,
    diffusion_chunk_size: Optional[int] = None,
    inplace_safe: bool = False,
    attn_chunk_size: Optional[int] = None,
    enable_efficient_fusion: bool = False,
) -> torch.Tensor:
    """
    Token-level diffusion sampling (Algorithm 18 adapted for tokens)

    Args:
        denoise_net: TokenDiffusionModule
        input_feature_dict: input features
        s_inputs: [..., N_token, c_s_inputs]
        s_trunk: [..., N_token, c_s]
        z_trunk: [..., N_token, N_token, c_z]
        pair_z: [..., N_token, N_token, c_z] (cached)
        noise_schedule: [N_iterations]
        N_sample: number of samples
        gamma0, gamma_min, noise_scale_lambda, step_scale_eta: sampling parameters
        diffusion_chunk_size: chunk size for diffusion
        inplace_safe: use inplace operations
        attn_chunk_size: chunk size for attention
        enable_efficient_fusion: enable fusion optimization

    Returns:
        token_coordinates: [..., N_sample, N_token, 3]
    """
    N_token = input_feature_dict["residue_index"].shape[-1]
    batch_shape = s_inputs.shape[:-2]
    device = s_inputs.device
    dtype = s_inputs.dtype

    def _chunk_sample_diffusion(chunk_n_sample, inplace_safe):
        # Initialize noise
        x_token = noise_schedule[0] * torch.randn(
            size=(*batch_shape, chunk_n_sample, N_token, 3), device=device, dtype=dtype
        )

        for _, (c_tau_last, c_tau) in enumerate(
            zip(noise_schedule[:-1], noise_schedule[1:])
        ):
            # Centre and randomly augment
            x_token = (
                centre_random_augmentation_token(x_input_coords=x_token, N_sample=1)
                .squeeze(dim=-3)
                .to(dtype)
            )

            # Predictor-corrector step
            # 1. Add noise
            gamma = float(gamma0) if c_tau > gamma_min else 0
            t_hat = c_tau_last * (gamma + 1)

            delta_noise_level = torch.sqrt(t_hat**2 - c_tau_last**2)
            x_noisy = x_token + noise_scale_lambda * delta_noise_level * torch.randn(
                size=x_token.shape, device=device, dtype=dtype
            )

            # 2. Denoise
            t_hat = (
                t_hat.reshape((1,) * (len(batch_shape) + 1))
                .expand(*batch_shape, chunk_n_sample)
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
                inplace_safe=inplace_safe,
                chunk_size=attn_chunk_size,
                use_conditioning=True,
                enable_efficient_fusion=enable_efficient_fusion,
            )

            # Euler step
            d_token = (x_noisy - x_denoised) / t_hat[..., None, None]
            dt = c_tau - t_hat
            x_token = x_noisy + step_scale_eta * dt[..., None, None] * d_token

        return x_token

    # Sample with optional chunking
    if diffusion_chunk_size is None or N_sample <= diffusion_chunk_size:
        return _chunk_sample_diffusion(N_sample, inplace_safe)
    else:
        # Chunk over N_sample dimension
        results = []
        for i in range(0, N_sample, diffusion_chunk_size):
            chunk_size = min(diffusion_chunk_size, N_sample - i)
            chunk_result = _chunk_sample_diffusion(chunk_size, inplace_safe)
            results.append(chunk_result)
        return torch.cat(results, dim=-3)


def sample_token_diffusion_training(
    noise_sampler: Any,
    denoise_net: Callable,
    label_dict: dict[str, Any],
    input_feature_dict: dict[str, Any],
    s_inputs: torch.Tensor,
    s_trunk: torch.Tensor,
    z_trunk: torch.Tensor,
    pair_z: torch.Tensor,
    N_sample: int,
    diffusion_chunk_size: Optional[int] = None,
    use_conditioning: bool = True,
    enable_efficient_fusion: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Token-level training diffusion sampling

    Args:
        noise_sampler: noise level sampler
        denoise_net: TokenDiffusionModule
        label_dict: contains "coordinate" [N_token, 3]
        input_feature_dict: input features
        s_inputs: [..., N_token, c_s_inputs]
        s_trunk: [..., N_token, c_s]
        z_trunk: [..., N_token, N_token, c_z]
        pair_z: [..., N_token, N_token, c_z] (cached)
        N_sample: number of noise samples
        diffusion_chunk_size: chunk size
        use_conditioning: whether to use conditioning
        enable_efficient_fusion: enable fusion

    Returns:
        x_noisy: [..., N_sample, N_token, 3]
        x_denoised: [..., N_sample, N_token, 3]
        noise_level: [..., N_sample]
    """
    batch_shape = s_inputs.shape[:-2]
    device = s_inputs.device
    dtype = s_inputs.dtype

    # Ground truth token coordinates
    x_true = label_dict["coordinate"]  # [..., N_token, 3]

    def _chunk_denoise(chunk_n_sample):
        # Sample noise levels
        noise_level_chunk = noise_sampler(
            size=(*batch_shape, chunk_n_sample), device=device
        ).to(dtype)

        # Add noise to ground truth
        noise = torch.randn(
            *batch_shape, chunk_n_sample, *x_true.shape[-2:], device=device, dtype=dtype
        )
        x_noisy_chunk = (
            x_true.unsqueeze(-3) + noise_level_chunk[..., None, None] * noise
        )

        # Denoise
        x_denoised_chunk = denoise_net(
            x_noisy=x_noisy_chunk,
            t_hat_noise_level=noise_level_chunk,
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            pair_z=pair_z,
            inplace_safe=False,
            chunk_size=None,
            use_conditioning=use_conditioning,
            enable_efficient_fusion=enable_efficient_fusion,
        )

        return x_noisy_chunk, x_denoised_chunk, noise_level_chunk

    # Sample with optional chunking
    if diffusion_chunk_size is None or N_sample <= diffusion_chunk_size:
        return _chunk_denoise(N_sample)
    else:
        noisy_list, denoised_list, noise_list = [], [], []
        for i in range(0, N_sample, diffusion_chunk_size):
            chunk_size = min(diffusion_chunk_size, N_sample - i)
            noisy, denoised, noise_lvl = _chunk_denoise(chunk_size)
            noisy_list.append(noisy)
            denoised_list.append(denoised)
            noise_list.append(noise_lvl)

        return (
            torch.cat(noisy_list, dim=-3),
            torch.cat(denoised_list, dim=-3),
            torch.cat(noise_list, dim=-1),
        )
