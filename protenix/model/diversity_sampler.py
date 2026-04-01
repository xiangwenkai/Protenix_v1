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

from typing import Optional

import torch
import torch.nn.functional as F


class DiversitySampler:
    """
    Injects repulsive bias during diffusion sampling to encourage diverse conformations.

    Based on AF3_ReD implementation with Kabsch alignment and noise-dependent weighting.
    """

    def __init__(
        self,
        weight: float = 1.0,
        sigma: float = 2.0,
        n_smooth: int = 1,
        bias_tmin: float = 0.0,
    ):
        """
        Args:
            weight: Strength of biasing potential
            sigma: Width of biasing Gaussian potential
            n_smooth: Number of neighbors for residue-level smoothing
            bias_tmin: Minimum noise level below which no bias is applied
        """
        self.weight = weight
        self.sigma = sigma
        self.n_smooth = n_smooth
        self.bias_tmin = bias_tmin
        self.structure_bank = []

    def add_structure(self, x: torch.Tensor) -> None:
        """Add a structure to the bank for repulsion."""
        self.structure_bank.append(x.detach().clone())

    def clear_bank(self) -> None:
        """Clear the structure bank."""
        self.structure_bank = []

    def _compute_optimal_rotation(
        self, x_centered: torch.Tensor, x_ref_centered: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute optimal rotation matrix via SVD (Kabsch algorithm).

        Args:
            x_centered: Centered coordinates [..., N_atom, 3]
            x_ref_centered: Centered reference [..., N_atom, 3]

        Returns:
            Rotation matrix [..., 3, 3]
        """
        # Compute covariance matrix: H = x_ref^T @ x / N
        H = torch.einsum("...ni,...nj->...ij", x_ref_centered, x_centered) / x_centered.shape[-2]

        # SVD decomposition
        U, _, Vt = torch.linalg.svd(H)

        # Rotation: R = U @ V^T
        R = torch.einsum("...ij,...kj->...ik", U, Vt)

        # Ensure proper rotation (det(R) = 1, not reflection)
        det = torch.linalg.det(R)
        correction = torch.where(det < 0, -1.0, 1.0)
        Vt_corrected = Vt * correction[..., None, None]
        R = torch.einsum("...ij,...jk->...ik", U, Vt_corrected)

        return R

    def _compute_rmsd_gradient(
        self, x: torch.Tensor, x_ref: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute RMSD gradient between x and x_ref with optimal alignment.

        Args:
            x: Current coordinates [..., N_atom, 3]
            x_ref: Reference coordinates [..., N_atom, 3]

        Returns:
            RMSD gradient [..., N_atom, 3]
        """
        # Center both structures
        x_mean = x.mean(dim=-2, keepdim=True)
        x_ref_mean = x_ref.mean(dim=-2, keepdim=True)
        x_centered = x - x_mean
        x_ref_centered = x_ref - x_ref_mean

        # Compute optimal rotation
        R = self._compute_optimal_rotation(x_centered, x_ref_centered)

        # Align x to x_ref: x_aligned = x_centered @ R^T
        x_aligned = torch.einsum("...ij,...jk->...ik", x_centered, R.transpose(-2, -1))

        # Compute RMSD
        diff = x_aligned - x_ref_centered
        rmsd_sq = (diff**2).sum(dim=-1).mean(dim=-1, keepdim=True)
        rmsd = torch.sqrt(rmsd_sq + 1e-8)

        # Gradient of RMSD w.r.t. x (chain rule through rotation)
        grad = torch.einsum("...ij,...jk->...ik", diff, R) / (rmsd[..., None] + 1e-8)
        return grad

    def _smooth_diffs(
        self, diffs: torch.Tensor, atom_to_token_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Smooth differences at token level using n_smooth neighbors.

        Args:
            diffs: Differences [..., N_atom, 3]
            atom_to_token_idx: Mapping from atoms to tokens [N_atom]

        Returns:
            Smoothed differences [..., N_atom, 3]
        """
        if self.n_smooth == 0:
            return diffs

        N_token = atom_to_token_idx.max().item() + 1
        token_diffs = torch.zeros(
            (*diffs.shape[:-2], N_token, 3), device=diffs.device, dtype=diffs.dtype
        )

        # Aggregate to token level
        for t in range(N_token):
            atom_mask = atom_to_token_idx == t
            if atom_mask.any():
                token_diffs[..., t, :] = diffs[..., atom_mask, :].mean(dim=-2)

        # Smooth at token level
        smoothed_token_diffs = torch.zeros_like(token_diffs)
        for t in range(N_token):
            neighbors = torch.arange(
                max(0, t - self.n_smooth),
                min(N_token, t + self.n_smooth + 1),
                device=diffs.device,
            )
            smoothed_token_diffs[..., t, :] = token_diffs[..., neighbors, :].mean(dim=-2)

        # Broadcast back to atom level
        smoothed_diffs = smoothed_token_diffs[..., atom_to_token_idx, :]
        return smoothed_diffs

    def compute_bias(
        self,
        x: torch.Tensor,
        noise_level: float,
        atom_to_token_idx: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """
        Compute biasing gradient using Gaussian potential.

        Args:
            x: Current coordinates [..., N_atom, 3]
            noise_level: Current noise level (time step)
            atom_to_token_idx: Mapping from atoms to tokens [N_atom]

        Returns:
            Biasing gradient [..., N_atom, 3] or None
        """
        if len(self.structure_bank) == 0 or noise_level < self.bias_tmin:
            return None

        # Center coordinates
        x_mean = x.mean(dim=-2, keepdim=True)
        x_centered = x - x_mean

        # Compute MSD for all reference structures
        msd_list = []
        diffs_list = []

        for x_ref in self.structure_bank:
            if x_ref.shape != x.shape:
                x_ref = x_ref.expand_as(x)

            x_ref_mean = x_ref.mean(dim=-2, keepdim=True)
            x_ref_centered = x_ref - x_ref_mean

            # Kabsch alignment (uses fit_mask if provided)
            R = self._compute_optimal_rotation(x_centered, x_ref_centered)
            x_aligned = torch.einsum(
                "...ij,...jk->...ik", x_centered, R.transpose(-2, -1)
            )

            # Compute MSD
            diffs = x_aligned - x_ref_centered
            msd = (diffs**2).sum(dim=-1).mean(dim=-1)
            msd_list.append(msd)
            diffs_list.append(diffs)

        msd = torch.stack(msd_list, dim=0)  # [N_bias, ...]
        diffs = torch.stack(diffs_list, dim=0)  # [N_bias, ..., N_atom, 3]

        # Gaussian potential weight
        sigma_eff = self.sigma + noise_level
        bias_exp = torch.exp(-msd / (2.0 * sigma_eff**2))  # [N_bias, ...]
        bias_exp = bias_exp[..., None, None]  # [N_bias, ..., 1, 1]

        # Smooth differences at token level
        diffs_smoothed = self._smooth_diffs(diffs, atom_to_token_idx)

        # Compute gradient
        grad = -self.weight * (bias_exp * diffs_smoothed).sum(dim=0) / (
            sigma_eff**2 + 1e-8
        )

        return grad
