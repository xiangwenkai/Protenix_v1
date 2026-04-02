# diversity_sampler.py
# AF3_ReD-style diversity bias (mask-free simplified version)

from typing import Optional, List
import torch


class DiversitySampler:
    """
    AF3_ReD-style repulsive bias during diffusion sampling.

    Differences from original:
    - No CA mask
    - No atom mask
    - No chain mask
    - Uses all atoms
    """

    def __init__(
        self,
        weight: float = 1.0,
        sigma: float = 2.0,
        n_smooth: int = 1,
        bias_tmin: float = 0.0,
    ):
        self.weight = weight
        self.sigma = sigma
        self.n_smooth = n_smooth
        self.bias_tmin = bias_tmin
        self.structure_bank: List[torch.Tensor] = []

    # ============================================================
    # Public API
    # ============================================================

    def add_structure(self, x: torch.Tensor) -> None:
        self.structure_bank.append(x.detach().clone())

    def clear_bank(self) -> None:
        self.structure_bank = []

    # ============================================================
    # Core math
    # ============================================================

    def _kabsch_align(self, x, x_ref):
        x_center = x.mean(dim=-2, keepdim=True)
        x_ref_center = x_ref.mean(dim=-2, keepdim=True)

        x_c = x - x_center
        x_ref_c = x_ref - x_ref_center

        # covariance: ref^T * x
        H = torch.einsum("...ni,...nj->...ij", x_ref_c, x_c)

        U, _, Vt = torch.linalg.svd(H.detach(), full_matrices=False)

        R = U @ Vt

        # ---- reflection correction (batch-safe) ----
        det = torch.linalg.det(R)              # shape [...]
        neg_mask = det < 0                    # shape [...]

        if neg_mask.any():
            Vt_corrected = Vt.clone()
            Vt_corrected[neg_mask, -1, :] *= -1
            R = U @ Vt_corrected

        # align reference onto x
        x_ref_aligned = torch.einsum(
            "...ni,...ij->...nj",
            x_ref_c,
            R,
        ) + x_center

        return x_ref_aligned

    def _smooth_diffs(
        self,
        diffs: torch.Tensor,  # [N_bias, ..., N_atom, 3]
        atom_to_token_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Token-level smoothing (same spirit as AF3 smoothing).
        """

        if self.n_smooth == 0:
            return diffs

        N_token = atom_to_token_idx.max().item() + 1

        token_diffs = torch.zeros(
            (*diffs.shape[:-2], N_token, 3),
            device=diffs.device,
            dtype=diffs.dtype,
        )

        # aggregate to token level
        for t in range(N_token):
            mask = atom_to_token_idx == t
            if mask.any():
                token_diffs[..., t, :] = diffs[..., mask, :].mean(dim=-2)

        # smooth neighbors
        smoothed = torch.zeros_like(token_diffs)
        for t in range(N_token):
            start = max(0, t - self.n_smooth)
            end = min(N_token, t + self.n_smooth + 1)
            smoothed[..., t, :] = token_diffs[..., start:end, :].mean(dim=-2)

        # broadcast back
        return smoothed[..., atom_to_token_idx, :]

    # ============================================================
    # Main bias computation
    # ============================================================

    def compute_bias(
        self,
        x: torch.Tensor,              # [..., N_atom, 3]
        noise_level: float,
        atom_to_token_idx: torch.Tensor,
    ) -> Optional[torch.Tensor]:

        if len(self.structure_bank) == 0:
            return None

        if noise_level <= self.bias_tmin:
            return None

        N_atom = x.shape[-2]
        sigma_eff = self.sigma + noise_level

        grads = []

        for x_ref in self.structure_bank:

            if x_ref.shape != x.shape:
                x_ref = x_ref.expand_as(x)

            # Kabsch alignment
            x_ref_aligned = self._kabsch_align(x, x_ref)

            # squared differences
            diffs = x - x_ref_aligned

            # MSD (mean over atoms)
            msd = (diffs**2).sum(dim=-1).mean(dim=-1)  # [...]

            # Gaussian weight
            weight_exp = torch.exp(-msd / (2.0 * sigma_eff**2))

            # smoothing
            diffs_smoothed = self._smooth_diffs(diffs.unsqueeze(0), atom_to_token_idx)[0]

            # AF3-style gradient (NO RMSD derivative)
            grad = (
                -self.weight
                * weight_exp[..., None, None]
                * diffs_smoothed
                / (N_atom * sigma_eff**2)
            )

            grads.append(grad)

        total_grad = torch.stack(grads, dim=0).sum(dim=0)

        return total_grad