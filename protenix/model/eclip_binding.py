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

"""Differentiable eCLIP binding signal losses for PPFT training.

These modules are intentionally external to :class:`protenix.model.protenix.Protenix`.
They may be saved as sidecar checkpoint state, but must not be added to the
Protenix model state_dict.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn


def _heavy_atom_mask(feat_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    """Return a best-effort heavy atom mask from Protenix features."""

    ref_mask = feat_dict.get("ref_mask")
    if ref_mask is None:
        mask = torch.ones_like(feat_dict["atom_to_token_idx"], dtype=torch.bool)
    else:
        mask = ref_mask.bool()

    ref_element = feat_dict.get("ref_element")
    if ref_element is not None and ref_element.ndim >= 2 and ref_element.shape[-1] > 0:
        # get_all_elems() starts from H, so one-hot index 0 corresponds to hydrogen.
        is_hydrogen = ref_element.argmax(dim=-1) == 0
        mask = mask & ~is_hydrogen
    return mask


def get_rna_token_indices(feat_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    """Return sorted token indices that contain at least one RNA atom."""

    atom_to_token = feat_dict["atom_to_token_idx"].long()
    rna_atom_mask = feat_dict["is_rna"].bool()
    if not rna_atom_mask.any():
        return atom_to_token.new_empty((0,))
    return atom_to_token[rna_atom_mask].unique(sorted=True)


def get_protein_token_indices(feat_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    """Return sorted token indices that contain at least one protein atom."""

    atom_to_token = feat_dict["atom_to_token_idx"].long()
    protein_atom_mask = feat_dict["is_protein"].bool()
    if not protein_atom_mask.any():
        return atom_to_token.new_empty((0,))
    return atom_to_token[protein_atom_mask].unique(sorted=True)


def compute_distogram_binding_score(
    contact_probs: torch.Tensor,
    feat_dict: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use Protenix token-token contact probabilities as RNA binding scores.

    For each RNA token r, binding probability is max_p contact_probs[r, p]
    over all protein tokens p.
    """

    rna_token_indices = get_rna_token_indices(feat_dict).to(contact_probs.device)
    protein_token_indices = get_protein_token_indices(feat_dict).to(contact_probs.device)
    if rna_token_indices.numel() == 0:
        raise ValueError("Cannot compute eCLIP binding score: no RNA tokens found.")
    if protein_token_indices.numel() == 0:
        raise ValueError("Cannot compute eCLIP binding score: no protein tokens found.")

    rna_protein_contacts = contact_probs[..., rna_token_indices, :].index_select(
        dim=-1,
        index=protein_token_indices,
    )
    p_bind = rna_protein_contacts.amax(dim=-1)
    return p_bind, rna_token_indices


def compute_soft_binding_score(
    coords: torch.Tensor,
    feat_dict: dict[str, torch.Tensor],
    *,
    cutoff: float = 5.0,
    temperature: float | torch.Tensor = 0.5,
    softmin_beta: float | torch.Tensor = 4.0,
    protein_atom_chunk_size: int = 4096,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute differentiable per-RNA-token binding probability.

    Args:
        coords: Predicted atom coordinates with shape ``[..., N_atom, 3]``.
        feat_dict: Protenix input feature dict for the same atoms.
        cutoff: Contact cutoff in Angstrom.
        temperature: Sigmoid temperature for ``sigmoid((cutoff - d) / temp)``.
        softmin_beta: Soft-min sharpness over RNA/protein heavy atom pairs.
        protein_atom_chunk_size: Chunk size over protein atoms to control memory.
        eps: Numeric lower bound.

    Returns:
        ``(p_bind, rna_token_indices)`` where ``p_bind`` has shape
        ``[..., N_rna_token]`` and token indices are sorted ascending.
    """

    if coords.shape[-1] != 3:
        raise ValueError(f"coords must end with xyz dimension, got {coords.shape}")
    if coords.ndim == 2:
        coords = coords.unsqueeze(0)

    atom_to_token = feat_dict["atom_to_token_idx"].to(coords.device).long()
    heavy_mask = _heavy_atom_mask(feat_dict).to(coords.device)
    protein_mask = feat_dict["is_protein"].to(coords.device).bool() & heavy_mask
    rna_mask = feat_dict["is_rna"].to(coords.device).bool() & heavy_mask

    if not protein_mask.any():
        raise ValueError("Cannot compute eCLIP binding score: no protein atoms found.")
    if not rna_mask.any():
        raise ValueError("Cannot compute eCLIP binding score: no RNA atoms found.")

    protein_coords = coords[..., protein_mask, :]
    rna_token_indices = atom_to_token[rna_mask].unique(sorted=True)

    beta = torch.as_tensor(softmin_beta, dtype=coords.dtype, device=coords.device).clamp_min(eps)
    temp = torch.as_tensor(temperature, dtype=coords.dtype, device=coords.device).clamp_min(eps)
    cutoff_t = torch.as_tensor(cutoff, dtype=coords.dtype, device=coords.device)

    scores = []
    leading_shape = coords.shape[:-2]
    for token_idx in rna_token_indices:
        token_atom_mask = rna_mask & (atom_to_token == token_idx)
        token_coords = coords[..., token_atom_mask, :]
        if token_coords.shape[-2] == 0:
            scores.append(torch.zeros(leading_shape, dtype=coords.dtype, device=coords.device))
            continue

        logsumexp_dist: Optional[torch.Tensor] = None
        for start in range(0, protein_coords.shape[-2], protein_atom_chunk_size):
            prot_chunk = protein_coords[..., start : start + protein_atom_chunk_size, :]
            distances = torch.linalg.vector_norm(
                token_coords[..., :, None, :] - prot_chunk[..., None, :, :],
                ord=2,
                dim=-1,
            )
            cur = torch.logsumexp((-beta * distances).reshape(*leading_shape, -1), dim=-1)
            logsumexp_dist = cur if logsumexp_dist is None else torch.logaddexp(logsumexp_dist, cur)

        soft_min_distance = -logsumexp_dist / beta
        scores.append(torch.sigmoid((cutoff_t - soft_min_distance) / temp))

    return torch.stack(scores, dim=-1), rna_token_indices


def align_signal_to_prediction(
    signal: torch.Tensor,
    pred_length: int,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad or truncate a 1D signal vector to match predicted RNA tokens."""

    signal = signal.to(device=device, dtype=torch.float32).flatten()
    aligned = torch.zeros(pred_length, dtype=torch.float32, device=device)
    mask = torch.zeros(pred_length, dtype=torch.bool, device=device)
    length = min(pred_length, signal.numel())
    if length > 0:
        aligned[:length] = signal[:length]
        mask[:length] = True
    return aligned, mask


def normalize_log_signal(signal: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Map one sample's non-negative eCLIP signal to a 0-1 log profile."""

    log_signal = torch.log1p(signal.float().clamp_min(0.0))
    return log_signal / log_signal.max().clamp_min(eps)


class EclipSignalLoss(nn.Module):
    """Multinomial count-profile eCLIP signal loss.

    This follows the RBPNet/parnet objective shape: the target is the original
    per-position count profile, while model scores are normalized across the
    RNA positions and used as multinomial logits. A profile contributes to the
    loss only when its maximum count reaches ``min_height``.
    """

    def __init__(
        self,
        *,
        profile_weight: float = 1.0,
        min_height: float = 3.0,
        binary_threshold: float = 2.0,
        signal_clip_value: float = 100.0,
        max_total_count: float = 100.0,
        eps: float = 1e-8,
        positive_weight: float | None = None,
        point_weight: float | None = None,
    ) -> None:
        super().__init__()
        self.profile_weight = profile_weight
        self.min_height = min_height
        self.binary_threshold = binary_threshold
        self.signal_clip_value = signal_clip_value
        self.max_total_count = max_total_count
        self.eps = eps
        # Kept only so older sidecar checkpoints/config calls do not break.
        _ = positive_weight, point_weight

    def forward(
        self,
        p_bind: torch.Tensor,
        target_signal: torch.Tensor,
        target_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if p_bind.ndim > 1 and p_bind.shape[0] == 1:
            p_bind = p_bind.squeeze(0)
        p_bind = p_bind.float()
        target_signal = target_signal.to(device=p_bind.device, dtype=torch.float32)
        if target_mask is None:
            target_mask = torch.ones_like(p_bind, dtype=torch.bool)
        else:
            target_mask = target_mask.to(device=p_bind.device, dtype=torch.bool)

        valid_p = p_bind[target_mask].clamp_min(self.eps)
        valid_signal = target_signal[target_mask].clamp_min(0.0)
        if valid_p.numel() == 0:
            zero = p_bind.sum() * 0.0
            return zero, {"loss": zero.detach()}

        raw_signal = valid_signal
        loss_signal = valid_signal
        if self.signal_clip_value is not None and float(self.signal_clip_value) > 0.0:
            loss_signal = loss_signal.clamp_max(float(self.signal_clip_value))
        clipped_total = loss_signal.sum()
        if self.max_total_count is not None and float(self.max_total_count) > 0.0:
            max_total = torch.as_tensor(
                float(self.max_total_count),
                dtype=loss_signal.dtype,
                device=loss_signal.device,
            )
            scale = torch.minimum(
                torch.ones((), dtype=loss_signal.dtype, device=loss_signal.device),
                max_total / clipped_total.clamp_min(self.eps),
            )
            loss_signal = loss_signal * scale

        profile_pass = raw_signal.max() >= float(self.min_height)
        profile_pass_bool = bool(profile_pass.detach().item())
        pred_logits = valid_p.log()
        if profile_pass_bool:
            profile_loss = -torch.distributions.Multinomial(
                logits=pred_logits.float(),
                validate_args=False,
            ).log_prob(loss_signal.float())
        else:
            profile_loss = pred_logits.sum() * 0.0

        weighted_profile_loss = self.profile_weight * profile_loss
        loss = weighted_profile_loss
        metrics = {
            "loss": loss.detach(),
            "profile_multinomial_nll": profile_loss.detach(),
            "weighted_profile_multinomial_nll": weighted_profile_loss.detach(),
        }
        return loss, metrics


class StructureSignalKLLoss(nn.Module):
    """KL loss for structure-derived soft RNA contact profiles.

    The PDB target signal is a bounded soft contact score, not a count profile.
    Both prediction and target are therefore normalized over valid RNA tokens
    before computing KL(target || prediction).
    """

    def __init__(
        self,
        *,
        profile_weight: float = 1.0,
        target_threshold: float = 0.0,
        min_peak: float = 0.0,
        binary_threshold: float = 0.5,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.profile_weight = profile_weight
        self.target_threshold = target_threshold
        self.min_peak = min_peak
        self.binary_threshold = binary_threshold
        self.eps = eps

    def forward(
        self,
        p_bind: torch.Tensor,
        target_signal: torch.Tensor,
        target_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if p_bind.ndim > 1 and p_bind.shape[0] == 1:
            p_bind = p_bind.squeeze(0)
        p_bind = p_bind.float()
        target_signal = target_signal.to(device=p_bind.device, dtype=torch.float32)
        if target_mask is None:
            target_mask = torch.ones_like(p_bind, dtype=torch.bool)
        else:
            target_mask = target_mask.to(device=p_bind.device, dtype=torch.bool)

        valid_p = p_bind[target_mask].clamp_min(self.eps)
        raw_signal = target_signal[target_mask].clamp_min(0.0)
        if valid_p.numel() == 0:
            zero = p_bind.sum() * 0.0
            return zero, {"loss": zero.detach()}

        target_threshold = float(self.target_threshold)
        filtered_signal = torch.where(
            raw_signal >= target_threshold,
            raw_signal,
            torch.zeros_like(raw_signal),
        )
        target_sum = filtered_signal.sum()
        raw_peak = raw_signal.max()
        filter_pass = (raw_peak >= float(self.min_peak)) & (target_sum > self.eps)
        filter_pass_bool = bool(filter_pass.detach().item())

        pred_profile = valid_p / valid_p.sum().clamp_min(self.eps)
        if filter_pass_bool:
            target_profile = filtered_signal / target_sum.clamp_min(self.eps)
            kl_loss = torch.nn.functional.kl_div(
                pred_profile.clamp_min(self.eps).log(),
                target_profile,
                reduction="sum",
            )
        else:
            target_profile = torch.zeros_like(filtered_signal)
            kl_loss = pred_profile.sum() * 0.0

        weighted_kl_loss = self.profile_weight * kl_loss
        metrics = {
            "loss": weighted_kl_loss.detach(),
            "profile_kl": kl_loss.detach(),
            "weighted_profile_kl": weighted_kl_loss.detach(),
        }
        return weighted_kl_loss, metrics


class EclipBindingScorer(nn.Module):
    """Sidecar module that maps rollout coordinates to eCLIP binding scores."""

    def __init__(
        self,
        *,
        cutoff: float = 5.0,
        temperature: float = 0.5,
        softmin_beta: float = 4.0,
        learn_temperature: bool = False,
        learn_softmin_beta: bool = False,
        protein_atom_chunk_size: int = 4096,
    ) -> None:
        super().__init__()
        self.cutoff = float(cutoff)
        self.protein_atom_chunk_size = protein_atom_chunk_size
        if learn_temperature:
            self.log_temperature = nn.Parameter(torch.log(torch.tensor(float(temperature))))
        else:
            self.register_buffer("log_temperature", torch.log(torch.tensor(float(temperature))))
        if learn_softmin_beta:
            self.log_softmin_beta = nn.Parameter(torch.log(torch.tensor(float(softmin_beta))))
        else:
            self.register_buffer("log_softmin_beta", torch.log(torch.tensor(float(softmin_beta))))

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp()

    @property
    def softmin_beta(self) -> torch.Tensor:
        return self.log_softmin_beta.exp()

    def forward(
        self,
        coords: torch.Tensor,
        feat_dict: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return compute_soft_binding_score(
            coords=coords,
            feat_dict=feat_dict,
            cutoff=self.cutoff,
            temperature=self.temperature,
            softmin_beta=self.softmin_beta,
            protein_atom_chunk_size=self.protein_atom_chunk_size,
        )

@torch.no_grad()
def masked_pearson(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    pred = pred.float()[mask]
    target = target.float()[mask]
    if pred.numel() < 2:
        return torch.tensor(0.0, device=mask.device)
    pred = pred - pred.mean()
    target = target - target.mean()
    denom = pred.norm() * target.norm()
    if denom <= eps:
        return torch.tensor(0.0, device=mask.device)
    return (pred * target).sum() / denom


@torch.no_grad()
def masked_std(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    value = value.float()[mask]
    if value.numel() == 0:
        return torch.tensor(float("nan"), device=mask.device)
    return value.std(unbiased=False)


@torch.no_grad()
def topk_overlap(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, k: int = 20) -> torch.Tensor:
    pred = pred.float().masked_fill(~mask, float("-inf"))
    target = target.float().masked_fill(~mask, float("-inf"))
    valid_n = int(mask.sum().item())
    if valid_n == 0:
        return torch.tensor(float("nan"), device=mask.device)
    k = min(k, valid_n)
    pred_idx = torch.topk(pred, k=k).indices
    target_idx = torch.topk(target, k=k).indices
    overlap = torch.isin(pred_idx, target_idx).float().mean()
    return overlap


@torch.no_grad()
def binary_auprc(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    threshold: float = 0.0,
) -> torch.Tensor:
    """Average precision/AUPRC for binary signal targets."""

    pred = pred.float()[mask]
    valid_target = target.float()[mask]
    if threshold <= 0.0:
        target = (valid_target > 0.0).float()
    else:
        target = (valid_target >= float(threshold)).float()
    if pred.numel() == 0:
        return torch.tensor(0.0, device=mask.device)
    positive_count = target.sum()
    if positive_count <= 0:
        return torch.tensor(0.0, device=mask.device)

    order = torch.argsort(pred, descending=True)
    sorted_target = target[order]
    tp_cumsum = torch.cumsum(sorted_target, dim=0)
    rank = torch.arange(1, sorted_target.numel() + 1, device=pred.device, dtype=torch.float32)
    precision_at_k = tp_cumsum / rank
    return (precision_at_k * sorted_target).sum() / positive_count
