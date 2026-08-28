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

"""eCLIP peak-region F1 reward computed from generated Protenix coordinates."""

import math
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from protenix.nft.reward import ConditionCheck, NFTReward, RewardOutput


def contiguous_intervals(mask: torch.Tensor) -> list[tuple[int, int]]:
    """Convert a one-dimensional boolean mask to half-open intervals."""
    values = mask.detach().bool().cpu().tolist()
    intervals = []
    start = None
    for index, value in enumerate(values):
        if value and start is None:
            start = index
        elif not value and start is not None:
            intervals.append((start, index))
            start = None
    if start is not None:
        intervals.append((start, len(values)))
    return intervals


def dynamic_peak_slack(
    interval: tuple[int, int], target_width: int, min_slack: int, max_slack: int
) -> int:
    """Return per-side dynamic slack using the reference screening rule."""
    width = max(0, int(interval[1]) - int(interval[0]))
    if width >= int(target_width):
        return 0
    needed = int(math.ceil((int(target_width) - width) / 2.0))
    return max(int(min_slack), min(int(max_slack), needed))


def moving_average(signal: torch.Tensor, window: int) -> torch.Tensor:
    """Edge-padded odd-window moving average matching the reference script."""
    if window <= 1:
        return signal.clone()
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = F.pad(signal[None, None], (pad, pad), mode="replicate")
    kernel = torch.ones(1, 1, window, device=signal.device, dtype=signal.dtype) / window
    return F.conv1d(padded, kernel).flatten()


def call_eclip_peak_mask(
    smoothed_signal: torch.Tensor, top_fraction: float
) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    """Call positive eCLIP peaks from the top fraction of smoothed signal."""
    if smoothed_signal.numel() == 0 or smoothed_signal.max() <= 0:
        empty = torch.zeros_like(smoothed_signal, dtype=torch.bool)
        return empty, []
    top_fraction = min(max(float(top_fraction), 1.0 / smoothed_signal.numel()), 1.0)
    threshold = torch.quantile(smoothed_signal.float(), 1.0 - top_fraction)
    peak_mask = (smoothed_signal >= threshold) & (smoothed_signal > 0)
    return peak_mask, contiguous_intervals(peak_mask)


def region_peak_metrics(
    predicted_signal: torch.Tensor,
    peak_intervals: list[tuple[int, int]],
    *,
    predicted_positive_threshold: float,
    peak_slack: int,
    peak_slack_mode: str,
    dynamic_peak_target_width: int,
    dynamic_peak_min_slack: int,
    dynamic_peak_max_slack: int,
) -> tuple[float, float, float]:
    """Compute reference-compatible region precision, recall, and F1."""
    target_hits, predicted_hits, target_count, predicted_count = region_peak_counts(
        predicted_signal,
        peak_intervals,
        predicted_positive_threshold=predicted_positive_threshold,
        peak_slack=peak_slack,
        peak_slack_mode=peak_slack_mode,
        dynamic_peak_target_width=dynamic_peak_target_width,
        dynamic_peak_min_slack=dynamic_peak_min_slack,
        dynamic_peak_max_slack=dynamic_peak_max_slack,
    )
    precision = predicted_hits / predicted_count if predicted_count else 0.0
    recall = target_hits / target_count if target_count else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return float(precision), float(recall), float(f1)


def region_peak_counts(
    predicted_signal: torch.Tensor,
    peak_intervals: list[tuple[int, int]],
    *,
    predicted_positive_threshold: float,
    peak_slack: int,
    peak_slack_mode: str,
    dynamic_peak_target_width: int,
    dynamic_peak_min_slack: int,
    dynamic_peak_max_slack: int,
) -> tuple[int, int, int, int]:
    """Return region hit/count statistics so disjoint RNA segments can be combined."""
    predicted_intervals = contiguous_intervals(
        predicted_signal >= float(predicted_positive_threshold)
    )
    expanded_peaks = expand_peak_intervals(
        predicted_signal.numel(),
        peak_intervals,
        peak_slack=peak_slack,
        peak_slack_mode=peak_slack_mode,
        dynamic_peak_target_width=dynamic_peak_target_width,
        dynamic_peak_min_slack=dynamic_peak_min_slack,
        dynamic_peak_max_slack=dynamic_peak_max_slack,
    )

    def overlaps(first: tuple[int, int], second: tuple[int, int]) -> bool:
        return max(first[0], second[0]) < min(first[1], second[1])

    target_hits = sum(
        any(overlaps(predicted, peak) for predicted in predicted_intervals)
        for peak in expanded_peaks
    )
    predicted_hits = sum(
        any(overlaps(predicted, peak) for peak in expanded_peaks)
        for predicted in predicted_intervals
    )
    return target_hits, predicted_hits, len(expanded_peaks), len(predicted_intervals)


def expand_peak_intervals(
    length: int,
    peak_intervals: list[tuple[int, int]],
    *,
    peak_slack: int,
    peak_slack_mode: str,
    dynamic_peak_target_width: int,
    dynamic_peak_min_slack: int,
    dynamic_peak_max_slack: int,
) -> list[tuple[int, int]]:
    """Expand peak intervals with the same fixed/dynamic rule used for region F1."""
    expanded_peaks = []
    for interval in peak_intervals:
        if peak_slack_mode == "dynamic":
            slack = dynamic_peak_slack(
                interval,
                dynamic_peak_target_width,
                dynamic_peak_min_slack,
                dynamic_peak_max_slack,
            )
        elif peak_slack_mode == "fixed":
            slack = max(0, int(peak_slack))
        else:
            raise ValueError(f"Unknown peak_slack_mode: {peak_slack_mode}")
        expanded_peaks.append(
            (max(0, interval[0] - slack), min(int(length), interval[1] + slack))
        )
    return expanded_peaks


def average_ranks(values: torch.Tensor) -> torch.Tensor:
    """Return average ranks with exact tie handling for Spearman correlation."""
    _, inverse, counts = torch.unique(
        values.float(), sorted=True, return_inverse=True, return_counts=True
    )
    starts = counts.cumsum(0) - counts
    unique_ranks = starts.float() + (counts.float() - 1.0) / 2.0
    return unique_ranks.index_select(0, inverse)


def spearman_correlation(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Compute a finite Spearman correlation, returning zero for constant profiles."""
    if first.numel() < 2 or second.shape != first.shape:
        return first.new_tensor(0.0, dtype=torch.float32)
    first_rank = average_ranks(first)
    second_rank = average_ranks(second)
    first_centered = first_rank - first_rank.mean()
    second_centered = second_rank - second_rank.mean()
    denominator = first_centered.square().sum().sqrt() * second_centered.square().sum().sqrt()
    if denominator <= 1e-8:
        return first.new_tensor(0.0, dtype=torch.float32)
    return (first_centered * second_centered).sum() / denominator


def contiguous_segment_slices(
    residue_indices: torch.Tensor, chain_ids: torch.Tensor
) -> list[slice]:
    """Split cropped RNA positions at chain boundaries and missing residue-index gaps."""
    if residue_indices.ndim != 1 or chain_ids.shape != residue_indices.shape:
        raise ValueError("residue_indices and chain_ids must be one-dimensional tensors of equal shape")
    if residue_indices.numel() == 0:
        return []
    boundaries = (
        (chain_ids[1:] != chain_ids[:-1])
        | ((residue_indices[1:] - residue_indices[:-1]).abs() != 1)
    ).nonzero(as_tuple=False).flatten()
    starts = [0, *(int(index) + 1 for index in boundaries)]
    ends = [*(int(index) + 1 for index in boundaries), residue_indices.numel()]
    return [slice(start, end) for start, end in zip(starts, ends, strict=True)]


class EclipPeakF1Reward(NFTReward):
    """Reward structures by eCLIP peak-region contact F1."""

    def __init__(
        self,
        eclip_smooth_window: int = 5,
        eclip_top_fraction: float = 0.10,
        primary_peak_slack: int = 5,
        peak_slack_mode: str = "dynamic",
        dynamic_peak_target_width: int = 5,
        dynamic_peak_min_slack: int = 0,
        dynamic_peak_max_slack: int = 2,
        predicted_positive_threshold: float = 0.5,
        contact_threshold: float = 8.0,
        distogram_min_bin: float = 2.3125,
        distogram_max_bin: float = 21.6875,
        distogram_no_bins: int = 64,
        distance_softmax_temperature: float = 2.0,
        region_f1_weight: float = 0.45,
        position_jaccard_weight: float = 0.20,
        top_enrichment_weight: float = 0.20,
        rank_correlation_weight: float = 0.15,
        coverage_penalty_weight: float = 0.20,
        min_contact_fraction: float = 0.02,
        max_contact_fraction: float = 0.35,
        ensemble_marginal_weight: float = 0.30,
    ) -> None:
        if peak_slack_mode not in {"fixed", "dynamic"}:
            raise ValueError("peak_slack_mode must be 'fixed' or 'dynamic'")
        if not 0 < eclip_top_fraction <= 1:
            raise ValueError("eclip_top_fraction must be in (0, 1]")
        self.eclip_smooth_window = eclip_smooth_window
        self.eclip_top_fraction = eclip_top_fraction
        self.primary_peak_slack = primary_peak_slack
        self.peak_slack_mode = peak_slack_mode
        self.dynamic_peak_target_width = dynamic_peak_target_width
        self.dynamic_peak_min_slack = dynamic_peak_min_slack
        self.dynamic_peak_max_slack = dynamic_peak_max_slack
        self.predicted_positive_threshold = predicted_positive_threshold
        self.contact_threshold = contact_threshold
        self.distogram_min_bin = distogram_min_bin
        self.distogram_max_bin = distogram_max_bin
        self.distogram_no_bins = distogram_no_bins
        self.distance_softmax_temperature = distance_softmax_temperature
        if not 0.0 <= min_contact_fraction <= max_contact_fraction <= 1.0:
            raise ValueError("contact-fraction bounds must satisfy 0 <= min <= max <= 1")
        self.region_f1_weight = region_f1_weight
        self.position_jaccard_weight = position_jaccard_weight
        self.top_enrichment_weight = top_enrichment_weight
        self.rank_correlation_weight = rank_correlation_weight
        self.coverage_penalty_weight = coverage_penalty_weight
        self.min_contact_fraction = min_contact_fraction
        self.max_contact_fraction = max_contact_fraction
        self.ensemble_marginal_weight = ensemble_marginal_weight

    def _profile_metrics(
        self,
        predicted_signal: torch.Tensor,
        smoothed_segments: list[torch.Tensor],
        segment_slices: list[slice],
        segment_peak_intervals: list[list[tuple[int, int]]],
    ) -> dict[str, torch.Tensor]:
        """Score one contact profile with region and anti-reward-hacking metrics."""
        target_hits = predicted_hits = target_count = predicted_count = 0
        peak_masks = []
        for segment, peak_intervals in zip(
            segment_slices, segment_peak_intervals, strict=True
        ):
            segment_prediction = predicted_signal[segment]
            counts = region_peak_counts(
                segment_prediction,
                peak_intervals,
                predicted_positive_threshold=self.predicted_positive_threshold,
                peak_slack=self.primary_peak_slack,
                peak_slack_mode=self.peak_slack_mode,
                dynamic_peak_target_width=self.dynamic_peak_target_width,
                dynamic_peak_min_slack=self.dynamic_peak_min_slack,
                dynamic_peak_max_slack=self.dynamic_peak_max_slack,
            )
            target_hits += counts[0]
            predicted_hits += counts[1]
            target_count += counts[2]
            predicted_count += counts[3]
            expanded = expand_peak_intervals(
                segment_prediction.numel(),
                peak_intervals,
                peak_slack=self.primary_peak_slack,
                peak_slack_mode=self.peak_slack_mode,
                dynamic_peak_target_width=self.dynamic_peak_target_width,
                dynamic_peak_min_slack=self.dynamic_peak_min_slack,
                dynamic_peak_max_slack=self.dynamic_peak_max_slack,
            )
            peak_mask = torch.zeros_like(segment_prediction, dtype=torch.bool)
            for start, end in expanded:
                peak_mask[start:end] = True
            peak_masks.append(peak_mask)

        precision = predicted_hits / predicted_count if predicted_count else 0.0
        recall = target_hits / target_count if target_count else 0.0
        region_f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
        peak_mask = torch.cat(peak_masks)
        predicted_positive = predicted_signal >= self.predicted_positive_threshold
        intersection = (peak_mask & predicted_positive).sum().float()
        union = (peak_mask | predicted_positive).sum().float()
        position_jaccard = intersection / union.clamp_min(1.0)
        peak_density = (
            predicted_signal[peak_mask].mean() if peak_mask.any() else predicted_signal.new_tensor(0.0)
        )
        background_mask = ~peak_mask
        background_density = (
            predicted_signal[background_mask].mean()
            if background_mask.any()
            else predicted_signal.new_tensor(0.0)
        )
        log_enrichment = torch.log(
            (peak_density.float() + 1e-6) / (background_density.float() + 1e-6)
        )
        top_enrichment = 0.5 * (torch.tanh(log_enrichment) + 1.0)
        smoothed_target = torch.cat(smoothed_segments)
        rank_correlation = 0.5 * (
            spearman_correlation(smoothed_target, predicted_signal).clamp(-1.0, 1.0) + 1.0
        )
        contact_fraction = predicted_positive.float().mean()
        coverage_penalty = (
            torch.relu(contact_fraction.new_tensor(self.min_contact_fraction) - contact_fraction)
            + torch.relu(contact_fraction - contact_fraction.new_tensor(self.max_contact_fraction))
        )
        enhanced_score = (
            self.region_f1_weight * predicted_signal.new_tensor(region_f1)
            + self.position_jaccard_weight * position_jaccard
            + self.top_enrichment_weight * top_enrichment
            + self.rank_correlation_weight * rank_correlation
            - self.coverage_penalty_weight * coverage_penalty
        )
        return {
            "peak_precision": predicted_signal.new_tensor(precision),
            "peak_recall": predicted_signal.new_tensor(recall),
            "peak_f1": predicted_signal.new_tensor(region_f1),
            "position_jaccard": position_jaccard,
            "top_enrichment": top_enrichment,
            "rank_correlation": rank_correlation,
            "contact_fraction": contact_fraction,
            "coverage_penalty": coverage_penalty,
            "background_contact_density": background_density,
            "enhanced_score": enhanced_score,
        }

    def _structure_signal(
        self,
        coordinates: torch.Tensor,
        protein_atom_mask: torch.Tensor,
        rna_atom_mask: torch.Tensor,
    ) -> torch.Tensor:
        protein_coordinates = coordinates[:, protein_atom_mask]
        rna_coordinates = coordinates[:, rna_atom_mask]
        min_distance = torch.cdist(rna_coordinates.float(), protein_coordinates.float()).amin(dim=-1)

        bin_width = (self.distogram_max_bin - self.distogram_min_bin) / self.distogram_no_bins
        bin_centers = torch.linspace(
            self.distogram_min_bin,
            self.distogram_max_bin - bin_width,
            self.distogram_no_bins,
            device=coordinates.device,
            dtype=torch.float32,
        ) + 0.5 * bin_width
        logits = -0.5 * (
            (bin_centers[None, None] - min_distance[..., None])
            / max(float(self.distance_softmax_temperature), 1e-6)
        ).square()
        probabilities = torch.softmax(logits, dim=-1)
        return probabilities[..., bin_centers < self.contact_threshold].sum(dim=-1)

    @staticmethod
    def _label_tensor(
        label_dict: Mapping[str, Any], names: tuple[str, ...]
    ) -> torch.Tensor:
        for name in names:
            if name in label_dict:
                value = label_dict[name]
                return value.squeeze(0) if value.ndim > 1 and value.shape[0] == 1 else value
        raise KeyError(f"Missing required eCLIP label; expected one of {names}")

    @torch.no_grad()
    def check_condition(
        self,
        input_feature_dict: Mapping[str, Any],
        label_dict: Mapping[str, Any],
    ) -> ConditionCheck:
        """Reject molecular content or labels that cannot yield an eCLIP reward."""
        representative_mask = input_feature_dict["distogram_rep_atom_mask"].squeeze().bool()
        atom_to_token_idx = input_feature_dict["atom_to_token_idx"].squeeze().long()
        valid_representative = representative_mask & (atom_to_token_idx >= 0)
        protein_atom_mask = (
            input_feature_dict["is_protein"].squeeze().bool() & valid_representative
        )
        rna_atom_mask = (
            input_feature_dict["is_rna"].squeeze().bool() & valid_representative
        )
        if not protein_atom_mask.any():
            return ConditionCheck(valid=False, reason="missing_protein")
        if not rna_atom_mask.any():
            return ConditionCheck(valid=False, reason="missing_rna")

        token_signal = self._label_tensor(
            label_dict, ("eclip_rna_binding_signal", "rna_binding_signal")
        ).to(device=atom_to_token_idx.device, dtype=torch.float32)
        token_signal_mask = self._label_tensor(
            label_dict,
            ("eclip_rna_binding_signal_mask", "rna_binding_signal_mask"),
        ).to(device=atom_to_token_idx.device, dtype=torch.bool)
        rna_token_indices = atom_to_token_idx[rna_atom_mask]
        valid_signal_positions = token_signal_mask.index_select(0, rna_token_indices)
        if not valid_signal_positions.any():
            return ConditionCheck(valid=False, reason="missing_eclip_signal")
        target_signal = token_signal.index_select(0, rna_token_indices)[
            valid_signal_positions
        ]
        if target_signal.numel() == 0 or target_signal.max() <= 0:
            return ConditionCheck(valid=False, reason="no_positive_eclip_peak")
        return ConditionCheck(valid=True)

    @torch.no_grad()
    def __call__(
        self,
        coordinates: torch.Tensor,
        input_feature_dict: Mapping[str, Any],
        label_dict: Mapping[str, Any],
    ) -> RewardOutput:
        if coordinates.ndim != 3:
            raise ValueError("eCLIP reward expects coordinates with shape [N_sample, N_atom, 3]")
        representative_mask = input_feature_dict["distogram_rep_atom_mask"].squeeze().bool()
        atom_to_token_idx = input_feature_dict["atom_to_token_idx"].squeeze().long()
        # Generated CIFs contain coordinates for every model atom. Use the
        # token mapping to exclude only padding, rather than the experimental
        # structure's resolved-coordinate mask.
        valid_representative = representative_mask & (atom_to_token_idx >= 0)
        protein_atom_mask = input_feature_dict["is_protein"].squeeze().bool() & valid_representative
        rna_atom_mask = input_feature_dict["is_rna"].squeeze().bool() & valid_representative
        if not protein_atom_mask.any() or not rna_atom_mask.any():
            rewards = torch.zeros(coordinates.shape[0], device=coordinates.device)
            return RewardOutput(
                rewards,
                {"eclip_valid": rewards.new_tensor(0.0)},
                torch.zeros_like(rewards, dtype=torch.bool),
            )

        token_signal = self._label_tensor(
            label_dict, ("eclip_rna_binding_signal", "rna_binding_signal")
        ).to(device=coordinates.device, dtype=torch.float32)
        token_signal_mask = self._label_tensor(
            label_dict,
            ("eclip_rna_binding_signal_mask", "rna_binding_signal_mask"),
        ).to(device=coordinates.device, dtype=torch.bool)
        rna_token_indices = atom_to_token_idx[rna_atom_mask]
        valid_signal_positions = token_signal_mask.index_select(0, rna_token_indices)
        if not valid_signal_positions.any():
            rewards = torch.zeros(coordinates.shape[0], device=coordinates.device)
            return RewardOutput(
                rewards,
                {"eclip_valid": rewards.new_tensor(0.0)},
                torch.zeros_like(rewards, dtype=torch.bool),
            )

        target_signal = token_signal.index_select(0, rna_token_indices)[valid_signal_positions]
        token_residue_indices = input_feature_dict.get("residue_index")
        if token_residue_indices is None:
            token_residue_indices = torch.arange(
                token_signal.numel(), device=coordinates.device
            )
        else:
            token_residue_indices = token_residue_indices.squeeze().long()
        token_chain_ids = input_feature_dict.get("asym_id")
        if token_chain_ids is None:
            token_chain_ids = torch.zeros_like(token_residue_indices)
        else:
            token_chain_ids = token_chain_ids.squeeze().long()
        selected_rna_token_indices = rna_token_indices[valid_signal_positions]
        segment_slices = contiguous_segment_slices(
            token_residue_indices.index_select(0, selected_rna_token_indices),
            token_chain_ids.index_select(0, selected_rna_token_indices),
        )
        selected_rna_atom_mask = rna_atom_mask.clone()
        selected_rna_atom_mask[rna_atom_mask] = valid_signal_positions
        predicted_signals = self._structure_signal(
            coordinates, protein_atom_mask, selected_rna_atom_mask
        )

        smoothed_segments = [
            moving_average(target_signal[segment], self.eclip_smooth_window)
            for segment in segment_slices
        ]
        all_smoothed = torch.cat(smoothed_segments)
        if all_smoothed.numel() == 0 or all_smoothed.max() <= 0:
            rewards = torch.zeros(coordinates.shape[0], device=coordinates.device)
            return RewardOutput(
                rewards,
                {
                    "eclip_valid": rewards.new_tensor(0.0),
                    "eclip_peak_count": rewards.new_tensor(0.0),
                },
                torch.zeros_like(rewards, dtype=torch.bool),
            )
        top_fraction = min(
            max(float(self.eclip_top_fraction), 1.0 / all_smoothed.numel()), 1.0
        )
        peak_threshold = torch.quantile(all_smoothed.float(), 1.0 - top_fraction)
        segment_peak_intervals = [
            contiguous_intervals((segment >= peak_threshold) & (segment > 0))
            for segment in smoothed_segments
        ]
        peak_count = sum(len(intervals) for intervals in segment_peak_intervals)
        if peak_count == 0:
            rewards = torch.zeros(coordinates.shape[0], device=coordinates.device)
            return RewardOutput(
                rewards,
                {
                    "eclip_valid": rewards.new_tensor(0.0),
                    "eclip_peak_count": rewards.new_tensor(0.0),
                },
                torch.zeros_like(rewards, dtype=torch.bool),
            )

        per_sample_metrics = [
            self._profile_metrics(
                predicted_signal,
                smoothed_segments,
                segment_slices,
                segment_peak_intervals,
            )
            for predicted_signal in predicted_signals
        ]
        metric_tensors = {
            name: torch.stack([metrics[name].float() for metrics in per_sample_metrics])
            for name in per_sample_metrics[0]
        }
        rewards_tensor = metric_tensors["enhanced_score"]
        if predicted_signals.shape[0] > 1:
            profile_sum = predicted_signals.sum(dim=0)
            ensemble_profile = profile_sum / predicted_signals.shape[0]
            ensemble_score = self._profile_metrics(
                ensemble_profile,
                smoothed_segments,
                segment_slices,
                segment_peak_intervals,
            )["enhanced_score"]
            leave_one_out_scores = torch.stack(
                [
                    self._profile_metrics(
                        (profile_sum - predicted_signal) / (predicted_signals.shape[0] - 1),
                        smoothed_segments,
                        segment_slices,
                        segment_peak_intervals,
                    )["enhanced_score"]
                    for predicted_signal in predicted_signals
                ]
            )
            ensemble_marginal = ensemble_score - leave_one_out_scores
        else:
            ensemble_score = rewards_tensor[0]
            ensemble_marginal = torch.zeros_like(rewards_tensor)
        return RewardOutput(
            rewards=rewards_tensor,
            metrics={
                "eclip_valid": rewards_tensor.new_tensor(1.0),
                "eclip_peak_precision": metric_tensors["peak_precision"].mean(),
                "eclip_peak_recall": metric_tensors["peak_recall"].mean(),
                "eclip_peak_f1": metric_tensors["peak_f1"].mean(),
                "eclip_position_jaccard": metric_tensors["position_jaccard"].mean(),
                "eclip_top_enrichment": metric_tensors["top_enrichment"].mean(),
                "eclip_rank_correlation": metric_tensors["rank_correlation"].mean(),
                "eclip_contact_fraction": metric_tensors["contact_fraction"].mean(),
                "eclip_coverage_penalty": metric_tensors["coverage_penalty"].mean(),
                "eclip_background_contact_density": metric_tensors[
                    "background_contact_density"
                ].mean(),
                "eclip_enhanced_score": rewards_tensor.mean(),
                "eclip_ensemble_score": ensemble_score,
                "eclip_ensemble_marginal": ensemble_marginal.mean(),
                "eclip_peak_count": rewards_tensor.new_tensor(float(peak_count)),
                "eclip_segment_count": rewards_tensor.new_tensor(float(len(segment_slices))),
                "eclip_signal_position_count": rewards_tensor.new_tensor(
                    float(valid_signal_positions.sum())
                ),
            },
            valid_mask=torch.ones_like(rewards_tensor, dtype=torch.bool),
            components={
                "eclip": rewards_tensor,
                "ensemble_marginal": ensemble_marginal,
            },
            component_weights={
                "eclip": 1.0,
                "ensemble_marginal": self.ensemble_marginal_weight,
            },
        )
