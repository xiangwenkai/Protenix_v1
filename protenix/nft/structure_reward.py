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

"""Rewards for physically valid, signal-compatible distillation ensembles."""

from typing import Any, Mapping

import torch

from protenix.nft.eclip_reward import EclipPeakF1Reward
from protenix.nft.reward import ConditionCheck, NFTReward, RewardOutput


def _squeeze_batch(value: torch.Tensor) -> torch.Tensor:
    """Remove the single dataloader batch dimension without collapsing feature axes."""
    return value.squeeze(0) if value.ndim > 1 and value.shape[0] == 1 else value


class PseudoLocalLDDTReward(NFTReward):
    """Weakly preserve high-confidence, same-chain geometry from a pseudo structure."""

    def __init__(
        self,
        confidence_threshold: float = 70.0,
        distance_cutoff: float = 30.0,
        min_pair_count: int = 8,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.distance_cutoff = distance_cutoff
        self.min_pair_count = min_pair_count

    @torch.no_grad()
    def __call__(
        self,
        coordinates: torch.Tensor,
        input_feature_dict: Mapping[str, Any],
        label_dict: Mapping[str, Any],
    ) -> RewardOutput:
        sample_count = coordinates.shape[0]
        zeros = torch.zeros(sample_count, device=coordinates.device, dtype=torch.float32)
        if "distillation_residue_plddt" not in label_dict:
            return RewardOutput(
                zeros,
                {"pseudo_local_valid": zeros.new_tensor(0.0)},
                torch.ones_like(zeros, dtype=torch.bool),
                {"pseudo_local_lddt": zeros},
                {"pseudo_local_lddt": 1.0},
            )

        representative_mask = _squeeze_batch(
            input_feature_dict["distogram_rep_atom_mask"]
        ).bool()
        atom_to_token = _squeeze_batch(input_feature_dict["atom_to_token_idx"]).long()
        coordinate_mask = _squeeze_batch(label_dict["coordinate_mask"]).bool()
        valid_atoms = representative_mask & coordinate_mask & (atom_to_token >= 0)
        if valid_atoms.sum() < 2:
            return RewardOutput(
                zeros,
                {"pseudo_local_valid": zeros.new_tensor(0.0)},
                torch.ones_like(zeros, dtype=torch.bool),
                {"pseudo_local_lddt": zeros},
                {"pseudo_local_lddt": 1.0},
            )

        token_indices = atom_to_token[valid_atoms]
        true_coordinates = _squeeze_batch(label_dict["coordinate"])[valid_atoms].float()
        predicted_coordinates = coordinates[:, valid_atoms].float()
        asym_id = _squeeze_batch(input_feature_dict["asym_id"]).long().index_select(
            0, token_indices
        )
        confidence = _squeeze_batch(label_dict["distillation_residue_plddt"]).float()
        confidence = confidence.index_select(0, token_indices)
        confidence_mask = label_dict.get("distillation_residue_plddt_mask")
        if confidence_mask is None:
            confidence_mask = torch.isfinite(confidence)
        else:
            confidence_mask = _squeeze_batch(confidence_mask).bool().index_select(
                0, token_indices
            )
        if confidence.max() <= 1.5:
            confidence = confidence * 100.0
        high_confidence = confidence_mask & (confidence >= self.confidence_threshold)

        true_distance = torch.cdist(true_coordinates, true_coordinates)
        pair_mask = (
            high_confidence[:, None]
            & high_confidence[None, :]
            & (asym_id[:, None] == asym_id[None, :])
            & (true_distance > 1e-3)
            & (true_distance <= self.distance_cutoff)
        )
        pair_mask = torch.triu(pair_mask, diagonal=1)
        pair_count = int(pair_mask.sum().item())
        if pair_count < self.min_pair_count:
            return RewardOutput(
                zeros,
                {
                    "pseudo_local_valid": zeros.new_tensor(0.0),
                    "pseudo_local_pair_count": zeros.new_tensor(float(pair_count)),
                },
                torch.ones_like(zeros, dtype=torch.bool),
                {"pseudo_local_lddt": zeros},
                {"pseudo_local_lddt": 1.0},
            )

        normalized_confidence = (confidence / 100.0).clamp(0.0, 1.0)
        pair_weight = (
            normalized_confidence[:, None] * normalized_confidence[None, :]
        )[pair_mask]
        predicted_distance = torch.cdist(predicted_coordinates, predicted_coordinates)
        distance_error = (
            predicted_distance - true_distance.unsqueeze(0)
        ).abs()[:, pair_mask]
        per_pair_lddt = sum(
            (distance_error < threshold).float() for threshold in (0.5, 1.0, 2.0, 4.0)
        ) / 4.0
        rewards = (per_pair_lddt * pair_weight).sum(dim=-1) / pair_weight.sum().clamp_min(1e-6)
        return RewardOutput(
            rewards,
            {
                "pseudo_local_valid": rewards.new_tensor(1.0),
                "pseudo_local_lddt": rewards.mean(),
                "pseudo_local_pair_count": rewards.new_tensor(float(pair_count)),
                "pseudo_local_high_confidence_token_count": rewards.new_tensor(
                    float(high_confidence.sum())
                ),
            },
            torch.ones_like(rewards, dtype=torch.bool),
            {"pseudo_local_lddt": rewards},
            {"pseudo_local_lddt": 1.0},
        )


class PhysicalValidityReward(NFTReward):
    """Penalize non-finite coordinates, steric clashes, and polymer chain breaks."""

    def __init__(
        self,
        clash_distance: float = 1.5,
        max_clash_fraction: float = 0.02,
        clash_chunk_size: int = 512,
        chain_break_tolerance: float = 3.0,
        severe_chain_break_tolerance: float = 8.0,
        hard_reject_chain_breaks: bool = False,
        clash_penalty_scale: float = 10.0,
    ) -> None:
        self.clash_distance = clash_distance
        self.max_clash_fraction = max_clash_fraction
        self.clash_chunk_size = clash_chunk_size
        self.chain_break_tolerance = chain_break_tolerance
        self.severe_chain_break_tolerance = severe_chain_break_tolerance
        self.hard_reject_chain_breaks = hard_reject_chain_breaks
        self.clash_penalty_scale = clash_penalty_scale

    def _clash_fraction(
        self,
        coordinates: torch.Tensor,
        atom_to_token: torch.Tensor,
        token_asym_id: torch.Tensor,
        token_residue_index: torch.Tensor,
        token_bonds: torch.Tensor | None,
    ) -> torch.Tensor:
        """Count unique nonlocal atom pairs closer than the configured threshold."""
        atom_count = coordinates.shape[-2]
        global_indices = torch.arange(atom_count, device=coordinates.device)
        atom_asym_id = token_asym_id.index_select(0, atom_to_token)
        atom_residue_index = token_residue_index.index_select(0, atom_to_token)
        clash_counts = torch.zeros(coordinates.shape[0], device=coordinates.device)
        for start in range(0, atom_count, self.clash_chunk_size):
            end = min(start + self.clash_chunk_size, atom_count)
            row_indices = global_indices[start:end]
            distances = torch.cdist(coordinates[:, start:end].float(), coordinates.float())
            different_token = atom_to_token[start:end, None] != atom_to_token[None, :]
            same_chain = atom_asym_id[start:end, None] == atom_asym_id[None, :]
            adjacent_residue = (
                atom_residue_index[start:end, None] - atom_residue_index[None, :]
            ).abs() <= 1
            nonlocal_pair = different_token & ~(same_chain & adjacent_residue)
            if token_bonds is not None:
                bonded_token = token_bonds[
                    atom_to_token[start:end, None], atom_to_token[None, :]
                ].bool()
                nonlocal_pair &= ~bonded_token
            unique_pair = global_indices[None, :] > row_indices[:, None]
            clash_mask = (distances < self.clash_distance) & nonlocal_pair & unique_pair
            clash_counts += clash_mask.sum(dim=(-2, -1))
        return clash_counts / max(atom_count, 1)

    @torch.no_grad()
    def __call__(
        self,
        coordinates: torch.Tensor,
        input_feature_dict: Mapping[str, Any],
        label_dict: Mapping[str, Any],
    ) -> RewardOutput:
        finite_mask = torch.isfinite(coordinates).all(dim=(-2, -1))
        safe_coordinates = torch.nan_to_num(coordinates.float())
        atom_to_token_full = _squeeze_batch(input_feature_dict["atom_to_token_idx"]).long()
        ref_mask = _squeeze_batch(input_feature_dict["ref_mask"]).bool()
        valid_atom_mask = ref_mask & (atom_to_token_full >= 0)
        ref_element = input_feature_dict.get("ref_element")
        if ref_element is not None:
            atomic_number = _squeeze_batch(ref_element).argmax(dim=-1)
            valid_atom_mask &= atomic_number != 1
        atom_to_token = atom_to_token_full[valid_atom_mask]
        token_asym_id = _squeeze_batch(input_feature_dict["asym_id"]).long()
        token_residue_index = _squeeze_batch(input_feature_dict["residue_index"]).long()
        token_bonds = input_feature_dict.get("token_bonds")
        if token_bonds is not None:
            token_bonds = _squeeze_batch(token_bonds)
        clash_fraction = self._clash_fraction(
            safe_coordinates[:, valid_atom_mask],
            atom_to_token,
            token_asym_id,
            token_residue_index,
            token_bonds,
        )

        representative_mask = _squeeze_batch(
            input_feature_dict["distogram_rep_atom_mask"]
        ).bool()
        coordinate_mask = _squeeze_batch(label_dict["coordinate_mask"]).bool()
        valid_representative = representative_mask & coordinate_mask & (atom_to_token_full >= 0)
        rep_token = atom_to_token_full[valid_representative]
        rep_asym = token_asym_id.index_select(0, rep_token)
        rep_residue = token_residue_index.index_select(0, rep_token)
        consecutive = (rep_asym[1:] == rep_asym[:-1]) & (
            (rep_residue[1:] - rep_residue[:-1]).abs() == 1
        )
        true_rep = _squeeze_batch(label_dict["coordinate"])[valid_representative].float()
        predicted_rep = safe_coordinates[:, valid_representative]
        if consecutive.any():
            true_step = torch.linalg.vector_norm(true_rep[1:] - true_rep[:-1], dim=-1)[
                consecutive
            ]
            predicted_step = torch.linalg.vector_norm(
                predicted_rep[:, 1:] - predicted_rep[:, :-1], dim=-1
            )[:, consecutive]
            step_error = (predicted_step - true_step).abs()
            chain_break_penalty = torch.relu(step_error - self.chain_break_tolerance).mean(
                dim=-1
            ) / max(self.chain_break_tolerance, 1e-6)
            severe_chain_break = (step_error > self.severe_chain_break_tolerance).any(dim=-1)
        else:
            chain_break_penalty = torch.zeros_like(clash_fraction)
            severe_chain_break = torch.zeros_like(finite_mask)

        penalty = (
            self.clash_penalty_scale * clash_fraction.clamp(max=1.0)
            + chain_break_penalty.clamp(max=1.0)
            + (~finite_mask).float()
        )
        valid_mask = finite_mask & (clash_fraction <= self.max_clash_fraction)
        if self.hard_reject_chain_breaks:
            valid_mask &= ~severe_chain_break
        return RewardOutput(
            rewards=-penalty,
            metrics={
                "physical_valid_fraction": valid_mask.float().mean(),
                "physical_penalty": penalty.mean(),
                "physical_clash_fraction": clash_fraction.mean(),
                "physical_chain_break_penalty": chain_break_penalty.mean(),
                "physical_nonfinite_fraction": (~finite_mask).float().mean(),
            },
            valid_mask=valid_mask,
            components={"physical_penalty": penalty},
            component_weights={"physical_penalty": -1.0},
        )


class DistillationEclipReward(NFTReward):
    """Composite reward for pseudo structures paired with experimental eCLIP signal."""

    def __init__(
        self,
        eclip: Mapping[str, Any] | None = None,
        physical: Mapping[str, Any] | None = None,
        pseudo_local: Mapping[str, Any] | None = None,
        component_weights: Mapping[str, float] | None = None,
    ) -> None:
        self.eclip = EclipPeakF1Reward(**dict(eclip or {}))
        self.physical = PhysicalValidityReward(**dict(physical or {}))
        self.pseudo_local = PseudoLocalLDDTReward(**dict(pseudo_local or {}))
        self.component_weights = {
            "eclip": 1.0,
            "ensemble_marginal": 0.30,
            "pseudo_local_lddt": 0.15,
            "physical_penalty": -0.30,
            **dict(component_weights or {}),
        }

    def check_condition(
        self,
        input_feature_dict: Mapping[str, Any],
        label_dict: Mapping[str, Any],
    ) -> ConditionCheck:
        """Use the primary experimental signal to reject unusable conditions early."""
        return self.eclip.check_condition(input_feature_dict, label_dict)

    @torch.no_grad()
    def __call__(
        self,
        coordinates: torch.Tensor,
        input_feature_dict: Mapping[str, Any],
        label_dict: Mapping[str, Any],
    ) -> RewardOutput:
        safe_coordinates = torch.nan_to_num(coordinates.float())
        eclip_output = self.eclip(safe_coordinates, input_feature_dict, label_dict)
        eclip_valid = eclip_output.valid_mask
        if eclip_valid is None:
            eclip_valid = torch.ones(
                coordinates.shape[0], device=coordinates.device, dtype=torch.bool
            )
        if not eclip_valid.any():
            zeros = torch.zeros(
                coordinates.shape[0], device=coordinates.device, dtype=torch.float32
            )
            components = {
                "eclip": eclip_output.components.get("eclip", zeros),
                "ensemble_marginal": eclip_output.components.get(
                    "ensemble_marginal", zeros
                ),
                "pseudo_local_lddt": zeros,
                "physical_penalty": zeros,
            }
            return RewardOutput(
                rewards=zeros,
                metrics={
                    **eclip_output.metrics,
                    "combined_raw_reward": zeros.new_tensor(0.0),
                },
                valid_mask=eclip_valid,
                components=components,
                component_weights=self.component_weights,
            )

        physical_output = self.physical(coordinates, input_feature_dict, label_dict)
        pseudo_output = self.pseudo_local(safe_coordinates, input_feature_dict, label_dict)
        sample_count = coordinates.shape[0]
        components = {
            "eclip": eclip_output.components.get(
                "eclip", torch.zeros(sample_count, device=coordinates.device)
            ),
            "ensemble_marginal": eclip_output.components.get(
                "ensemble_marginal", torch.zeros(sample_count, device=coordinates.device)
            ),
            "pseudo_local_lddt": pseudo_output.components["pseudo_local_lddt"],
            "physical_penalty": physical_output.components["physical_penalty"],
        }
        rewards = sum(
            float(self.component_weights[name]) * value for name, value in components.items()
        )
        physical_valid = physical_output.valid_mask
        if physical_valid is None:
            physical_valid = torch.ones_like(eclip_valid)
        metrics = {
            **eclip_output.metrics,
            **pseudo_output.metrics,
            **physical_output.metrics,
            "combined_raw_reward": rewards.mean(),
        }
        return RewardOutput(
            rewards=rewards,
            metrics=metrics,
            valid_mask=eclip_valid & physical_valid,
            components=components,
            component_weights=self.component_weights,
        )
