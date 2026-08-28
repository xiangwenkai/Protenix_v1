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

"""Default configuration for online Protenix DiffusionNFT training."""

from protenix.config.extend_types import ListValue

nft_configs = {
    "data": {
        "train_sets": ListValue(["distillation_eclip_v1"]),
        "test_sets": ListValue([], dtype=str),
        "distillation_eclip_v1": {
            "base_info": {
                "inclusion": {"eval_type": ListValue(["rna_prot"])},
            },
        },
    },
    "nft": {
        "samples_per_condition": 4,
        "rollout_steps": 20,
        "rollout_groups_per_outer_step": 4,
        "forward_noise_samples": 4,
        "max_rollout_attempts": 64,
        "inner_steps": 1,
        "learning_rate": 1e-5,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "weight_decay": 1e-4,
        "max_grad_norm": 1.0,
        "interpolation_beta": 1.0,
        "reference_weight": 1e-4,
        "advantage_clip": 5.0,
        "global_reward_std": True,
        "old_policy_decay": 0.9,
        "old_policy_decay_type": 1,
        "random_coordinate_augmentation": True,
        "save_at_end": True,
        "resume_from": "",
        "reward": {
            "name": "distillation_eclip",
            "kwargs": {
                "component_weights": {
                    "eclip": 1.0,
                    "ensemble_marginal": 0.30,
                    "pseudo_local_lddt": 0.15,
                    "physical_penalty": -0.30,
                },
                "eclip": {
                    "eclip_smooth_window": 5,
                    "eclip_top_fraction": 0.10,
                    "primary_peak_slack": 5,
                    "peak_slack_mode": "dynamic",
                    "dynamic_peak_target_width": 5,
                    "dynamic_peak_min_slack": 0,
                    "dynamic_peak_max_slack": 2,
                    "predicted_positive_threshold": 0.5,
                    "contact_threshold": 8.0,
                    "distogram_min_bin": 2.3125,
                    "distogram_max_bin": 21.6875,
                    "distogram_no_bins": 64,
                    "distance_softmax_temperature": 2.0,
                    "region_f1_weight": 0.45,
                    "position_jaccard_weight": 0.20,
                    "top_enrichment_weight": 0.20,
                    "rank_correlation_weight": 0.15,
                    "coverage_penalty_weight": 0.20,
                    "min_contact_fraction": 0.02,
                    "max_contact_fraction": 0.35,
                    "ensemble_marginal_weight": 0.30,
                },
                "physical": {
                    "clash_distance": 1.5,
                    "max_clash_fraction": 0.02,
                    "clash_chunk_size": 512,
                    "chain_break_tolerance": 3.0,
                    "severe_chain_break_tolerance": 8.0,
                    "hard_reject_chain_breaks": False,
                    "clash_penalty_scale": 10.0,
                },
                "pseudo_local": {
                    "confidence_threshold": 70.0,
                    "distance_cutoff": 30.0,
                    "min_pair_count": 8,
                },
            },
        },
    }
}
