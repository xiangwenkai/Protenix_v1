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

"""Default configuration for eCLIP PPFT training."""

from protenix.config.extend_types import DefaultNoneWithType, ListValue, ValueMaybeNone


eclip_ppft_configs = {
    "eclip_ppft": {
        "data_dir": "/inspire/ssd/project/sais-bio/public/xiangwenkai/GITHUB/parnet/data_process/high_quality_positive",
        "protein_sequence_tsv": "/inspire/ssd/project/sais-bio/public/xiangwenkai/GITHUB/parnet/parnet/assets/ENCODE.protein_symbol2sequence.uniprot.tsv",
        "train_split": "train",
        "eval_split": "validation",
        "num_workers": 0,
        "batch_size": 1,
        "train_limit": DefaultNoneWithType(int),
        "eval_limit": ValueMaybeNone(128),
        "eval_max_steps": ValueMaybeNone(32),
        "max_rna_length": ValueMaybeNone(600),
        "rna_crop_size": ValueMaybeNone(256),
        "max_protein_length": ValueMaybeNone(600),
        "min_signal_max": DefaultNoneWithType(float),
        "zero_signal_keep_prob": 1.0,
        "shuffle_files": True,
        "shuffle_buffer": 256,
        "parquet_batch_size": 1024,
        "confidence_rollout_steps": 20,
        "n_cycle": DefaultNoneWithType(int),
        "inplace_safe": False,
        "diffusion_attn_chunk_size": DefaultNoneWithType(int),
        "distogram_contact_threshold": 8.0,
        "signal_profile_weight": 1.0,
        "signal_positive_weight": 1.0,
        "signal_point_weight": 0.2,
        "signal_loss_weight": 1.0,
        "confidence_quality_weight": 0.2,
        "confidence_quality_target": 0.8,
        "confidence_monitor_clash": True,
        "sidecar_lr": 1e-4,
        "train_sidecar": False,
        "train_last_pairformer_blocks": -1,
        "extra_trainable_substrings": ListValue([""]),
        "save_every_steps": 100,
        "eval_every_steps": 100,
        "log_every_steps": 10,
        "skip_bad_samples": True,
    }
}
