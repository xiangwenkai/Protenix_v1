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

"""Configuration for Protenix structure + RNA binding signal SFT."""


rna_signal_sft_configs = {
    "rna_signal_sft": {
        "signal_profile_weight": 1.0,
        "signal_target_threshold": 0.0,
        "signal_min_peak": 0.5,
        "signal_binary_threshold": 0.5,
        "signal_loss_weight": 0.05,
        # Deprecated count-profile options kept for compatibility with older
        # command lines. PDB signal loss now uses normalized KL divergence.
        "signal_multinomial_min_height": 3.0,
        "signal_clip_value": 100.0,
        "signal_multinomial_max_total": 100.0,
    }
}
