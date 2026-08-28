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

"""Online DiffusionNFT building blocks for Protenix."""

from protenix.nft.advantage import (
    combine_reward_component_advantages,
    compute_grouped_advantages,
)
from protenix.nft.eclip_reward import EclipPeakF1Reward
from protenix.nft.loss import DiffusionNFTLoss, NFTLossOutput
from protenix.nft.policy import PolicyCopies, scheduled_old_policy_decay
from protenix.nft.reward import (
    ConditionCheck,
    NFTReward,
    NullReward,
    RewardOutput,
    WeightedReward,
    build_reward,
    register_reward,
)
from protenix.nft.structure_reward import (
    DistillationEclipReward,
    PhysicalValidityReward,
    PseudoLocalLDDTReward,
)

__all__ = [
    "ConditionCheck",
    "DiffusionNFTLoss",
    "DistillationEclipReward",
    "EclipPeakF1Reward",
    "NFTLossOutput",
    "NFTReward",
    "NullReward",
    "PolicyCopies",
    "PhysicalValidityReward",
    "PseudoLocalLDDTReward",
    "RewardOutput",
    "WeightedReward",
    "build_reward",
    "combine_reward_component_advantages",
    "compute_grouped_advantages",
    "register_reward",
    "scheduled_old_policy_decay",
]
