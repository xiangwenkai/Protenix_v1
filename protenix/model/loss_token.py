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
Token-Level Loss Functions

Token级别的损失函数，只计算token中心原子坐标的损失。
"""

from functools import lru_cache
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from protenix.model.loss import (
    loss_reduction,
    _get_off_diagonal_mask,
    softmax_cross_entropy,
)
from protenix.model.utils import get_checkpoint_fn
from protenix.utils.logger import get_logger
from protenix.utils.torch_utils import cdist

logger = get_logger(__name__)


class TokenLevelMSELoss(nn.Module):
    """
    Token级别的MSE损失

    计算token中心原子坐标的均方误差
    """

    def __init__(
        self,
        eps: float = 1e-8,
        reduction: str = "mean",
    ) -> None:
        super(TokenLevelMSELoss, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(
        self,
        pred_token_coordinate: torch.Tensor,
        true_token_coordinate: torch.Tensor,
        token_mask: torch.Tensor,
        is_rna: torch.Tensor,
        is_dna: torch.Tensor,
        is_ligand: torch.Tensor,
        per_sample_scale: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Token级别MSE损失

        Args:
            pred_token_coordinate: [..., N_sample, N_token, 3]
            true_token_coordinate: [..., N_token, 3]
            token_mask: [N_token] token是否有效
            is_rna: [N_token] 是否是RNA
            is_dna: [N_token] 是否是DNA
            is_ligand: [N_token] 是否是配体
            per_sample_scale: [..., N_sample] 每个样本的缩放因子

        Returns:
            loss: scalar
        """
        # 计算坐标差
        diff = pred_token_coordinate - true_token_coordinate.unsqueeze(-3)
        squared_diff = torch.sum(diff**2, dim=-1)  # [..., N_sample, N_token]

        # 根据分子类型加权
        # RNA/DNA权重较低，配体权重较高
        weight = torch.ones_like(token_mask, dtype=torch.float32)
        weight = weight * (1 - 0.5 * (is_rna + is_dna))  # RNA/DNA权重0.5
        weight = weight * token_mask  # 应用mask

        # 计算加权MSE
        weighted_squared_diff = squared_diff * weight.unsqueeze(-2)
        mse = torch.sum(weighted_squared_diff, dim=-1) / (
            torch.sum(weight) + self.eps
        )  # [..., N_sample]

        # 应用per-sample缩放
        if per_sample_scale is not None:
            mse = mse * per_sample_scale

        # 对N_sample维度求平均
        mse = mse.mean(dim=-1)  # [...]

        return loss_reduction(mse, method=self.reduction)


class TokenLevelSmoothLDDTLoss(nn.Module):
    """
    Token级别的Smooth LDDT损失

    计算token对之间距离的LDDT
    """

    def __init__(
        self,
        eps: float = 1e-10,
        reduction: str = "mean",
    ) -> None:
        super(TokenLevelSmoothLDDTLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def _chunk_forward(
        self,
        pred_distance: torch.Tensor,
        true_distance: torch.Tensor,
        lddt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """计算LDDT"""
        dist_diff = torch.abs(pred_distance - true_distance)
        dist_diff_epsilon = 0
        for threshold in [0.5, 1, 2, 4]:
            dist_diff_epsilon += 0.25 * torch.sigmoid(threshold - dist_diff)

        if lddt_mask is not None:
            lddt = torch.sum(lddt_mask * dist_diff_epsilon, dim=(-1, -2)) / (
                torch.sum(lddt_mask, dim=(-1, -2)) + self.eps
            )
        else:
            lddt = torch.mean(dist_diff_epsilon, dim=(-1, -2))

        return lddt

    def forward(
        self,
        pred_token_coordinate: torch.Tensor,
        true_token_coordinate: torch.Tensor,
        token_mask: torch.Tensor,
        is_nucleotide: torch.Tensor,
        is_nucleotide_threshold: float = 30.0,
        is_not_nucleotide_threshold: float = 15.0,
        diffusion_chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Token级别Smooth LDDT损失

        Args:
            pred_token_coordinate: [..., N_sample, N_token, 3]
            true_token_coordinate: [..., N_token, 3]
            token_mask: [N_token]
            is_nucleotide: [N_token]
            is_nucleotide_threshold: 核苷酸距离阈值
            is_not_nucleotide_threshold: 非核苷酸距离阈值
            diffusion_chunk_size: chunk大小

        Returns:
            loss: scalar
        """
        # 计算真实距离
        distance_mask = token_mask[..., None] * token_mask[..., None, :]
        true_distance = torch.cdist(true_token_coordinate, true_token_coordinate)

        # 构建LDDT mask
        is_nucleotide_mask = is_nucleotide.bool()
        lddt_mask = (
            (true_distance < is_nucleotide_threshold) * is_nucleotide_mask[..., None]
            + (true_distance < is_not_nucleotide_threshold)
            * (~is_nucleotide_mask[..., None])
        )
        lddt_mask = lddt_mask * _get_off_diagonal_mask(
            lddt_mask.size(-1), lddt_mask.device, true_distance.dtype
        )
        lddt_mask = lddt_mask * distance_mask
        lddt_mask = lddt_mask.unsqueeze(dim=-3)  # [..., 1, N_token, N_token]

        # 计算LDDT
        if diffusion_chunk_size is None:
            pred_distance = torch.cdist(pred_token_coordinate, pred_token_coordinate)
            lddt = self._chunk_forward(pred_distance, true_distance, lddt_mask)
        else:
            checkpoint_fn = get_checkpoint_fn()
            lddt = []
            N_sample = pred_token_coordinate.shape[-3]
            no_chunks = N_sample // diffusion_chunk_size + (
                N_sample % diffusion_chunk_size != 0
            )
            for i in range(no_chunks):
                pred_distance_i = torch.cdist(
                    pred_token_coordinate[
                        i * diffusion_chunk_size : (i + 1) * diffusion_chunk_size, :, :
                    ],
                    pred_token_coordinate[
                        i * diffusion_chunk_size : (i + 1) * diffusion_chunk_size, :, :
                    ],
                )
                lddt_i = checkpoint_fn(
                    self._chunk_forward, pred_distance_i, true_distance, lddt_mask
                )
                lddt.append(lddt_i)
            lddt = torch.cat(lddt, dim=-1)

        lddt = lddt.mean(dim=-1)  # [...]
        return 1 - loss_reduction(lddt, method=self.reduction)


class TokenLevelBondLoss(nn.Module):
    """
    Token级别的Bond损失

    只考虑相邻token之间的骨架连接
    """

    def __init__(self, eps: float = 1e-6, reduction: str = "mean") -> None:
        super(TokenLevelBondLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(
        self,
        pred_token_coordinate: torch.Tensor,
        true_token_coordinate: torch.Tensor,
        token_bonds: torch.Tensor,
        token_mask: torch.Tensor,
        per_sample_scale: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Token级别Bond损失

        Args:
            pred_token_coordinate: [..., N_sample, N_token, 3]
            true_token_coordinate: [..., N_token, 3]
            token_bonds: [N_token, N_token] token之间的连接关系
            token_mask: [N_token]
            per_sample_scale: [..., N_sample]

        Returns:
            loss: scalar
        """
        # 计算距离
        bond_mask = token_bonds * token_mask[..., None] * token_mask[..., None, :]
        bond_indices = torch.nonzero(bond_mask, as_tuple=True)

        if len(bond_indices[0]) == 0:
            return torch.tensor(
                0.0, device=pred_token_coordinate.device, requires_grad=True
            )

        # 提取有bond的token对
        pred_coords_i = pred_token_coordinate.index_select(-2, bond_indices[0])
        pred_coords_j = pred_token_coordinate.index_select(-2, bond_indices[1])
        true_coords_i = true_token_coordinate.index_select(-2, bond_indices[0])
        true_coords_j = true_token_coordinate.index_select(-2, bond_indices[1])

        # 计算距离
        pred_distance = torch.linalg.vector_norm(
            pred_coords_i - pred_coords_j, ord=2, dim=-1
        )
        true_distance = torch.linalg.vector_norm(
            true_coords_i - true_coords_j, ord=2, dim=-1
        )

        # 计算损失
        squared_diff = (pred_distance - true_distance) ** 2
        bond_loss = torch.mean(squared_diff, dim=-1)  # [..., N_sample]

        if per_sample_scale is not None:
            bond_loss = bond_loss * per_sample_scale

        bond_loss = bond_loss.mean(dim=-1)  # [...]
        return loss_reduction(bond_loss, method=self.reduction)


class TokenLevelDistogramLoss(nn.Module):
    """
    Token级别的Distogram损失

    预测token对之间的距离分布
    """

    def __init__(
        self,
        min_bin: float = 3.25,
        max_bin: float = 50.75,
        num_bins: int = 64,
        eps: float = 1e-6,
        reduction: str = "mean",
    ) -> None:
        super(TokenLevelDistogramLoss, self).__init__()
        self.min_bin = min_bin
        self.max_bin = max_bin
        self.num_bins = num_bins
        self.eps = eps
        self.reduction = reduction

        # 创建bin boundaries
        boundaries = torch.linspace(min_bin, max_bin, num_bins - 1)
        self.register_buffer("boundaries", boundaries)

    def forward(
        self,
        logits: torch.Tensor,
        true_token_coordinate: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Token级别Distogram损失

        Args:
            logits: [N_token, N_token, num_bins]
            true_token_coordinate: [N_token, 3]
            token_mask: [N_token]

        Returns:
            loss: scalar
        """
        # 计算真实距离
        true_distance = torch.cdist(true_token_coordinate, true_token_coordinate)

        # 将距离离散化到bins
        true_distance_clipped = torch.clamp(true_distance, self.min_bin, self.max_bin)
        bin_index = torch.searchsorted(
            self.boundaries, true_distance_clipped.contiguous()
        )

        # Mask
        mask = token_mask[..., None] * token_mask[..., None, :]
        mask = mask * _get_off_diagonal_mask(mask.size(-1), mask.device, mask.dtype)

        # 计算交叉熵
        loss = softmax_cross_entropy(logits, bin_index)
        loss = (loss * mask).sum() / (mask.sum() + self.eps)

        return loss_reduction(loss, method=self.reduction)


class TokenLevelProtenixLoss(nn.Module):
    """
    Token级别的总损失

    整合所有token-level损失
    """

    def __init__(self, configs) -> None:
        super(TokenLevelProtenixLoss, self).__init__()
        self.configs = configs

        # 损失权重
        self.alpha_diffusion = self.configs.loss.weight.alpha_diffusion
        self.alpha_distogram = self.configs.loss.weight.alpha_distogram
        self.alpha_bond = self.configs.loss.weight.alpha_bond
        self.weight_smooth_lddt = self.configs.loss.weight.smooth_lddt

        self.loss_weight = {
            "mse_loss": self.alpha_diffusion,
            "bond_loss": self.alpha_diffusion * self.alpha_bond,
            "smooth_lddt_loss": self.alpha_diffusion * self.weight_smooth_lddt,
            "distogram_loss": self.alpha_distogram,
        }

        # 损失函数
        self.mse_loss = TokenLevelMSELoss(**configs.loss.diffusion.mse)
        self.bond_loss = TokenLevelBondLoss(**configs.loss.diffusion.bond)
        self.smooth_lddt_loss = TokenLevelSmoothLDDTLoss(
            **configs.loss.diffusion.smooth_lddt
        )
        self.distogram_loss = TokenLevelDistogramLoss(**configs.loss.distogram)

    def forward(
        self,
        feat_dict: dict[str, Any],
        pred_dict: dict[str, torch.Tensor],
        label_dict: dict[str, Any],
        mode: str = "train",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        计算token-level总损失

        Args:
            feat_dict: 特征字典，包含token_mask, is_rna, is_dna, is_ligand等
            pred_dict: 预测字典，包含coordinate (token坐标), distogram等
            label_dict: 标签字典，包含coordinate (token坐标)
            mode: 'train' or 'eval'

        Returns:
            cum_loss: 总损失
            losses: 各项损失的字典
        """
        assert mode in ["train", "eval"]

        # 提取token相关特征
        token_mask = feat_dict["token_mask"]
        is_rna = feat_dict.get("is_rna", torch.zeros_like(token_mask))
        is_dna = feat_dict.get("is_dna", torch.zeros_like(token_mask))
        is_ligand = feat_dict.get("is_ligand", torch.zeros_like(token_mask))
        is_nucleotide = is_rna + is_dna
        token_bonds = feat_dict.get("token_bonds", torch.zeros(len(token_mask), len(token_mask)))

        # Per-sample缩放（仅训练时）
        if mode == "train" and "noise_level" in pred_dict:
            per_sample_scale = (
                pred_dict["noise_level"] ** 2 + self.configs.sigma_data**2
            ) / (self.configs.sigma_data * pred_dict["noise_level"]) ** 2
        else:
            per_sample_scale = None

        # 定义损失函数
        loss_fns = {
            "mse_loss": lambda: self.mse_loss(
                pred_token_coordinate=pred_dict["coordinate"],
                true_token_coordinate=label_dict["coordinate"],
                token_mask=token_mask,
                is_rna=is_rna,
                is_dna=is_dna,
                is_ligand=is_ligand,
                per_sample_scale=per_sample_scale,
            ),
            "smooth_lddt_loss": lambda: self.smooth_lddt_loss(
                pred_token_coordinate=pred_dict["coordinate"],
                true_token_coordinate=label_dict["coordinate"],
                token_mask=token_mask,
                is_nucleotide=is_nucleotide.bool(),
                diffusion_chunk_size=self.configs.loss.diffusion_lddt_chunk_size,
            ),
            "bond_loss": lambda: self.bond_loss(
                pred_token_coordinate=pred_dict["coordinate"],
                true_token_coordinate=label_dict["coordinate"],
                token_bonds=token_bonds,
                token_mask=token_mask,
                per_sample_scale=per_sample_scale,
            ),
        }

        # Distogram loss (如果有预测)
        if "distogram" in pred_dict:
            loss_fns["distogram_loss"] = lambda: self.distogram_loss(
                logits=pred_dict["distogram"],
                true_token_coordinate=label_dict["coordinate"],
                token_mask=token_mask,
            )

        # 聚合损失
        cum_loss = 0.0
        losses = {}
        for loss_name, loss_fn in loss_fns.items():
            weight = self.loss_weight[loss_name]
            loss = loss_fn()

            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"{loss_name} is NaN. Skipping...")
                continue

            losses[loss_name] = loss.detach().clone()
            losses[f"weighted_{loss_name}"] = weight * loss.detach().clone()
            cum_loss = cum_loss + weight * loss

        losses["loss"] = cum_loss.detach().clone()

        return cum_loss, losses
