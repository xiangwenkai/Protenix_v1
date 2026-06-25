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

"""Train Protenix on PDB structures with an additional RNA binding signal loss."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Mapping, Tuple

import torch

from configs.configs_base import configs as configs_base
from configs.configs_data import data_configs
from configs.configs_model_type import model_configs
from configs.configs_rna_signal_sft import rna_signal_sft_configs
from protenix.config.config import parse_configs, parse_sys_args
from protenix.model import sample_confidence
from protenix.model.eclip_binding import (
    EclipSignalLoss,
    compute_distogram_binding_score,
    get_protein_token_indices,
    get_rna_token_indices,
    topk_overlap,
)
from protenix.utils.torch_utils import autocasting_disable_decorator
from runner.train import AF3Trainer


def deep_update(target: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), Mapping):
            deep_update(target[key], value)
        else:
            target[key] = value
    return target


def _squeeze_optional_batch(value: torch.Tensor) -> torch.Tensor:
    if value.ndim > 1 and value.shape[0] == 1:
        return value.squeeze(0)
    return value


class RNASignalSFTTrainer(AF3Trainer):
    """AF3Trainer plus structure-derived RNA binding signal supervision."""

    def init_loss(self) -> None:
        super().init_loss()
        cfg = self.configs.rna_signal_sft
        self.signal_loss = EclipSignalLoss(
            profile_weight=cfg.signal_profile_weight,
            min_height=cfg.signal_multinomial_min_height,
            binary_threshold=cfg.signal_binary_threshold,
            signal_clip_value=cfg.signal_clip_value,
            max_total_count=cfg.signal_multinomial_max_total,
        ).to(self.device)
        for param in self.signal_loss.parameters():
            param.requires_grad_(False)

    def _zero_signal_loss(
        self,
        batch: Dict[str, Any],
        reason: str,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        pred_dict = batch["pred_dict"]
        ref_tensor = None
        for value in pred_dict.values():
            if torch.is_tensor(value):
                ref_tensor = value
                break
        if ref_tensor is None:
            ref_tensor = batch["input_feature_dict"]["atom_to_token_idx"].float()
        zero = ref_tensor.float().sum() * 0.0
        metrics = {
            "loss": zero.detach(),
            "skipped": torch.tensor(1.0, device=zero.device),
        }
        if reason == "missing":
            metrics["missing_label"] = torch.tensor(1.0, device=zero.device)
        elif reason == "empty":
            metrics["empty_target"] = torch.tensor(1.0, device=zero.device)
        elif reason == "no_rna":
            metrics["no_rna_token"] = torch.tensor(1.0, device=zero.device)
        elif reason == "no_protein":
            metrics["no_protein_token"] = torch.tensor(1.0, device=zero.device)
        return zero, metrics

    def _get_contact_probs(self, pred_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        if "distogram" in pred_dict:
            return autocasting_disable_decorator(True)(
                sample_confidence.compute_contact_prob
            )(
                distogram_logits=pred_dict["distogram"],
                **sample_confidence.get_bin_params(self.configs.loss.distogram),
            )
        if "contact_probs" in pred_dict:
            return pred_dict["contact_probs"]
        raise KeyError("Missing both pred_dict['distogram'] and pred_dict['contact_probs'].")

    def get_rna_signal_loss(
        self,
        batch: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        label_dict = batch["label_dict"]
        if "rna_binding_signal" not in label_dict:
            raise KeyError(
                "Missing label_dict['rna_binding_signal']. "
                "Run scripts/add_structure_rna_signal_to_protenix_pkl.py first "
                "and point data.*.base_info.bioassembly_dict_dir to the processed pkl dir."
            )
        if "rna_binding_signal_mask" not in label_dict:
            raise KeyError("Missing label_dict['rna_binding_signal_mask'].")

        feat_dict = batch["input_feature_dict"]
        if get_rna_token_indices(feat_dict).numel() == 0:
            return self._zero_signal_loss(batch, reason="no_rna")
        if get_protein_token_indices(feat_dict).numel() == 0:
            return self._zero_signal_loss(batch, reason="no_protein")

        contact_probs = self._get_contact_probs(batch["pred_dict"])
        p_bind, rna_token_indices = compute_distogram_binding_score(contact_probs, feat_dict)
        if p_bind.ndim > 1 and p_bind.shape[0] == 1:
            p_bind = p_bind.squeeze(0)

        token_signal = _squeeze_optional_batch(label_dict["rna_binding_signal"]).to(
            device=p_bind.device,
            dtype=torch.float32,
        )
        token_mask = _squeeze_optional_batch(label_dict["rna_binding_signal_mask"]).to(
            device=p_bind.device,
            dtype=torch.bool,
        )
        rna_token_indices = rna_token_indices.to(device=p_bind.device, dtype=torch.long)
        target = token_signal.index_select(0, rna_token_indices)
        target_mask = token_mask.index_select(0, rna_token_indices)
        if int(target_mask.sum().item()) < 1:
            return self._zero_signal_loss(batch, reason="empty")

        signal_loss, metrics = self.signal_loss(p_bind, target, target_mask)
        p_bind_1d = p_bind.squeeze(0) if p_bind.ndim > 1 and p_bind.shape[0] == 1 else p_bind
        metrics["topk_overlap"] = topk_overlap(p_bind_1d, target, target_mask)
        metrics["valid_tokens"] = target_mask.float().sum()
        metrics["skipped"] = torch.tensor(0.0, device=p_bind.device)
        return signal_loss, metrics

    def get_loss(
        self, batch: Dict[str, Any], mode: str = "train"
    ) -> Tuple[torch.Tensor, Dict[str, Any], Dict[str, Any]]:
        structure_loss, loss_dict, batch = super().get_loss(batch, mode=mode)
        signal_loss, signal_metrics = self.get_rna_signal_loss(batch)
        weighted_signal_loss = self.configs.rna_signal_sft.signal_loss_weight * signal_loss
        total_loss = structure_loss + weighted_signal_loss

        loss_dict = dict(loss_dict)
        loss_dict["structure_loss"] = structure_loss.detach()
        loss_dict["rna_signal_loss"] = signal_loss.detach()
        loss_dict["weighted_rna_signal_loss"] = weighted_signal_loss.detach()
        for key, value in signal_metrics.items():
            loss_dict[f"rna_signal/{key}"] = value
        loss_dict["loss"] = total_loss.detach()
        return total_loss, loss_dict, batch


def build_configs(arg_str: str) -> Any:
    configs_base["triangle_attention"] = os.environ.get(
        "TRIANGLE_ATTENTION", "cuequivariance"
    )
    configs_base["triangle_multiplicative"] = os.environ.get(
        "TRIANGLE_MULTIPLICATIVE", "cuequivariance"
    )
    first_pass = {
        **configs_base,
        **{"data": data_configs},
        **rna_signal_sft_configs,
    }
    parsed = parse_configs(first_pass, arg_str=arg_str, fill_required_with_null=True)
    model_name = parsed.model_name

    base_configs = {
        **configs_base,
        **{"data": data_configs},
        **rna_signal_sft_configs,
    }
    deep_update(base_configs, model_configs[model_name])
    return parse_configs(
        configs=base_configs,
        arg_str=arg_str,
        fill_required_with_null=True,
    )


def main() -> None:
    log_format = (
        "%(asctime)s,%(msecs)-3d %(levelname)-8s "
        "[%(filename)s:%(lineno)s %(funcName)s] %(message)s"
    )
    logging.basicConfig(
        format=log_format,
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filemode="w",
    )
    configs = build_configs(parse_sys_args())
    logging.info(
        "RNA signal SFT config: model=%s signal_weight=%s",
        configs.model_name,
        configs.rna_signal_sft.signal_loss_weight,
    )
    trainer = RNASignalSFTTrainer(configs)
    trainer.run()


if __name__ == "__main__":
    main()
