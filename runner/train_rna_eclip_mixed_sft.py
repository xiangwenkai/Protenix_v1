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

"""Mixed SFT on structure-labeled PDB RNA complexes and eCLIP signal data."""

from __future__ import annotations

import datetime
import hashlib
import logging
import os
import random
import time
import traceback
from argparse import Namespace
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import wandb
from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm

os.environ.setdefault("LAYERNORM_TYPE", "torch")
os.environ["WANDB_CONSOLE"] = "off"

from configs.configs_base import configs as configs_base
from configs.configs_data import data_configs
from configs.configs_eclip_ppft import eclip_ppft_configs
from configs.configs_model_type import model_configs
from configs.configs_rna_eclip_mixed_sft import rna_eclip_mixed_sft_configs
from configs.configs_rna_signal_sft import rna_signal_sft_configs
from protenix.config.config import parse_configs, parse_sys_args, save_config
from protenix.data.eclip_ppft_dataset import (
    EclipPPFTDataset,
    collate_eclip_ppft_samples,
)
from protenix.data.pipeline.dataloader import get_dataloaders
from protenix.metrics.lddt_metrics import LDDTMetrics
from protenix.model import sample_confidence
from protenix.model.eclip_binding import (
    EclipSignalLoss,
    align_signal_to_prediction,
    compute_distogram_binding_score,
    get_protein_token_indices,
    get_rna_token_indices,
    topk_overlap,
)
from protenix.model.generator_ppft import sample_diffusion_ppft
from protenix.model.loss import ProtenixLoss
from protenix.model.protenix import Protenix, update_input_feature_dict
from protenix.utils.distributed import DIST_WRAPPER
from protenix.utils.lr_scheduler import FinetuneLRScheduler, get_lr_scheduler
from protenix.utils.metrics import SimpleMetricAggregator
from protenix.utils.permutation.permutation import SymmetricPermutation
from protenix.utils.seed import seed_everything
from protenix.utils.torch_utils import autocasting_disable_decorator, to_device
from protenix.utils.training import get_optimizer, is_loss_nan_check
from runner.ema import EMAWrapper
from runner.train_eclip_ppft import featurize_eclip_sample

torch.serialization.add_safe_globals([Namespace])


def deep_update(target: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), Mapping):
            deep_update(target[key], value)
        else:
            target[key] = value
    return target


def _module_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    if hasattr(model, "module"):
        return model.module.state_dict()
    return model.state_dict()


def _squeeze_optional_batch(value: torch.Tensor) -> torch.Tensor:
    if value.ndim > 1 and value.shape[0] == 1:
        return value.squeeze(0)
    return value


class MixedSFTForwardModule(nn.Module):
    """DDP-visible forward module for both PDB and eCLIP tasks."""

    def __init__(
        self,
        model: Protenix,
        configs: Any,
        eclip_cfg: Any,
    ) -> None:
        super().__init__()
        self.model = model
        self.configs = configs
        self.eclip_cfg = eclip_cfg

    def forward(self, task: str, **kwargs: Any) -> Any:
        if task == "pdb":
            return self.forward_pdb(**kwargs)
        if task == "eclip":
            return self.forward_eclip(**kwargs)
        raise ValueError(f"Unknown mixed SFT task: {task}")

    def forward_pdb(
        self,
        *,
        input_feature_dict: Dict[str, torch.Tensor],
        label_dict: Dict[str, torch.Tensor],
        label_full_dict: Dict[str, torch.Tensor],
        mode: str,
        current_step: int | None,
        symmetric_permutation: SymmetricPermutation,
        mc_dropout_apply_rate: float,
    ) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any]]:
        return self.model(
            input_feature_dict=input_feature_dict,
            label_dict=label_dict,
            label_full_dict=label_full_dict,
            mode=mode,
            current_step=current_step,
            symmetric_permutation=symmetric_permutation,
            mc_dropout_apply_rate=mc_dropout_apply_rate,
        )

    def forward_eclip(
        self,
        *,
        feat_dict: Dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        n_cycle = self.eclip_cfg.n_cycle
        if n_cycle is None:
            n_cycle = self.model.N_cycle

        s_inputs, s, z = self.model.get_pairformer_output(
            input_feature_dict=feat_dict,
            N_cycle=int(n_cycle),
            inplace_safe=self.eclip_cfg.inplace_safe,
            chunk_size=self.eclip_cfg.diffusion_attn_chunk_size,
        )

        distogram_logits = self.model.distogram_head(z)
        contact_probs = sample_confidence.compute_contact_prob(
            distogram_logits=distogram_logits,
            **sample_confidence.get_bin_params(self.configs.loss.distogram),
            thres=self.eclip_cfg.distogram_contact_threshold,
        )
        p_bind, _ = compute_distogram_binding_score(contact_probs, feat_dict)

        if self.eclip_cfg.confidence_quality_weight <= 0.0:
            return p_bind, p_bind.sum() * 0.0, {}

        cache = {"pair_z": None, "p_lm/c_l": [None, None]}
        if self.model.enable_diffusion_shared_vars_cache:
            cache["pair_z"] = (
                self.model.diffusion_module.diffusion_conditioning.prepare_cache(
                    feat_dict["relp"], z, False
                )
            )
            cache["p_lm/c_l"] = (
                self.model.diffusion_module.atom_attention_encoder.prepare_cache(
                    ref_pos=feat_dict["ref_pos"],
                    ref_charge=feat_dict["ref_charge"],
                    ref_mask=feat_dict["ref_mask"],
                    ref_element=feat_dict["ref_element"],
                    ref_atom_name_chars=feat_dict["ref_atom_name_chars"],
                    atom_to_token_idx=feat_dict["atom_to_token_idx"],
                    d_lm=feat_dict["d_lm"],
                    v_lm=feat_dict["v_lm"],
                    pad_info=feat_dict["pad_info"],
                    r_l=True,
                    z=cache["pair_z"],
                    inplace_safe=False,
                )
            )

        noise_schedule = self.model.inference_noise_scheduler(
            N_step=self.eclip_cfg.confidence_rollout_steps,
            device=s_inputs.device,
            dtype=s_inputs.dtype,
        )
        coords = sample_diffusion_ppft(
            denoise_net=self.model.diffusion_module,
            input_feature_dict=feat_dict,
            s_inputs=s_inputs,
            s_trunk=s,
            z_trunk=None if cache["pair_z"] is not None else z,
            pair_z=cache["pair_z"],
            p_lm=cache["p_lm/c_l"][0],
            c_l=cache["p_lm/c_l"][1],
            noise_schedule=noise_schedule,
            N_sample=1,
            record_grad_steps=(),
            detach_unrecorded_steps=True,
            gamma0=self.configs.sample_diffusion.gamma0,
            gamma_min=self.configs.sample_diffusion.gamma_min,
            noise_scale_lambda=self.configs.sample_diffusion.noise_scale_lambda,
            step_scale_eta=self.configs.sample_diffusion.step_scale_eta,
            inplace_safe=self.eclip_cfg.inplace_safe,
            attn_chunk_size=self.eclip_cfg.diffusion_attn_chunk_size,
            enable_efficient_fusion=self.model.enable_efficient_fusion,
        )
        quality_loss, quality_metrics = self.confidence_quality_loss(
            feat_dict=feat_dict,
            coords=coords,
            s_inputs=s_inputs,
            s=s,
            z=z,
        )
        return p_bind, quality_loss, quality_metrics

    def confidence_quality_loss(
        self,
        *,
        feat_dict: Dict[str, torch.Tensor],
        coords: torch.Tensor,
        s_inputs: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        plddt_logits, _, _, _ = self.model.run_confidence_head(
            input_feature_dict=feat_dict,
            s_inputs=s_inputs,
            s_trunk=s,
            z_trunk=z,
            pair_mask=None,
            x_pred_coords=coords,
            triangle_multiplicative=self.configs.triangle_multiplicative,
            triangle_attention=self.configs.triangle_attention,
            inplace_safe=self.eclip_cfg.inplace_safe,
            chunk_size=self.eclip_cfg.diffusion_attn_chunk_size,
        )
        atom_plddt = sample_confidence.logits_to_score(
            plddt_logits,
            **sample_confidence.get_bin_params(self.configs.loss.plddt),
        )
        plddt_mean = atom_plddt.mean()
        target = torch.as_tensor(
            self.eclip_cfg.confidence_quality_target,
            dtype=plddt_mean.dtype,
            device=plddt_mean.device,
        )
        quality_loss = float(self.eclip_cfg.confidence_quality_weight) * torch.relu(
            target - plddt_mean
        )
        metrics = {
            "plddt_mean": plddt_mean.detach(),
            "plddt_min": atom_plddt.detach().amin(),
            "quality_loss": quality_loss.detach(),
        }
        if self.eclip_cfg.confidence_monitor_clash:
            with torch.no_grad():
                is_polymer = (
                    feat_dict["is_protein"].bool()
                    | feat_dict["is_rna"].bool()
                    | feat_dict["is_dna"].bool()
                )
                has_clash = sample_confidence.calculate_clash(
                    pred_coordinate=coords.detach().float(),
                    asym_id=feat_dict["asym_id"].long(),
                    atom_to_token_idx=feat_dict["atom_to_token_idx"].long(),
                    is_polymer=is_polymer.long(),
                    threshold=self.configs.metrics.clash.af3_clash_threshold,
                )
                metrics["has_clash"] = has_clash.float().mean()
        return quality_loss, metrics


class RNAEclipMixedSFTTrainer:
    """Train with PDB structure batches and eCLIP signal-only batches."""

    def __init__(self, configs: Any) -> None:
        self.configs = configs
        self.eclip_cfg = configs.eclip_ppft
        self.rna_cfg = configs.rna_signal_sft
        self.mixed_cfg = configs.mixed_sft
        self.init_env()
        self.init_basics()
        self.init_log()
        self.init_model()
        self.init_loss()
        self.init_data()
        self.try_load_checkpoint()

    def init_env(self) -> None:
        logging.info(
            "Distributed environment: world size: %s, global rank: %s, local rank: %s",
            DIST_WRAPPER.world_size,
            DIST_WRAPPER.rank,
            DIST_WRAPPER.local_rank,
        )
        self.use_cuda = torch.cuda.device_count() > 0
        self.use_ddp = DIST_WRAPPER.world_size > 1
        self.device = (
            torch.device(f"cuda:{DIST_WRAPPER.local_rank}")
            if self.use_cuda
            else torch.device("cpu")
        )
        if self.use_cuda:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            if DIST_WRAPPER.local_rank >= torch.cuda.device_count():
                raise RuntimeError(
                    "DDP local_rank is larger than available CUDA devices: "
                    f"local_rank={DIST_WRAPPER.local_rank}, "
                    f"cuda_device_count={torch.cuda.device_count()}."
                )
            torch.cuda.set_device(self.device)
        if self.use_ddp:
            timeout_seconds = int(os.environ.get("NCCL_TIMEOUT_SECOND", 600))
            backend = "nccl" if self.use_cuda else "gloo"
            dist.init_process_group(
                backend=backend, timeout=datetime.timedelta(seconds=timeout_seconds)
            )
        if not self.configs.deterministic_seed:
            hash_string = f"({self.configs.seed},{DIST_WRAPPER.rank},mixed_sft_seed)"
            rank_seed = int(hashlib.sha256(hash_string.encode("utf8")).hexdigest(), 16)
            rank_seed = rank_seed % (2**32)
        else:
            rank_seed = self.configs.seed
        seed_everything(rank_seed, deterministic=self.configs.deterministic)

    def init_basics(self) -> None:
        self.step = 0
        self.global_step = 0
        self.start_step = 0
        self.iters_to_accumulate = self.configs.iters_to_accumulate
        self.best_eval_loss = float("inf")
        self.best_eval_step = -1

        self.run_name = self.configs.run_name + "_" + time.strftime("%Y%m%d_%H%M%S")
        run_names = DIST_WRAPPER.all_gather_object(
            self.run_name if DIST_WRAPPER.rank == 0 else None
        )
        self.run_name = [name for name in run_names if name is not None][0]
        self.run_dir = Path(self.configs.base_dir) / self.run_name
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.error_dir = self.run_dir / "errors"
        if DIST_WRAPPER.rank == 0:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.error_dir.mkdir(parents=True, exist_ok=True)
            save_config(self.configs, str(self.run_dir / "config.yaml"))
        if self.use_ddp:
            dist.barrier()
        self.print(f"Using run dir: {self.run_dir}")

    def init_log(self) -> None:
        if self.configs.use_wandb and DIST_WRAPPER.rank == 0:
            wandb.init(
                project=self.configs.project,
                name=self.run_name,
                config=vars(self.configs),
                id=self.configs.wandb_id or None,
            )
        self.train_metrics = SimpleMetricAggregator(["avg"])

    def init_model(self) -> None:
        self.raw_model = Protenix(self.configs).to(self.device)
        if self.mixed_cfg.freeze_confidence_head:
            for param in self.raw_model.confidence_head.parameters():
                param.requires_grad_(False)
            self.print("Frozen confidence_head parameters for mixed SFT.")

        self.train_module: nn.Module = MixedSFTForwardModule(
            model=self.raw_model,
            configs=self.configs,
            eclip_cfg=self.eclip_cfg,
        ).to(self.device)
        if self.use_ddp:
            self.print("Using DistributedDataParallel (DDP) for mixed SFT")
            ddp_kwargs: dict[str, Any] = {
                "find_unused_parameters": True,
                "static_graph": False,
                "broadcast_buffers": False,
            }
            if self.use_cuda:
                ddp_kwargs.update(
                    {
                        "device_ids": [DIST_WRAPPER.local_rank],
                        "output_device": DIST_WRAPPER.local_rank,
                    }
                )
            self.train_module = DDP(self.train_module, **ddp_kwargs)

        self.optimizer = get_optimizer(
            self.configs,
            self.train_module,
            param_names=self.configs.get("finetune_params_with_substring", [""]),
        )
        self.init_scheduler()
        if self.configs.get("ema_decay", -1) > 0:
            assert self.configs.ema_decay < 1
            self.ema_wrapper = EMAWrapper(
                self.train_module,
                self.configs.ema_decay,
                self.configs.ema_mutable_param_keywords,
            )
            self.ema_wrapper.register()

        n_trainable = sum(p.numel() for p in self.raw_model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.raw_model.parameters())
        self.print(f"Trainable params: {n_trainable / 1e6:.2f}M / {n_total / 1e6:.2f}M")

    def init_scheduler(self, **kwargs: Any) -> None:
        finetune_params = self.configs.get("finetune_params_with_substring", [""])
        is_finetune = len(finetune_params[0]) > 0
        if is_finetune:
            self.lr_scheduler = FinetuneLRScheduler(
                self.optimizer,
                self.configs,
                self.configs.finetune,
                **kwargs,
            )
        else:
            self.lr_scheduler = get_lr_scheduler(self.configs, self.optimizer, **kwargs)

    def init_loss(self) -> None:
        self.structure_loss = ProtenixLoss(self.configs)
        self.symmetric_permutation = SymmetricPermutation(
            self.configs, error_dir=str(self.error_dir)
        )
        self.lddt_metrics = LDDTMetrics(self.configs)
        self.pdb_signal_loss = EclipSignalLoss(
            profile_weight=self.rna_cfg.signal_profile_weight,
            min_height=self.rna_cfg.signal_multinomial_min_height,
            binary_threshold=self.rna_cfg.signal_binary_threshold,
            signal_clip_value=self.rna_cfg.signal_clip_value,
            max_total_count=self.rna_cfg.signal_multinomial_max_total,
        ).to(self.device)
        self.eclip_signal_loss = EclipSignalLoss(
            profile_weight=self.eclip_cfg.signal_profile_weight,
            min_height=self.eclip_cfg.signal_multinomial_min_height,
            binary_threshold=self.eclip_cfg.signal_binary_threshold,
            signal_clip_value=self.eclip_cfg.signal_clip_value,
            max_total_count=self.eclip_cfg.signal_multinomial_max_total,
        ).to(self.device)

    def init_data(self) -> None:
        self.pdb_train_dl, self.pdb_test_dls = get_dataloaders(
            self.configs,
            DIST_WRAPPER.world_size,
            seed=self.configs.seed,
            error_dir=str(self.error_dir),
        )
        common_kwargs = dict(
            data_dir=self.eclip_cfg.data_dir,
            protein_sequence_tsv=self.eclip_cfg.protein_sequence_tsv,
            max_rna_length=self.eclip_cfg.max_rna_length,
            rna_crop_size=self.eclip_cfg.rna_crop_size,
            max_protein_length=self.eclip_cfg.max_protein_length,
            min_signal_max=self.eclip_cfg.min_signal_max,
            zero_signal_keep_prob=self.eclip_cfg.zero_signal_keep_prob,
            parquet_batch_size=self.eclip_cfg.parquet_batch_size,
            seed=self.configs.seed,
            rank=DIST_WRAPPER.rank,
            world_size=DIST_WRAPPER.world_size,
        )
        self.eclip_train_dataset = EclipPPFTDataset(
            split=self.eclip_cfg.train_split,
            limit=self.eclip_cfg.train_limit,
            shuffle_files=self.eclip_cfg.shuffle_files,
            shuffle_buffer=self.eclip_cfg.shuffle_buffer,
            **common_kwargs,
        )
        self.eclip_eval_dataset = EclipPPFTDataset(
            split=self.eclip_cfg.eval_split,
            limit=self.eclip_cfg.eval_limit,
            shuffle_files=False,
            shuffle_buffer=0,
            **common_kwargs,
        )
        self.eclip_train_dl = DataLoader(
            self.eclip_train_dataset,
            batch_size=self.eclip_cfg.batch_size,
            num_workers=self.eclip_cfg.num_workers,
            collate_fn=collate_eclip_ppft_samples,
        )
        self.eclip_eval_dl = DataLoader(
            self.eclip_eval_dataset,
            batch_size=1,
            num_workers=0,
            collate_fn=collate_eclip_ppft_samples,
        )
        self._pdb_iter = iter(self.pdb_train_dl)
        self._eclip_iter = iter(self.eclip_train_dl)

    def try_load_checkpoint(self) -> None:
        if not self.configs.load_checkpoint_path:
            return
        checkpoint_path = self.configs.load_checkpoint_path
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint["model"]
        first_key = next(iter(state_dict))
        if first_key.startswith("module."):
            state_dict = {key[len("module.") :]: value for key, value in state_dict.items()}
        self.raw_model.load_state_dict(state_dict, strict=self.configs.load_strict)
        if not self.configs.load_params_only:
            if not self.configs.skip_load_optimizer and checkpoint.get("optimizer") is not None:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
            if not self.configs.skip_load_scheduler and checkpoint.get("scheduler") is not None:
                self.lr_scheduler.load_state_dict(checkpoint["scheduler"])
            if not self.configs.skip_load_step:
                self.step = int(checkpoint.get("step", -1)) + 1
                self.start_step = self.step
                self.global_step = self.step * self.iters_to_accumulate
            self.best_eval_loss = float(checkpoint.get("best_eval_loss", self.best_eval_loss))
            self.best_eval_step = int(checkpoint.get("best_eval_step", self.best_eval_step))
        self.print(f"Loaded checkpoint {checkpoint_path} at step {self.step}")

    def save_checkpoint(self, filename: str | None = None) -> None:
        if DIST_WRAPPER.rank != 0:
            return
        path = self.checkpoint_dir / (filename or f"{self.step}.pt")
        checkpoint = {
            "model": _module_state_dict(self.raw_model),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            "step": self.step,
            "best_eval_loss": self.best_eval_loss,
            "best_eval_step": self.best_eval_step,
            "config": dict(self.configs),
        }
        torch.save(checkpoint, path)
        self.print(f"Saved checkpoint to {path}")

    def print(self, msg: str) -> None:
        if DIST_WRAPPER.rank == 0:
            logging.info(msg)

    def _next_pdb_batch(self) -> Dict[str, Any]:
        try:
            return next(self._pdb_iter)
        except StopIteration:
            self._pdb_iter = iter(self.pdb_train_dl)
            return next(self._pdb_iter)

    def _next_eclip_samples(self) -> list[dict[str, Any]]:
        try:
            return next(self._eclip_iter)
        except StopIteration:
            self._eclip_iter = iter(self.eclip_train_dl)
            return next(self._eclip_iter)

    def choose_task(self) -> str:
        rng = random.Random(self.configs.seed + self.step)
        return "pdb" if rng.random() < float(self.mixed_cfg.pdb_sample_prob) else "eclip"

    def prepare_eclip_feature_dict(self, sample: dict[str, Any]) -> Dict[str, torch.Tensor]:
        feat_dict, _ = featurize_eclip_sample(sample)
        feat_dict = to_device(feat_dict, self.device)
        feat_dict = self.raw_model.relative_position_encoding.generate_relp(feat_dict)
        return update_input_feature_dict(feat_dict)

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
        metrics[f"{reason}_target"] = torch.tensor(1.0, device=zero.device)
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

    def get_pdb_signal_loss(
        self,
        batch: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        label_dict = batch["label_dict"]
        if "rna_binding_signal" not in label_dict:
            raise KeyError(
                "Missing label_dict['rna_binding_signal']. "
                "Run scripts/add_structure_rna_signal_to_protenix_pkl.py first."
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

        signal_loss, metrics = self.pdb_signal_loss(p_bind, target, target_mask)
        p_bind_1d = p_bind.squeeze(0) if p_bind.ndim > 1 and p_bind.shape[0] == 1 else p_bind
        metrics["topk_overlap"] = topk_overlap(p_bind_1d, target, target_mask)
        metrics["valid_tokens"] = target_mask.float().sum()
        metrics["skipped"] = torch.tensor(0.0, device=p_bind.device)
        return signal_loss, metrics

    def forward_pdb_batch(
        self, batch: Dict[str, Any], mode: str = "train"
    ) -> tuple[torch.Tensor, Dict[str, Any], Dict[str, Any]]:
        batch["pred_dict"], batch["label_dict"], _ = self.train_module(
            task="pdb",
            input_feature_dict=batch["input_feature_dict"],
            label_dict=batch["label_dict"],
            label_full_dict=batch["label_full_dict"],
            mode=mode,
            current_step=self.step if mode == "train" else None,
            symmetric_permutation=self.symmetric_permutation,
            mc_dropout_apply_rate=(
                0 if mode == "train" else self.configs.mc_dropout_apply_rate
            ),
        )
        structure_loss, structure_loss_dict = autocasting_disable_decorator(
            self.configs.skip_amp.loss
        )(self.structure_loss)(
            feat_dict=batch["input_feature_dict"],
            pred_dict=batch["pred_dict"],
            label_dict=batch["label_dict"],
            mode=mode,
        )
        signal_loss, signal_metrics = self.get_pdb_signal_loss(batch)
        weighted_signal_loss = self.rna_cfg.signal_loss_weight * signal_loss
        total_loss = structure_loss + weighted_signal_loss

        metrics = {f"pdb/{key}": value for key, value in structure_loss_dict.items()}
        metrics["pdb/structure_loss"] = structure_loss.detach()
        metrics["pdb/rna_signal_loss"] = signal_loss.detach()
        metrics["pdb/weighted_rna_signal_loss"] = weighted_signal_loss.detach()
        for key, value in signal_metrics.items():
            metrics[f"pdb/rna_signal/{key}"] = value
        metrics["loss"] = total_loss.detach()
        return total_loss, metrics, batch

    def forward_eclip_sample(
        self,
        sample: dict[str, Any],
    ) -> tuple[torch.Tensor, Dict[str, Any]]:
        feat_dict = self.prepare_eclip_feature_dict(sample)
        p_bind, quality_loss, quality_metrics = self.train_module(
            task="eclip",
            feat_dict=feat_dict,
        )
        target, target_mask = align_signal_to_prediction(
            sample["signal_vector"],
            p_bind.shape[-1],
            device=p_bind.device,
        )
        signal_loss, metrics = self.eclip_signal_loss(p_bind, target, target_mask)
        weighted_signal_loss = self.eclip_cfg.signal_loss_weight * signal_loss
        total_loss = weighted_signal_loss + quality_loss

        metrics = {f"eclip/{key}": value for key, value in metrics.items()}
        metrics.update({f"eclip/confidence/{key}": value for key, value in quality_metrics.items()})
        metrics["eclip/weighted_signal_loss"] = weighted_signal_loss.detach()
        metrics["eclip/topk_overlap"] = topk_overlap(p_bind.squeeze(0), target, target_mask)
        metrics["eclip/rna_tokens"] = torch.tensor(float(p_bind.shape[-1]), device=p_bind.device)
        metrics["loss"] = total_loss.detach()
        return total_loss, metrics

    def train_pdb_batch(self, batch: Dict[str, Any]) -> tuple[torch.Tensor, Dict[str, Any]]:
        batch = to_device(batch, self.device)
        loss, metrics, _ = self.forward_pdb_batch(batch, mode="train")
        return loss, metrics

    def train_eclip_samples(
        self,
        samples: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, Dict[str, Any]]:
        loss_sum = None
        merged_metrics: Dict[str, Any] = {}
        valid_count = 0
        for sample in samples:
            try:
                loss, metrics = self.forward_eclip_sample(sample)
            except Exception:
                if self.use_ddp or not self.eclip_cfg.skip_bad_samples:
                    raise
                self.dump_sample_error(sample, traceback.format_exc())
                continue
            loss_sum = loss if loss_sum is None else loss_sum + loss
            valid_count += 1
            merged_metrics.update(metrics)
        if valid_count == 0:
            raise RuntimeError(f"Rank {DIST_WRAPPER.rank} has no valid eCLIP samples.")
        merged_metrics["eclip/valid"] = torch.tensor(
            float(valid_count), device=loss_sum.device
        )
        return loss_sum / valid_count, merged_metrics

    def train_one_mixed_step(self, task: str) -> dict[str, float]:
        self.train_module.train()
        train_precision = {
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }[self.configs.dtype]
        enable_amp = (
            torch.autocast(device_type="cuda", dtype=train_precision, cache_enabled=False)
            if self.use_cuda
            else nullcontext()
        )

        with enable_amp:
            if task == "pdb":
                loss, metrics = self.train_pdb_batch(self._next_pdb_batch())
            else:
                loss, metrics = self.train_eclip_samples(self._next_eclip_samples())

        if self.configs.dtype in ["bf16", "fp32"] and is_loss_nan_check(loss):
            self.print(f"Skip {task} iteration with NaN/Inf loss at step {self.step}")
            self.optimizer.zero_grad(set_to_none=True)
            return {"skipped": 1.0, "task": task}

        loss_to_backward = loss / self.iters_to_accumulate
        loss_to_backward.backward()

        should_update = (self.global_step + 1) % self.iters_to_accumulate == 0
        if should_update:
            if self.configs.grad_clip_norm != 0.0:
                torch.nn.utils.clip_grad_norm_(
                    self.train_module.parameters(), self.configs.grad_clip_norm
                )
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.lr_scheduler.step()
            if hasattr(self, "ema_wrapper"):
                self.ema_wrapper.update()

        for key, value in metrics.items():
            if torch.is_tensor(value) or isinstance(value, (float, int)):
                self.train_metrics.add(
                    key,
                    value.detach() if torch.is_tensor(value) else value,
                    namespace="train",
                )
        return {
            "loss": float(loss.detach().cpu()),
            "task": task,
            "updated": float(should_update),
        }

    def dump_sample_error(self, sample: dict[str, Any], error: str) -> None:
        sample_id = str(sample.get("sample_id", "unknown")).replace("/", "_")
        digest = hashlib.sha1(sample_id.encode("utf-8")).hexdigest()[:10]
        path = self.error_dir / f"{self.step}_{digest}.txt"
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(f"sample_id={sample_id}\n")
            handle.write(error)

    @torch.no_grad()
    def evaluate_pdb(self, mode: str = "eval") -> dict[str, float]:
        metric_wrapper = SimpleMetricAggregator(["avg"])
        self.train_module.eval()
        eval_precision = {
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }[self.configs.dtype]
        enable_amp = (
            torch.autocast(device_type="cuda", dtype=eval_precision)
            if self.use_cuda
            else nullcontext()
        )
        for test_name, test_dl in self.pdb_test_dls.items():
            self.print(f"Testing PDB on {test_name}")
            evaluated_pids = []
            total_batch_num = len(test_dl)
            for index, batch in enumerate(tqdm(test_dl, disable=DIST_WRAPPER.rank != 0)):
                batch = to_device(batch, self.device)
                pid = batch["basic"]["pdb_id"]
                if index + 1 == total_batch_num and DIST_WRAPPER.world_size > 1:
                    all_data_ids = DIST_WRAPPER.all_gather_object(evaluated_pids)
                    dedup_ids = set(sum(all_data_ids, []))
                    if pid in dedup_ids:
                        break
                evaluated_pids.append(pid)
                with enable_amp:
                    _, loss_dict, batch = self.forward_pdb_batch(batch, mode=mode)
                    lddt_dict = self.lddt_metrics.compute_lddt(
                        batch["pred_dict"], batch["label_dict"]
                    )
                    lddt_metrics, _ = self.lddt_metrics.aggregate_lddt(
                        lddt_dict, batch["pred_dict"]["summary_confidence"]
                    )
                    simple_metrics = {
                        k: v for k, v in lddt_metrics.items() if "diff" not in k
                    }
                    simple_metrics.update(loss_dict)
                for key, value in simple_metrics.items():
                    metric_key = key if key.startswith("pdb/") else f"pdb/{key}"
                    metric_wrapper.add(metric_key, value, namespace=test_name)
                del batch, simple_metrics
                if index % 5 == 0:
                    torch.cuda.empty_cache()
        return metric_wrapper.calc()

    @torch.no_grad()
    def evaluate_eclip(self) -> dict[str, float]:
        self.train_module.eval()
        metric_wrapper = SimpleMetricAggregator(["avg"])
        for eval_step, samples in enumerate(
            tqdm(self.eclip_eval_dl, desc="eclip eval", leave=False, disable=DIST_WRAPPER.rank != 0)
        ):
            if (
                self.eclip_cfg.eval_max_steps is not None
                and eval_step >= self.eclip_cfg.eval_max_steps
            ):
                break
            for sample in samples:
                try:
                    loss, metrics = self.forward_eclip_sample(sample)
                except Exception:
                    if self.use_ddp or not self.eclip_cfg.skip_bad_samples:
                        raise
                    self.dump_sample_error(sample, traceback.format_exc())
                    continue
                metrics["loss"] = loss.detach()
                for key, value in metrics.items():
                    if torch.is_tensor(value) or isinstance(value, (float, int)):
                        metric_wrapper.add(
                            key,
                            value.detach() if torch.is_tensor(value) else value,
                            namespace="eclip_eval",
                        )
        return metric_wrapper.calc()

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        metrics = {}
        metrics.update(self.evaluate_pdb())
        metrics.update(self.evaluate_eclip())
        eval_loss_values = [
            float(value) for key, value in metrics.items() if key.endswith("/loss.avg")
        ]
        if eval_loss_values:
            metrics["eval/loss.avg"] = sum(float(v) for v in eval_loss_values) / len(
                eval_loss_values
            )
        self.print(f"Step {self.step} eval metrics: {metrics}")
        if self.configs.use_wandb and DIST_WRAPPER.rank == 0:
            wandb.log(metrics, step=self.step)
        return metrics

    def save_best_eval_checkpoint(self, metrics: dict[str, float]) -> None:
        eval_loss = metrics.get("eval/loss.avg")
        if eval_loss is None:
            return
        eval_loss = float(eval_loss)
        if eval_loss >= self.best_eval_loss:
            return
        self.best_eval_loss = eval_loss
        self.best_eval_step = self.step
        self.print(
            f"New best eval loss {self.best_eval_loss:.6f} at step {self.best_eval_step}"
        )
        self.save_checkpoint("best_eval.pt")

    def run(self) -> None:
        if self.configs.eval_only or self.configs.eval_first:
            self.evaluate()
            if self.configs.eval_only:
                return

        pbar = tqdm(
            total=self.configs.max_steps,
            initial=self.step,
            desc="RNA+eCLIP mixed SFT",
            dynamic_ncols=True,
            disable=DIST_WRAPPER.rank != 0,
        )
        self.optimizer.zero_grad(set_to_none=True)
        try:
            while self.step < self.configs.max_steps:
                task = self.choose_task()
                step_metrics = self.train_one_mixed_step(task)
                self.global_step += 1

                is_update_step = self.global_step % self.iters_to_accumulate == 0
                if is_update_step:
                    self.step += 1
                    postfix = {
                        "task": task,
                        "lr": f"{self.lr_scheduler.get_last_lr()[0]:.2e}",
                    }
                    if "loss" in step_metrics:
                        postfix["loss"] = f"{step_metrics['loss']:.4f}"
                    if "skipped" in step_metrics:
                        postfix["skipped"] = int(step_metrics["skipped"])
                    pbar.set_postfix(postfix)
                    pbar.update(1)

                    if self.configs.log_interval > 0 and self.step % self.configs.log_interval == 0:
                        metrics = self.train_metrics.calc()
                        self.print(f"Step {self.step} train metrics: {metrics}")
                        if self.configs.use_wandb and DIST_WRAPPER.rank == 0:
                            metrics["train/lr"] = self.lr_scheduler.get_last_lr()[0]
                            wandb.log(metrics, step=self.step)

                    if self.configs.eval_interval > 0 and self.step % self.configs.eval_interval == 0:
                        eval_metrics = self.evaluate()
                        self.save_best_eval_checkpoint(eval_metrics)

                    if (
                        self.configs.checkpoint_interval > 0
                        and self.step % self.configs.checkpoint_interval == 0
                    ):
                        self.save_checkpoint()
        finally:
            pbar.close()

        self.save_checkpoint()
        if self.use_ddp:
            dist.barrier()
            dist.destroy_process_group()


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
        **eclip_ppft_configs,
        **rna_signal_sft_configs,
        **rna_eclip_mixed_sft_configs,
    }
    parsed = parse_configs(first_pass, arg_str=arg_str, fill_required_with_null=True)
    model_name = parsed.model_name

    base_configs = {
        **configs_base,
        **{"data": data_configs},
        **eclip_ppft_configs,
        **rna_signal_sft_configs,
        **rna_eclip_mixed_sft_configs,
    }
    deep_update(base_configs, model_configs[model_name])
    configs = parse_configs(
        configs=base_configs,
        arg_str=arg_str,
        fill_required_with_null=True,
    )
    configs.sample_diffusion.N_sample = 1
    configs.sample_diffusion.N_sample_mini_rollout = 1
    return configs


@record
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
        "RNA+eCLIP mixed SFT config: model=%s pdb_sample_prob=%s",
        configs.model_name,
        configs.mixed_sft.pdb_sample_prob,
    )
    trainer = RNAEclipMixedSFTTrainer(configs)
    trainer.run()


if __name__ == "__main__":
    main()
