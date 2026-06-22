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

"""eCLIP PPFT trainer for Protenix.

This runner keeps the Protenix module unchanged. The differentiable eCLIP
binding scorer/loss is an external training sidecar and is saved outside
``checkpoint["model"]``.
"""

from __future__ import annotations

import datetime
import hashlib
import logging
import os
import time
import traceback
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.distributed as dist
import torch.nn as nn
import wandb
from biotite.structure import AtomArray
from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm

os.environ.setdefault("LAYERNORM_TYPE", "torch")

from configs.configs_base import configs as configs_base
from configs.configs_data import data_configs
from configs.configs_eclip_ppft import eclip_ppft_configs
from configs.configs_model_type import model_configs
from protenix.config.config import parse_configs, parse_sys_args, save_config
from protenix.data.eclip_ppft_dataset import (
    EclipPPFTDataset,
    build_protenix_sample_dict,
    collate_eclip_ppft_samples,
)
from protenix.data.inference.json_to_feature import SampleDictToFeatures
from protenix.data.utils import data_type_transform, make_dummy_feature
from protenix.model import sample_confidence
from protenix.model.eclip_binding import (
    EclipBindingScorer,
    EclipSignalLoss,
    align_signal_to_prediction,
    binary_auprc,
    masked_pearson,
    masked_std,
    normalize_log_signal,
    topk_overlap,
)
from protenix.model.generator_ppft import sample_diffusion_ppft
from protenix.model.protenix import Protenix, update_input_feature_dict
from protenix.utils.distributed import DIST_WRAPPER
from protenix.utils.lr_scheduler import get_lr_scheduler
from protenix.utils.metrics import SimpleMetricAggregator
from protenix.utils.seed import seed_everything
from protenix.utils.torch_utils import to_device
from protenix.utils.training import is_loss_nan_check

os.environ["WANDB_CONSOLE"] = "off"


def deep_update(target: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), Mapping):
            deep_update(target[key], value)
        else:
            target[key] = value
    return target


def build_eclip_ppft_checkpoint(
    *,
    model: nn.Module,
    binding_scorer: EclipBindingScorer,
    signal_loss: EclipSignalLoss,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    step: int,
    config: Any | None = None,
) -> dict[str, Any]:
    """Build a checkpoint whose ``model`` key contains only Protenix weights."""

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": None if optimizer is None else optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "step": step,
        "eclip_binding_scorer": binding_scorer.state_dict(),
        "eclip_signal_loss": signal_loss.state_dict(),
    }
    if config is not None:
        checkpoint["config"] = dict(config)
    return checkpoint


def featurize_eclip_sample(sample: dict[str, Any]) -> tuple[dict[str, torch.Tensor], AtomArray]:
    """Build Protenix inference-style features without structure labels."""

    sample_dict = build_protenix_sample_dict(sample)
    sample2feat = SampleDictToFeatures(sample_dict)
    features_dict, atom_array, _ = sample2feat.get_feature_dict()
    features_dict["distogram_rep_atom_mask"] = torch.tensor(
        atom_array.distogram_rep_atom_mask
    ).long()
    features_dict = make_dummy_feature(features_dict=features_dict, dummy_feats=["msa", "template"])
    return data_type_transform(features_dict), atom_array


def _module_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    if hasattr(model, "module"):
        return model.module.state_dict()
    return model.state_dict()


class EclipPPFTForwardModule(nn.Module):
    """DDP-visible eCLIP forward path over Protenix plus binding scorer."""

    def __init__(
        self,
        model: Protenix,
        binding_scorer: EclipBindingScorer,
        configs: Any,
        eclip_cfg: Any,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.model = model
        self.binding_scorer = binding_scorer
        self.configs = configs
        self.eclip_cfg = eclip_cfg
        self.device = device

    def _sync_if_cuda(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _timer_start(self) -> float:
        self._sync_if_cuda()
        return time.perf_counter()

    def _timer_elapsed(self, start: float) -> float:
        self._sync_if_cuda()
        return time.perf_counter() - start

    def forward(
        self, feat_dict: dict[str, torch.Tensor]
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
        dict[str, float],
    ]:
        timings: dict[str, float] = {}
        n_cycle = self.eclip_cfg.n_cycle
        if n_cycle is None:
            n_cycle = self.model.N_cycle

        start = self._timer_start()
        s_inputs, s, z = self.model.get_pairformer_output(
            input_feature_dict=feat_dict,
            N_cycle=int(n_cycle),
            inplace_safe=self.eclip_cfg.inplace_safe,
            chunk_size=self.eclip_cfg.diffusion_attn_chunk_size,
        )
        timings["time/pairformer_s"] = self._timer_elapsed(start)

        start = self._timer_start()
        cache = {"pair_z": None, "p_lm/c_l": [None, None]}
        if self.model.enable_diffusion_shared_vars_cache:
            cache["pair_z"] = self.model.diffusion_module.diffusion_conditioning.prepare_cache(
                feat_dict["relp"], z, False
            )
            cache["p_lm/c_l"] = self.model.diffusion_module.atom_attention_encoder.prepare_cache(
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
        timings["time/diffusion_cache_s"] = self._timer_elapsed(start)

        start = self._timer_start()
        noise_schedule = self.model.inference_noise_scheduler(
            N_step=self.eclip_cfg.n_rollout_steps,
            device=s_inputs.device,
            dtype=s_inputs.dtype,
        )
        timings["time/noise_schedule_s"] = self._timer_elapsed(start)

        start = self._timer_start()
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
            record_grad_steps=set(self.eclip_cfg.record_grad_steps),
            detach_unrecorded_steps=self.eclip_cfg.detach_unrecorded_steps,
            gamma0=self.configs.sample_diffusion.gamma0,
            gamma_min=self.configs.sample_diffusion.gamma_min,
            noise_scale_lambda=self.configs.sample_diffusion.noise_scale_lambda,
            step_scale_eta=self.configs.sample_diffusion.step_scale_eta,
            inplace_safe=self.eclip_cfg.inplace_safe,
            attn_chunk_size=self.eclip_cfg.diffusion_attn_chunk_size,
            enable_efficient_fusion=self.model.enable_efficient_fusion,
        )
        timings["time/diffusion_rollout_s"] = self._timer_elapsed(start)

        start = self._timer_start()
        p_bind, _ = self.binding_scorer(coords.float(), feat_dict)
        timings["time/binding_score_s"] = self._timer_elapsed(start)

        start = self._timer_start()
        quality_loss, quality_metrics = self.confidence_quality_loss(
            feat_dict=feat_dict,
            coords=coords,
            s_inputs=s_inputs,
            s=s,
            z=z,
        )
        timings["time/confidence_quality_s"] = self._timer_elapsed(start)
        return coords, p_bind, quality_loss, quality_metrics, timings

    def confidence_quality_loss(
        self,
        *,
        feat_dict: dict[str, torch.Tensor],
        coords: torch.Tensor,
        s_inputs: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        zero = coords.sum() * 0.0
        if self.eclip_cfg.confidence_quality_weight <= 0.0:
            return zero, {}

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
        quality_loss = torch.relu(target - plddt_mean)
        quality_loss = float(self.eclip_cfg.confidence_quality_weight) * quality_loss
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


class EclipPPFTTrainer:
    def __init__(self, configs: Any) -> None:
        self.configs = configs
        self.eclip_cfg = configs.eclip_ppft
        self.init_env()
        self.init_dirs()
        self.init_log()
        self.init_model()
        self.init_loss()
        self.init_train_module()
        self.init_optimizer()
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
                    f"cuda_device_count={torch.cuda.device_count()}, "
                    f"WORLD_SIZE={DIST_WRAPPER.world_size}, "
                    f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}. "
                    "Make sure the job exposes one GPU per local rank, or reduce "
                    "--nproc_per_node."
                )
            torch.cuda.set_device(self.device)
        if self.use_ddp:
            timeout_seconds = int(os.environ.get("NCCL_TIMEOUT_SECOND", 600))
            backend = "nccl" if self.use_cuda else "gloo"
            dist.init_process_group(
                backend=backend, timeout=datetime.timedelta(seconds=timeout_seconds)
            )
        if not self.configs.deterministic_seed:
            hash_string = f"({self.configs.seed},{DIST_WRAPPER.rank},eclip_ppft_seed)"
            rank_seed = int(hashlib.sha256(hash_string.encode("utf8")).hexdigest(), 16)
            rank_seed = rank_seed % (2**32)
        else:
            rank_seed = self.configs.seed
        seed_everything(rank_seed, deterministic=self.configs.deterministic)

    def init_dirs(self) -> None:
        self.step = 0
        self.global_step = 0
        self.run_name = self.configs.run_name + "_" + time.strftime("%Y%m%d_%H%M%S")
        run_names = DIST_WRAPPER.all_gather_object(
            self.run_name if DIST_WRAPPER.rank == 0 else None
        )
        self.run_name = [name for name in run_names if name is not None][0]
        self.run_dir = Path(self.configs.base_dir) / self.run_name
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.error_dir = self.run_dir / "errors"
        self.best_eval_loss = float("inf")
        self.best_eval_step = -1
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
        self.model = self.raw_model
        self.freeze_and_select_trainable_parameters()
        for param in self.raw_model.confidence_head.parameters():
            param.requires_grad_(False)
        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.model.parameters())
        self.print(
            f"Trainable Protenix params: {n_trainable / 1e6:.2f}M / {n_total / 1e6:.2f}M"
        )

    def init_loss(self) -> None:
        self.binding_scorer = EclipBindingScorer(
            cutoff=self.eclip_cfg.binding_cutoff,
            temperature=self.eclip_cfg.binding_temperature,
            softmin_beta=self.eclip_cfg.binding_softmin_beta,
            learn_temperature=self.eclip_cfg.learn_binding_temperature,
            learn_softmin_beta=self.eclip_cfg.learn_binding_softmin_beta,
            protein_atom_chunk_size=self.eclip_cfg.binding_protein_atom_chunk_size,
        ).to(self.device)
        self.signal_loss = EclipSignalLoss(
            profile_weight=self.eclip_cfg.signal_profile_weight,
            positive_weight=self.eclip_cfg.signal_positive_weight,
            point_weight=self.eclip_cfg.signal_point_weight,
        ).to(self.device)
        if not self.eclip_cfg.train_sidecar:
            for param in self.binding_scorer.parameters():
                param.requires_grad_(False)
            for param in self.signal_loss.parameters():
                param.requires_grad_(False)

    def init_train_module(self) -> None:
        self.train_module = EclipPPFTForwardModule(
            model=self.raw_model,
            binding_scorer=self.binding_scorer,
            configs=self.configs,
            eclip_cfg=self.eclip_cfg,
            device=self.device,
        ).to(self.device)
        if self.use_ddp:
            self.print("Using DistributedDataParallel (DDP) for eCLIP PPFT")
            if self.eclip_cfg.skip_bad_samples:
                self.print(
                    "DDP mode will fail fast on bad samples to keep ranks synchronized."
                )
            ddp_kwargs: dict[str, Any] = {
                "find_unused_parameters": self.configs.find_unused_parameters,
                "static_graph": not self.configs.find_unused_parameters,
                "broadcast_buffers": False,
            }
            if self.use_cuda:
                ddp_kwargs.update(
                    {
                        "device_ids": [DIST_WRAPPER.local_rank],
                        "output_device": DIST_WRAPPER.local_rank,
                    }
                )
            self.train_module = DDP(
                self.train_module,
                **ddp_kwargs,
            )

    def init_optimizer(self) -> None:
        self.optimizer = self.build_optimizer()
        self.scheduler = get_lr_scheduler(self.configs, self.optimizer)

    def init_data(self) -> None:
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
        self.train_dataset = EclipPPFTDataset(
            split=self.eclip_cfg.train_split,
            limit=self.eclip_cfg.train_limit,
            shuffle_files=self.eclip_cfg.shuffle_files,
            shuffle_buffer=self.eclip_cfg.shuffle_buffer,
            **common_kwargs,
        )
        self.eval_dataset = EclipPPFTDataset(
            split=self.eclip_cfg.eval_split,
            limit=self.eclip_cfg.eval_limit,
            shuffle_files=False,
            shuffle_buffer=0,
            **common_kwargs,
        )
        self.train_dl = DataLoader(
            self.train_dataset,
            batch_size=self.eclip_cfg.batch_size,
            num_workers=self.eclip_cfg.num_workers,
            collate_fn=collate_eclip_ppft_samples,
        )
        self.eval_dl = DataLoader(
            self.eval_dataset,
            batch_size=1,
            num_workers=0,
            collate_fn=collate_eclip_ppft_samples,
        )

    def freeze_and_select_trainable_parameters(self) -> None:
        for param in self.model.parameters():
            param.requires_grad_(False)
        patterns = self.trainable_name_patterns()
        selected = []
        for name, param in self.model.named_parameters():
            if any(pattern in name for pattern in patterns):
                param.requires_grad_(True)
                selected.append(name)
        if not selected:
            raise ValueError(f"No Protenix parameters matched trainable patterns: {patterns}")
        self.print(f"Trainable pattern list: {patterns}")
        self.print(f"First trainable params: {selected[:20]}")

    def trainable_name_patterns(self) -> list[str]:
        patterns = [
            "diffusion_module.diffusion_conditioning",
            "diffusion_module.atom_attention_decoder",
            "diffusion_module.layernorm_s",
            "diffusion_module.linear_no_bias_s",
            "diffusion_module.layernorm_a",
        ]
        n_diff_blocks = int(self.configs.model.diffusion_module.transformer.n_blocks)
        keep_diff = min(int(self.eclip_cfg.train_last_diffusion_blocks), n_diff_blocks)
        patterns.extend(
            f"diffusion_module.diffusion_transformer.blocks.{idx}."
            for idx in range(n_diff_blocks - keep_diff, n_diff_blocks)
        )
        keep_pair = int(self.eclip_cfg.train_last_pairformer_blocks)
        if self.eclip_cfg.train_stage == "diffusion_pairformer" or keep_pair > 0:
            n_pair_blocks = int(self.configs.model.pairformer.n_blocks)
            keep_pair = min(keep_pair, n_pair_blocks)
            patterns.extend(
                f"pairformer_stack.blocks.{idx}."
                for idx in range(n_pair_blocks - keep_pair, n_pair_blocks)
            )
            patterns.extend(["linear_no_bias_z_cycle", "linear_no_bias_s"])
        patterns.extend(list(self.eclip_cfg.extra_trainable_substrings))
        return [pattern for pattern in patterns if pattern]

    def build_optimizer(self) -> torch.optim.Optimizer:
        param_groups = [
            {
                "params": [p for p in self.model.parameters() if p.requires_grad],
                "lr": self.configs.lr,
            }
        ]
        sidecar_params = [
            p
            for module in (getattr(self, "binding_scorer", None), getattr(self, "signal_loss", None))
            if module is not None
            for p in module.parameters()
            if p.requires_grad
        ]
        if sidecar_params:
            param_groups.append({"params": sidecar_params, "lr": self.eclip_cfg.sidecar_lr})
        return torch.optim.Adam(
            param_groups,
            lr=self.configs.adam.lr,
            weight_decay=self.configs.adam.weight_decay,
            betas=(self.configs.adam.beta1, self.configs.adam.beta2),
        )

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
                self.scheduler.load_state_dict(checkpoint["scheduler"])
            if not self.configs.skip_load_step:
                self.step = int(checkpoint.get("step", -1)) + 1
                self.global_step = self.step
            self.best_eval_loss = float(checkpoint.get("best_eval_loss", self.best_eval_loss))
            self.best_eval_step = int(checkpoint.get("best_eval_step", self.best_eval_step))
        if "eclip_binding_scorer" in checkpoint:
            self.binding_scorer.load_state_dict(checkpoint["eclip_binding_scorer"], strict=False)
        if "eclip_signal_loss" in checkpoint:
            self.signal_loss.load_state_dict(checkpoint["eclip_signal_loss"], strict=False)
        self.print(f"Loaded checkpoint {checkpoint_path} at step {self.step}")

    def save_checkpoint(self, filename: str | None = None) -> None:
        if DIST_WRAPPER.rank != 0:
            return
        path = self.checkpoint_dir / (filename or f"{self.step}.pt")
        checkpoint = build_eclip_ppft_checkpoint(
            model=self.raw_model,
            binding_scorer=self.binding_scorer,
            signal_loss=self.signal_loss,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            step=self.step,
            config=self.configs,
        )
        checkpoint["model"] = _module_state_dict(self.raw_model)
        checkpoint["best_eval_loss"] = self.best_eval_loss
        checkpoint["best_eval_step"] = self.best_eval_step
        torch.save(checkpoint, path)
        logging.info("Saved checkpoint to %s", path)

    def save_best_eval_checkpoint(self, metrics: dict[str, float]) -> None:
        eval_loss = metrics.get("eval/loss.avg")
        if eval_loss is None:
            self.print("Skip best checkpoint update: eval/loss.avg is missing.")
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

    def print(self, msg: str) -> None:
        if DIST_WRAPPER.rank == 0:
            logging.info(msg)

    def _sync_if_cuda(self) -> None:
        if self.use_cuda:
            torch.cuda.synchronize(self.device)

    def _timer_start(self) -> float:
        self._sync_if_cuda()
        return time.perf_counter()

    def _timer_elapsed(self, start: float) -> float:
        self._sync_if_cuda()
        return time.perf_counter() - start

    @staticmethod
    def _format_timing(metrics: dict[str, Any], namespace: str = "train") -> str:
        parts = []
        for key, label in [
            (f"{namespace}/time/feature_cpu_s.avg", "feature_cpu"),
            (f"{namespace}/time/feature_update_s.avg", "feature_update"),
            (f"{namespace}/time/pairformer_s.avg", "pairformer"),
            (f"{namespace}/time/diffusion_rollout_s.avg", "rollout"),
            (f"{namespace}/time/binding_score_s.avg", "binding"),
            (f"{namespace}/time/confidence_quality_s.avg", "confidence"),
            (f"{namespace}/time/loss_s.avg", "loss"),
            (f"{namespace}/time/backward_s.avg", "backward"),
            (f"{namespace}/time/optimizer_s.avg", "optimizer"),
        ]:
            if key in metrics:
                parts.append(f"{label}={metrics[key]:.3f}s")
        return ", ".join(parts)

    def prepare_input_feature_dict(
        self, sample: dict[str, Any]
    ) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        timings: dict[str, float] = {}
        start = self._timer_start()
        feat_dict, _ = featurize_eclip_sample(sample)
        timings["time/feature_cpu_s"] = self._timer_elapsed(start)

        start = self._timer_start()
        feat_dict = to_device(feat_dict, self.device)
        feat_dict = self.model.relative_position_encoding.generate_relp(feat_dict)
        feat_dict = update_input_feature_dict(feat_dict)
        timings["time/feature_update_s"] = self._timer_elapsed(start)
        return feat_dict, timings

    def rollout_and_score(
        self, feat_dict: dict[str, torch.Tensor]
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
        dict[str, float],
    ]:
        coords, p_bind, quality_loss, quality_metrics, timings = self.train_module(feat_dict)
        return coords, p_bind, quality_loss, quality_metrics, timings

    def forward_sample(self, sample: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        forward_start = self._timer_start()
        feat_dict, timings = self.prepare_input_feature_dict(sample)
        _, p_bind, quality_loss, quality_metrics, rollout_timings = self.rollout_and_score(feat_dict)
        timings.update(rollout_timings)

        start = self._timer_start()
        target, target_mask = align_signal_to_prediction(
            sample["signal_vector"],
            p_bind.shape[-1],
            device=p_bind.device,
        )
        signal_loss, metrics = self.signal_loss(p_bind, target, target_mask)
        total_loss = self.eclip_cfg.signal_loss_weight * signal_loss + quality_loss
        timings["time/loss_s"] = self._timer_elapsed(start)
        timings["time/forward_total_s"] = self._timer_elapsed(forward_start)

        metrics = {f"eclip/{key}": value for key, value in metrics.items()}
        metrics.update({f"confidence/{key}": value for key, value in quality_metrics.items()})
        metrics.update(timings)
        metrics["loss"] = total_loss.detach()
        p_bind_1d = p_bind.squeeze(0)
        binary_target = (target > 0.0).to(dtype=target.dtype)
        point_target = normalize_log_signal(target.float().clamp_min(0.0))
        metrics["pearson"] = masked_pearson(p_bind_1d, binary_target, target_mask)
        metrics["signal_pearson"] = masked_pearson(p_bind_1d, target, target_mask)
        metrics["point_pearson"] = masked_pearson(p_bind_1d, point_target, target_mask)
        metrics["topk_overlap"] = topk_overlap(p_bind.squeeze(0), target, target_mask)
        metrics["profile_auprc"] = binary_auprc(p_bind_1d, target, target_mask)
        metrics["profile_positive_rate"] = binary_target[target_mask].mean()
        metrics["target_point_signal_std"] = masked_std(point_target, target_mask)
        metrics["pred_signal_std"] = masked_std(p_bind_1d, target_mask)
        metrics["target_signal_std"] = masked_std(target, target_mask)
        metrics["rna_tokens"] = torch.tensor(float(p_bind.shape[-1]), device=p_bind.device)
        return total_loss, metrics

    def train_batch(self, samples: list[dict[str, Any]]) -> dict[str, float]:
        self.train_module.train()
        self.signal_loss.train()
        train_precision = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[self.configs.dtype]
        enable_amp = (
            torch.autocast(device_type="cuda", dtype=train_precision, cache_enabled=False)
            if self.use_cuda
            else nullcontext()
        )
        self.optimizer.zero_grad(set_to_none=True)
        loss_sum = None
        valid_count = 0
        timing_sums: dict[str, float] = {}
        for sample in samples:
            try:
                with enable_amp:
                    loss, metrics = self.forward_sample(sample)
            except Exception:
                if self.use_ddp or not self.eclip_cfg.skip_bad_samples:
                    raise
                self.dump_sample_error(sample, traceback.format_exc())
                continue
            if loss_sum is None:
                loss_sum = loss
            else:
                loss_sum = loss_sum + loss
            valid_count += 1
            for key, value in metrics.items():
                if torch.is_tensor(value) or isinstance(value, (float, int)):
                    self.train_metrics.add(key, value.detach() if torch.is_tensor(value) else value, namespace="train")
                if key.startswith("time/"):
                    timing_sums[key] = timing_sums.get(key, 0.0) + float(value)
        if valid_count == 0:
            if self.use_ddp:
                raise RuntimeError(f"Rank {DIST_WRAPPER.rank} has no valid samples at step {self.step}.")
            logging.warning("Skipping empty/failed batch at step %d", self.step)
            return {"skipped": 1.0}
        loss_mean = loss_sum / valid_count
        if is_loss_nan_check(loss_mean):
            if self.use_ddp:
                raise RuntimeError(f"Rank {DIST_WRAPPER.rank} got NaN/Inf loss at step {self.step}.")
            logging.warning("Skipping NaN/Inf loss at step %d", self.step)
            return {"skipped": 1.0}

        start = self._timer_start()
        loss_mean.backward()
        backward_s = self._timer_elapsed(start)
        self.train_metrics.add("time/backward_s", backward_s, namespace="train")

        start = self._timer_start()
        if self.configs.grad_clip_norm != 0.0:
            params = [p for group in self.optimizer.param_groups for p in group["params"]]
            torch.nn.utils.clip_grad_norm_(params, self.configs.grad_clip_norm)
        self.optimizer.step()
        self.scheduler.step()
        optimizer_s = self._timer_elapsed(start)
        self.train_metrics.add("time/optimizer_s", optimizer_s, namespace="train")

        step_metrics = {
            "loss": float(loss_mean.detach().cpu()),
            "valid": float(valid_count),
        }
        step_metrics.update(
            {key: value / valid_count for key, value in timing_sums.items()}
        )
        step_metrics["time/backward_s"] = backward_s
        step_metrics["time/optimizer_s"] = optimizer_s
        return step_metrics

    def dump_sample_error(self, sample: dict[str, Any], error: str) -> None:
        sample_id = str(sample.get("sample_id", "unknown")).replace("/", "_")
        digest = hashlib.sha1(sample_id.encode("utf-8")).hexdigest()[:10]
        path = self.error_dir / f"{self.step}_{digest}.txt"
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(f"sample_id={sample_id}\n")
            handle.write(error)

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        self.train_module.eval()
        self.signal_loss.eval()
        metric_wrapper = SimpleMetricAggregator(["avg"])
        eval_iter = tqdm(
            self.eval_dl,
            desc="eval",
            leave=False,
            disable=DIST_WRAPPER.rank != 0,
        )
        for eval_step, samples in enumerate(eval_iter):
            if (
                self.eclip_cfg.eval_max_steps is not None
                and eval_step >= self.eclip_cfg.eval_max_steps
            ):
                break
            for sample in samples:
                try:
                    loss, metrics = self.forward_sample(sample)
                except Exception:
                    if self.use_ddp or not self.eclip_cfg.skip_bad_samples:
                        raise
                    self.dump_sample_error(sample, traceback.format_exc())
                    continue
                metrics["loss"] = loss.detach()
                for key, value in metrics.items():
                    if torch.is_tensor(value) or isinstance(value, (float, int)):
                        metric_wrapper.add(key, value.detach() if torch.is_tensor(value) else value, namespace="eval")
        metrics = metric_wrapper.calc()
        self.print(f"Step {self.step} eval metrics: {metrics}")
        timing_msg = self._format_timing(metrics, namespace="eval")
        if timing_msg:
            self.print(f"Step {self.step} eval timing: {timing_msg}")
        if self.configs.use_wandb and DIST_WRAPPER.rank == 0:
            wandb.log(metrics, step=self.step)
        return metrics

    def run(self) -> None:
        pbar = tqdm(
            total=self.configs.max_steps,
            initial=self.step,
            desc="eCLIP PPFT",
            dynamic_ncols=True,
            disable=DIST_WRAPPER.rank != 0,
        )
        try:
            while self.step < self.configs.max_steps:
                for samples in self.train_dl:
                    step_metrics = self.train_batch(samples)
                    self.global_step += 1
                    self.step += 1
                    postfix = {
                        "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
                    }
                    if "loss" in step_metrics:
                        postfix["loss"] = f"{step_metrics['loss']:.4f}"
                    for key, label in [
                        ("time/feature_cpu_s", "feat"),
                        ("time/pairformer_s", "pf"),
                        ("time/diffusion_rollout_s", "roll"),
                        ("time/binding_score_s", "bind"),
                        ("time/confidence_quality_s", "conf"),
                        ("time/backward_s", "bwd"),
                    ]:
                        if key in step_metrics:
                            postfix[label] = f"{step_metrics[key]:.1f}s"
                    if "skipped" in step_metrics:
                        postfix["skipped"] = int(step_metrics["skipped"])
                    pbar.set_postfix(postfix)
                    pbar.update(1)
                    if self.step % self.eclip_cfg.log_every_steps == 0:
                        metrics = self.train_metrics.calc()
                        self.print(f"Step {self.step} train metrics: {metrics}")
                        timing_msg = self._format_timing(metrics, namespace="train")
                        if timing_msg:
                            self.print(f"Step {self.step} train timing: {timing_msg}")
                        if self.configs.use_wandb and DIST_WRAPPER.rank == 0:
                            metrics["train/lr"] = self.scheduler.get_last_lr()[0]
                            wandb.log(metrics, step=self.step)
                    if self.eclip_cfg.eval_every_steps > 0 and self.step % self.eclip_cfg.eval_every_steps == 0:
                        eval_metrics = self.evaluate()
                        self.save_best_eval_checkpoint(eval_metrics)
                    if self.eclip_cfg.save_every_steps > 0 and self.step % self.eclip_cfg.save_every_steps == 0:
                        self.save_checkpoint()
                    if self.step >= self.configs.max_steps:
                        break
        finally:
            pbar.close()
        self.save_checkpoint()
        if self.use_ddp:
            dist.barrier()
            dist.destroy_process_group()


def build_configs(arg_str: str) -> Any:
    configs_base["triangle_attention"] = os.environ.get("TRIANGLE_ATTENTION", "cuequivariance")
    configs_base["triangle_multiplicative"] = os.environ.get("TRIANGLE_MULTIPLICATIVE", "cuequivariance")
    first_pass = {**configs_base, **{"data": data_configs}, **eclip_ppft_configs}
    parsed = parse_configs(first_pass, arg_str=arg_str, fill_required_with_null=True)
    model_name = parsed.model_name
    base_configs = {**configs_base, **{"data": data_configs}, **eclip_ppft_configs}
    deep_update(base_configs, model_configs[model_name])
    configs = parse_configs(base_configs, arg_str=arg_str, fill_required_with_null=True)
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
        "eCLIP PPFT config: model=%s rollout_samples=1 rollout_steps=%s",
        configs.model_name,
        configs.eclip_ppft.n_rollout_steps,
    )
    trainer = EclipPPFTTrainer(configs)
    trainer.run()


if __name__ == "__main__":
    main()
