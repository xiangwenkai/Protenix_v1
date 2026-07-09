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
import json
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
    collect_cell_vocab,
    collate_eclip_ppft_samples,
)
from protenix.data.inference.json_to_feature import SampleDictToFeatures
from protenix.data.msa.msa_featurizer import InferenceMSAFeaturizer
from protenix.data.utils import data_type_transform, make_dummy_feature
from protenix.model import sample_confidence
from protenix.model.eclip_binding import (
    EclipSignalLoss,
    align_signal_to_prediction,
    compute_distogram_binding_score,
)
from protenix.model.generator_ppft import sample_diffusion_ppft
from protenix.model.protenix import Protenix, update_input_feature_dict
from protenix.utils.distributed import DIST_WRAPPER
from protenix.utils.lr_scheduler import get_lr_scheduler
from protenix.utils.metrics import SimpleMetricAggregator
from protenix.utils.seed import seed_everything
from protenix.utils.torch_utils import dict_to_tensor, to_device
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
        "eclip_signal_loss": signal_loss.state_dict(),
    }
    if config is not None:
        checkpoint["config"] = dict(config)
    return checkpoint


ProteinMsaLookup = dict[str, dict[str, str]]


def _resolve_msa_path(msa_path: str, base_dir: Path) -> str:
    path = Path(msa_path)
    if not path.is_absolute():
        path = base_dir / path
    return str(path)


def _normalize_msa_info(key: str, value: Any, base_dir: Path) -> dict[str, str]:
    if isinstance(value, str):
        msa_info = {"unpairedMsaPath": _resolve_msa_path(value, base_dir)}
    elif isinstance(value, Mapping):
        key_map = {
            "unpairedMsaPath": "unpairedMsaPath",
            "pairedMsaPath": "pairedMsaPath",
            "unpaired_msa_path": "unpairedMsaPath",
            "paired_msa_path": "pairedMsaPath",
            "non_pairing": "unpairedMsaPath",
            "pairing": "pairedMsaPath",
            "non_pairing_a3m": "unpairedMsaPath",
            "pairing_a3m": "pairedMsaPath",
        }
        msa_info = {
            target_key: _resolve_msa_path(str(value[source_key]), base_dir)
            for source_key, target_key in key_map.items()
            if value.get(source_key)
        }
        msa_dir = value.get("precomputed_msa_dir") or value.get("msa_dir")
        if msa_dir:
            msa_dir = _resolve_msa_path(str(msa_dir), base_dir)
            unpaired = os.path.join(msa_dir, "non_pairing.a3m")
            paired = os.path.join(msa_dir, "pairing.a3m")
            if os.path.exists(unpaired):
                msa_info.setdefault("unpairedMsaPath", unpaired)
            if os.path.exists(paired):
                msa_info.setdefault("pairedMsaPath", paired)
    else:
        raise TypeError(f"Unsupported MSA map value for {key!r}: {type(value).__name__}")

    if not msa_info:
        raise ValueError(f"MSA map entry {key!r} has no usable MSA path.")
    for msa_key, msa_path in msa_info.items():
        if not os.path.exists(msa_path):
            raise FileNotFoundError(f"MSA path for {key!r} ({msa_key}) does not exist: {msa_path}")
    return msa_info


def load_protein_msa_lookup(path: str | None) -> ProteinMsaLookup:
    if not path:
        return {}
    lookup_path = Path(path)
    with open(lookup_path, "r", encoding="utf-8") as f:
        raw_lookup = json.load(f)
    if not isinstance(raw_lookup, dict):
        raise TypeError("protein_msa_map_json must contain a JSON object.")
    return {
        str(key): _normalize_msa_info(str(key), value, lookup_path.parent)
        for key, value in raw_lookup.items()
    }


def configure_eclip_cell_condition(configs: Any, eclip_cfg: Any) -> dict[str, int]:
    use_cell_condition = bool(eclip_cfg.use_cell_condition)
    cell_vocab = collect_cell_vocab(eclip_cfg.data_dir) if use_cell_condition else {}
    cell_adapter = configs.model.diffusion_module.cell_adapter
    cell_adapter.enable = use_cell_condition
    cell_adapter.num_cells = len(cell_vocab) + 1 if use_cell_condition else 0
    cell_adapter.embedding_dim = int(eclip_cfg.cell_condition_dim)
    cell_adapter.target = str(eclip_cfg.cell_condition_target)
    return cell_vocab


def load_model_state_dict_allowing_cell_adapter(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    *,
    strict: bool,
    allow_cell_adapter_missing: bool,
) -> None:
    if not strict or not allow_cell_adapter_missing:
        model.load_state_dict(state_dict, strict=strict)
        return

    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    disallowed_missing = [
        key
        for key in missing
        if not key.startswith("diffusion_module.cell_adapter.")
    ]
    if disallowed_missing or unexpected:
        raise RuntimeError(
            "Checkpoint is not strict-compatible after allowing cell adapter params. "
            f"missing={disallowed_missing}, unexpected={unexpected}"
        )


def _lookup_protein_msa_info(
    sample: dict[str, Any],
    protein_msa_lookup: ProteinMsaLookup | None,
) -> dict[str, str] | None:
    if not protein_msa_lookup:
        return None
    protein_sequence = str(sample.get("protein_sequence", ""))
    sequence_sha1 = hashlib.sha1(protein_sequence.encode("utf-8")).hexdigest()
    candidate_keys = [
        str(sample.get("protein_symbol", "")),
        protein_sequence,
        sequence_sha1,
        f"sha1:{sequence_sha1}",
    ]
    for key in candidate_keys:
        if key and key in protein_msa_lookup:
            return protein_msa_lookup[key]
    return None


def _attach_protein_msa_info(
    sample_dict: dict[str, Any],
    sample: dict[str, Any],
    protein_msa_lookup: ProteinMsaLookup | None,
) -> bool:
    msa_info = _lookup_protein_msa_info(sample, protein_msa_lookup)
    if msa_info is None:
        return False
    for sequence_entry in sample_dict["sequences"]:
        protein_chain = sequence_entry.get("proteinChain")
        if protein_chain is not None:
            protein_chain.update(msa_info)
            return True
    return False


def featurize_eclip_sample(
    sample: dict[str, Any],
    protein_msa_lookup: ProteinMsaLookup | None = None,
    *,
    protein_msa_pair_as_unpair: bool = True,
    protein_msa_use_rna_msa: bool = False,
) -> tuple[dict[str, torch.Tensor], AtomArray]:
    """Build Protenix inference-style features without structure labels."""

    sample_dict = build_protenix_sample_dict(sample)
    has_protein_msa = _attach_protein_msa_info(
        sample_dict=sample_dict,
        sample=sample,
        protein_msa_lookup=protein_msa_lookup,
    )
    sample2feat = SampleDictToFeatures(sample_dict)
    features_dict, atom_array, _ = sample2feat.get_feature_dict()
    features_dict["distogram_rep_atom_mask"] = torch.tensor(
        atom_array.distogram_rep_atom_mask
    ).long()
    features_dict["cell_id"] = torch.tensor(
        int(sample.get("cell_id", 0)),
        dtype=torch.long,
    )
    dummy_feats = ["template"]
    if has_protein_msa:
        msa_features = InferenceMSAFeaturizer.make_msa_feature(
            bioassembly=sample_dict["sequences"],
            atom_array=atom_array,
            msa_pair_as_unpair=protein_msa_pair_as_unpair,
            use_rna_msa=protein_msa_use_rna_msa,
        )
        features_dict.update(dict_to_tensor(msa_features))
    else:
        dummy_feats.append("msa")
    features_dict = make_dummy_feature(features_dict=features_dict, dummy_feats=dummy_feats)
    return data_type_transform(features_dict), atom_array


def _module_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    if hasattr(model, "module"):
        return model.module.state_dict()
    return model.state_dict()


class EclipPPFTForwardModule(nn.Module):
    """DDP-visible eCLIP forward path over Protenix distogram contacts."""

    def __init__(
        self,
        model: Protenix,
        configs: Any,
        eclip_cfg: Any,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.model = model
        self.configs = configs
        self.eclip_cfg = eclip_cfg
        self.device = device

    def forward(
        self, feat_dict: dict[str, torch.Tensor]
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
        torch.Tensor | None,
    ]:
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
            return p_bind, p_bind.sum() * 0.0, {}, None

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
        return p_bind, quality_loss, quality_metrics, coords.detach()

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
            "quality_loss": quality_loss.detach(),
        }
        return quality_loss, metrics


class EclipPPFTTrainer:
    def __init__(self, configs: Any) -> None:
        self.configs = configs
        self.eclip_cfg = configs.eclip_ppft
        self.protein_msa_lookup = load_protein_msa_lookup(
            self.eclip_cfg.protein_msa_map_json
        )
        self.cell_vocab = configure_eclip_cell_condition(self.configs, self.eclip_cfg)
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
        self.signal_loss = EclipSignalLoss(
            profile_weight=self.eclip_cfg.signal_profile_weight,
            min_height=self.eclip_cfg.signal_multinomial_min_height,
            binary_threshold=self.eclip_cfg.signal_binary_threshold,
            signal_clip_value=self.eclip_cfg.signal_clip_value,
            max_total_count=self.eclip_cfg.signal_multinomial_max_total,
        ).to(self.device)
        if not self.eclip_cfg.train_sidecar:
            for param in self.signal_loss.parameters():
                param.requires_grad_(False)

    def init_train_module(self) -> None:
        self.train_module = EclipPPFTForwardModule(
            model=self.raw_model,
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
            cell_vocab=self.cell_vocab,
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
        if self.protein_msa_lookup:
            self.print(
                f"Loaded protein MSA map entries: {len(self.protein_msa_lookup)}"
            )
        if self.eclip_cfg.use_cell_condition:
            self.print(
                f"Using eCLIP cell condition: cells={len(self.cell_vocab)} "
                f"target={self.eclip_cfg.cell_condition_target}"
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
        patterns = []
        keep_pair = int(self.eclip_cfg.train_last_pairformer_blocks)
        n_pair_blocks = int(self.configs.model.pairformer.n_blocks)
        if keep_pair < 0:
            patterns.append("pairformer_stack.")
        elif keep_pair > 0:
            keep_pair = min(keep_pair, n_pair_blocks)
            patterns.extend(
                f"pairformer_stack.blocks.{idx}."
                for idx in range(n_pair_blocks - keep_pair, n_pair_blocks)
            )
        if self.eclip_cfg.use_cell_condition:
            patterns.append("diffusion_module.cell_adapter.")
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
            for module in (getattr(self, "signal_loss", None),)
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
        load_model_state_dict_allowing_cell_adapter(
            self.raw_model,
            state_dict,
            strict=self.configs.load_strict,
            allow_cell_adapter_missing=bool(self.eclip_cfg.use_cell_condition),
        )
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
        if "eclip_signal_loss" in checkpoint:
            self.signal_loss.load_state_dict(checkpoint["eclip_signal_loss"], strict=False)
        self.print(f"Loaded checkpoint {checkpoint_path} at step {self.step}")

    def save_checkpoint(self, filename: str | None = None) -> None:
        if DIST_WRAPPER.rank != 0:
            return
        path = self.checkpoint_dir / (filename or f"{self.step}.pt")
        checkpoint = build_eclip_ppft_checkpoint(
            model=self.raw_model,
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

    def prepare_input_feature_dict(
        self, sample: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        feat_dict, _ = featurize_eclip_sample(
            sample,
            protein_msa_lookup=self.protein_msa_lookup,
            protein_msa_pair_as_unpair=self.eclip_cfg.protein_msa_pair_as_unpair,
            protein_msa_use_rna_msa=self.eclip_cfg.protein_msa_use_rna_msa,
        )
        feat_dict = to_device(feat_dict, self.device)
        feat_dict = self.model.relative_position_encoding.generate_relp(feat_dict)
        feat_dict = update_input_feature_dict(feat_dict)
        return feat_dict

    def rollout_and_score(
        self, feat_dict: dict[str, torch.Tensor]
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
        torch.Tensor | None,
    ]:
        p_bind, quality_loss, quality_metrics, rollout_coords = self.train_module(feat_dict)
        return p_bind, quality_loss, quality_metrics, rollout_coords

    def forward_sample(
        self,
        sample: dict[str, Any],
        *,
        include_rollout_metrics: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        feat_dict = self.prepare_input_feature_dict(sample)
        p_bind, quality_loss, quality_metrics, rollout_coords = self.rollout_and_score(feat_dict)

        target, target_mask = align_signal_to_prediction(
            sample["signal_vector"],
            p_bind.shape[-1],
            device=p_bind.device,
        )
        signal_loss, metrics = self.signal_loss(p_bind, target, target_mask)
        total_loss = self.eclip_cfg.signal_loss_weight * signal_loss + quality_loss

        metrics = {f"eclip/{key}": value for key, value in metrics.items()}
        metrics.update({f"confidence/{key}": value for key, value in quality_metrics.items()})
        metrics["loss"] = total_loss.detach()
        _ = include_rollout_metrics, rollout_coords
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

        loss_mean.backward()

        if self.configs.grad_clip_norm != 0.0:
            params = [p for group in self.optimizer.param_groups for p in group["params"]]
            torch.nn.utils.clip_grad_norm_(params, self.configs.grad_clip_norm)
        self.optimizer.step()
        self.scheduler.step()

        step_metrics = {
            "loss": float(loss_mean.detach().cpu()),
            "valid": float(valid_count),
        }
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
                    loss, metrics = self.forward_sample(
                        sample,
                        include_rollout_metrics=True,
                    )
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
                    if "skipped" in step_metrics:
                        postfix["skipped"] = int(step_metrics["skipped"])
                    pbar.set_postfix(postfix)
                    pbar.update(1)
                    if self.step % self.eclip_cfg.log_every_steps == 0:
                        metrics = self.train_metrics.calc()
                        self.print(f"Step {self.step} train metrics: {metrics}")
                        if self.configs.use_wandb and DIST_WRAPPER.rank == 0:
                            metrics["train/lr"] = self.scheduler.get_last_lr()[0]
                            wandb.log(metrics, step=self.step)
                    if self.configs.eval_interval > 0 and self.step % self.configs.eval_interval == 0:
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
        "eCLIP PPFT config: model=%s binding=distogram_contact confidence_rollout_steps=%s",
        configs.model_name,
        configs.eclip_ppft.confidence_rollout_steps,
    )
    trainer = EclipPPFTTrainer(configs)
    trainer.run()


if __name__ == "__main__":
    main()
