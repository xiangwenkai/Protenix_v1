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

"""Online DiffusionNFT training entry point for Protenix coordinate diffusion."""

import logging
import os
from collections import Counter
from copy import deepcopy
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.distributed as dist
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from configs.configs_base import configs as configs_base
from configs.configs_data import data_configs
from configs.configs_model_type import model_configs
from configs.configs_nft import nft_configs
from protenix.config.config import parse_configs, parse_sys_args
from protenix.model.protenix import Protenix, update_input_feature_dict
from protenix.model.utils import centre_random_augmentation
from protenix.nft import (
    DiffusionNFTLoss,
    PolicyCopies,
    build_reward,
    combine_reward_component_advantages,
    compute_grouped_advantages,
    scheduled_old_policy_decay,
)
from protenix.utils.distributed import DIST_WRAPPER
from protenix.utils.torch_utils import to_device
from runner.train import AF3Trainer, FOLDBENCH_EVAL_CONFIGS

logger = logging.getLogger(__name__)


@dataclass
class RolloutGroup:
    """One condition, its old-policy structures, rewards, and normalized advantages."""

    batch: dict[str, Any]
    coordinates: torch.Tensor
    rewards: torch.Tensor
    reward_components: dict[str, torch.Tensor]
    component_weights: dict[str, float]
    reward_metrics: dict[str, torch.Tensor]
    advantages: torch.Tensor | None = None
    component_advantages: dict[str, torch.Tensor] | None = None


def deep_update(base: dict[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge nested configuration mappings."""
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


class ProtenixNFTTrainer(AF3Trainer):
    """Run old-policy rollouts followed by EDM-coordinate NFT updates."""

    def init_model(self) -> None:
        """Build a frozen Protenix trunk and three diffusion policy states."""
        self.raw_model = Protenix(self.configs).to(self.device)
        self.raw_model.requires_grad_(False)
        self.raw_model.eval()
        self.raw_model.diffusion_module.requires_grad_(True)

        self.current_policy = self.raw_model.diffusion_module
        self.use_ddp = DIST_WRAPPER.world_size > 1
        if self.use_ddp:
            self.current_policy = DDP(
                self.current_policy,
                device_ids=[DIST_WRAPPER.local_rank],
                output_device=DIST_WRAPPER.local_rank,
                find_unused_parameters=False,
            )
        self.model = self.current_policy
        self.base_checkpoint_path = str(self.configs.load_checkpoint_path or "")
        # Policy copies are created only after the supervised checkpoint is loaded.
        # This avoids allocating and immediately discarding two copies of random weights.
        self.policy_copies: PolicyCopies | None = None
        self.optimizer = torch.optim.AdamW(
            self.current_policy.parameters(),
            lr=self.configs.nft.learning_rate,
            betas=(self.configs.nft.adam_beta1, self.configs.nft.adam_beta2),
            weight_decay=self.configs.nft.weight_decay,
        )
        self.scaler = torch.GradScaler(
            device="cuda" if self.use_cuda else "cpu",
            enabled=self.configs.dtype == "fp16",
        )
        self.lr_scheduler = None

    def init_loss(self) -> None:
        """Initialize the NFT objective and configured experimental-signal reward."""
        super().init_loss()
        self.nft_loss = DiffusionNFTLoss(
            interpolation_beta=self.configs.nft.interpolation_beta,
            reference_weight=self.configs.nft.reference_weight,
            advantage_clip=self.configs.nft.advantage_clip,
        )
        reward_kwargs = self.configs.nft.reward.kwargs.to_dict()
        self.reward = build_reward(self.configs.nft.reward.name, reward_kwargs)

    def _unwrapped_current_policy(self) -> torch.nn.Module:
        return self.current_policy.module if isinstance(self.current_policy, DDP) else self.current_policy

    def _reset_policy_copies(self) -> None:
        self.policy_copies = PolicyCopies(self._unwrapped_current_policy())

    def _amp_context(self):
        """Match the mixed-precision context used by standard Protenix training."""
        precision = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }[self.configs.dtype]
        if self.use_cuda and self.configs.dtype != "fp32":
            return torch.autocast(
                device_type="cuda", dtype=precision, cache_enabled=False
            )
        return nullcontext()

    @staticmethod
    def _strip_ddp_prefix(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if state_dict and next(iter(state_dict)).startswith("module."):
            return {key[len("module.") :]: value for key, value in state_dict.items()}
        return dict(state_dict)

    def try_load_checkpoint(self) -> None:
        """Load either a supervised Protenix base or a complete NFT checkpoint."""
        resume_path = self.configs.nft.resume_from
        if resume_path:
            checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
            base_checkpoint_path = checkpoint.get("base_checkpoint_path") or self.base_checkpoint_path
            if "current_policy" in checkpoint:
                if not base_checkpoint_path:
                    raise ValueError(
                        "A base checkpoint is required to resume a compact NFT checkpoint"
                    )
                self._load_supervised_base(base_checkpoint_path)
                self._unwrapped_current_policy().load_state_dict(
                    self._strip_ddp_prefix(checkpoint["current_policy"]), strict=True
                )
            else:
                self.raw_model.load_state_dict(
                    self._strip_ddp_prefix(checkpoint["model"]), strict=True
                )
            self.base_checkpoint_path = str(base_checkpoint_path or "")
            self._reset_policy_copies()
            self.policy_copies.load_state_dict(checkpoint["policy_copies"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.scaler.load_state_dict(checkpoint["scaler"])
            self.step = int(checkpoint["step"])
            self.global_step = self.step
            self.start_step = self.step
            if "torch_rng_state" in checkpoint:
                torch.set_rng_state(checkpoint["torch_rng_state"])
            if self.use_cuda and "cuda_rng_state_all" in checkpoint:
                torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
            self.print(f"Resumed NFT training from {resume_path} at step {self.step}")
            return

        if self.configs.load_checkpoint_path:
            self._load_supervised_base(self.configs.load_checkpoint_path)
            self.base_checkpoint_path = str(self.configs.load_checkpoint_path)
            self._reset_policy_copies()
            self.print(f"Loaded supervised base model from {self.configs.load_checkpoint_path}")
        else:
            logger.warning("No base checkpoint was provided; NFT policies start from random initialization.")
            self._reset_policy_copies()

    def _load_supervised_base(self, checkpoint_path: str) -> None:
        """Load a supervised model through CPU memory to reduce peak GPU allocation."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model_state = checkpoint.get("model", checkpoint)
        self.raw_model.load_state_dict(
            self._strip_ddp_prefix(model_state), strict=self.configs.load_strict
        )

    def save_checkpoint(self, ema_suffix: str = "") -> None:
        """Save current, old, and reference policies as one resumable checkpoint."""
        del ema_suffix
        if DIST_WRAPPER.rank != 0:
            return
        if self.policy_copies is None:
            raise RuntimeError("NFT policy copies have not been initialized")
        checkpoint_path = os.path.join(self.checkpoint_dir, f"nft_step_{self.step}.pt")
        torch.save(
            {
                "format_version": 2,
                "base_checkpoint_path": self.base_checkpoint_path,
                "current_policy": self._unwrapped_current_policy().state_dict(),
                "policy_copies": self.policy_copies.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scaler": self.scaler.state_dict(),
                "step": self.step,
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state_all": torch.cuda.get_rng_state_all() if self.use_cuda else None,
            },
            checkpoint_path,
        )
        self.print(f"Saved NFT checkpoint to {checkpoint_path}")

    @torch.no_grad()
    def _prepare_conditioning(self, input_feature_dict: dict[str, Any]) -> dict[str, Any]:
        self.raw_model.eval()
        input_feature_dict = self.raw_model.relative_position_encoding.generate_relp(
            input_feature_dict
        )
        input_feature_dict = update_input_feature_dict(input_feature_dict)
        with self._amp_context():
            s_inputs, s_trunk, z_trunk = self.raw_model.get_pairformer_output(
                input_feature_dict=input_feature_dict,
                N_cycle=self.configs.model.N_cycle,
                inplace_safe=False,
                chunk_size=None,
            )
        return {
            "input_feature_dict": input_feature_dict,
            "s_inputs": s_inputs.detach(),
            "s_trunk": s_trunk.detach(),
            "z_trunk": z_trunk.detach(),
            "pair_z": None,
            "p_lm": None,
            "c_l": None,
            "enable_efficient_fusion": self.configs.enable_efficient_fusion,
        }

    @torch.no_grad()
    def _rollout(self, conditioning: dict[str, Any]) -> torch.Tensor:
        if self.policy_copies is None:
            raise RuntimeError("NFT policy copies have not been initialized")
        noise_schedule = self.raw_model.inference_noise_scheduler(
            N_step=self.configs.nft.rollout_steps,
            device=self.device,
            dtype=conditioning["s_inputs"].dtype,
        )
        with self._amp_context():
            coordinates = self.raw_model.sample_diffusion(
                denoise_net=self.policy_copies.old,
                **conditioning,
                noise_schedule=noise_schedule,
                N_sample=self.configs.nft.samples_per_condition,
            )
        if coordinates.ndim != 3:
            raise ValueError(
                "NFT currently expects one condition per dataloader item and rollout coordinates with "
                f"shape [N_sample, N_atom, 3], got {tuple(coordinates.shape)}"
            )
        return coordinates.detach()

    def _augment_coordinates(
        self, coordinates: torch.Tensor, coordinate_mask: torch.Tensor
    ) -> torch.Tensor:
        if not self.configs.nft.random_coordinate_augmentation:
            return coordinates
        return centre_random_augmentation(
            x_input_coords=coordinates,
            N_sample=1,
            mask=coordinate_mask,
        ).squeeze(-3)

    def _policy_predictions(
        self,
        noisy_coordinates: torch.Tensor,
        noise_level: torch.Tensor,
        conditioning: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.policy_copies is None:
            raise RuntimeError("NFT policy copies have not been initialized")
        policy_kwargs = {
            **conditioning,
            "x_noisy": noisy_coordinates,
            "t_hat_noise_level": noise_level,
        }
        with torch.no_grad():
            old_denoised = self.policy_copies.old(**policy_kwargs)
            reference_denoised = self.policy_copies.reference(**policy_kwargs)
        current_denoised = self.current_policy(**policy_kwargs)
        return current_denoised, old_denoised, reference_denoised

    @torch.no_grad()
    def _collect_rollout_group(
        self, batch: dict[str, Any]
    ) -> tuple[RolloutGroup | None, str | None]:
        """Collect and score one condition, returning CPU-resident rollout data."""
        device_batch = to_device(batch, self.device)
        input_feature_dict = device_batch["input_feature_dict"]
        label_dict = device_batch["label_dict"]
        condition_check = self.reward.check_condition(input_feature_dict, label_dict)
        if not condition_check.valid:
            return None, f"condition/{condition_check.reason or 'invalid'}"
        conditioning = self._prepare_conditioning(input_feature_dict)
        coordinates = self._rollout(conditioning)
        reward_output = self.reward(coordinates, input_feature_dict, label_dict)
        valid_mask = reward_output.valid_mask
        if valid_mask is None:
            valid_mask = torch.ones_like(reward_output.rewards, dtype=torch.bool)
        valid_mask = valid_mask.to(device=coordinates.device, dtype=torch.bool)
        if valid_mask.sum() < 2:
            valid_count = int(valid_mask.sum().item())
            return None, f"rollout/valid_{valid_count}_of_{valid_mask.numel()}"
        return (
            RolloutGroup(
                batch=batch,
                coordinates=coordinates[valid_mask].detach().cpu(),
                rewards=reward_output.rewards[valid_mask].detach().float().cpu(),
                reward_components={
                    name: value[valid_mask].detach().float().cpu()
                    for name, value in (
                        reward_output.components
                        or {"reward": reward_output.rewards}
                    ).items()
                },
                component_weights=(
                    reward_output.component_weights
                    or {"reward": 1.0}
                ),
                reward_metrics={
                    name: value.detach().float().cpu()
                    for name, value in reward_output.metrics.items()
                },
            ),
            None,
        )

    def _assign_global_advantages(
        self, groups: list[RolloutGroup]
    ) -> dict[str, torch.Tensor]:
        """Normalize every reward component independently before scalarization."""
        component_names = sorted(groups[0].reward_components)
        component_weights = groups[0].component_weights
        if set(component_weights) != set(component_names):
            raise ValueError(
                "Reward component weights must match reward components: "
                f"{sorted(component_weights)} != {component_names}"
            )
        for group in groups[1:]:
            if sorted(group.reward_components) != component_names:
                raise ValueError("Every rollout group must expose the same reward components")
            if group.component_weights != component_weights:
                raise ValueError("Reward component weights changed within one rollout buffer")
        for group in groups:
            group.advantages = torch.zeros_like(group.rewards)
            group.component_advantages = {}

        diagnostics = {}
        for name in component_names:
            local_values = torch.cat(
                [group.reward_components[name] for group in groups]
            ).to(self.device)
            statistics = torch.stack(
                [
                    local_values.sum(),
                    local_values.square().sum(),
                    local_values.new_tensor(float(local_values.numel())),
                ]
            )
            if dist.is_initialized():
                dist.all_reduce(statistics, op=dist.ReduceOp.SUM)
            global_mean = statistics[0] / statistics[2].clamp_min(1.0)
            global_variance = (
                statistics[1] / statistics[2].clamp_min(1.0) - global_mean.square()
            )
            global_scale = global_variance.clamp_min(0.0).sqrt().clamp_min(1e-4)
            component_advantages = []
            for group in groups:
                values = group.reward_components[name].to(self.device)
                if self.configs.nft.global_reward_std:
                    _, normalized = combine_reward_component_advantages(
                        {name: values},
                        {name: 1.0},
                        component_scales={name: global_scale},
                    )
                    advantage = normalized[name]
                else:
                    advantage = compute_grouped_advantages(values, global_std=False)
                advantage_cpu = advantage.detach().cpu()
                group.component_advantages[name] = advantage_cpu
                group.advantages += float(component_weights[name]) * advantage_cpu
                component_advantages.append(advantage)
            diagnostics[f"component/{name}_mean"] = local_values.mean()
            diagnostics[f"component/{name}_std"] = local_values.std(unbiased=False)
            diagnostics[f"component/{name}_advantage_abs_mean"] = torch.cat(
                component_advantages
            ).abs().mean()
            diagnostics[f"component/{name}_weight"] = local_values.new_tensor(
                float(component_weights[name])
            )
        diagnostics["combined_advantage_abs_mean"] = torch.cat(
            [group.advantages.to(self.device) for group in groups]
        ).abs().mean()
        return diagnostics

    def _collect_outer_rollouts(self) -> tuple[list[RolloutGroup], dict[str, torch.Tensor]]:
        """Collect a fixed number of valid condition groups on every rank."""
        target = int(self.configs.nft.rollout_groups_per_outer_step)
        max_attempts = int(self.configs.nft.max_rollout_attempts)
        if target < 1 or max_attempts < target:
            raise ValueError("rollout group settings require max_rollout_attempts >= target >= 1")

        groups: list[RolloutGroup] = []
        attempts = 0
        skip_reasons: Counter[str] = Counter()
        while True:
            if len(groups) < target and attempts < max_attempts:
                try:
                    batch = next(self._train_iterator)
                except StopIteration:
                    self._train_iterator = iter(self.train_dl)
                    batch = next(self._train_iterator)
                group, skip_reason = self._collect_rollout_group(batch)
                attempts += 1
                if group is not None:
                    groups.append(group)
                elif skip_reason is not None:
                    skip_reasons[skip_reason] += 1

            ready = torch.tensor(
                int(len(groups) >= target), device=self.device, dtype=torch.int32
            )
            if dist.is_initialized():
                dist.all_reduce(ready, op=dist.ReduceOp.MIN)
            if ready.item() == 1:
                break

            exhausted = torch.tensor(
                int(attempts >= max_attempts), device=self.device, dtype=torch.int32
            )
            if dist.is_initialized():
                dist.all_reduce(exhausted, op=dist.ReduceOp.MAX)
            if exhausted.item() == 1:
                raise RuntimeError(
                    "Unable to collect valid NFT reward groups on every rank within "
                    f"{max_attempts} attempts"
                )

        groups = groups[:target]
        if skip_reasons and DIST_WRAPPER.rank == 0:
            logger.info(
                "NFT rollout filtering summary after %d attempts: %s",
                attempts,
                dict(sorted(skip_reasons.items())),
            )
        component_metrics = self._assign_global_advantages(groups)
        rewards = torch.cat([group.rewards for group in groups])
        return groups, {
            "reward_mean": rewards.mean(),
            "reward_std": rewards.std(unbiased=False),
            "rollout_attempts": rewards.new_tensor(float(attempts)),
            "valid_rollout_groups": rewards.new_tensor(float(len(groups))),
            **component_metrics,
        }

    def _edm_reference_output_scale(self, noise_level: torch.Tensor) -> torch.Tensor:
        """Return EDM c_out so denoised-coordinate differences recover raw-network MSE."""
        sigma_data = float(self.raw_model.diffusion_module.sigma_data)
        ratio = noise_level.float() / sigma_data
        return noise_level.float() / torch.sqrt(1.0 + ratio.square())

    def _optimizer_step(self) -> None:
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            self.current_policy.parameters(), self.configs.nft.max_grad_norm
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.step += 1
        self.global_step = self.step

    def train_step(
        self, groups: list[RolloutGroup], collection_metrics: Mapping[str, torch.Tensor]
    ) -> dict[str, float]:
        """Optimize a frozen-old-policy rollout buffer, then update old policy once."""
        metric_values: dict[str, list[torch.Tensor]] = {}
        forward_noise_samples = int(self.configs.nft.forward_noise_samples)
        accumulation_groups = max(1, int(self.iters_to_accumulate))
        if forward_noise_samples < 1:
            raise ValueError("forward_noise_samples must be positive")

        stop_training = False
        for _ in range(int(self.configs.nft.inner_steps)):
            permutation = torch.randperm(len(groups)).tolist()
            ordered_groups = [groups[index] for index in permutation]
            for chunk_start in range(0, len(ordered_groups), accumulation_groups):
                chunk = ordered_groups[chunk_start : chunk_start + accumulation_groups]
                denominator = float(len(chunk) * forward_noise_samples)
                self.optimizer.zero_grad(set_to_none=True)

                for group in chunk:
                    device_batch = to_device(group.batch, self.device)
                    input_feature_dict = device_batch["input_feature_dict"]
                    label_dict = device_batch["label_dict"]
                    coordinate_mask = label_dict["coordinate_mask"].float()
                    coordinates = group.coordinates.to(self.device)
                    advantages = group.advantages
                    if advantages is None:
                        raise RuntimeError("Advantages must be assigned before NFT optimization")
                    advantages = advantages.to(self.device)
                    conditioning = self._prepare_conditioning(input_feature_dict)
                    self.current_policy.train()

                    for _ in range(forward_noise_samples):
                        clean_coordinates = self._augment_coordinates(
                            coordinates, coordinate_mask
                        )
                        noise_level = self.raw_model.train_noise_sampler(
                            size=clean_coordinates.shape[:-2], device=self.device
                        ).to(dtype=clean_coordinates.dtype)
                        noisy_coordinates = clean_coordinates + (
                            torch.randn_like(clean_coordinates)
                            * noise_level[..., None, None]
                        )

                        with self._amp_context():
                            current, old, reference = self._policy_predictions(
                                noisy_coordinates, noise_level, conditioning
                            )
                            loss_output = self.nft_loss(
                                current_denoised=current,
                                old_denoised=old,
                                reference_denoised=reference,
                                target_coordinates=clean_coordinates,
                                advantages=advantages,
                                coordinate_mask=coordinate_mask,
                                reference_output_scale=self._edm_reference_output_scale(
                                    noise_level
                                ),
                            )
                        self.scaler.scale(loss_output.loss / denominator).backward()
                        for name, value in loss_output.metrics.items():
                            metric_values.setdefault(name, []).append(value.detach())

                self._optimizer_step()
                if self.step >= self.configs.max_steps:
                    stop_training = True
                    break
            if stop_training:
                break

        if self.policy_copies is None:
            raise RuntimeError("NFT policy copies have not been initialized")
        decay = scheduled_old_policy_decay(
            self.step,
            int(self.configs.nft.old_policy_decay_type),
            float(self.configs.nft.old_policy_decay),
        )
        self.policy_copies.update_old(self.current_policy, decay=decay)
        metric_values.setdefault("old_policy_decay", []).append(
            torch.tensor(decay, device=self.device)
        )
        for group in groups:
            for name, value in group.reward_metrics.items():
                metric_values.setdefault(name, []).append(value.to(self.device))
        for name, value in collection_metrics.items():
            metric_values.setdefault(name, []).append(value.to(self.device))
        averaged = {
            name: torch.stack([value.float().mean() for value in values]).mean()
            for name, values in metric_values.items()
        }
        return self._reduce_metrics(averaged)

    def _reduce_metrics(self, metrics: Mapping[str, torch.Tensor]) -> dict[str, float]:
        local_metrics = {
            name: value.detach().float().mean().item() for name, value in metrics.items()
        }
        if not dist.is_initialized():
            return local_metrics

        # Reward diagnostics can legitimately differ between ranks. For example,
        # pseudo-local lDDT exposes fewer metrics when a condition has too few
        # high-confidence pairs. Reducing each local dictionary entry separately
        # would then issue a different number/order of collectives on every rank
        # and eventually deadlock NCCL. Gather once, then aggregate the union of
        # metric names locally so every rank always executes the same collective.
        gathered_metrics = DIST_WRAPPER.all_gather_object(local_metrics)
        metric_names = sorted(
            {name for rank_metrics in gathered_metrics for name in rank_metrics}
        )
        return {
            name: sum(
                rank_metrics[name]
                for rank_metrics in gathered_metrics
                if name in rank_metrics
            )
            / sum(name in rank_metrics for rank_metrics in gathered_metrics)
            for name in metric_names
        }

    def _foldbench_validation_enabled(self) -> bool:
        """Return whether FoldBench validation is enabled in the merged config."""
        fold_cfg = getattr(self.configs, "foldbench_eval", None)
        return bool(fold_cfg is not None and fold_cfg.enable)

    def _has_validation_targets(self) -> bool:
        """Return whether regular test sets or FoldBench require validation."""
        return bool(self.test_dls) or self._foldbench_validation_enabled()

    @staticmethod
    def _next_validation_step(step: int, interval: int) -> int | None:
        """Return the first validation boundary strictly after the current step."""
        if interval <= 0:
            return None
        return (max(int(step), 0) // int(interval) + 1) * int(interval)

    @torch.no_grad()
    def evaluate(self, mode: str = "eval") -> None:
        """Run native Protenix test-set metrics and optional FoldBench validation."""
        training_model = self.model
        raw_model_was_training = self.raw_model.training
        policy_was_training = self.current_policy.training
        try:
            # AF3Trainer._evaluate expects self.model to be the complete Protenix
            # model. NFT trains only its diffusion module through self.model.
            self.model = self.raw_model
            if self.test_dls:
                self._evaluate(mode=mode)
            if self._foldbench_validation_enabled():
                foldbench_metrics = self.evaluate_foldbench()
                if foldbench_metrics and self.configs.use_wandb and DIST_WRAPPER.rank == 0:
                    wandb.log(foldbench_metrics, step=self.step)
        finally:
            self.model = training_model
            self.raw_model.train(raw_model_was_training)
            self.current_policy.train(policy_was_training)

    def run(self) -> None:
        """Run the online NFT outer loop."""
        self._train_iterator = iter(self.train_dl)
        has_validation_targets = self._has_validation_targets()
        if self.configs.eval_only or self.configs.eval_first:
            if has_validation_targets:
                self.evaluate()
            else:
                self.print("NFT validation skipped: no data.test_sets or FoldBench target configured")
            if self.configs.eval_only:
                return

        eval_interval = int(self.configs.eval_interval)
        next_validation_step = self._next_validation_step(self.step, eval_interval)
        progress = tqdm(
            total=int(self.configs.max_steps),
            initial=int(self.step),
            desc="NFT training",
            unit="step",
            dynamic_ncols=True,
            disable=DIST_WRAPPER.rank != 0,
        )
        try:
            while self.step < self.configs.max_steps:
                previous_step = self.step
                groups, collection_metrics = self._collect_outer_rollouts()
                metrics = self.train_step(groups, collection_metrics)
                progress.update(self.step - previous_step)
                progress.set_postfix(
                    loss=f"{metrics['loss']:.4f}",
                    reward=f"{metrics['reward_mean']:.4f}",
                    attempts=int(metrics["rollout_attempts"]),
                )
                if DIST_WRAPPER.rank == 0 and self.step % self.configs.log_interval == 0:
                    self.print(f"NFT step {self.step}: {metrics}")
                    if self.configs.use_wandb:
                        wandb.log(
                            {f"nft/{key}": value for key, value in metrics.items()},
                            step=self.step,
                        )

                should_save = (
                    self.configs.checkpoint_interval > 0
                    and self.step % self.configs.checkpoint_interval == 0
                )
                should_save_at_end = (
                    self.step >= self.configs.max_steps and self.configs.nft.save_at_end
                )
                if should_save or should_save_at_end:
                    self.save_checkpoint()

                reached_validation_interval = (
                    next_validation_step is not None and self.step >= next_validation_step
                )
                reached_last_step = self.step >= self.configs.max_steps
                if has_validation_targets and (reached_validation_interval or reached_last_step):
                    self.evaluate()
                    progress.refresh()
                if reached_validation_interval:
                    next_validation_step = self._next_validation_step(self.step, eval_interval)
        finally:
            progress.close()


def build_configs(arg_str: str) -> Any:
    """Merge base, model-specific, data, and NFT configurations."""
    initial = {
        **configs_base,
        "data": deepcopy(data_configs),
        **deepcopy(FOLDBENCH_EVAL_CONFIGS),
    }
    deep_update(initial, nft_configs)
    first_pass = parse_configs(initial, arg_str=arg_str, fill_required_with_null=True)
    merged = {
        **configs_base,
        "data": deepcopy(data_configs),
        **deepcopy(FOLDBENCH_EVAL_CONFIGS),
    }
    deep_update(merged, model_configs[first_pass.model_name])
    deep_update(merged, nft_configs)
    return parse_configs(merged, arg_str=arg_str, fill_required_with_null=True)


def main() -> None:
    """Parse command-line configuration and start NFT training."""
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
    )
    configs_base["triangle_attention"] = os.environ.get("TRIANGLE_ATTENTION", "cuequivariance")
    configs_base["triangle_multiplicative"] = os.environ.get(
        "TRIANGLE_MULTIPLICATIVE", "cuequivariance"
    )
    trainer = ProtenixNFTTrainer(build_configs(parse_sys_args()))
    trainer.run()


if __name__ == "__main__":
    main()
