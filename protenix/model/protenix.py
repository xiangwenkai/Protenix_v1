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

import copy
import random
import time
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from protenix.model import sample_confidence
from protenix.model.generator import (
    InferenceNoiseScheduler,
    sample_diffusion,
    sample_diffusion_training,
    TrainingNoiseSampler,
)
from protenix.model.modules.confidence import ConfidenceHead
from protenix.model.modules.cross_pair_proposal import (
    atom_mask_to_token_mask,
    build_cross_pair_pair_feature_tensors,
    build_cross_pair_target_contact_map,
    build_cross_pair_token_feature_tensor,
    compute_per_head_contact_losses,
    KWayCrossPairProposal,
)
from protenix.model.modules.diffusion import DiffusionModule
from protenix.model.modules.embedders import (
    ConstraintEmbedder,
    InputFeatureEmbedder,
    RelativePositionEncoding,
)
from protenix.model.modules.head import DistogramHead
from protenix.model.modules.pairformer import (
    MSAModule,
    PairformerStack,
    TemplateEmbedder,
)
from protenix.model.modules.primitives import LinearNoBias
from protenix.model.triangular.layers import LayerNorm
from protenix.model.utils import simple_merge_dict_list
from protenix.utils.logger import get_logger
from protenix.utils.permutation.permutation import SymmetricPermutation
from protenix.utils.torch_utils import autocasting_disable_decorator

logger = get_logger(__name__)


def update_input_feature_dict(input_feature_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Lines 1-3 of Algorithm 5 compute d_lm, v_lm, and pad_info utilized in the AtomAttentionEncoder.
    Args:
            input_feature_dict (dict[str, Any]): input features
    Returns:
            input_feature_dict (dict[str, Any]): input features
    """
    from protenix.model.modules.transformer import rearrange_qk_to_dense_trunk

    with torch.no_grad():
        # Prepare tensors in dense trunks for local operations
        q_trunked_list, k_trunked_list, pad_info = rearrange_qk_to_dense_trunk(
            q=[input_feature_dict["ref_pos"], input_feature_dict["ref_space_uid"]],
            k=[input_feature_dict["ref_pos"], input_feature_dict["ref_space_uid"]],
            dim_q=[-2, -1],
            dim_k=[-2, -1],
            n_queries=32,
            n_keys=128,
            compute_mask=True,
        )
        # Compute atom pair feature
        d_lm = (
            q_trunked_list[0][..., None, :] - k_trunked_list[0][..., None, :, :]
        )  # [..., n_blocks, n_queries, n_keys, 3]
        v_lm = (
            q_trunked_list[1][..., None].int() == k_trunked_list[1][..., None, :].int()
        ).unsqueeze(
            dim=-1
        )  # [..., n_blocks, n_queries, n_keys, 1]
        input_feature_dict["d_lm"] = d_lm
        input_feature_dict["v_lm"] = v_lm
        input_feature_dict["pad_info"] = pad_info
        return input_feature_dict


class Protenix(nn.Module):
    """
    Implements Algorithm 1 [Main Inference/Train Loop] in AF3
    """

    def __init__(self, configs: Any) -> None:
        super(Protenix, self).__init__()
        self.configs = configs
        torch.backends.cuda.matmul.allow_tf32 = self.configs.enable_tf32
        # Some constants
        self.enable_diffusion_shared_vars_cache = (
            self.configs.enable_diffusion_shared_vars_cache
        )
        self.enable_efficient_fusion = self.configs.enable_efficient_fusion
        self.N_cycle = self.configs.model.N_cycle
        self.N_model_seed = self.configs.model.N_model_seed
        self.train_confidence_only = configs.train_confidence_only
        if self.train_confidence_only:  # the final finetune stage
            assert configs.loss.weight.alpha_diffusion == 0.0
            assert configs.loss.weight.alpha_distogram == 0.0

        # Diffusion scheduler
        self.train_noise_sampler = TrainingNoiseSampler(**configs.train_noise_sampler)
        self.inference_noise_scheduler = InferenceNoiseScheduler(
            **configs.inference_noise_scheduler
        )
        self.diffusion_batch_size = self.configs.diffusion_batch_size

        # Model
        esm_configs = configs.get("esm", {})  # This is used in InputFeatureEmbedder
        self.input_embedder = InputFeatureEmbedder(
            **configs.model.input_embedder, esm_configs=esm_configs
        )
        self.relative_position_encoding = RelativePositionEncoding(
            **configs.model.relative_position_encoding
        )
        self.template_embedder = TemplateEmbedder(**configs.model.template_embedder)
        self.msa_module = MSAModule(
            **configs.model.msa_module,
            msa_configs=configs.data.get("msa", {}),
        )
        self.constraint_embedder = ConstraintEmbedder(
            **configs.model.constraint_embedder
        )
        self.pairformer_stack = PairformerStack(**configs.model.pairformer)
        self.diffusion_module = DiffusionModule(**configs.model.diffusion_module)
        self.distogram_head = DistogramHead(**configs.model.distogram_head)
        self.confidence_head = ConfidenceHead(**configs.model.confidence_head)
        self.c_s, self.c_z, self.c_s_inputs = (
            configs.c_s,
            configs.c_z,
            configs.c_s_inputs,
        )
        self.cross_pair_proposal_enabled = configs.model.cross_pair_proposal.enable
        self.cross_pair_proposal = (
            KWayCrossPairProposal(c_s=self.c_s, **configs.model.cross_pair_proposal)
            if self.cross_pair_proposal_enabled
            else None
        )

        self.linear_no_bias_sinit = LinearNoBias(
            in_features=self.c_s_inputs, out_features=self.c_s
        )
        self.linear_no_bias_zinit1 = LinearNoBias(
            in_features=self.c_s, out_features=self.c_z
        )
        self.linear_no_bias_zinit2 = LinearNoBias(
            in_features=self.c_s, out_features=self.c_z
        )
        self.linear_no_bias_token_bond = LinearNoBias(
            in_features=1, out_features=self.c_z
        )
        self.linear_no_bias_z_cycle = LinearNoBias(
            in_features=self.c_z, out_features=self.c_z
        )
        self.linear_no_bias_s = LinearNoBias(
            in_features=self.c_s, out_features=self.c_s
        )
        self.layernorm_z_cycle = LayerNorm(self.c_z)
        self.layernorm_s = LayerNorm(self.c_s)

        # Zero init the recycling layer
        nn.init.zeros_(self.linear_no_bias_z_cycle.weight)
        nn.init.zeros_(self.linear_no_bias_s.weight)

    def get_pairformer_output(
        self,
        input_feature_dict: dict[str, Any],
        N_cycle: int,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
        mc_dropout: bool = False,
        mc_dropout_rate: float = 0.4,
    ) -> tuple[torch.Tensor, ...]:
        """
        The forward pass from the input to pairformer output

        Args:
            input_feature_dict (dict[str, Any]): input features
            N_cycle (int): number of cycles
            inplace_safe (bool): Whether it is safe to use inplace operations. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.

        Returns:
            Tuple[torch.Tensor, ...]: s_inputs, s, z
        """
        if self.train_confidence_only:
            self.input_embedder.eval()
            self.template_embedder.eval()
            self.msa_module.eval()
            self.pairformer_stack.eval()

        # Line 1-5
        s_inputs = self.input_embedder(
            input_feature_dict, inplace_safe=False, chunk_size=chunk_size
        )  # [..., N_token, 449]
        z_constraint = None

        if "constraint_feature" in input_feature_dict:
            z_constraint = self.constraint_embedder(
                input_feature_dict["constraint_feature"]
            )

        s_init = self.linear_no_bias_sinit(s_inputs)  # [..., N_token, c_s]
        z_init = (
            self.linear_no_bias_zinit1(s_init)[..., None, :]
            + self.linear_no_bias_zinit2(s_init)[..., None, :, :]
        )  # [..., N_token, N_token, c_z]
        if inplace_safe:
            z_init += self.relative_position_encoding(input_feature_dict["relp"])
            z_init += self.linear_no_bias_token_bond(
                input_feature_dict["token_bonds"].unsqueeze(dim=-1)
            )
            if z_constraint is not None:
                z_init += z_constraint
        else:
            z_init = z_init + self.relative_position_encoding(
                input_feature_dict["relp"]
            )
            z_init = z_init + self.linear_no_bias_token_bond(
                input_feature_dict["token_bonds"].unsqueeze(dim=-1)
            )
            if z_constraint is not None:
                z_init = z_init + z_constraint
        # Line 6
        z = torch.zeros_like(z_init)
        s = torch.zeros_like(s_init)

        # Line 7-13 recycling
        for cycle_no in range(N_cycle):
            with torch.set_grad_enabled(
                self.training
                and (not self.train_confidence_only)
                and cycle_no == (N_cycle - 1)
            ):
                if mc_dropout:
                    z = z_init + F.dropout(
                        self.linear_no_bias_z_cycle(self.layernorm_z_cycle(z)),
                        p=self.configs.mc_dropout_rate,
                    )
                else:
                    z = z_init + self.linear_no_bias_z_cycle(self.layernorm_z_cycle(z))
                if inplace_safe:
                    if self.template_embedder.n_blocks > 0:
                        z += self.template_embedder(
                            input_feature_dict,
                            z,
                            triangle_multiplicative=self.configs.triangle_multiplicative,
                            triangle_attention=self.configs.triangle_attention,
                            inplace_safe=inplace_safe,
                            chunk_size=chunk_size,
                        )
                    z = self.msa_module(
                        input_feature_dict,
                        z,
                        s_inputs,
                        pair_mask=None,
                        triangle_multiplicative=self.configs.triangle_multiplicative,
                        triangle_attention=self.configs.triangle_attention,
                        inplace_safe=inplace_safe,
                        chunk_size=chunk_size,
                    )
                else:
                    if self.template_embedder.n_blocks > 0:
                        z = z + self.template_embedder(
                            input_feature_dict,
                            z,
                            triangle_multiplicative=self.configs.triangle_multiplicative,
                            triangle_attention=self.configs.triangle_attention,
                            inplace_safe=inplace_safe,
                            chunk_size=chunk_size,
                        )
                    z = self.msa_module(
                        input_feature_dict,
                        z,
                        s_inputs,
                        pair_mask=None,
                        triangle_multiplicative=self.configs.triangle_multiplicative,
                        triangle_attention=self.configs.triangle_attention,
                        inplace_safe=inplace_safe,
                        chunk_size=chunk_size,
                    )
                s = s_init + self.linear_no_bias_s(self.layernorm_s(s))
                s, z = self.pairformer_stack(
                    s,
                    z,
                    pair_mask=None,
                    triangle_multiplicative=self.configs.triangle_multiplicative,
                    triangle_attention=self.configs.triangle_attention,
                    inplace_safe=inplace_safe,
                    chunk_size=chunk_size,
                )

        if self.train_confidence_only:
            self.input_embedder.train()
            self.template_embedder.train()
            self.msa_module.train()
            self.pairformer_stack.train()

        return s_inputs, s, z

    def sample_diffusion(self, **kwargs: Any) -> torch.Tensor:
        """
        Samples diffusion process based on the provided configurations.

        Returns:
            torch.Tensor: The result of the diffusion sampling process.
        """
        _configs = {
            key: self.configs.sample_diffusion.get(key)
            for key in [
                "gamma0",
                "gamma_min",
                "noise_scale_lambda",
                "step_scale_eta",
            ]
        }
        _configs.update(
            {
                "attn_chunk_size": (
                    self.configs.infer_setting.chunk_size if not self.training else None
                ),
                "diffusion_chunk_size": (
                    self.configs.infer_setting.sample_diffusion_chunk_size
                    if not self.training
                    else None
                ),
            }
        )
        return autocasting_disable_decorator(self.configs.skip_amp.sample_diffusion)(
            sample_diffusion
        )(**_configs, **kwargs)

    def run_confidence_head(self, *args: Any, **kwargs: Any) -> Any:
        """
        Runs the confidence head with optional automatic mixed precision (AMP) disabled.

        Returns:
            Any: The output of the confidence head.
        """
        return autocasting_disable_decorator(self.configs.skip_amp.confidence_head)(
            self.confidence_head
        )(*args, **kwargs)

    def _prepare_diffusion_cache(
        self,
        input_feature_dict: dict[str, Any],
        z: torch.Tensor,
    ) -> dict[str, Any]:
        cache = dict()
        if self.enable_diffusion_shared_vars_cache:
            cache["pair_z"] = autocasting_disable_decorator(
                self.configs.skip_amp.sample_diffusion
            )(self.diffusion_module.diffusion_conditioning.prepare_cache)(
                input_feature_dict["relp"], z, False
            )
            cache["p_lm/c_l"] = autocasting_disable_decorator(
                self.configs.skip_amp.sample_diffusion
            )(self.diffusion_module.atom_attention_encoder.prepare_cache)(
                ref_pos=input_feature_dict["ref_pos"],
                ref_charge=input_feature_dict["ref_charge"],
                ref_mask=input_feature_dict["ref_mask"],
                ref_element=input_feature_dict["ref_element"],
                ref_atom_name_chars=input_feature_dict["ref_atom_name_chars"],
                atom_to_token_idx=input_feature_dict["atom_to_token_idx"],
                d_lm=input_feature_dict["d_lm"],
                v_lm=input_feature_dict["v_lm"],
                pad_info=input_feature_dict["pad_info"],
                r_l=True,
                z=cache["pair_z"],
                inplace_safe=False,
            )
        else:
            cache["pair_z"] = None
            cache["p_lm/c_l"] = [None, None]
        return cache

    def _inject_cross_pair_transition(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        prot_idx: torch.Tensor,
        rna_idx: torch.Tensor,
        delta_pr: torch.Tensor,
        delta_rp: torch.Tensor,
        delta_s_prot: torch.Tensor,
        delta_s_rna: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        s_mod = s.clone()
        z_mod = z.clone()
        delta_pr = delta_pr.to(dtype=z_mod.dtype, device=z_mod.device)
        delta_rp = delta_rp.to(dtype=z_mod.dtype, device=z_mod.device)
        delta_s_prot = delta_s_prot.to(dtype=s_mod.dtype, device=s_mod.device)
        delta_s_rna = delta_s_rna.to(dtype=s_mod.dtype, device=s_mod.device)
        z_mod[prot_idx[:, None], rna_idx[None, :], :] = (
            z_mod[prot_idx[:, None], rna_idx[None, :], :] + delta_pr
        )
        z_mod[rna_idx[:, None], prot_idx[None, :], :] = (
            z_mod[rna_idx[:, None], prot_idx[None, :], :] + delta_rp.transpose(0, 1)
        )
        s_mod.index_add_(0, prot_idx, delta_s_prot)
        s_mod.index_add_(0, rna_idx, delta_s_rna)
        return s_mod, z_mod

    def _masked_cross_pair_mean(
        self,
        value: torch.Tensor,
        pair_valid_mask: Optional[torch.Tensor],
        eps: float = 1e-6,
    ) -> torch.Tensor:
        value = value.to(torch.float32)
        if pair_valid_mask is None:
            return value.mean()
        mask = pair_valid_mask.to(device=value.device, dtype=torch.float32)
        if mask.dim() == 3:
            mask = mask[0]
        return (value * mask).sum() / (mask.sum() + eps)

    def _soft_contact_jaccard(
        self,
        contact_a: torch.Tensor,
        contact_b: torch.Tensor,
        pair_valid_mask: Optional[torch.Tensor],
        eps: float = 1e-6,
    ) -> torch.Tensor:
        contact_a = contact_a.to(torch.float32)
        contact_b = contact_b.to(torch.float32)
        if pair_valid_mask is None:
            mask = torch.ones_like(contact_a, dtype=torch.float32)
        else:
            mask = pair_valid_mask.to(device=contact_a.device, dtype=torch.float32)
            if mask.dim() == 3:
                mask = mask[0]
        intersection = (torch.minimum(contact_a, contact_b) * mask).sum()
        union = (torch.maximum(contact_a, contact_b) * mask).sum()
        return intersection / (union + eps)

    def _high_prob_region_l1(
        self,
        contact_a: torch.Tensor,
        contact_b: torch.Tensor,
        pair_valid_mask: Optional[torch.Tensor],
        prob_threshold: float = 0.5,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        contact_a = contact_a.to(torch.float32)
        contact_b = contact_b.to(torch.float32)
        if pair_valid_mask is None:
            valid_mask = torch.ones_like(contact_a, dtype=torch.bool)
        else:
            valid_mask = pair_valid_mask.to(device=contact_a.device, dtype=torch.bool)
            if valid_mask.dim() == 3:
                valid_mask = valid_mask[0]

        active_mask = (
            (contact_a >= prob_threshold) | (contact_b >= prob_threshold)
        ) & valid_mask
        if active_mask.sum() == 0:
            return contact_a.new_zeros(())

        diff = torch.abs(contact_a - contact_b)
        active_mask = active_mask.to(dtype=diff.dtype)
        return (diff * active_mask).sum() / (active_mask.sum() + eps)

    def _mean_delta_norm(
        self,
        delta: torch.Tensor,
    ) -> torch.Tensor:
        return delta.to(torch.float32).norm(dim=-1).mean()

    def _decode_cross_pair_head_deltas(
        self,
        proposal_data: dict[str, Any],
        head_indices: int | list[int] | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.cross_pair_proposal.decode_selected_heads(
            proposal_state=proposal_data["state"],
            head_indices=head_indices,
        )

    def _get_cross_pair_proposal_data(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        input_feature_dict: dict[str, Any],
    ) -> Optional[dict[str, Any]]:
        if (not self.cross_pair_proposal_enabled) or (self.cross_pair_proposal is None):
            return None
        atom_to_token_idx = input_feature_dict["atom_to_token_idx"].long()
        n_token = int(input_feature_dict["token_index"].shape[-1])
        prot_token_mask = atom_mask_to_token_mask(
            input_feature_dict["is_protein"].bool(), atom_to_token_idx, n_token
        )
        rna_token_mask = atom_mask_to_token_mask(
            input_feature_dict["is_rna"].bool(), atom_to_token_idx, n_token
        )
        prot_idx = torch.nonzero(prot_token_mask, as_tuple=False).squeeze(-1)
        rna_idx = torch.nonzero(rna_token_mask, as_tuple=False).squeeze(-1)
        if prot_idx.numel() == 0 or rna_idx.numel() == 0:
            return None

        z_pr = z[prot_idx[:, None], rna_idx[None, :], :]
        prot_single = s.index_select(0, prot_idx)
        rna_single = s.index_select(0, rna_idx)
        has_frame = input_feature_dict.get("has_frame")
        prot_has_frame = (
            has_frame.index_select(0, prot_idx) if has_frame is not None else None
        )
        rna_has_frame = (
            has_frame.index_select(0, rna_idx) if has_frame is not None else None
        )
        prot_token_feat = build_cross_pair_token_feature_tensor(
            restype=input_feature_dict["restype"].index_select(0, prot_idx),
            has_frame=prot_has_frame,
            is_protein=True,
            is_rna=False,
        )
        rna_token_feat = build_cross_pair_token_feature_tensor(
            restype=input_feature_dict["restype"].index_select(0, rna_idx),
            has_frame=rna_has_frame,
            is_protein=False,
            is_rna=True,
        )
        bio_pair_feat, geom_pair_feat = build_cross_pair_pair_feature_tensors(
            prot_token_feat=prot_token_feat,
            rna_token_feat=rna_token_feat,
        )
        logits, proposal_state = self.cross_pair_proposal(
            z_pr,
            prot_single=prot_single,
            rna_single=rna_single,
            prot_token_feat=prot_token_feat,
            rna_token_feat=rna_token_feat,
            bio_pair_feat=bio_pair_feat,
            geom_pair_feat=geom_pair_feat,
        )
        probs = torch.sigmoid(logits)
        return {
            "logits": logits,
            "probs": probs,
            "state": proposal_state,
            "prot_idx": prot_idx,
            "rna_idx": rna_idx,
        }

    def _select_training_cross_pair_route(
        self,
        proposal_data: Optional[dict[str, Any]],
        feat_dict: dict[str, Any],
        label_dict: dict[str, Any],
    ) -> tuple[Optional[dict[str, torch.Tensor]], Optional[dict[str, torch.Tensor]]]:
        if proposal_data is None:
            return None, None
        target_dict = build_cross_pair_target_contact_map(
            feat_dict=feat_dict,
            label_dict=label_dict,
            contact_threshold=self.configs.loss.cross_pair_proposal.contact_threshold,
        )
        if target_dict is None:
            return None, None
        per_head_loss = compute_per_head_contact_losses(
            logits=proposal_data["logits"],
            target=target_dict["target"],
            pair_valid_mask=target_dict["pair_valid_mask"],
            dice_weight=self.configs.loss.cross_pair_proposal.dice_weight,
            pos_weight=self.configs.loss.cross_pair_proposal.pos_weight,
            eps=self.configs.loss.cross_pair_proposal.eps,
        )
        best_idx = torch.argmin(per_head_loss)
        return {
            "head_indices": best_idx.view(1),
            "per_head_loss": per_head_loss,
        }, target_dict

    def _run_training_diffusion_branch(
        self,
        input_feature_dict: dict[str, Any],
        label_dict: dict[str, Any],
        s_inputs: torch.Tensor,
        s: torch.Tensor,
        z_branch: torch.Tensor,
        use_conditioning: bool,
        inplace_safe: bool,
        diffusion_batch_size: Optional[int] = None,
        compute_confidence: bool = False,
    ) -> dict[str, Any]:
        branch_pred = {}
        cache = self._prepare_diffusion_cache(input_feature_dict, z_branch)
        n_sample = (
            self.diffusion_batch_size
            if diffusion_batch_size is None
            else diffusion_batch_size
        )
        _, x_denoised, x_noise_level = autocasting_disable_decorator(
            self.configs.skip_amp.sample_diffusion_training
        )(sample_diffusion_training)(
            noise_sampler=self.train_noise_sampler,
            denoise_net=self.diffusion_module,
            label_dict=label_dict,
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s,
            z_trunk=None if cache["pair_z"] is not None else z_branch,
            pair_z=cache["pair_z"],
            p_lm=cache["p_lm/c_l"][0],
            c_l=cache["p_lm/c_l"][1],
            N_sample=n_sample,
            diffusion_chunk_size=self.configs.diffusion_chunk_size,
            use_conditioning=use_conditioning,
            enable_efficient_fusion=self.enable_efficient_fusion,
        )
        branch_pred["coordinate"] = x_denoised
        branch_pred["noise_level"] = x_noise_level
        branch_pred["distogram"] = autocasting_disable_decorator(True)(
            self.distogram_head
        )(z_branch)

        if compute_confidence:
            contact_probs = autocasting_disable_decorator(True)(
                sample_confidence.compute_contact_prob
            )(
                distogram_logits=branch_pred["distogram"],
                **sample_confidence.get_bin_params(self.configs.loss.distogram),
            )
            (
                branch_pred["plddt"],
                branch_pred["pae"],
                branch_pred["pde"],
                branch_pred["resolved"],
            ) = self.run_confidence_head(
                input_feature_dict=input_feature_dict,
                s_inputs=s_inputs,
                s_trunk=s,
                z_trunk=z_branch,
                pair_mask=None,
                x_pred_coords=branch_pred["coordinate"],
                triangle_multiplicative=self.configs.triangle_multiplicative,
                triangle_attention=self.configs.triangle_attention,
                inplace_safe=inplace_safe,
                chunk_size=None,
            )
            summary_confidence, _ = autocasting_disable_decorator(True)(
                sample_confidence.compute_full_data_and_summary
            )(
                configs=self.configs,
                pae_logits=branch_pred["pae"],
                plddt_logits=branch_pred["plddt"],
                pde_logits=branch_pred["pde"],
                contact_probs=contact_probs.unsqueeze(0).expand(
                    branch_pred["coordinate"].shape[0], -1, -1
                ),
                token_asym_id=input_feature_dict["asym_id"],
                token_has_frame=input_feature_dict["has_frame"],
                atom_coordinate=branch_pred["coordinate"],
                atom_to_token_idx=input_feature_dict["atom_to_token_idx"],
                atom_is_polymer=1 - input_feature_dict["is_ligand"],
                N_recycle=self.N_cycle,
                interested_atom_mask=None,
                return_full_data=False,
                mol_id=input_feature_dict["mol_id"],
                elements_one_hot=input_feature_dict["ref_element"],
            )
            branch_pred["summary_confidence_scores"] = (
                sample_confidence.merge_per_sample_confidence_scores(summary_confidence)
            )
        return branch_pred

    def _run_quality_only_diffusion_branch(
        self,
        input_feature_dict: dict[str, Any],
        s_inputs: torch.Tensor,
        s: torch.Tensor,
        z_branch: torch.Tensor,
        inplace_safe: bool,
        diffusion_batch_size: Optional[int] = None,
    ) -> dict[str, Any]:
        branch_pred = {}
        cache = self._prepare_diffusion_cache(input_feature_dict, z_branch)
        n_sample = (
            self.configs.model.cross_pair_proposal.random_branch_diffusion_batch_size
            if diffusion_batch_size is None
            else diffusion_batch_size
        )
        noise_schedule = self.inference_noise_scheduler(
            N_step=self.configs.sample_diffusion["N_step"],
            device=s_inputs.device,
            dtype=s_inputs.dtype,
        )
        branch_pred["coordinate"] = self.sample_diffusion(
            denoise_net=self.diffusion_module,
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s,
            z_trunk=None if cache["pair_z"] is not None else z_branch,
            pair_z=cache["pair_z"],
            p_lm=cache["p_lm/c_l"][0],
            c_l=cache["p_lm/c_l"][1],
            N_sample=n_sample,
            noise_schedule=noise_schedule,
            inplace_safe=inplace_safe,
            enable_efficient_fusion=self.enable_efficient_fusion,
        )
        branch_pred["distogram"] = autocasting_disable_decorator(True)(
            self.distogram_head
        )(z_branch)
        contact_probs = autocasting_disable_decorator(True)(
            sample_confidence.compute_contact_prob
        )(
            distogram_logits=branch_pred["distogram"],
            **sample_confidence.get_bin_params(self.configs.loss.distogram),
        )
        (
            branch_pred["plddt"],
            branch_pred["pae"],
            branch_pred["pde"],
            branch_pred["resolved"],
        ) = self.run_confidence_head(
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s,
            z_trunk=z_branch,
            pair_mask=None,
            x_pred_coords=branch_pred["coordinate"],
            triangle_multiplicative=self.configs.triangle_multiplicative,
            triangle_attention=self.configs.triangle_attention,
            inplace_safe=inplace_safe,
            chunk_size=None,
        )
        summary_confidence, _ = autocasting_disable_decorator(True)(
            sample_confidence.compute_full_data_and_summary
        )(
            configs=self.configs,
            pae_logits=branch_pred["pae"],
            plddt_logits=branch_pred["plddt"],
            pde_logits=branch_pred["pde"],
            contact_probs=contact_probs.unsqueeze(0).expand(
                branch_pred["coordinate"].shape[0], -1, -1
            ),
            token_asym_id=input_feature_dict["asym_id"],
            token_has_frame=input_feature_dict["has_frame"],
            atom_coordinate=branch_pred["coordinate"],
            atom_to_token_idx=input_feature_dict["atom_to_token_idx"],
            atom_is_polymer=1 - input_feature_dict["is_ligand"],
            N_recycle=self.N_cycle,
            interested_atom_mask=None,
            return_full_data=False,
            mol_id=input_feature_dict["mol_id"],
            elements_one_hot=input_feature_dict["ref_element"],
        )
        branch_pred["summary_confidence_scores"] = (
            sample_confidence.merge_per_sample_confidence_scores(summary_confidence)
        )
        return branch_pred

    def _compute_cross_pair_quality_samples(
        self, summary_scores: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        cfg = self.configs.loss.cross_pair_quality
        ranking_weight = getattr(cfg, "ranking_weight", 0.50)
        iptm_weight = getattr(cfg, "iptm_weight", 0.25)
        ptm_weight = getattr(cfg, "ptm_weight", 0.15)
        plddt_weight = getattr(cfg, "plddt_weight", 0.10)
        clash_penalty = getattr(cfg, "clash_penalty", 1.0)

        plddt = summary_scores["plddt"].to(torch.float32) / 100.0
        ptm = summary_scores["ptm"].to(torch.float32)
        iptm = summary_scores["iptm"].to(torch.float32)
        ranking_score = summary_scores["ranking_score"].to(torch.float32)
        has_clash = summary_scores["has_clash"].to(torch.float32)
        return (
            ranking_weight * ranking_score
            + iptm_weight * iptm
            + ptm_weight * ptm
            + plddt_weight * plddt
            - clash_penalty * has_clash
        )

    def _score_cross_pair_heads_for_inference(
        self,
        input_feature_dict: dict[str, Any],
        s_inputs: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        proposal_data: dict[str, Any],
        N_cycle: int,
        inplace_safe: bool,
        chunk_size: Optional[int],
    ) -> torch.Tensor:
        n_head = proposal_data["logits"].shape[0]
        if n_head == 1:
            return proposal_data["logits"].new_zeros((1,))

        with torch.no_grad():
            N_sample_mini_rollout = self.configs.sample_diffusion["N_sample_mini_rollout"]
            N_step_mini_rollout = self.configs.sample_diffusion["N_step_mini_rollout"]
            noise_schedule = self.inference_noise_scheduler(
                N_step=N_step_mini_rollout,
                device=s_inputs.device,
                dtype=s_inputs.dtype,
            )

            head_scores = []
            for head_idx in range(n_head):
                delta_pr, delta_rp, delta_s_prot, delta_s_rna = self._decode_cross_pair_head_deltas(
                    proposal_data=proposal_data,
                    head_indices=head_idx,
                )
                s_k, z_k = self._inject_cross_pair_transition(
                    s=s,
                    z=z,
                    prot_idx=proposal_data["prot_idx"],
                    rna_idx=proposal_data["rna_idx"],
                    delta_pr=delta_pr[0],
                    delta_rp=delta_rp[0],
                    delta_s_prot=delta_s_prot[0],
                    delta_s_rna=delta_s_rna[0],
                )
                cache = self._prepare_diffusion_cache(input_feature_dict, z_k)
                coordinate_mini = self.sample_diffusion(
                    denoise_net=self.diffusion_module,
                    input_feature_dict=input_feature_dict,
                    s_inputs=s_inputs,
                    s_trunk=s_k,
                    z_trunk=None if cache["pair_z"] is not None else z_k,
                    pair_z=None if cache["pair_z"] is None else cache["pair_z"],
                    p_lm=cache["p_lm/c_l"][0],
                    c_l=cache["p_lm/c_l"][1],
                    N_sample=N_sample_mini_rollout,
                    noise_schedule=noise_schedule,
                    inplace_safe=inplace_safe,
                    enable_efficient_fusion=self.enable_efficient_fusion,
                )
                distogram_k = autocasting_disable_decorator(True)(self.distogram_head)(
                    z_k
                )
                contact_probs = autocasting_disable_decorator(True)(
                    sample_confidence.compute_contact_prob
                )(
                    distogram_logits=distogram_k,
                    **sample_confidence.get_bin_params(self.configs.loss.distogram),
                )
                plddt_k, pae_k, pde_k, _ = self.run_confidence_head(
                    input_feature_dict=input_feature_dict,
                    s_inputs=s_inputs,
                    s_trunk=s_k,
                    z_trunk=z_k,
                    pair_mask=None,
                    x_pred_coords=coordinate_mini,
                    triangle_multiplicative=self.configs.triangle_multiplicative,
                    triangle_attention=self.configs.triangle_attention,
                    inplace_safe=inplace_safe,
                    chunk_size=chunk_size,
                )
                summary_confidence, _ = autocasting_disable_decorator(True)(
                    sample_confidence.compute_full_data_and_summary
                )(
                    configs=self.configs,
                    pae_logits=pae_k,
                    plddt_logits=plddt_k,
                    pde_logits=pde_k,
                    contact_probs=contact_probs.unsqueeze(0).expand(
                        coordinate_mini.shape[0], -1, -1
                    ),
                    token_asym_id=input_feature_dict["asym_id"],
                    token_has_frame=input_feature_dict["has_frame"],
                    atom_coordinate=coordinate_mini,
                    atom_to_token_idx=input_feature_dict["atom_to_token_idx"],
                    atom_is_polymer=1 - input_feature_dict["is_ligand"],
                    N_recycle=N_cycle,
                    interested_atom_mask=None,
                    return_full_data=False,
                    mol_id=input_feature_dict["mol_id"],
                    elements_one_hot=input_feature_dict["ref_element"],
                )
                summary_scores = sample_confidence.merge_per_sample_confidence_scores(
                    summary_confidence
                )
                q_samples = self._compute_cross_pair_quality_samples(summary_scores)
                head_scores.append(q_samples.mean())

            return torch.stack(head_scores, dim=0)

    def _run_inference_for_z(
        self,
        input_feature_dict: dict[str, Any],
        s_inputs: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        N_cycle: int,
        mode: str,
        n_sample: int,
        step_st: float,
        step_trunk: float,
        inplace_safe: bool,
        chunk_size: Optional[int],
        diversity_sampler: Any,
    ) -> tuple[dict[str, Any], dict[str, float]]:
        pred_dict = {}
        time_tracker = {}
        noise_schedule = self.inference_noise_scheduler(
            N_step=self.configs.sample_diffusion["N_step"],
            device=s_inputs.device,
            dtype=s_inputs.dtype,
        )
        cache = self._prepare_diffusion_cache(input_feature_dict, z)
        pred_dict["coordinate"] = self.sample_diffusion(
            denoise_net=self.diffusion_module,
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s,
            z_trunk=None if cache["pair_z"] is not None else z,
            pair_z=cache["pair_z"],
            p_lm=cache["p_lm/c_l"][0],
            c_l=cache["p_lm/c_l"][1],
            N_sample=n_sample,
            noise_schedule=noise_schedule,
            inplace_safe=inplace_safe,
            enable_efficient_fusion=self.enable_efficient_fusion,
            diversity_sampler=diversity_sampler,
        )
        step_diffusion = time.time()
        time_tracker.update({"diffusion": step_diffusion - step_trunk})
        contact_probs = autocasting_disable_decorator(True)(
            sample_confidence.compute_contact_prob
        )(
            distogram_logits=self.distogram_head(z),
            **sample_confidence.get_bin_params(self.configs.loss.distogram),
        )
        pred_dict["contact_probs"] = contact_probs
        pred_dict["per_sample_contact_probs"] = contact_probs.unsqueeze(0).expand(
            n_sample, -1, -1
        )
        (
            pred_dict["plddt"],
            pred_dict["pae"],
            pred_dict["pde"],
            pred_dict["resolved"],
        ) = self.run_confidence_head(
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s,
            z_trunk=z,
            pair_mask=None,
            x_pred_coords=pred_dict["coordinate"],
            triangle_multiplicative=self.configs.triangle_multiplicative,
            triangle_attention=self.configs.triangle_attention,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )
        step_confidence = time.time()
        time_tracker.update({"confidence": step_confidence - step_diffusion})
        time_tracker.update({"model_forward": time.time() - step_st})
        if mode == "inference":
            interested_atom_mask = None
            mol_id = None
            elements_one_hot = None
        else:
            interested_atom_mask = None
            mol_id = input_feature_dict["mol_id"]
            elements_one_hot = input_feature_dict["ref_element"]
        (
            pred_dict["summary_confidence"],
            pred_dict["full_data"],
        ) = autocasting_disable_decorator(True)(
            sample_confidence.compute_full_data_and_summary
        )(
            configs=self.configs,
            pae_logits=pred_dict["pae"],
            plddt_logits=pred_dict["plddt"],
            pde_logits=pred_dict["pde"],
            contact_probs=pred_dict["per_sample_contact_probs"],
            token_asym_id=input_feature_dict["asym_id"],
            token_has_frame=input_feature_dict["has_frame"],
            atom_coordinate=pred_dict["coordinate"],
            atom_to_token_idx=input_feature_dict["atom_to_token_idx"],
            atom_is_polymer=1 - input_feature_dict["is_ligand"],
            N_recycle=N_cycle,
            interested_atom_mask=interested_atom_mask,
            return_full_data=True,
            mol_id=mol_id,
            elements_one_hot=elements_one_hot,
        )
        return pred_dict, time_tracker

    def main_inference_loop(
        self,
        input_feature_dict: dict[str, Any],
        label_dict: dict[str, Any],
        N_cycle: int,
        mode: str,
        inplace_safe: bool = True,
        chunk_size: Optional[int] = 4,
        N_model_seed: int = 1,
        symmetric_permutation: SymmetricPermutation = None,
        mc_dropout_apply_rate: float = 0.4,
        diversity_sampler = None
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        """
        Main inference loop (multiple model seeds) for the Alphafold3 model.

        Args:
            input_feature_dict (dict[str, Any]): Input features dictionary.
            label_dict (dict[str, Any]): Label dictionary.
            N_cycle (int): Number of cycles.
            mode (str): Mode of operation (e.g., 'inference').
            inplace_safe (bool): Whether to use inplace operations safely. Defaults to True.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to 4.
            N_model_seed (int): Number of model seeds. Defaults to 1.
            symmetric_permutation (SymmetricPermutation): Symmetric permutation object. Defaults to None.
            mc_dropout_apply_rate (float): Only for inference mode

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]: Prediction, log, and time dictionaries.
        """
        # For backward compatibility, if N_model_seed > 1, process multiple seeds here
        # But in evaluation mode, this should be handled externally
        if N_model_seed > 1 and mode in ["inference"]:
            pred_dicts = []
            log_dicts = []
            time_trackers = []
            for _ in range(N_model_seed):
                pred_dict, log_dict, time_tracker = self._main_inference_loop(
                    input_feature_dict=(
                        copy.deepcopy(input_feature_dict)
                        if (N_model_seed > 1 and mode == "inference")
                        else input_feature_dict
                    ),  # the input_feature_dict is modified when mode is "inference"
                    label_dict=label_dict,
                    N_cycle=N_cycle,
                    mode=mode,
                    inplace_safe=inplace_safe,
                    chunk_size=chunk_size,
                    symmetric_permutation=symmetric_permutation,
                    mc_dropout=random.random() < mc_dropout_apply_rate,
                    diversity_sampler=diversity_sampler
                )
                pred_dicts.append(pred_dict)
                log_dicts.append(log_dict)
                time_trackers.append(time_tracker)

            # Combine outputs of multiple models
            def _cat(dict_list, key):
                return torch.cat([x[key] for x in dict_list], dim=0)

            def _list_join(dict_list, key):
                return sum([x[key] for x in dict_list], [])

            all_pred_dict = {
                "coordinate": _cat(pred_dicts, "coordinate"),
                "summary_confidence": _list_join(pred_dicts, "summary_confidence"),
                "full_data": _list_join(pred_dicts, "full_data"),
                "plddt": _cat(pred_dicts, "plddt"),
                "pae": _cat(pred_dicts, "pae"),
                "pde": _cat(pred_dicts, "pde"),
                "resolved": _cat(pred_dicts, "resolved"),
            }

            all_log_dict = simple_merge_dict_list(log_dicts)
            all_time_dict = simple_merge_dict_list(time_trackers)
            return all_pred_dict, all_log_dict, all_time_dict
        else:
            # Single seed inference - delegate to _main_inference_loop
            return self._main_inference_loop(
                input_feature_dict=input_feature_dict,
                label_dict=label_dict,
                N_cycle=N_cycle,
                mode=mode,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
                symmetric_permutation=symmetric_permutation,
                mc_dropout=random.random() < mc_dropout_apply_rate,
                diversity_sampler=diversity_sampler
            )

    def _get_dynamic_chunk_size(self, N_token: int) -> Optional[int]:
        """
        Get dynamic chunk_size based on token count

        Args:
            N_token (int): Number of tokens

        Returns:
            Optional[int]: Optimal chunk_size for the given token count
        """
        if not hasattr(self.configs.infer_setting, "chunk_size_thresholds"):
            return self.configs.infer_setting.chunk_size

        thresholds = self.configs.infer_setting.chunk_size_thresholds

        # Convert string keys to integers and sort in ascending order
        threshold_pairs = [(int(k), v) for k, v in thresholds.items()]
        sorted_thresholds = sorted(threshold_pairs, key=lambda x: x[0])

        # Find the appropriate chunk_size for the given token count
        for threshold, chunk_size in sorted_thresholds:
            if N_token <= threshold:
                return None if chunk_size == -1 else chunk_size

        # For token counts larger than the largest threshold, use smallest chunk_size
        return 32  # extreme case for very large proteins

    def _main_inference_loop(
        self,
        input_feature_dict: dict[str, Any],
        label_dict: dict[str, Any],
        N_cycle: int,
        mode: str,
        inplace_safe: bool = True,
        chunk_size: Optional[int] = 4,
        symmetric_permutation: SymmetricPermutation = None,
        mc_dropout: bool = False,
        diversity_sampler = None
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        """
        Main inference loop (single model seed) for the Alphafold3 model.
        mc_dropout: do not use by default

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]: Prediction, log, and time dictionaries.
        """
        step_st = time.time()
        N_token = input_feature_dict["residue_index"].shape[-1]

        # Apply dynamic chunk_size if enabled (otherwise keep the passed chunk_size)
        if (
            hasattr(self.configs.infer_setting, "dynamic_chunk_size")
            and self.configs.infer_setting.dynamic_chunk_size
        ):
            chunk_size = self._get_dynamic_chunk_size(N_token)
        # If dynamic chunking is disabled, chunk_size keeps its original value from the function parameter

        log_dict = {}
        pred_dict = {}
        time_tracker = {}

        s_inputs, s, z = self.get_pairformer_output(
            input_feature_dict=input_feature_dict,
            N_cycle=N_cycle,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
            mc_dropout=mc_dropout,
        )

        keys_to_delete = []
        for key in input_feature_dict.keys():
            if "template_" in key or key in [
                "msa",
                "has_deletion",
                "deletion_value",
                "profile",
                "deletion_mean",
                # "token_bonds",
            ]:
                keys_to_delete.append(key)

        for key in keys_to_delete:
            del input_feature_dict[key]
        step_trunk = time.time()
        time_tracker.update({"pairformer": step_trunk - step_st})
        N_sample = self.configs.sample_diffusion["N_sample"]
        if mode != "inference":
            pred_dict, branch_time_tracker = self._run_inference_for_z(
                input_feature_dict=input_feature_dict,
                s_inputs=s_inputs,
                s=s,
                z=z,
                N_cycle=N_cycle,
                mode=mode,
                n_sample=N_sample,
                step_st=step_st,
                step_trunk=step_trunk,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
                diversity_sampler=diversity_sampler,
            )
            time_tracker.update(branch_time_tracker)

            if label_dict is not None and symmetric_permutation is not None:
                perm_start = time.time()
                pred_dict, log_dict = symmetric_permutation.permute_inference_pred_dict(
                    input_feature_dict=input_feature_dict,
                    pred_dict=pred_dict,
                    label_dict=label_dict,
                    permute_by_pocket=("pocket_mask" in label_dict)
                    and ("interested_ligand_mask" in label_dict),
                )
                time_tracker.update({"permutation": time.time() - perm_start})

            if label_dict is None:
                interested_atom_mask = None
            else:
                interested_atom_mask = label_dict.get("interested_ligand_mask", None)
            (
                pred_dict["summary_confidence"],
                pred_dict["full_data"],
            ) = autocasting_disable_decorator(True)(
                sample_confidence.compute_full_data_and_summary
            )(
                configs=self.configs,
                pae_logits=pred_dict["pae"],
                plddt_logits=pred_dict["plddt"],
                pde_logits=pred_dict["pde"],
                contact_probs=pred_dict.get(
                    "per_sample_contact_probs", pred_dict["contact_probs"]
                ),
                token_asym_id=input_feature_dict["asym_id"],
                token_has_frame=input_feature_dict["has_frame"],
                atom_coordinate=pred_dict["coordinate"],
                atom_to_token_idx=input_feature_dict["atom_to_token_idx"],
                atom_is_polymer=1 - input_feature_dict["is_ligand"],
                N_recycle=N_cycle,
                interested_atom_mask=interested_atom_mask,
                return_full_data=True,
                mol_id=input_feature_dict["mol_id"],
                elements_one_hot=input_feature_dict["ref_element"],
            )
            return pred_dict, log_dict, time_tracker
        proposal_data = self._get_cross_pair_proposal_data(s, z, input_feature_dict)
        if proposal_data is None:
            pred_dict, branch_time_tracker = self._run_inference_for_z(
                input_feature_dict=input_feature_dict,
                s_inputs=s_inputs,
                s=s,
                z=z,
                N_cycle=N_cycle,
                mode=mode,
                n_sample=N_sample,
                step_st=step_st,
                step_trunk=step_trunk,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
                diversity_sampler=diversity_sampler,
            )
            time_tracker.update(branch_time_tracker)
        else:
            proposal_scores = self._score_cross_pair_heads_for_inference(
                input_feature_dict=input_feature_dict,
                s_inputs=s_inputs,
                s=s,
                z=z,
                proposal_data=proposal_data,
                N_cycle=N_cycle,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
            )
            n_head = proposal_scores.size(0)
            num_active_heads = min(n_head, N_sample)
            top_ids = torch.argsort(proposal_scores, descending=True)[:num_active_heads]
            top_delta_pr, top_delta_rp, top_delta_s_prot, top_delta_s_rna = self._decode_cross_pair_head_deltas(
                proposal_data=proposal_data,
                head_indices=top_ids,
            )
            base_samples = N_sample // num_active_heads
            residual = N_sample % num_active_heads

            merged_pred = {}
            merged_time = {}
            all_proposal_ids = []
            pred_chunks = []
            for rank, head_idx in enumerate(top_ids.tolist()):
                cur_n_sample = base_samples + int(rank < residual)
                if cur_n_sample == 0:
                    continue
                s_k, z_k = self._inject_cross_pair_transition(
                    s=s,
                    z=z,
                    prot_idx=proposal_data["prot_idx"],
                    rna_idx=proposal_data["rna_idx"],
                    delta_pr=top_delta_pr[rank],
                    delta_rp=top_delta_rp[rank],
                    delta_s_prot=top_delta_s_prot[rank],
                    delta_s_rna=top_delta_s_rna[rank],
                )
                pred_k, time_k = self._run_inference_for_z(
                    input_feature_dict=input_feature_dict,
                    s_inputs=s_inputs,
                    s=s_k,
                    z=z_k,
                    N_cycle=N_cycle,
                    mode=mode,
                    n_sample=cur_n_sample,
                    step_st=step_st,
                    step_trunk=step_trunk,
                    inplace_safe=inplace_safe,
                    chunk_size=chunk_size,
                    diversity_sampler=diversity_sampler,
                )
                pred_chunks.append(pred_k)
                all_proposal_ids.extend([head_idx] * cur_n_sample)
                merged_time = time_k

            pred_dict["coordinate"] = torch.cat(
                [chunk["coordinate"] for chunk in pred_chunks], dim=0
            )
            pred_dict["plddt"] = torch.cat([chunk["plddt"] for chunk in pred_chunks], dim=0)
            pred_dict["pae"] = torch.cat([chunk["pae"] for chunk in pred_chunks], dim=0)
            pred_dict["pde"] = torch.cat([chunk["pde"] for chunk in pred_chunks], dim=0)
            pred_dict["resolved"] = torch.cat(
                [chunk["resolved"] for chunk in pred_chunks], dim=0
            )
            pred_dict["per_sample_contact_probs"] = torch.cat(
                [chunk["per_sample_contact_probs"] for chunk in pred_chunks], dim=0
            )
            pred_dict["contact_probs"] = pred_chunks[0]["contact_probs"]
            pred_dict["summary_confidence"] = sum(
                [chunk["summary_confidence"] for chunk in pred_chunks], []
            )
            pred_dict["full_data"] = sum([chunk["full_data"] for chunk in pred_chunks], [])
            pred_dict["cross_pair_head_ids"] = torch.tensor(
                all_proposal_ids, device=z.device, dtype=torch.long
            )
            pred_dict["cross_pair_logits"] = proposal_data["logits"]
            pred_dict["cross_pair_head_quality_scores"] = proposal_scores
            time_tracker.update(merged_time)

        # Permutation: when label is given, permute coordinates and other heads
        if label_dict is not None and symmetric_permutation is not None:
            pred_dict, log_dict = symmetric_permutation.permute_inference_pred_dict(
                input_feature_dict=input_feature_dict,
                pred_dict=pred_dict,
                label_dict=label_dict,
                permute_by_pocket=("pocket_mask" in label_dict)
                and ("interested_ligand_mask" in label_dict),
            )
            last_step_seconds = step_confidence
            time_tracker.update({"permutation": time.time() - last_step_seconds})

        return pred_dict, log_dict, time_tracker

    def main_train_loop(
        self,
        input_feature_dict: dict[str, Any],
        label_full_dict: dict[str, Any],
        label_dict: dict[str, Any],
        N_cycle: int,
        symmetric_permutation: SymmetricPermutation,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        """
        Main training loop for the Alphafold3 model.

        Args:
            input_feature_dict (dict[str, Any]): Input features dictionary.
            label_full_dict (dict[str, Any]): Full label dictionary (uncropped).
            label_dict (dict): Label dictionary (cropped).
            N_cycle (int): Number of cycles.
            symmetric_permutation (SymmetricPermutation): Symmetric permutation object.
            inplace_safe (bool): Whether to use inplace operations safely. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
                Prediction, updated label, and log dictionaries.
        """
        z_base = None
        s_inputs, s, z = self.get_pairformer_output(
            input_feature_dict=input_feature_dict,
            N_cycle=N_cycle,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )

        log_dict = {}
        pred_dict = {}

        proposal_data = self._get_cross_pair_proposal_data(s, z, input_feature_dict)
        pred_dict["cross_pair_logits"] = (
            proposal_data["logits"] if proposal_data is not None else None
        )
        selected_route, target_contact_dict = self._select_training_cross_pair_route(
            proposal_data=proposal_data,
            feat_dict=input_feature_dict,
            label_dict=label_dict,
        )
        primary_head_idx = None
        selected_contact_probs = None
        pair_valid_mask = None
        if selected_route is not None:
            route_head_indices = selected_route["head_indices"]
            primary_head_idx = int(route_head_indices[0].item())
            selected_delta_pr, selected_delta_rp, selected_delta_s_prot, selected_delta_s_rna = self._decode_cross_pair_head_deltas(
                proposal_data=proposal_data,
                head_indices=route_head_indices,
            )
            s_base, z_base = s, z
            s, z = self._inject_cross_pair_transition(
                s=s,
                z=z,
                prot_idx=proposal_data["prot_idx"],
                rna_idx=proposal_data["rna_idx"],
                delta_pr=selected_delta_pr[0],
                delta_rp=selected_delta_rp[0],
                delta_s_prot=selected_delta_s_prot[0],
                delta_s_rna=selected_delta_s_rna[0],
            )
            selected_contact_probs = torch.sigmoid(
                proposal_data["logits"][primary_head_idx]
            )
            pair_valid_mask = target_contact_dict["pair_valid_mask"]
            pred_dict["cross_pair_selected_idx"] = route_head_indices[0]
            pred_dict["cross_pair_target"] = target_contact_dict["target"]
            pred_dict["cross_pair_valid_mask"] = pair_valid_mask
            per_head_loss = selected_route["per_head_loss"].detach()
            sorted_per_head_loss, _ = torch.sort(per_head_loss)
            second_best_loss = (
                sorted_per_head_loss[1]
                if sorted_per_head_loss.numel() > 1
                else sorted_per_head_loss[0]
            )
            log_dict.update(
                {
                    "cross_pair_selected_idx": route_head_indices[0].detach(),
                    "cross_pair_selected_loss": per_head_loss[primary_head_idx],
                    "cross_pair_second_best_loss": second_best_loss,
                    "cross_pair_selection_margin": (
                        second_best_loss - per_head_loss[primary_head_idx]
                    ),
                    "cross_pair_target_density": self._masked_cross_pair_mean(
                        target_contact_dict["target"], pair_valid_mask
                    ),
                    "cross_pair_selected_density": self._masked_cross_pair_mean(
                        selected_contact_probs, pair_valid_mask
                    ),
                    "cross_pair_selected_target_jaccard": self._soft_contact_jaccard(
                        selected_contact_probs,
                        target_contact_dict["target"],
                        pair_valid_mask,
                    ),
                    "cross_pair_selected_delta_pr_norm": self._mean_delta_norm(
                        selected_delta_pr[0]
                    ),
                    "cross_pair_selected_delta_rp_norm": self._mean_delta_norm(
                        selected_delta_rp[0]
                    ),
                    "cross_pair_selected_delta_s_prot_norm": self._mean_delta_norm(
                        selected_delta_s_prot[0]
                    ),
                    "cross_pair_selected_delta_s_rna_norm": self._mean_delta_norm(
                        selected_delta_s_rna[0]
                    ),
                }
            )

        cache = self._prepare_diffusion_cache(input_feature_dict, z)
        # Mini-rollout: used for confidence and label permutation
        with torch.no_grad():
            # [..., 1, N_atom, 3]
            N_sample_mini_rollout = self.configs.sample_diffusion[
                "N_sample_mini_rollout"
            ]  # =1
            N_step_mini_rollout = self.configs.sample_diffusion["N_step_mini_rollout"]
            self.diffusion_module.eval()  # use eval mode for mini-rollout
            coordinate_mini = self.sample_diffusion(
                denoise_net=self.diffusion_module,
                input_feature_dict=input_feature_dict,
                s_inputs=s_inputs.detach(),
                s_trunk=s.detach(),
                z_trunk=None if cache["pair_z"] is not None else z.detach(),
                pair_z=None if cache["pair_z"] is None else cache["pair_z"].detach(),
                p_lm=(
                    None
                    if cache["p_lm/c_l"][0] is None
                    else cache["p_lm/c_l"][0].detach()
                ),
                c_l=(
                    None
                    if cache["p_lm/c_l"][1] is None
                    else cache["p_lm/c_l"][1].detach()
                ),
                N_sample=N_sample_mini_rollout,
                noise_schedule=self.inference_noise_scheduler(
                    N_step=N_step_mini_rollout,
                    device=s_inputs.device,
                    dtype=s_inputs.dtype,
                ),
                enable_efficient_fusion=self.enable_efficient_fusion,
            )
            self.diffusion_module.train()
            coordinate_mini.detach_()
            pred_dict["coordinate_mini"] = coordinate_mini

            # Permute ground truth to match mini-rollout prediction
            (
                label_dict,
                perm_log_dict,
            ) = symmetric_permutation.permute_label_to_match_mini_rollout(
                coordinate_mini,
                input_feature_dict,
                label_dict,
                label_full_dict,
            )
            log_dict.update(perm_log_dict)

        # Confidence: use mini-rollout prediction, and detach token embeddings
        drop_embedding = (
            random.random() < self.configs.model.confidence_embedding_drop_rate
        )
        plddt_pred, pae_pred, pde_pred, resolved_pred = self.run_confidence_head(
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s,
            z_trunk=z,
            pair_mask=None,
            x_pred_coords=coordinate_mini,
            use_embedding=not drop_embedding,
            triangle_multiplicative=self.configs.triangle_multiplicative,
            triangle_attention=self.configs.triangle_attention,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )
        pred_dict.update(
            {
                "plddt": plddt_pred,
                "pae": pae_pred,
                "pde": pde_pred,
                "resolved": resolved_pred,
            }
        )

        if self.train_confidence_only:
            # Skip diffusion loss and distogram loss. Return now.
            return pred_dict, label_dict, log_dict

        # Denoising: use permuted coords to generate noisy samples and perform denoising
        # x_denoised: [..., N_sample, N_atom, 3]
        # x_noise_level: [..., N_sample]
        N_sample = self.diffusion_batch_size
        drop_conditioning = (
            random.random() < self.configs.model.condition_embedding_drop_rate
        )
        _, x_denoised, x_noise_level = autocasting_disable_decorator(
            self.configs.skip_amp.sample_diffusion_training
        )(sample_diffusion_training)(
            noise_sampler=self.train_noise_sampler,
            denoise_net=self.diffusion_module,
            label_dict=label_dict,
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s,
            z_trunk=None if cache["pair_z"] is not None else z,
            pair_z=cache["pair_z"],
            p_lm=cache["p_lm/c_l"][0],
            c_l=cache["p_lm/c_l"][1],
            N_sample=N_sample,
            diffusion_chunk_size=self.configs.diffusion_chunk_size,
            use_conditioning=not drop_conditioning,
            enable_efficient_fusion=self.enable_efficient_fusion,
        )
        pred_dict.update(
            {
                "distogram": autocasting_disable_decorator(True)(self.distogram_head)(
                    z
                ),
                # [..., N_sample=48, N_atom, 3]: diffusion loss
                "coordinate": x_denoised,
                "noise_level": x_noise_level,
            }
        )

        quality_aux_enabled = (
            self.configs.loss.weight.alpha_cross_pair_quality > 0.0
            and self.configs.loss.cross_pair_quality.enable
        )
        charge_aux_enabled = (
            self.configs.loss.weight.alpha_cross_pair_charge > 0.0
            and self.configs.loss.cross_pair_charge.enable
        )
        patch_smoothness_aux_enabled = (
            self.configs.loss.weight.alpha_cross_pair_patch_smoothness > 0.0
            and self.configs.loss.cross_pair_patch_smoothness.enable
        )
        random_head_aux_enabled = (
            quality_aux_enabled
            or charge_aux_enabled
            or patch_smoothness_aux_enabled
        )
        if (
            proposal_data is not None
            and proposal_data["logits"].shape[0] > 1
            and random_head_aux_enabled
        ):
            candidate_heads = [
                idx
                for idx in range(proposal_data["logits"].shape[0])
                if idx != primary_head_idx
            ]
            if len(candidate_heads) == 0:
                candidate_heads = list(range(proposal_data["logits"].shape[0]))
            random_head_idx = random.choice(candidate_heads)
            pred_dict.update(
                {
                    "cross_pair_random_head_idx": torch.tensor(
                        random_head_idx, device=z.device, dtype=torch.long
                    )
                }
            )
            random_contact_probs = torch.sigmoid(proposal_data["logits"][random_head_idx])
            log_dict.update(
                {
                    "cross_pair_random_head_idx": pred_dict[
                        "cross_pair_random_head_idx"
                    ].detach(),
                    "cross_pair_random_density": self._masked_cross_pair_mean(
                        random_contact_probs, pair_valid_mask
                    ),
                }
            )
            if target_contact_dict is not None:
                log_dict["cross_pair_random_target_jaccard"] = (
                    self._soft_contact_jaccard(
                        random_contact_probs,
                        target_contact_dict["target"],
                        pair_valid_mask,
                    )
                )
            if selected_contact_probs is not None:
                log_dict["cross_pair_selected_random_jaccard"] = (
                    self._soft_contact_jaccard(
                        selected_contact_probs,
                        random_contact_probs,
                        pair_valid_mask,
                    )
                )
                log_dict["cross_pair_selected_random_l1"] = (
                    self._high_prob_region_l1(
                        selected_contact_probs,
                        random_contact_probs,
                        pair_valid_mask,
                    )
                )
            if quality_aux_enabled and z_base is not None:
                random_delta_pr, random_delta_rp, random_delta_s_prot, random_delta_s_rna = self._decode_cross_pair_head_deltas(
                    proposal_data=proposal_data,
                    head_indices=random_head_idx,
                )
                log_dict.update(
                    {
                        "cross_pair_random_delta_pr_norm": self._mean_delta_norm(
                            random_delta_pr[0]
                        ),
                        "cross_pair_random_delta_rp_norm": self._mean_delta_norm(
                            random_delta_rp[0]
                        ),
                        "cross_pair_random_delta_s_prot_norm": self._mean_delta_norm(
                            random_delta_s_prot[0]
                        ),
                        "cross_pair_random_delta_s_rna_norm": self._mean_delta_norm(
                            random_delta_s_rna[0]
                        ),
                    }
                )
                s_random, z_random = self._inject_cross_pair_transition(
                    s=s_base,
                    z=z_base,
                    prot_idx=proposal_data["prot_idx"],
                    rna_idx=proposal_data["rna_idx"],
                    delta_pr=random_delta_pr[0],
                    delta_rp=random_delta_rp[0],
                    delta_s_prot=random_delta_s_prot[0],
                    delta_s_rna=random_delta_s_rna[0],
                )
                random_branch_pred = self._run_quality_only_diffusion_branch(
                    input_feature_dict=input_feature_dict,
                    s_inputs=s_inputs,
                    s=s_random,
                    z_branch=z_random,
                    inplace_safe=inplace_safe,
                    diffusion_batch_size=self.configs.model.cross_pair_proposal.random_branch_diffusion_batch_size,
                )
                pred_dict["summary_confidence_scores_random_head"] = random_branch_pred[
                    "summary_confidence_scores"
                ]

        # Permute symmetric atom/chain in each sample to match true structure
        # Note: currently chains cannot be permuted since label is cropped
        (
            pred_dict,
            perm_log_dict,
            _,
            _,
        ) = symmetric_permutation.permute_diffusion_sample_to_match_label(
            input_feature_dict, pred_dict, label_dict, stage="train"
        )
        log_dict.update(perm_log_dict)
        log_dict.update({"noise_level": x_noise_level})

        return pred_dict, label_dict, log_dict

    def forward(
        self,
        input_feature_dict: dict[str, Any],
        label_full_dict: dict[str, Any],
        label_dict: dict[str, Any],
        mode: str = "inference",
        current_step: Optional[int] = None,
        symmetric_permutation: SymmetricPermutation = None,
        disable_inplace: bool = False,
        mc_dropout_apply_rate: float = 0.4,
        diversity_sampler = None
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        """
        Forward pass of the Alphafold3 model.

        Args:
            input_feature_dict (dict[str, Any]): Input features dictionary.
            label_full_dict (dict[str, Any]): Full label dictionary (uncropped).
            label_dict (dict[str, Any]): Label dictionary (cropped).
            mode (str): Mode of operation ('train', 'inference', 'eval'). Defaults to 'inference'.
            current_step (Optional[int]): Current training step. Defaults to None.
            symmetric_permutation (SymmetricPermutation): Symmetric permutation object. Defaults to None.

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
                Prediction, updated label, and log dictionaries.
        """

        assert mode in ["train", "eval", "inference"]
        not_use_gradient = not (self.training or torch.is_grad_enabled())
        inplace_safe = not_use_gradient and (not disable_inplace)

        input_feature_dict = self.relative_position_encoding.generate_relp(
            input_feature_dict
        )
        input_feature_dict = update_input_feature_dict(input_feature_dict)

        if mode == "train":
            nc_rng = np.random.RandomState(current_step)
            N_cycle = nc_rng.randint(1, self.N_cycle + 1)
            assert self.training
            assert label_dict is not None
            assert symmetric_permutation is not None

            pred_dict, label_dict, log_dict = self.main_train_loop(
                input_feature_dict=input_feature_dict,
                label_full_dict=label_full_dict,
                label_dict=label_dict,
                N_cycle=N_cycle,
                symmetric_permutation=symmetric_permutation,
                inplace_safe=inplace_safe,
                chunk_size=None,
            )
            log_dict["N_cycle"] = N_cycle
        elif mode == "inference":
            pred_dict, log_dict, time_tracker = self.main_inference_loop(
                input_feature_dict=input_feature_dict,
                label_dict=None,
                N_cycle=self.N_cycle,
                mode=mode,
                inplace_safe=inplace_safe,
                chunk_size=self.configs.infer_setting.chunk_size,
                N_model_seed=self.N_model_seed,
                symmetric_permutation=None,
                mc_dropout_apply_rate=mc_dropout_apply_rate,
                diversity_sampler = diversity_sampler
            )
            log_dict.update({"time": time_tracker})
        elif mode == "eval":
            if label_dict is not None:
                assert (
                    label_dict["coordinate"].size()
                    == label_full_dict["coordinate"].size()
                )
                label_dict.update(label_full_dict)

            pred_dict, log_dict, time_tracker = self.main_inference_loop(
                input_feature_dict=input_feature_dict,
                label_dict=label_dict,
                N_cycle=self.N_cycle,
                mode=mode,
                inplace_safe=inplace_safe,
                chunk_size=self.configs.infer_setting.chunk_size,
                N_model_seed=1,
                symmetric_permutation=symmetric_permutation,
                mc_dropout_apply_rate=mc_dropout_apply_rate,
            )
            log_dict.update({"time": time_tracker})

        return pred_dict, label_dict, log_dict
