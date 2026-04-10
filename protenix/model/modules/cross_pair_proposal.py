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

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def atom_mask_to_token_mask(
    atom_mask: torch.Tensor,
    atom_to_token_idx: torch.Tensor,
    n_token: int,
) -> torch.Tensor:
    """Aggregate an atom-level mask into token space."""
    token_mask = torch.zeros(
        n_token, device=atom_mask.device, dtype=torch.bool
    )
    active_tokens = atom_to_token_idx[atom_mask.bool()].long()
    if active_tokens.numel() > 0:
        token_mask[active_tokens.unique()] = True
    return token_mask


def build_token_coordinate_tensor(
    coordinate: torch.Tensor,
    coordinate_mask: torch.Tensor,
    atom_to_token_idx: torch.Tensor,
    rep_atom_mask: torch.Tensor,
    n_token: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build one representative coordinate per token using the existing distogram
    representative atom mask.
    """
    valid_rep_mask = rep_atom_mask.bool() & coordinate_mask.bool()
    token_coord = coordinate.new_zeros((n_token, 3))
    token_valid = torch.zeros(n_token, device=coordinate.device, dtype=torch.bool)
    if valid_rep_mask.any():
        rep_token_idx = atom_to_token_idx[valid_rep_mask].long()
        token_coord[rep_token_idx] = coordinate[valid_rep_mask]
        token_valid[rep_token_idx] = True
    return token_coord, token_valid


def build_cross_pair_target_contact_map(
    feat_dict: dict,
    label_dict: dict,
    contact_threshold: float,
) -> Optional[dict[str, torch.Tensor]]:
    """Construct a protein-token x RNA-token target contact map."""
    atom_to_token_idx = feat_dict["atom_to_token_idx"].long()
    n_token = int(feat_dict["token_index"].shape[-1])
    prot_token_mask = atom_mask_to_token_mask(
        feat_dict["is_protein"].bool(), atom_to_token_idx, n_token
    )
    rna_token_mask = atom_mask_to_token_mask(
        feat_dict["is_rna"].bool(), atom_to_token_idx, n_token
    )
    prot_idx = torch.nonzero(prot_token_mask, as_tuple=False).squeeze(-1)
    rna_idx = torch.nonzero(rna_token_mask, as_tuple=False).squeeze(-1)
    if prot_idx.numel() == 0 or rna_idx.numel() == 0:
        return None

    token_coord, token_valid = build_token_coordinate_tensor(
        coordinate=label_dict["coordinate"],
        coordinate_mask=label_dict["coordinate_mask"],
        atom_to_token_idx=atom_to_token_idx,
        rep_atom_mask=feat_dict["distogram_rep_atom_mask"],
        n_token=n_token,
    )
    prot_valid = token_valid[prot_idx]
    rna_valid = token_valid[rna_idx]
    pair_valid = prot_valid[:, None] & rna_valid[None, :]
    if pair_valid.sum() == 0:
        return None

    prot_coord = token_coord[prot_idx].float()
    rna_coord = token_coord[rna_idx].float()
    pair_distance = torch.cdist(prot_coord, rna_coord)
    target = (pair_distance <= contact_threshold).to(pair_distance.dtype)
    target = target * pair_valid.to(target.dtype)
    return {
        "target": target,
        "pair_valid_mask": pair_valid,
        "prot_idx": prot_idx,
        "rna_idx": rna_idx,
    }


def compute_per_head_contact_losses(
    logits: torch.Tensor,
    target: torch.Tensor,
    pair_valid_mask: Optional[torch.Tensor] = None,
    dice_weight: float = 1.0,
    pos_weight: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Per-head proposal loss used for head selection and coverage loss."""
    target = target.to(logits.dtype)
    if target.dim() == 2:
        target = target.unsqueeze(0).expand_as(logits)
    if pair_valid_mask is None:
        pair_valid_mask = torch.ones_like(target, dtype=logits.dtype)
    else:
        pair_valid_mask = pair_valid_mask.to(logits.dtype)
        if pair_valid_mask.dim() == 2:
            pair_valid_mask = pair_valid_mask.unsqueeze(0).expand_as(logits)

    pos_weight_tensor = logits.new_tensor(pos_weight)
    bce = F.binary_cross_entropy_with_logits(
        logits, target, reduction="none", pos_weight=pos_weight_tensor
    )
    bce = (bce * pair_valid_mask).sum(dim=(-1, -2)) / (
        pair_valid_mask.sum(dim=(-1, -2)) + eps
    )

    probs = torch.sigmoid(logits)
    intersection = (probs * target * pair_valid_mask).sum(dim=(-1, -2))
    denom = ((probs + target) * pair_valid_mask).sum(dim=(-1, -2))
    dice = 1.0 - ((2.0 * intersection + eps) / (denom + eps))
    return bce + dice_weight * dice


def compute_diversity_margin_loss(
    logits: torch.Tensor,
    margin: float,
    confidence_margin: float = 0.25,
    ambiguity_weight: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Penalize near-duplicate proposals and strongly penalize inactive proposals
    whose contact probabilities collapse toward zero.
    """
    n_head = logits.shape[0]
    if n_head <= 1:
        return logits.new_zeros(())

    probs = torch.sigmoid(logits).float()
    flat = probs.reshape(n_head, -1)
    flat_norm = flat / (flat.norm(dim=-1, keepdim=True) + eps)
    pairwise = torch.cdist(flat_norm, flat_norm, p=2)
    upper = torch.triu_indices(n_head, n_head, offset=1, device=probs.device)
    pairwise = pairwise[upper[0], upper[1]]
    duplicate_loss = torch.relu(margin - pairwise).mean()

    # Sparse contact maps should not be penalized just because the global mean
    # probability is small. Instead, require that each head has at least a small
    # set of confident contact candidates. If even the top-scoring entries are
    # near zero, the proposal has effectively collapsed to the all-negative mode.
    topk = min(max(flat.shape[-1] // 100, 1), 64)
    topk_mean_prob = flat.topk(k=topk, dim=-1).values.mean(dim=-1)
    normalized_activity = topk_mean_prob / max(confidence_margin, eps)
    ambiguity_loss = -torch.log(normalized_activity.clamp_min(eps))
    ambiguity_loss = torch.relu(ambiguity_loss).mean()

    return duplicate_loss + ambiguity_weight * ambiguity_loss


class KWayCrossPairProposal(nn.Module):
    """
    Generate K alternative protein-RNA cross-pair proposals from the trunk pair
    features. Each head predicts both a contact map and a feature delta that can
    be injected back into the cross-pair block.
    """

    def __init__(
        self,
        c_z: int = 128,
        hidden_dim: int = 128,
        num_heads: int = 4,
        slot_attn_heads: int = 4,
        delta_scale: float = 1.0,
        gate_floor: float = 0.0,
        random_branch_diffusion_batch_size: int = 8,
        contact_threshold: float = 8.0,
        enable: bool = False,
    ) -> None:
        super(KWayCrossPairProposal, self).__init__()
        self.c_z = c_z
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.slot_attn_heads = slot_attn_heads
        self.delta_scale = delta_scale
        self.gate_floor = gate_floor
        self.random_branch_diffusion_batch_size = random_branch_diffusion_batch_size
        self.contact_threshold = contact_threshold
        self.enable = enable

        self.input_ln = nn.LayerNorm(c_z)
        # v2.1 residual context encoder:
        # keep a direct projection path from z_pr while the MLP learns a task-
        # specific refinement. This avoids the proposal branch shrinking the
        # signal into an almost-silent representation.
        self.res_proj = (
            nn.Identity()
            if hidden_dim == c_z
            else nn.Linear(c_z, hidden_dim)
        )
        self.shared = nn.Sequential(
            nn.Linear(c_z, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.context_ln = nn.LayerNorm(hidden_dim)
        self.context_scale = nn.Parameter(torch.tensor(1.0))

        # v2.2 slot-style input-conditioned proposals:
        # learned base slots + an input-dependent query initializer, followed by
        # cross-attention over the cross-pair context map.
        self.head_slots = nn.Parameter(torch.randn(num_heads, hidden_dim))
        self.query_ln = nn.LayerNorm(hidden_dim)
        self.query_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_heads * hidden_dim),
        )
        self.slot_query_ln = nn.LayerNorm(hidden_dim)
        self.slot_kv_ln = nn.LayerNorm(hidden_dim)
        self.slot_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=slot_attn_heads,
            batch_first=True,
        )
        self.slot_post_ln = nn.LayerNorm(hidden_dim)
        self.slot_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.contact_head = nn.Linear(hidden_dim, 1)
        self.delta_head_pr = nn.Linear(hidden_dim, c_z)
        self.delta_head_rp = nn.Linear(hidden_dim, c_z)

    def forward(
        self, z_pr: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z_pr: [N_protein_token, N_rna_token, c_z]

        Returns:
            contact_logits: [K, N_protein_token, N_rna_token]
            delta_pr: [K, N_protein_token, N_rna_token, c_z]
            delta_rp: [K, N_protein_token, N_rna_token, c_z]
        """
        x = self.input_ln(z_pr)
        h = self.res_proj(z_pr) + self.context_scale * self.shared(x)
        h = self.context_ln(h)
        # Match the injected residual to the local pair-feature scale instead of
        # adding a unit-norm vector whose per-channel amplitude is only
        # O(1 / sqrt(c_z)). This keeps proposal deltas small but actually visible
        # to the pretrained pair representation.
        local_z_rms = z_pr.float().pow(2).mean(dim=-1, keepdim=True).sqrt()
        local_z_rms = local_z_rms.to(dtype=z_pr.dtype)
        delta_feature_scale = math.sqrt(self.c_z) * local_z_rms

        pooled = h.mean(dim=(0, 1))
        dynamic_queries = self.query_mlp(self.query_ln(pooled)).view(
            self.num_heads, self.hidden_dim
        )
        slot_init = self.head_slots + dynamic_queries

        kv = h.reshape(1, -1, self.hidden_dim)
        slot_input = slot_init.unsqueeze(0)
        slot_attn_out, _ = self.slot_attn(
            query=self.slot_query_ln(slot_input),
            key=self.slot_kv_ln(kv),
            value=self.slot_kv_ln(kv),
            need_weights=False,
        )
        slots = slot_input + slot_attn_out
        slots = slots + self.slot_mlp(self.slot_post_ln(slots))
        head_queries = slots.squeeze(0)
        logits, deltas_pr, deltas_rp = [], [], []
        for head_idx in range(self.num_heads):
            h_k = h + head_queries[head_idx].view(1, 1, -1)
            contact_logits = self.contact_head(h_k).squeeze(-1)
            contact_probs = torch.sigmoid(contact_logits)
            delta_gate = (
                self.gate_floor + (1.0 - self.gate_floor) * contact_probs
            ).unsqueeze(-1)
            delta_pr_raw = self.delta_head_pr(h_k)
            delta_rp_raw = self.delta_head_rp(h_k)
            delta_pr_unit = F.normalize(delta_pr_raw, p=2, dim=-1, eps=1e-6)
            delta_rp_unit = F.normalize(delta_rp_raw, p=2, dim=-1, eps=1e-6)
            deltas_pr.append(
                delta_gate * self.delta_scale * delta_feature_scale * delta_pr_unit
            )
            deltas_rp.append(
                delta_gate * self.delta_scale * delta_feature_scale * delta_rp_unit
            )
            logits.append(contact_logits)
        return (
            torch.stack(logits, dim=0),
            torch.stack(deltas_pr, dim=0),
            torch.stack(deltas_rp, dim=0),
        )
