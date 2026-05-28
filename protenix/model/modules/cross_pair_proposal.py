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

from protenix.data.constants import (
    DNA_STD_RESIDUES,
    PRO_STD_RESIDUES,
    RNA_STD_RESIDUES,
    STD_RESIDUES_WITH_GAP,
)
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


def build_cross_pair_distance_target_map(
    feat_dict: dict,
    label_dict: dict,
    contact_threshold: float,
    contact_temperature: float = 1.5,
) -> Optional[dict[str, torch.Tensor]]:
    """Construct a soft token-pair target aligned to the full proposal grid."""
    atom_to_token_idx = feat_dict['atom_to_token_idx'].long()
    n_token = int(feat_dict['token_index'].shape[-1])

    prot_token_mask = atom_mask_to_token_mask(
        feat_dict['is_protein'].bool(), atom_to_token_idx, n_token
    )
    rna_token_mask = atom_mask_to_token_mask(
        feat_dict['is_rna'].bool(), atom_to_token_idx, n_token
    )
    prot_idx = torch.nonzero(prot_token_mask, as_tuple=False).squeeze(-1)
    rna_idx = torch.nonzero(rna_token_mask, as_tuple=False).squeeze(-1)
    if prot_idx.numel() == 0 or rna_idx.numel() == 0:
        return None

    coord_mask = label_dict['coordinate_mask'].bool()
    prot_atom_mask = feat_dict['is_protein'].bool() & coord_mask
    rna_atom_mask = feat_dict['is_rna'].bool() & coord_mask
    prot_atom_idx = torch.nonzero(prot_atom_mask, as_tuple=False).squeeze(-1)
    rna_atom_idx = torch.nonzero(rna_atom_mask, as_tuple=False).squeeze(-1)

    pair_shape = (prot_idx.shape[0], rna_idx.shape[0])
    min_distance = label_dict['coordinate'].new_full(pair_shape, float('inf')).float()
    pair_valid = torch.zeros(pair_shape, device=prot_idx.device, dtype=torch.bool)

    if prot_atom_idx.numel() > 0 and rna_atom_idx.numel() > 0:
        prot_atom_coord = label_dict['coordinate'].index_select(0, prot_atom_idx).float()
        rna_atom_coord = label_dict['coordinate'].index_select(0, rna_atom_idx).float()
        atom_pair_distance = torch.cdist(prot_atom_coord, rna_atom_coord)

        prot_pair_token_idx = atom_to_token_idx.index_select(0, prot_atom_idx)
        rna_pair_token_idx = atom_to_token_idx.index_select(0, rna_atom_idx)
        n_rna = rna_idx.shape[0]
        prot_token_to_local = torch.full((n_token,), -1, dtype=torch.long, device=prot_idx.device)
        rna_token_to_local = torch.full((n_token,), -1, dtype=torch.long, device=rna_idx.device)
        prot_token_to_local[prot_idx] = torch.arange(prot_idx.shape[0], device=prot_idx.device)
        rna_token_to_local[rna_idx] = torch.arange(rna_idx.shape[0], device=rna_idx.device)

        local_prot_token = prot_token_to_local[prot_pair_token_idx]
        local_rna_token = rna_token_to_local[rna_pair_token_idx]
        flat_pair_idx = (local_prot_token[:, None] * n_rna + local_rna_token[None, :]).reshape(-1)
        flat_distance = atom_pair_distance.reshape(-1)

        min_distance_flat = flat_distance.new_full((prot_idx.shape[0] * n_rna,), float('inf'))
        min_distance_flat.scatter_reduce_(
            0, flat_pair_idx, flat_distance, reduce='amin', include_self=True
        )
        min_distance = min_distance_flat.view(*pair_shape)
        pair_valid = torch.isfinite(min_distance)

    soft_target = torch.sigmoid((contact_threshold - min_distance) / contact_temperature)
    soft_target = torch.where(pair_valid, soft_target, torch.zeros_like(soft_target))
    if pair_valid.sum() == 0:
        return None
    return {
        'target': soft_target,
        'distance_target': min_distance,
        'pair_valid_mask': pair_valid,
        'prot_idx': prot_idx,
        'rna_idx': rna_idx,
    }


def build_cross_pair_target_contact_map(
    feat_dict: dict,
    label_dict: dict,
    contact_threshold: float,
) -> Optional[dict[str, torch.Tensor]]:
    """Construct a protein-token x RNA-token soft contact target map."""
    return build_cross_pair_distance_target_map(
        feat_dict=feat_dict,
        label_dict=label_dict,
        contact_threshold=contact_threshold,
    )


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
    contact_probs: torch.Tensor,
    margin: float,
    pair_valid_mask: Optional[torch.Tensor] = None,
    confidence_margin: float = 0.25,
    ambiguity_weight: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Penalize near-duplicate proposals and strongly penalize inactive proposals
    whose contact probabilities collapse toward zero.
    """
    n_head = contact_probs.shape[0]
    if n_head <= 1:
        return contact_probs.new_zeros(())

    probs = contact_probs.float()
    if pair_valid_mask is None:
        valid_flat = torch.ones(
            probs.shape[-2] * probs.shape[-1],
            device=probs.device,
            dtype=torch.bool,
        )
    else:
        if pair_valid_mask.dim() == 3:
            pair_valid_mask = pair_valid_mask[0]
        valid_flat = pair_valid_mask.reshape(-1).bool()
    if valid_flat.sum() == 0:
        return probs.new_zeros(())

    flat = probs.reshape(n_head, -1)[:, valid_flat]
    flat_norm = flat / (flat.norm(dim=-1, keepdim=True) + eps)
    pairwise = torch.cdist(flat_norm, flat_norm, p=2)
    upper = torch.triu_indices(n_head, n_head, offset=1, device=flat.device)
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


def build_regular_simplex_slots(
    num_heads: int,
    hidden_dim: int,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Build approximately uniform head prototypes by embedding the vertices of a
    regular simplex into ``hidden_dim`` dimensions and L2-normalizing them.
    """
    if num_heads <= 0 or hidden_dim <= 0:
        raise ValueError("num_heads and hidden_dim must be positive")
    if num_heads == 1:
        return torch.ones((1, hidden_dim), device=device, dtype=dtype)

    base = torch.eye(num_heads, dtype=torch.float32)
    base = base - base.mean(dim=0, keepdim=True)
    u, s, _ = torch.linalg.svd(base, full_matrices=False)
    rank = min(hidden_dim, num_heads - 1)
    coords = u[:, :rank] * s[:rank]
    if hidden_dim > rank:
        coords = F.pad(coords, (0, hidden_dim - rank))
    coords = F.normalize(coords, p=2, dim=-1, eps=1e-6)
    return coords.to(device=device, dtype=dtype)


_RESTYPE_DIM = len(STD_RESIDUES_WITH_GAP)
_TOKEN_FEATURE_LAYOUT = {
    "is_protein": _RESTYPE_DIM + 0,
    "is_rna": _RESTYPE_DIM + 1,
    "has_frame": _RESTYPE_DIM + 2,
    "positive_charge": _RESTYPE_DIM + 3,
    "negative_charge": _RESTYPE_DIM + 4,
    "aromatic": _RESTYPE_DIM + 5,
    "nucleic_base": _RESTYPE_DIM + 6,
    "purine": _RESTYPE_DIM + 7,
    "pyrimidine": _RESTYPE_DIM + 8,
}
_TOKEN_FEATURE_DIM = _RESTYPE_DIM + len(_TOKEN_FEATURE_LAYOUT)
_BIO_PAIR_FEATURE_DIM = 6
_GEOMETRY_PAIR_FEATURE_DIM = 4


def build_cross_pair_token_feature_tensor(
    restype: torch.Tensor,
    has_frame: Optional[torch.Tensor] = None,
    *,
    is_protein: bool,
    is_rna: bool,
) -> torch.Tensor:
    """
    Build a compact token descriptor for cross-pair content/geometry heads.
    """
    if restype.shape[-1] != _RESTYPE_DIM:
        raise ValueError(
            f"Expected restype feature dim {_RESTYPE_DIM}, got {restype.shape[-1]}"
        )

    restype = restype.to(torch.float32)
    token_feat = restype.new_zeros((restype.shape[0], _TOKEN_FEATURE_DIM))
    token_feat[:, :_RESTYPE_DIM] = restype

    if has_frame is not None:
        token_feat[:, _TOKEN_FEATURE_LAYOUT["has_frame"]] = has_frame.to(
            dtype=restype.dtype
        ).reshape(-1)

    if is_protein:
        token_feat[:, _TOKEN_FEATURE_LAYOUT["is_protein"]] = 1.0
    if is_rna:
        token_feat[:, _TOKEN_FEATURE_LAYOUT["is_rna"]] = 1.0

    positive = (
        restype[:, PRO_STD_RESIDUES["ARG"]]
        + restype[:, PRO_STD_RESIDUES["LYS"]]
        + 0.5 * restype[:, PRO_STD_RESIDUES["HIS"]]
    )
    negative = restype[:, PRO_STD_RESIDUES["ASP"]] + restype[:, PRO_STD_RESIDUES["GLU"]]
    aromatic = (
        restype[:, PRO_STD_RESIDUES["PHE"]]
        + restype[:, PRO_STD_RESIDUES["TRP"]]
        + restype[:, PRO_STD_RESIDUES["TYR"]]
        + 0.5 * restype[:, PRO_STD_RESIDUES["HIS"]]
    )
    nucleic_base = (
        restype[:, RNA_STD_RESIDUES["A"]]
        + restype[:, RNA_STD_RESIDUES["G"]]
        + restype[:, RNA_STD_RESIDUES["C"]]
        + restype[:, RNA_STD_RESIDUES["U"]]
        + restype[:, DNA_STD_RESIDUES["DA"]]
        + restype[:, DNA_STD_RESIDUES["DG"]]
        + restype[:, DNA_STD_RESIDUES["DC"]]
        + restype[:, DNA_STD_RESIDUES["DT"]]
    )
    purine = (
        restype[:, RNA_STD_RESIDUES["A"]]
        + restype[:, RNA_STD_RESIDUES["G"]]
        + restype[:, DNA_STD_RESIDUES["DA"]]
        + restype[:, DNA_STD_RESIDUES["DG"]]
    )
    pyrimidine = (
        restype[:, RNA_STD_RESIDUES["C"]]
        + restype[:, RNA_STD_RESIDUES["U"]]
        + restype[:, DNA_STD_RESIDUES["DC"]]
        + restype[:, DNA_STD_RESIDUES["DT"]]
    )

    token_feat[:, _TOKEN_FEATURE_LAYOUT["positive_charge"]] = positive
    token_feat[:, _TOKEN_FEATURE_LAYOUT["negative_charge"]] = negative
    token_feat[:, _TOKEN_FEATURE_LAYOUT["aromatic"]] = aromatic
    token_feat[:, _TOKEN_FEATURE_LAYOUT["nucleic_base"]] = nucleic_base
    token_feat[:, _TOKEN_FEATURE_LAYOUT["purine"]] = purine
    token_feat[:, _TOKEN_FEATURE_LAYOUT["pyrimidine"]] = pyrimidine
    return token_feat


def build_cross_pair_pair_feature_tensors(
    prot_token_feat: torch.Tensor,
    rna_token_feat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build lightweight pairwise biological and geometry proxy features."""
    prot_positive = prot_token_feat[:, _TOKEN_FEATURE_LAYOUT["positive_charge"]]
    prot_negative = prot_token_feat[:, _TOKEN_FEATURE_LAYOUT["negative_charge"]]
    prot_aromatic = prot_token_feat[:, _TOKEN_FEATURE_LAYOUT["aromatic"]]
    prot_has_frame = prot_token_feat[:, _TOKEN_FEATURE_LAYOUT["has_frame"]]

    rna_base = rna_token_feat[:, _TOKEN_FEATURE_LAYOUT["nucleic_base"]]
    rna_purine = rna_token_feat[:, _TOKEN_FEATURE_LAYOUT["purine"]]
    rna_pyrimidine = rna_token_feat[:, _TOKEN_FEATURE_LAYOUT["pyrimidine"]]
    rna_has_frame = rna_token_feat[:, _TOKEN_FEATURE_LAYOUT["has_frame"]]

    prot_positive_pair = prot_positive[:, None].expand(-1, rna_token_feat.shape[0])
    prot_negative_pair = prot_negative[:, None].expand(-1, rna_token_feat.shape[0])
    prot_aromatic_pair = prot_aromatic[:, None].expand(-1, rna_token_feat.shape[0])
    prot_frame_pair = prot_has_frame[:, None].expand(-1, rna_token_feat.shape[0])
    rna_base_pair = rna_base[None, :].expand(prot_token_feat.shape[0], -1)
    rna_purine_pair = rna_purine[None, :].expand(prot_token_feat.shape[0], -1)
    rna_pyrimidine_pair = rna_pyrimidine[None, :].expand(
        prot_token_feat.shape[0], -1
    )
    rna_frame_pair = rna_has_frame[None, :].expand(prot_token_feat.shape[0], -1)

    frame_pair = prot_frame_pair * rna_frame_pair
    bio_pair_feat = torch.stack(
        [
            prot_positive_pair * rna_base_pair,
            prot_negative_pair * rna_base_pair,
            prot_aromatic_pair * rna_base_pair,
            prot_aromatic_pair * rna_purine_pair,
            prot_aromatic_pair * rna_pyrimidine_pair,
            frame_pair,
        ],
        dim=-1,
    )
    geometry_pair_feat = torch.stack(
        [
            prot_frame_pair,
            rna_frame_pair,
            frame_pair,
            torch.abs(prot_frame_pair - rna_frame_pair),
        ],
        dim=-1,
    )
    return bio_pair_feat, geometry_pair_feat


class KWayCrossPairProposal(nn.Module):
    """
    Generate K alternative protein-RNA cross-pair proposals from the trunk pair
    features. Each head predicts both a contact map and a feature delta that can
    be injected back into the cross-pair block.
    """

    def __init__(
        self,
        c_s: int = 384,
        c_z: int = 128,
        hidden_dim: int = 128,
        num_heads: int = 4,
        slot_attn_heads: int = 4,
        delta_scale: float = 1.0,
        single_delta_scale: float = 1.0,
        delta_clip_norm: float = 4.0,
        training_routing_top_k: int = 4,
        training_routing_temperature: float = 0.25,
        head_slot_init: str = "random",
        head_slot_init_scale: float = 1.0,
        token_feature_dim: int = _TOKEN_FEATURE_DIM,
        bio_pair_feature_dim: int = _BIO_PAIR_FEATURE_DIM,
        geometry_pair_feature_dim: int = _GEOMETRY_PAIR_FEATURE_DIM,
        token_hidden_dim: Optional[int] = None,
        use_content_branch: bool = True,
        use_geometry_branch: bool = True,
        use_pair_bio_feat: bool = True,
        use_pair_geom_feat: bool = True,
        content_logit_scale: float = 1.0,
        bio_pair_feat_scale: float = 1.0,
        geom_pair_feat_scale: float = 1.0,
        num_content_slot_bank: int = 4,
        content_prior_hidden_dim: Optional[int] = None,
        content_prior_bias_scale: float = 1.0,
        content_delta_scale: float = 1.0,
        geometry_delta_scale: float = 1.0,
        freeze_geometry_for_head0: bool = False,
        gate_floor: float = 0.0,
        random_branch_diffusion_batch_size: int = 8,
        contact_threshold: float = 8.0,
        enable: bool = False,
    ) -> None:
        super(KWayCrossPairProposal, self).__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.slot_attn_heads = slot_attn_heads
        self.delta_scale = delta_scale
        self.single_delta_scale = single_delta_scale
        self.delta_clip_norm = delta_clip_norm
        self.training_routing_top_k = training_routing_top_k
        self.training_routing_temperature = training_routing_temperature
        self.head_slot_init = head_slot_init
        self.head_slot_init_scale = head_slot_init_scale
        self.token_feature_dim = token_feature_dim
        self.bio_pair_feature_dim = bio_pair_feature_dim
        self.geometry_pair_feature_dim = geometry_pair_feature_dim
        self.token_hidden_dim = (
            max(hidden_dim // 2, 32) if token_hidden_dim is None else token_hidden_dim
        )
        self.num_content_slot_bank = num_content_slot_bank
        self.content_prior_hidden_dim = (
            hidden_dim
            if content_prior_hidden_dim is None
            else content_prior_hidden_dim
        )
        self.use_content_branch = use_content_branch
        self.use_geometry_branch = use_geometry_branch
        self.use_pair_bio_feat = use_pair_bio_feat
        self.use_pair_geom_feat = use_pair_geom_feat
        self.content_logit_scale = content_logit_scale
        self.bio_pair_feat_scale = bio_pair_feat_scale
        self.geom_pair_feat_scale = geom_pair_feat_scale
        self.content_prior_bias_scale = content_prior_bias_scale
        self.content_delta_scale = content_delta_scale
        self.geometry_delta_scale = geometry_delta_scale
        self.freeze_geometry_for_head0 = freeze_geometry_for_head0
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
        self.head_slots = nn.Parameter(torch.empty(num_heads, hidden_dim))
        self.head_content_mix = nn.Parameter(
            torch.empty(num_heads, num_content_slot_bank)
        )
        self.content_prior_input_ln = nn.LayerNorm(hidden_dim + 1)
        self.content_prior_bank = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim + 1, self.content_prior_hidden_dim),
                    nn.GELU(),
                    nn.Linear(self.content_prior_hidden_dim, 1),
                )
                for _ in range(num_content_slot_bank)
            ]
        )
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
        self.prot_content_tower = nn.Sequential(
            nn.Linear(token_feature_dim, self.token_hidden_dim),
            nn.GELU(),
            nn.Linear(self.token_hidden_dim, self.token_hidden_dim),
        )
        self.rna_content_tower = nn.Sequential(
            nn.Linear(token_feature_dim, self.token_hidden_dim),
            nn.GELU(),
            nn.Linear(self.token_hidden_dim, self.token_hidden_dim),
        )
        content_input_dim = (
            hidden_dim + 4 * self.token_hidden_dim + bio_pair_feature_dim
        )
        self.content_pair_ln = nn.LayerNorm(content_input_dim)
        self.content_pair_mlp = nn.Sequential(
            nn.Linear(content_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.content_out_ln = nn.LayerNorm(hidden_dim)
        self.content_head_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.content_head_ln = nn.LayerNorm(hidden_dim)
        self.geometry_seed_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.geometry_seed_ln = nn.LayerNorm(hidden_dim)
        self.geometry_pair_proj = nn.Linear(geometry_pair_feature_dim, hidden_dim)
        self.geometry_pair_ln = nn.LayerNorm(hidden_dim)
        self.geometry_gate = nn.Linear(2 * hidden_dim, hidden_dim)
        self.geometry_out_ln = nn.LayerNorm(hidden_dim)
        self.geometry_head_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.geometry_head_ln = nn.LayerNorm(hidden_dim)
        self.prot_single_proj = nn.Linear(c_s, hidden_dim, bias=False)
        self.rna_single_proj = nn.Linear(c_s, hidden_dim, bias=False)
        self.single_pair_pool_ln = nn.LayerNorm(hidden_dim)
        self.single_head_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.single_head_ln = nn.LayerNorm(hidden_dim)
        self.prot_single_delta_proj = nn.Linear(hidden_dim, c_s, bias=False)
        self.prot_single_delta_ln = nn.LayerNorm(c_s)
        self.rna_single_delta_proj = nn.Linear(hidden_dim, c_s, bias=False)
        self.rna_single_delta_ln = nn.LayerNorm(c_s)
        self.single_delta_head_proj = nn.Linear(hidden_dim, c_s, bias=False)
        self.single_delta_head_ln = nn.LayerNorm(c_s)
        self.content_delta_pair_pr_proj = nn.Linear(hidden_dim, c_z, bias=False)
        self.content_delta_pair_pr_ln = nn.LayerNorm(c_z)
        self.content_delta_head_pr_proj = nn.Linear(hidden_dim, c_z, bias=False)
        self.content_delta_head_pr_ln = nn.LayerNorm(c_z)
        self.content_delta_pair_rp_proj = nn.Linear(hidden_dim, c_z, bias=False)
        self.content_delta_pair_rp_ln = nn.LayerNorm(c_z)
        self.content_delta_head_rp_proj = nn.Linear(hidden_dim, c_z, bias=False)
        self.content_delta_head_rp_ln = nn.LayerNorm(c_z)
        self.geometry_delta_pair_pr_proj = nn.Linear(hidden_dim, c_z, bias=False)
        self.geometry_delta_pair_pr_ln = nn.LayerNorm(c_z)
        self.geometry_delta_head_pr_proj = nn.Linear(hidden_dim, c_z, bias=False)
        self.geometry_delta_head_pr_ln = nn.LayerNorm(c_z)
        self.geometry_delta_pair_rp_proj = nn.Linear(hidden_dim, c_z, bias=False)
        self.geometry_delta_pair_rp_ln = nn.LayerNorm(c_z)
        self.geometry_delta_head_rp_proj = nn.Linear(hidden_dim, c_z, bias=False)
        self.geometry_delta_head_rp_ln = nn.LayerNorm(c_z)
        self.delta_pair_pr_proj = nn.Linear(hidden_dim, c_z, bias=False)
        self.delta_pair_pr_ln = nn.LayerNorm(c_z)
        self.delta_head_pr_proj = nn.Linear(hidden_dim, c_z, bias=False)
        self.delta_head_pr_ln = nn.LayerNorm(c_z)
        self.delta_pair_rp_proj = nn.Linear(hidden_dim, c_z, bias=False)
        self.delta_pair_rp_ln = nn.LayerNorm(c_z)
        self.delta_head_rp_proj = nn.Linear(hidden_dim, c_z, bias=False)
        self.delta_head_rp_ln = nn.LayerNorm(c_z)
        self.delta_head_pr = nn.Linear(hidden_dim, c_z)
        self.delta_head_rp = nn.Linear(hidden_dim, c_z)
        self._reset_head_slots()

    def _build_slot_vectors(
        self,
        num_vectors: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        mode = self.head_slot_init.lower()
        scale = float(self.head_slot_init_scale)
        if num_vectors <= 0:
            raise ValueError("slot bank sizes must be positive")
        if mode == "random":
            slots = torch.randn(num_vectors, self.hidden_dim, device=device, dtype=dtype)
            return scale * slots

        if mode == "orthogonal":
            if num_vectors > self.hidden_dim:
                raise ValueError(
                    "orthogonal head slot init requires slot bank size <= hidden_dim"
                )
            basis = torch.randn(
                self.hidden_dim,
                num_vectors,
                device=device,
                dtype=torch.float32,
            )
            slots, _ = torch.linalg.qr(basis, mode="reduced")
            slots = slots.transpose(0, 1)
        elif mode == "simplex":
            slots = build_regular_simplex_slots(
                num_heads=num_vectors,
                hidden_dim=self.hidden_dim,
                device=device,
                dtype=torch.float32,
            )
        else:
            raise ValueError(
                f"Unsupported head_slot_init: {self.head_slot_init}. "
                "Expected one of {'random', 'orthogonal', 'simplex'}."
            )

        return scale * slots.to(device=device, dtype=dtype)

    def _reset_head_slots(self) -> None:
        with torch.no_grad():
            self.head_slots.copy_(
                self._build_slot_vectors(
                    self.num_heads,
                    device=self.head_slots.device,
                    dtype=self.head_slots.dtype,
                )
            )
            self.head_content_mix.fill_(-2.0)
            content_assign = torch.arange(
                self.num_heads, device=self.head_content_mix.device
            ) % self.num_content_slot_bank
            self.head_content_mix.scatter_(1, content_assign[:, None], 2.0)

    def _compute_pair_local_head_shift(
        self,
        pair_feature: torch.Tensor,
        head_feature: torch.Tensor,
    ) -> torch.Tensor:
        return torch.einsum("prc,kc->kpr", pair_feature, head_feature) / math.sqrt(
            self.hidden_dim
        )

    def _compute_pair_local_delta_shift(
        self,
        pair_delta_bias: torch.Tensor,
        head_delta_bias: torch.Tensor,
    ) -> torch.Tensor:
        return pair_delta_bias.unsqueeze(0) * head_delta_bias[:, None, None, :]

    def _apply_delta_clip(self, delta: torch.Tensor, clip_norm: float, eps: float = 1e-6) -> torch.Tensor:
        if clip_norm <= 0:
            return delta
        delta_norm = delta.norm(dim=-1, keepdim=True)
        scale = torch.clamp(clip_norm / (delta_norm + eps), max=1.0)
        return delta * scale

    def _build_single_transition_feature(
        self,
        pair_feature: torch.Tensor,
        single_feature: Optional[torch.Tensor],
        single_proj: nn.Module,
        reduce_dim: int,
    ) -> torch.Tensor:
        pooled_pair = pair_feature.mean(dim=reduce_dim)
        if single_feature is None:
            single_term = pooled_pair.new_zeros(pooled_pair.shape)
        else:
            single_term = single_proj(single_feature.to(dtype=pooled_pair.dtype))
        return self.single_pair_pool_ln(pooled_pair + single_term)


    def _compute_content_prior_bias_bank(
        self,
        pair_content: torch.Tensor,
        base_contact_logits: torch.Tensor,
    ) -> torch.Tensor:
        prior_input = torch.cat(
            [pair_content, base_contact_logits.unsqueeze(-1)], dim=-1
        )
        prior_input = self.content_prior_input_ln(prior_input)
        prior_bias_bank = []
        for prior_net in self.content_prior_bank:
            prior_bias_bank.append(prior_net(prior_input).squeeze(-1))
        return torch.stack(prior_bias_bank, dim=0)

    def _build_content_pair_feature(
        self,
        h: torch.Tensor,
        prot_token_feat: Optional[torch.Tensor],
        rna_token_feat: Optional[torch.Tensor],
        bio_pair_feat: Optional[torch.Tensor],
    ) -> torch.Tensor:
        n_prot, n_rna, _ = h.shape
        if not self.use_content_branch:
            return h.new_zeros(h.shape)
        if prot_token_feat is None:
            prot_token_feat = h.new_zeros((n_prot, self.token_feature_dim))
        if rna_token_feat is None:
            rna_token_feat = h.new_zeros((n_rna, self.token_feature_dim))
        if bio_pair_feat is None:
            bio_pair_feat = h.new_zeros((n_prot, n_rna, self.bio_pair_feature_dim))
        elif not self.use_pair_bio_feat:
            bio_pair_feat = h.new_zeros((n_prot, n_rna, self.bio_pair_feature_dim))

        prot_ctx = self.prot_content_tower(prot_token_feat.to(dtype=h.dtype))
        rna_ctx = self.rna_content_tower(rna_token_feat.to(dtype=h.dtype))
        prot_ctx_pair = prot_ctx[:, None, :].expand(-1, n_rna, -1)
        rna_ctx_pair = rna_ctx[None, :, :].expand(n_prot, -1, -1)
        pair_input = torch.cat(
            [
                h,
                prot_ctx_pair,
                rna_ctx_pair,
                prot_ctx_pair * rna_ctx_pair,
                torch.abs(prot_ctx_pair - rna_ctx_pair),
                self.bio_pair_feat_scale * bio_pair_feat.to(dtype=h.dtype),
            ],
            dim=-1,
        )
        return self.content_out_ln(
            h + self.content_pair_mlp(self.content_pair_ln(pair_input))
        )

    def _build_geometry_pair_feature(
        self,
        h: torch.Tensor,
        geom_pair_feat: Optional[torch.Tensor],
    ) -> torch.Tensor:
        n_prot, n_rna, _ = h.shape
        if not self.use_geometry_branch:
            return h.new_zeros(h.shape)
        if geom_pair_feat is None:
            geom_pair_feat = h.new_zeros(
                (n_prot, n_rna, self.geometry_pair_feature_dim)
            )
        elif not self.use_pair_geom_feat:
            geom_pair_feat = h.new_zeros(
                (n_prot, n_rna, self.geometry_pair_feature_dim)
            )
        geom_seed = self.geometry_seed_ln(self.geometry_seed_proj(h))
        geom_aux = self.geometry_pair_ln(
            self.geometry_pair_proj(
                self.geom_pair_feat_scale * geom_pair_feat.to(dtype=h.dtype)
            )
        )
        geom_gate = torch.sigmoid(
            self.geometry_gate(torch.cat([geom_seed, geom_aux], dim=-1))
        )
        return self.geometry_out_ln(geom_seed + geom_gate * geom_aux)

    def _compute_contact_logits(
        self,
        base_contact_logits: torch.Tensor,
        content_prior_shift: torch.Tensor,
    ) -> torch.Tensor:
        return (
            base_contact_logits.unsqueeze(0)
            + self.content_logit_scale
            * (self.content_prior_bias_scale * content_prior_shift)
        )

    def forward(
        self,
        z_pr: torch.Tensor,
        prot_single: Optional[torch.Tensor] = None,
        rna_single: Optional[torch.Tensor] = None,
        prot_token_feat: Optional[torch.Tensor] = None,
        rna_token_feat: Optional[torch.Tensor] = None,
        bio_pair_feat: Optional[torch.Tensor] = None,
        geom_pair_feat: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Args:
            z_pr: [N_protein_token, N_rna_token, c_z]

        Returns:
            contact_logits: [K, N_protein_token, N_rna_token]
            proposal_state: lightweight state used to lazily decode deltas only
                for the selected heads.
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
        if prot_single is None:
            prot_single_rms = z_pr.new_ones((z_pr.shape[0], 1))
        else:
            prot_single_rms = prot_single.float().pow(2).mean(dim=-1, keepdim=True).sqrt()
            prot_single_rms = prot_single_rms.to(dtype=z_pr.dtype)
        if rna_single is None:
            rna_single_rms = z_pr.new_ones((z_pr.shape[1], 1))
        else:
            rna_single_rms = rna_single.float().pow(2).mean(dim=-1, keepdim=True).sqrt()
            rna_single_rms = rna_single_rms.to(dtype=z_pr.dtype)

        pooled = h.mean(dim=(0, 1))
        dynamic_queries = self.query_mlp(self.query_ln(pooled)).view(
            self.num_heads, self.hidden_dim
        )
        head_content_mix = F.softmax(self.head_content_mix, dim=-1)
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
        head_state = slots.squeeze(0)
        base_contact_logits = F.linear(
            h, self.contact_head.weight, self.contact_head.bias
        ).squeeze(-1)
        pair_content = self._build_content_pair_feature(
            h=h,
            prot_token_feat=prot_token_feat,
            rna_token_feat=rna_token_feat,
            bio_pair_feat=bio_pair_feat,
        )
        pair_geometry = self._build_geometry_pair_feature(
            h=h,
            geom_pair_feat=geom_pair_feat,
        )
        prot_transition = self._build_single_transition_feature(
            pair_feature=pair_content + pair_geometry,
            single_feature=prot_single,
            single_proj=self.prot_single_proj,
            reduce_dim=1,
        )
        rna_transition = self._build_single_transition_feature(
            pair_feature=pair_content + pair_geometry,
            single_feature=rna_single,
            single_proj=self.rna_single_proj,
            reduce_dim=0,
        )
        content_prior_bias_bank = self._compute_content_prior_bias_bank(
            pair_content=pair_content,
            base_contact_logits=base_contact_logits,
        )
        content_prior_shift = torch.einsum(
            "kb,bpr->kpr", head_content_mix, content_prior_bias_bank
        )
        head_content = self.content_head_ln(self.content_head_proj(head_state))
        head_geometry = self.geometry_head_ln(self.geometry_head_proj(head_state))
        if self.freeze_geometry_for_head0 and head_geometry.shape[0] > 0:
            head_geometry = head_geometry.clone()
            head_geometry[0] = 0.0
        logits = self._compute_contact_logits(
            base_contact_logits=base_contact_logits,
            content_prior_shift=content_prior_shift,
        )
        proposal_state = {
            "h": h,
            "head_state": head_state,
            "head_queries": head_state,
            "head_content_mix": head_content_mix,
            "content_prior_bias_bank": content_prior_bias_bank,
            "content_prior_shift": content_prior_shift,
            "head_content": head_content,
            "head_geometry": head_geometry,
            "delta_feature_scale": delta_feature_scale,
            "base_contact_logits": base_contact_logits,
            "pair_content": pair_content,
            "pair_geometry": pair_geometry,
            "content_pair_delta_pr_bias": self.content_delta_pair_pr_ln(
                self.content_delta_pair_pr_proj(pair_content)
            ),
            "content_pair_delta_rp_bias": self.content_delta_pair_rp_ln(
                self.content_delta_pair_rp_proj(pair_content)
            ),
            "geometry_pair_delta_pr_bias": self.geometry_delta_pair_pr_ln(
                self.geometry_delta_pair_pr_proj(pair_geometry)
            ),
            "geometry_pair_delta_rp_bias": self.geometry_delta_pair_rp_ln(
                self.geometry_delta_pair_rp_proj(pair_geometry)
            ),
            "pair_delta_pr_bias": self.delta_pair_pr_ln(self.delta_pair_pr_proj(h)),
            "pair_delta_rp_bias": self.delta_pair_rp_ln(self.delta_pair_rp_proj(h)),
            "prot_transition": prot_transition,
            "rna_transition": rna_transition,
            "prot_single_delta_bias": self.prot_single_delta_ln(
                self.prot_single_delta_proj(prot_transition)
            ),
            "rna_single_delta_bias": self.rna_single_delta_ln(
                self.rna_single_delta_proj(rna_transition)
            ),
            "prot_single_rms": prot_single_rms,
            "rna_single_rms": rna_single_rms,
        }
        return logits, proposal_state

    def _ensure_base_delta(
        self, proposal_state: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        base_delta_pr = proposal_state.get("base_delta_pr")
        base_delta_rp = proposal_state.get("base_delta_rp")
        if base_delta_pr is None or base_delta_rp is None:
            h = proposal_state["h"]
            base_delta_pr = F.linear(
                h, self.delta_head_pr.weight, self.delta_head_pr.bias
            )
            base_delta_rp = F.linear(
                h, self.delta_head_rp.weight, self.delta_head_rp.bias
            )
            proposal_state["base_delta_pr"] = base_delta_pr
            proposal_state["base_delta_rp"] = base_delta_rp
        return base_delta_pr, base_delta_rp

    def decode_selected_heads(
        self,
        proposal_state: dict[str, torch.Tensor],
        head_indices: int | list[int] | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Lazily decode deltas only for the selected heads.

        Args:
            proposal_state: state returned by ``forward``.
            head_indices: scalar head id or a list/tensor of head ids.

        Returns:
            delta_pr: [K_sel, N_protein_token, N_rna_token, c_z]
            delta_rp: [K_sel, N_protein_token, N_rna_token, c_z]
            delta_s_prot: [K_sel, N_protein_token, c_s]
            delta_s_rna: [K_sel, N_rna_token, c_s]
        """
        if not torch.is_tensor(head_indices):
            head_indices = torch.as_tensor(
                head_indices, device=proposal_state["head_queries"].device
            )
        head_indices = head_indices.long().reshape(-1)

        base_delta_pr, base_delta_rp = self._ensure_base_delta(proposal_state)
        head_queries = proposal_state["head_state"].index_select(0, head_indices)
        head_content = proposal_state["head_content"].index_select(0, head_indices)
        head_geometry = proposal_state["head_geometry"].index_select(0, head_indices)
        head_delta_pr_bias = self.delta_head_pr_ln(
            self.delta_head_pr_proj(head_queries)
        )
        head_delta_rp_bias = self.delta_head_rp_ln(
            self.delta_head_rp_proj(head_queries)
        )
        head_single = self.single_head_ln(self.single_head_proj(head_queries))
        head_single_delta_bias = self.single_delta_head_ln(
            self.single_delta_head_proj(head_single)
        )
        content_head_delta_pr_bias = self.content_delta_head_pr_ln(
            self.content_delta_head_pr_proj(head_content)
        )
        content_head_delta_rp_bias = self.content_delta_head_rp_ln(
            self.content_delta_head_rp_proj(head_content)
        )
        geometry_head_delta_pr_bias = self.geometry_delta_head_pr_ln(
            self.geometry_delta_head_pr_proj(head_geometry)
        )
        geometry_head_delta_rp_bias = self.geometry_delta_head_rp_ln(
            self.geometry_delta_head_rp_proj(head_geometry)
        )
        selected_logits = self._compute_contact_logits(
            base_contact_logits=proposal_state["base_contact_logits"],
            content_prior_shift=proposal_state["content_prior_shift"].index_select(
                0, head_indices
            ),
        )
        delta_gate = (
            self.gate_floor + (1.0 - self.gate_floor) * torch.sigmoid(selected_logits)
        ).unsqueeze(-1)

        delta_pr_shift = self._compute_pair_local_delta_shift(
            pair_delta_bias=proposal_state["pair_delta_pr_bias"],
            head_delta_bias=head_delta_pr_bias,
        )
        delta_rp_shift = self._compute_pair_local_delta_shift(
            pair_delta_bias=proposal_state["pair_delta_rp_bias"],
            head_delta_bias=head_delta_rp_bias,
        )
        content_delta_pr_shift = self._compute_pair_local_delta_shift(
            pair_delta_bias=proposal_state["content_pair_delta_pr_bias"],
            head_delta_bias=content_head_delta_pr_bias,
        )
        content_delta_rp_shift = self._compute_pair_local_delta_shift(
            pair_delta_bias=proposal_state["content_pair_delta_rp_bias"],
            head_delta_bias=content_head_delta_rp_bias,
        )
        geometry_delta_pr_shift = self._compute_pair_local_delta_shift(
            pair_delta_bias=proposal_state["geometry_pair_delta_pr_bias"],
            head_delta_bias=geometry_head_delta_pr_bias,
        )
        geometry_delta_rp_shift = self._compute_pair_local_delta_shift(
            pair_delta_bias=proposal_state["geometry_pair_delta_rp_bias"],
            head_delta_bias=geometry_head_delta_rp_bias,
        )
        delta_pr_raw = (
            base_delta_pr.unsqueeze(0)
            + delta_pr_shift
            + self.content_delta_scale * content_delta_pr_shift
            + self.geometry_delta_scale * geometry_delta_pr_shift
        )
        delta_rp_raw = (
            base_delta_rp.unsqueeze(0)
            + delta_rp_shift
            + self.content_delta_scale * content_delta_rp_shift
            + self.geometry_delta_scale * geometry_delta_rp_shift
        )
        prot_single_bias = proposal_state["prot_single_delta_bias"].unsqueeze(0)
        rna_single_bias = proposal_state["rna_single_delta_bias"].unsqueeze(0)
        delta_s_prot_raw = prot_single_bias * head_single_delta_bias[:, None, :]
        delta_s_rna_raw = rna_single_bias * head_single_delta_bias[:, None, :]

        delta_pr_raw = self._apply_delta_clip(delta_pr_raw, self.delta_clip_norm)
        delta_rp_raw = self._apply_delta_clip(delta_rp_raw, self.delta_clip_norm)
        delta_s_prot_raw = self._apply_delta_clip(delta_s_prot_raw, self.delta_clip_norm)
        delta_s_rna_raw = self._apply_delta_clip(delta_s_rna_raw, self.delta_clip_norm)

        delta_scale = self.delta_scale * proposal_state["delta_feature_scale"].unsqueeze(0)
        prot_gate = delta_gate.mean(dim=2)
        rna_gate = delta_gate.mean(dim=1)
        delta_s_prot_scale = (
            self.single_delta_scale * proposal_state["prot_single_rms"].unsqueeze(0)
        )
        delta_s_rna_scale = (
            self.single_delta_scale * proposal_state["rna_single_rms"].unsqueeze(0)
        )
        return (
            delta_gate * delta_scale * delta_pr_raw,
            delta_gate * delta_scale * delta_rp_raw,
            prot_gate * delta_s_prot_scale * delta_s_prot_raw,
            rna_gate * delta_s_rna_scale * delta_s_rna_raw,
        )
