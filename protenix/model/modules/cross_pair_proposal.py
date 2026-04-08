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

    bce = F.binary_cross_entropy_with_logits(
        logits, target, reduction="none"
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
    probs: torch.Tensor,
    margin: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Penalize near-duplicate proposals."""
    n_head = probs.shape[0]
    if n_head <= 1:
        return probs.new_zeros(())
    flat = probs.reshape(n_head, -1).float()
    flat = flat / (flat.norm(dim=-1, keepdim=True) + eps)
    pairwise = torch.cdist(flat, flat, p=2)
    upper = torch.triu_indices(n_head, n_head, offset=1, device=probs.device)
    pairwise = pairwise[upper[0], upper[1]]
    return torch.relu(margin - pairwise).mean()


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
        delta_scale: float = 1.0,
        contact_threshold: float = 8.0,
        enable: bool = False,
    ) -> None:
        super(KWayCrossPairProposal, self).__init__()
        self.c_z = c_z
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.delta_scale = delta_scale
        self.contact_threshold = contact_threshold
        self.enable = enable

        self.input_ln = nn.LayerNorm(c_z)
        self.shared = nn.Sequential(
            nn.Linear(c_z, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.head_embed = nn.Parameter(torch.randn(num_heads, hidden_dim))
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
        h = self.shared(self.input_ln(z_pr))
        logits, deltas_pr, deltas_rp = [], [], []
        for head_idx in range(self.num_heads):
            h_k = h + self.head_embed[head_idx].view(1, 1, -1)
            contact_logits = self.contact_head(h_k).squeeze(-1)
            contact_probs = torch.sigmoid(contact_logits)
            delta_gate = contact_probs.unsqueeze(-1) * self.delta_scale
            deltas_pr.append(delta_gate * self.delta_head_pr(h_k))
            deltas_rp.append(delta_gate * self.delta_head_rp(h_k))
            logits.append(contact_logits)
        return (
            torch.stack(logits, dim=0),
            torch.stack(deltas_pr, dim=0),
            torch.stack(deltas_rp, dim=0),
        )
