import dataclasses
from os.path import exists as opexists, join as opjoin
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
from biotite.structure import AtomArray

from protenix.data.constants import (
    DNA_CHAIN,
    LIGAND_CHAIN_TYPES,
    PROTEIN_CHAIN,
    RNA_CHAIN,
)
from protenix.data.msa.msa_utils import map_to_standard
from protenix.data.ss.ss_utils import parse_dot_bracket, load_sec_struct_file
from protenix.utils.file_io import load_json_cached
from protenix.utils.logger import get_logger

logger = get_logger(__name__)

class SSSourceManager:
    """
    Manages Secondary Structure (SS) data retrieval.
    This class handles finding and loading SS files (e.g., .dbn)
    specifically for RNA chains.
    """
    def __init__(
        self,
        raw_paths: Sequence[str],
        mappings: Dict[int, Dict[Any, Any]],
        enabled: bool,
    ) -> None:
        self.raw_paths = raw_paths
        self.mappings = mappings
        self.enabled = enabled

    def fetch_ss(self, sequence: str, chain_type: str) -> str:
        """
        Fetches secondary structure (dot-bracket string) for the given sequence.
        ONLY loads if chain_type is RNA_CHAIN.
        """
        if not self.enabled or chain_type != RNA_CHAIN:
            return ""
        
        # Iterate through available source paths
        for path, m_key in zip(self.raw_paths, self.mappings):
            mapping = self.mappings[m_key]
            if sequence not in mapping:
                continue
            
            # Assuming mapping value is a list, first element is ID/Filename
            eid = str(mapping[sequence][0])
            
            # Construct file path: base_path/eid/eid.dbn
            # This structure mirrors common MSA directory layouts
            fpath = opjoin(path, eid, f"ss.txt")
            
            if opexists(fpath):
                ss_str = load_sec_struct_file(fpath)
                if ss_str:
                    return ss_str
                    
        return ""

class SSFeatureAssemblyLine:
    """
    Assembles Secondary Structure features into tensors.
    """
    def assemble(
        self, 
        bioassembly: Mapping[int, Mapping[str, Any]], 
        std_idxs: np.ndarray,
        chain_mask: Optional[np.ndarray] = None
    ) -> "SSFeat":
        """
        Assembles the secondary structure feature matrix.
        
        Args:
            bioassembly: Mapping of asymmetric IDs to chain information.
            std_idxs: Array of standardized residue indices for the final combined sequence.
            chain_mask: Optional boolean mask indicating which residues are valid (from MSA assembly).

        Returns:
            An SSFeat object containing the assembled matrix.
        """
        cropped_matrices = []
        
        # 1. Process each chain
        # Iterate directly to match the order in msa_featurizer.py (bioassembly.items())
        # which determines the concatenation order of the features.
        
        chain_lengths = []
        full_seq_len = 0
        
        for aid, info in bioassembly.items():
            seq = info["sequence"]
            ctype = info.get("chain_entity_type", "")
            chain_lengths.append(len(seq))
            full_seq_len += len(seq)
            
            # ONLY process if it's an RNA chain
            if ctype == RNA_CHAIN:
                ss_str = info.get("sec_struct", "")
                mat = parse_dot_bracket(ss_str)
            else:
                # For non-RNA (Protein/DNA/Ligand), use zero matrix
                mat = np.zeros((len(seq), len(seq)), dtype=np.int8)
                
            cropped_matrices.append(mat)
            
        # 2. Merge (Block Diagonal)
        from scipy.linalg import block_diag
        if not cropped_matrices:
            merged_matrix = np.zeros((0, 0), dtype=np.int8)
        else:
            merged_matrix = block_diag(*cropped_matrices).astype(np.int8)
            
        # 3. Map to standard indices
        # std_idxs are indices into the concatenated sequence of all chains (merged_matrix)
        # We need to extract the submatrix corresponding to these indices.
        
        final_dim = len(std_idxs)
        final_matrix = np.zeros((final_dim, final_dim), dtype=np.int8)
        
        if merged_matrix.shape[0] > 0:
            # Filter std_idxs that are within bounds of the merged matrix
            # (std_idxs might contain padding or indices for other entities if not handled carefully, 
            # though usually it should match the bioassembly context)
            
            valid_mask = std_idxs < merged_matrix.shape[0]
            if np.any(valid_mask):
                # We need to map:
                # Source indices: std_idxs[valid_mask] (indices in merged_matrix)
                # Target indices: where valid_mask is True (indices in final_matrix)
                
                src_indices = std_idxs[valid_mask]
                
                # Use np.ix_ to extract the subgrid
                sub_mat = merged_matrix[np.ix_(src_indices, src_indices)]
                
                # Use np.ix_ to place it in the target grid
                # Find indices in final_matrix where valid_mask is True
                tgt_indices = np.where(valid_mask)[0]
                final_matrix[np.ix_(tgt_indices, tgt_indices)] = sub_mat

        return SSFeat(sec_struct_matrix=final_matrix)

class SSFeaturizer:
    """
    Main entry point for Secondary Structure featurization.
    """

    def __init__(
        self,
        dataset_name: str = "",
        rna_seq_or_filename_to_ss_jsons: Sequence[str] = [""],
        rna_ss_raw_paths: Sequence[str] = [""],
        enable_rna_ss: bool = True,
    ) -> None:
        self.dataset_name = dataset_name
        self.mgr = SSSourceManager(
            rna_ss_raw_paths,
            {
                i: load_json_cached(p)
                for i, p in enumerate(rna_seq_or_filename_to_ss_jsons)
            },
            enable_rna_ss,
        )
        logger.info(f"SSFeaturizer for {dataset_name} initialized.")

    def make_ss_features(
        self,
        bioassembly_dict: Dict[str, Any],
        selected_indices: Optional[np.ndarray],
        entity_to_asym_id_int: Mapping[str, Sequence[int]],
    ) -> Dict[str, Any]:
        """
        Processes bioassembly information into a dictionary of Secondary Structure features.
        """
        atom_array, token_array = (
            bioassembly_dict["atom_array"],
            bioassembly_dict["token_array"],
        )
        sel_tokens = (
            token_array[selected_indices]
            if selected_indices is not None
            else token_array
        )
        sel_asyms = set(
            atom_array[sel_tokens.get_annotation("centre_atom_index")].asym_id_int
        )

        # 1. Resolve metadata and fetch SS
        meta = {}
        poly_map = {
            "polypeptide(L)": PROTEIN_CHAIN,
            "polyribonucleotide": RNA_CHAIN,
            "polydeoxyribonucleotide": DNA_CHAIN,
        }

        for eid, asyms in entity_to_asym_id_int.items():
            for aid in [a for a in asyms if a in sel_asyms]:
                seq = bioassembly_dict["sequences"].get(eid) or (
                    "X" * (atom_array.asym_id_int == aid).sum()
                )
                ctype = poly_map.get(
                    bioassembly_dict["entity_poly_type"].get(eid, "non-polymer"),
                    LIGAND_CHAIN_TYPES,
                )

                ss_str = ""
                if ctype == RNA_CHAIN:
                    ss_str = self.mgr.fetch_ss(seq, ctype)

                meta[aid] = {
                    "entity_id": eid,
                    "chain_id": atom_array.chain_id[atom_array.asym_id_int == aid][0],
                    "sequence": seq,
                    "chain_entity_type": ctype,
                    "sec_struct": ss_str,
                }

        # 2. Map coordinates and assemble features
        ca = atom_array[sel_tokens.get_annotation("centre_atom_index")]
        std_idxs = map_to_standard(ca.asym_id_int, ca.res_id, meta)

        res = SSFeatureAssemblyLine().assemble(meta, std_idxs).to_dict()
        return res

    def __call__(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Convenience method to call make_ss_features."""
        return self.make_ss_features(*args, **kwargs)


@dataclasses.dataclass(frozen=True)
class SSFeat:
    """Container for Secondary Structure features."""
    sec_struct_matrix: np.ndarray
    
    def to_dict(self) -> Dict[str, Any]:
        return {"rna_sec_struct": self.sec_struct_matrix}

class InferenceSSFeaturizer:
    """Specialized featurizer for inference scenarios for Secondary Structure."""

    @staticmethod
    def make_ss_feature(
        bioassembly: Sequence[Dict[str, Any]],
        atom_array: AtomArray,
        use_rna_ss: bool = True,
    ) -> Dict[str, Any]:
        """
        Prepares SS features during inference from bioassembly structure.

        Args:
            bioassembly: List of entities in the biological assembly.
            atom_array: Structural data array.
            use_rna_ss: Whether to use Secondary Structure for RNA chains.

        Returns:
            Dictionary of processed SS features.
        """
        meta, curr_aid = {}, 0
        for eid, info in enumerate(bioassembly):
            seq, count, ctype, ss_str = "", 0, LIGAND_CHAIN_TYPES, ""
            
            if "proteinChain" in info:
                c = info["proteinChain"]
                seq, count, ctype = c["sequence"], c["count"], PROTEIN_CHAIN
                # Usually no SS input for protein in this context, but placeholder kept
            
            elif "rnaSequence" in info:
                c = info["rnaSequence"]
                seq, count, ctype = c["sequence"], c["count"], RNA_CHAIN
                if use_rna_ss:
                    # 1. Try direct string content
                    ss_str = c.get("secStruct", "")
                    
                    # 2. Try file path
                    if not ss_str and c.get("secStructPath"):
                         # Basic file read, assuming file contains just the string or simple format
                         # Reusing load_sec_struct_file from utils would be safer if paths are complex
                        try:
                            with open(c["secStructPath"], "r") as f:
                                # Simple read; robust parsing is in assemble/utils
                                content = f.read().strip()
                                # Basic cleaning if it's a fasta-like or other format could go here
                                # But let's assume raw string or simple content for now
                                ss_str = content 
                        except Exception as e:
                            logger.warning(f"Failed to read SS file {c.get('secStructPath')}: {e}")

            elif "dnaSequence" in info:
                c = info["dnaSequence"]
                seq, count, ctype = c["sequence"], c["count"], DNA_CHAIN
            
            elif "ligand" in info:
                count, ctype, seq = (
                    info["ligand"]["count"],
                    LIGAND_CHAIN_TYPES,
                    "X" * (atom_array.asym_id_int == curr_aid).sum(),
                )

            for c_idx in range(count):
                aid = curr_aid + c_idx
                meta[aid] = {
                    "entity_id": eid,
                    "chain_id": atom_array.chain_id[atom_array.asym_id_int == aid][0],
                    "sequence": seq,
                    "chain_entity_type": ctype,
                    "sec_struct": ss_str,
                }
            curr_aid += count

        # We need the centre atoms to map to standard indices, consistent with MSA/InputFeature logic
        ca = atom_array[atom_array.centre_atom_mask.astype(bool)]
        std_idxs = map_to_standard(ca.asym_id_int, ca.res_id, meta)
        
        return SSFeatureAssemblyLine().assemble(meta, std_idxs).to_dict()
