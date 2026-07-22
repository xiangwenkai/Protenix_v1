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

"""Configuration for mixed PDB structure + eCLIP RNA signal SFT."""

from protenix.config.extend_types import DefaultNoneWithType

rna_eclip_mixed_sft_configs = {
    "mixed_sft": {
        "pdb_sample_prob": 0.5,
        "freeze_confidence_head": True,
        "rollout_metric_contact_cutoff": 5.0,
        "pdb_eval_max_steps": DefaultNoneWithType(int),
        "best_metric": "structure",
        "best_metric_mode": "auto",
        "save_training_state": False,
    },
    "foldbench_eval": {
        "enable": False,
        "start_index": 0,
        "seeds": "102",
        "cycle": 10,
        "step": 200,
        "sample": 1,
        "dtype": "bf16",
        "use_msa": True,
        "use_rna_msa": True,
        "msa_server_mode": "protenix",
        "metric_type": "rank",
        "skip_dockqv2": False,
        "dockq_allowed_mismatches": 8,
        "ground_truth_dir": (
            "/inspire/ssd/project/sais-bio/public/xiangwenkai/Protenix_v1/"
            "examples/sft/pxmeter_other_targets/interface_protein_rna/ground_truth_cif"
        ),
        "dockq_ground_truth_dir": (
            "/inspire/ssd/project/sais-bio/public/xiangwenkai/GITHUB/"
            "Protenix_v1/data/protein_rna"
        ),
        "foldbench_repo": (
            "/inspire/ssd/project/sais-bio/public/xiangwenkai/GITHUB/FoldBench"
        ),
        "foldbench_conda_env": "foldbench",
        "targets_dir": (
            "/inspire/ssd/project/sais-bio/public/xiangwenkai/GITHUB/FoldBench/targets"
        ),
        "mmcif_dir": (
            "/inspire/ssd/project/sais-bio/public/Protein/data/AI_Models/"
            "protenix_v1_dataset/mmcif"
        ),
        "seq_to_pdb_index": (
            "/inspire/ssd/project/sais-bio/public/Protein/data/AI_Models/protenix_v1_dataset/common/seq_to_pdb_index.json"
        ),
        "msa_template_dir": (
            "/inspire/ssd/project/sais-bio/public/Protein/data/AI_Models/protenix_v1_dataset/mmcif_msa_template"
        ),
        "rna_msa_cache_root": (
            "/inspire/ssd/project/sais-bio/public/xiangwenkai/GITHUB/"
            "Protenix/rna_data/rna_msa"
        ),
        "foldbench_rna_msa_cache_root": (
            "/inspire/ssd/project/sais-bio/public/xiangwenkai/GITHUB/"
            "FoldBench/targets/interface_protein_rna_rna_msa"
        ),
        "ntrna_database_path": (
            "/inspire/ssd/project/sais-bio/public/Protein/data/AI_Models/"
            "protenix_v1_dataset/search_database/"
            "nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta"
        ),
        "rfam_database_path": (
            "/inspire/ssd/project/sais-bio/public/Protein/data/AI_Models/"
            "protenix_v1_dataset/search_database/"
            "rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta"
        ),
        "rna_central_database_path": (
            "/inspire/ssd/project/sais-bio/public/Protein/data/AI_Models/"
            "protenix_v1_dataset/search_database/"
            "rnacentral_active_seq_id_90_cov_80_linclust.fasta"
        ),
        "nhmmer_n_cpu": 2,
    },
}
