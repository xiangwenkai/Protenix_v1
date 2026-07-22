import argparse
import gzip
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter a Protenix indices CSV down to higher-quality protein-RNA "
            "interface rows using resolution, per-chain resolved fractions, "
            "and resolved token contact count."
        )
    )
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--bioassembly-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--max-resolution", type=float, default=6.0)
    parser.add_argument("--min-rna-resolved-fraction", type=float, default=0.7)
    parser.add_argument("--min-prot-resolved-fraction", type=float, default=0.6)
    parser.add_argument("--contact-distance", type=float, default=8.0)
    parser.add_argument("--min-contact-pairs", type=int, default=5)
    return parser.parse_args()


def load_bioassembly(pdb_id: str, bioassembly_dir: Path) -> dict:
    with gzip.open(bioassembly_dir / f"{pdb_id}.pkl.gz", "rb") as f:
        return pickle.load(f)


def build_chain_stats(bioassembly_dict: dict) -> dict[str, dict[str, np.ndarray | float]]:
    atom_array = bioassembly_dict["atom_array"]
    token_array = bioassembly_dict["token_array"]
    centre_atom_indices = token_array.get_annotation("centre_atom_index")
    centre_atoms = atom_array[centre_atom_indices]

    stats = {}
    for chain_id in np.unique(centre_atoms.chain_id):
        mask = centre_atoms.chain_id == chain_id
        resolved_mask = mask & centre_atoms.is_resolved
        total_tokens = int(mask.sum())
        resolved_tokens = int(resolved_mask.sum())
        stats[str(chain_id)] = {
            "total_tokens": total_tokens,
            "resolved_tokens": resolved_tokens,
            "resolved_fraction": (
                float(resolved_tokens) / float(total_tokens) if total_tokens > 0 else 0.0
            ),
            "resolved_coords": np.asarray(centre_atoms.coord[resolved_mask], dtype=np.float32),
        }
    return stats


def count_contact_pairs(coords_a: np.ndarray, coords_b: np.ndarray, threshold: float) -> int:
    if coords_a.size == 0 or coords_b.size == 0:
        return 0
    tree_a = cKDTree(coords_a)
    tree_b = cKDTree(coords_b)
    return int(tree_a.count_neighbors(tree_b, threshold))


def is_protein_rna_interface(row: pd.Series) -> bool:
    if row["type"] != "interface":
        return False
    m1 = str(row["mol_1_type"])
    m2 = str(row["mol_2_type"])
    return (m1 == "prot" and m2 == "nuc") or (m1 == "nuc" and m2 == "prot")


def main() -> None:
    args = parse_args()

    df_all = pd.read_csv(args.input_csv)
    df_eval = df_all[df_all.apply(is_protein_rna_interface, axis=1)].copy()

    print(f"step0_all_rows={len(df_all)}")
    print(f"step0_all_pdbs={df_all['pdb_id'].nunique()}")
    print(f"step1_prot_nuc_interface_rows={len(df_eval)}")
    print(f"step1_prot_nuc_interface_pdbs={df_eval['pdb_id'].nunique()}")

    df_eval = df_eval[
        pd.to_numeric(df_eval["resolution"], errors="coerce") >= 0.0
    ].copy()
    print(f"step2_resolution_known_rows={len(df_eval)}")
    print(f"step2_resolution_known_pdbs={df_eval['pdb_id'].nunique()}")

    df_eval = df_eval[
        pd.to_numeric(df_eval["resolution"], errors="coerce") < args.max_resolution
    ].copy()
    print(f"step3_resolution_lt_{args.max_resolution:g}_rows={len(df_eval)}")
    print(f"step3_resolution_lt_{args.max_resolution:g}_pdbs={df_eval['pdb_id'].nunique()}")

    df_eval = df_eval.reset_index(drop=True)

    kept_rows: list[dict] = []
    skipped_missing_pkl = 0

    for pdb_id, sub_df in df_eval.groupby("pdb_id", sort=True):
        pkl_path = args.bioassembly_dir / f"{pdb_id}.pkl.gz"
        if not pkl_path.exists():
            skipped_missing_pkl += len(sub_df)
            continue

        bioassembly_dict = load_bioassembly(str(pdb_id), args.bioassembly_dir)
        chain_stats = build_chain_stats(bioassembly_dict)

        for _, row in sub_df.iterrows():
            row_dict = row.to_dict()

            if str(row["mol_1_type"]) == "prot":
                prot_chain = str(row["chain_1_id"])
                rna_chain = str(row["chain_2_id"])
            else:
                prot_chain = str(row["chain_2_id"])
                rna_chain = str(row["chain_1_id"])

            prot_stats = chain_stats.get(prot_chain)
            rna_stats = chain_stats.get(rna_chain)
            if prot_stats is None or rna_stats is None:
                continue

            prot_fraction = float(prot_stats["resolved_fraction"])
            rna_fraction = float(rna_stats["resolved_fraction"])
            contact_pairs = count_contact_pairs(
                prot_stats["resolved_coords"],
                rna_stats["resolved_coords"],
                threshold=args.contact_distance,
            )

            if rna_fraction < args.min_rna_resolved_fraction:
                continue
            if prot_fraction < args.min_prot_resolved_fraction:
                continue
            if contact_pairs < args.min_contact_pairs:
                continue

            row_dict["filter_rna_resolved_fraction"] = rna_fraction
            row_dict["filter_prot_resolved_fraction"] = prot_fraction
            row_dict["filter_contact_pairs"] = contact_pairs
            row_dict["filter_contact_distance"] = args.contact_distance
            kept_rows.append(row_dict)

    kept_eval_df = pd.DataFrame(kept_rows)
    kept_pdb_ids = (
        sorted(kept_eval_df["pdb_id"].unique().tolist()) if len(kept_eval_df) else []
    )
    out_df = df_all[df_all["pdb_id"].isin(kept_pdb_ids)].copy()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)

    print(f"step4_quality_filtered_interface_rows={len(kept_eval_df)}")
    print(
        "step4_quality_filtered_interface_pdbs="
        f"{kept_eval_df['pdb_id'].nunique() if len(kept_eval_df) else 0}"
    )
    print(f"skipped_missing_pkl_rows={skipped_missing_pkl}")
    print(f"step5_output_all_rows_from_kept_pdbs={len(out_df)}")
    print(f"step5_output_pdbs={out_df['pdb_id'].nunique() if len(out_df) else 0}")
    if len(kept_eval_df):
        print(
            "median_rna_fraction="
            f"{kept_eval_df['filter_rna_resolved_fraction'].median():.3f}"
        )
        print(
            "median_prot_fraction="
            f"{kept_eval_df['filter_prot_resolved_fraction'].median():.3f}"
        )
        print(
            "median_contact_pairs="
            f"{kept_eval_df['filter_contact_pairs'].median():.1f}"
        )
    print(f"saved={args.output_csv}")


if __name__ == "__main__":
    main()



"""
python scripts/filter_protein_rna_interface_indices.py --input-csv /inspire/ssd/project/sais-bio/public/xiangwenkai/GITHUB/Protenix_v1/data/protein_rna_train.csv --bioassembly-dir /inspire/ssd/project/sais-bio/public/xiangwenkai/GITHUB/Protenix_v1/data/train --output-csv /inspire/ssd/project/sais-bio/public/xiangwenkai/GITHUB/Protenix_v1/data/protein_rna_train_filter.csv

  1.resolution < 6A

  - 这是看整条结构的清晰度。
  - 数值越小，结构通常越可靠。
  - 6A 以上很多结构已经比较模糊了，尤其 RNA 和界面细节更容易不准。
  - 这条是在先砍掉一批“整体质量偏差”的样本。

  2.RNA resolved fraction >= 0.7

  - 意思是：这条 RNA 链里，至少 70% 的 token/残基是真的在结构里看得到、坐标不是缺失的。
  - 如果 RNA 缺了一大截，模型会学到错误的界面几何。
  - 这条是在防“RNA 明明不完整，却还拿来当真值”。

  3.protein resolved fraction >= 0.6

  - 跟上面一样，只不过对象换成蛋白链。
  - 蛋白如果有太多残基没解析出来，界面位置也会不稳定。
  - 这里阈值比 RNA 稍松一点，是因为蛋白很多时候局部缺失没那么致命，但太缺也不行。

  4.contact pairs >= 5，距离阈值 8A

  - 这是在问：这条蛋白链和 RNA 链之间，至少有没有像样的接触。
  - 我们只数“已经解析出来的” token 对，如果两边距离小于 8A，就算一对接触。
  - 至少要有 5 对，才认为这不是“擦边路过”或者“其实没形成稳定界面”的假样本。
  - 这条是在防“名义上是复合物，实际上几乎没接触”。
"""