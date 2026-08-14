"""Run FoldBench metric calculation inside the FoldBench conda environment."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Optional

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank-csv", type=Path, required=True)
    parser.add_argument("--rank-eval-dir", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--foldbench-repo", type=Path, required=True)
    parser.add_argument("--target-type", required=True)
    parser.add_argument("--skip-dockqv2", action="store_true")
    parser.add_argument("--dockq-allowed-mismatches", type=int, default=8)
    return parser.parse_args()


def run_dockqv2_case(
    row: dict[str, Any],
    native_dir: Path,
    detail_dir: Path,
    allowed_mismatches: int,
) -> Optional[dict[str, Any]]:
    try:
        from evaluation.eval_by_dockqv2 import NumpyEncoder, dockq

        pdb_id = row["pdb_id"]
        chain1 = row["interface_chain_id_1"]
        chain2 = row["interface_chain_id_2"]
        seed = row["seed"]
        sample = row["sample"]
        prediction_path = row["prediction_path"]
        native_path = native_dir / f"{pdb_id}.cif"
        if not Path(prediction_path).exists() or not native_path.exists():
            return None
        info = dockq(
            model_path=prediction_path,
            native_path=str(native_path),
            model_chains=[chain1, chain2],
            native_chains=[chain1, chain2],
            small_molecule=False,
            allowed_mismatches=allowed_mismatches,
        )
        if info is None or not info.get("best_result"):
            return None
        detail_dir.mkdir(parents=True, exist_ok=True)
        output_path = (
            detail_dir / f"{pdb_id}_{seed}_{sample}_{chain1}_{chain2}_dockqv2.json"
        )
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(info, handle, cls=NumpyEncoder)
        best_result = info["best_result"][list(info["best_result"].keys())[0]]
        result = dict(row)
        result.update(
            {
                "lrmsd": best_result["LRMSD"],
                "irmsd": best_result["iRMSD"],
                "dockq_score": best_result["DockQ"],
            }
        )
        return result
    except Exception:
        print(f"DockQv2 failed:\n{traceback.format_exc()}", file=sys.stderr)
        return None


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.foldbench_repo.resolve()))

    from evaluation import eval_by_ost

    rank_eval_dir = args.rank_eval_dir.resolve()
    raw_dir = rank_eval_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    rank_df = pd.read_csv(args.rank_csv) if args.rank_csv.exists() else pd.DataFrame()
    if rank_df.empty:
        pd.DataFrame().to_csv(raw_dir / f"{args.target_type}_ost.csv", index=False)
        pd.DataFrame().to_csv(raw_dir / f"{args.target_type}_dockqv2.csv", index=False)
        return

    eval_by_ost(
        rank_df,
        args.target_type,
        str(rank_eval_dir),
        str(args.eval_root / "ground_truths"),
        max_workers=1,
    )

    if args.skip_dockqv2:
        return

    results = []
    for _, row in rank_df.iterrows():
        result = run_dockqv2_case(
            row.to_dict(),
            args.eval_root / "dockq_ground_truths",
            rank_eval_dir / "detail",
            args.dockq_allowed_mismatches,
        )
        if result is not None:
            results.append(result)
    pd.DataFrame(results).to_csv(
        raw_dir / f"{args.target_type}_dockqv2.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
