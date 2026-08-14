#!/usr/bin/env python3
"""Fill missing MSA/template cache entries for train_rna_before202606.

The script scans the configured protein-RNA training set, compares unique
protein/RNA entity sequences against the Protenix public cache layout, and can
optionally search the missing entries in place.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_TRAIN_CSV = Path(
    "/inspire/ssd/project/sais-bio/public/xiangwenkai/GITHUB/Protenix_v1/data/"
    "protein_rna_train_filter.csv"
)
DEFAULT_BIOASSEMBLY_DIR = Path(
    "/inspire/ssd/project/sais-bio/public/xiangwenkai/GITHUB/Protenix_v1/data/"
    "train_signal"
)
DEFAULT_DATASET_ROOT = Path(
    "/inspire/ssd/project/sais-bio/public/Protein/data/AI_Models/"
    "protenix_v1_dataset"
)


def valid_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def normalize_protein(seq: str) -> str:
    return "".join(str(seq).split()).upper()


def normalize_rna(seq: str) -> str:
    return "".join(str(seq).split()).upper().replace("T", "U")


def sha_id(prefix: str, seq: str) -> str:
    return f"{prefix}_{hashlib.sha256(seq.encode()).hexdigest()[:16]}"


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as tmp:
        json.dump(data, tmp, indent=2, sort_keys=True)
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def backup_once(path: Path, tag: str) -> Path:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    backup = path.with_name(f"{path.name}.bak.{tag}.{stamp}")
    shutil.copy2(path, backup)
    return backup


def read_pdb_ids(train_csv: Path) -> list[str]:
    df = pd.read_csv(train_csv, usecols=["pdb_id"])
    return sorted(set(df["pdb_id"].astype(str).str.lower()))


def load_sequences_one(args: tuple[str, str]) -> tuple[str, list[str], list[str], str | None]:
    pdb_id, bioassembly_dir = args
    path = Path(bioassembly_dir) / f"{pdb_id}.pkl.gz"
    if not path.exists():
        return pdb_id, [], [], "missing_pickle"

    try:
        with gzip.open(path, "rb") as f:
            bio = pickle.load(f)
    except Exception as exc:  # pragma: no cover - diagnostic path
        return pdb_id, [], [], repr(exc)

    prot: list[str] = []
    rna: list[str] = []
    sequences = bio.get("sequences") or {}
    entity_poly_type = bio.get("entity_poly_type") or {}
    for entity_id, seq in sequences.items():
        poly_type = str(
            entity_poly_type.get(str(entity_id), entity_poly_type.get(entity_id, ""))
        ).lower()
        if "polypeptide" in poly_type:
            norm = normalize_protein(seq)
            if norm:
                prot.append(norm)
        elif "ribonucleotide" in poly_type:
            norm = normalize_rna(seq)
            if norm:
                rna.append(norm)
    return pdb_id, prot, rna, None


def collect_unique_sequences(
    train_csv: Path, bioassembly_dir: Path, workers: int
) -> tuple[set[str], set[str], list[tuple[str, str | None]]]:
    pdb_ids = read_pdb_ids(train_csv)
    prot: set[str] = set()
    rna: set[str] = set()
    errors: list[tuple[str, str | None]] = []
    print(f"Scanning {len(pdb_ids)} unique PDB ids from {train_csv}", flush=True)

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(load_sequences_one, (pdb_id, str(bioassembly_dir)))
            for pdb_id in pdb_ids
        ]
        for done, future in enumerate(as_completed(futures), 1):
            pdb_id, prot_seqs, rna_seqs, error = future.result()
            prot.update(prot_seqs)
            rna.update(rna_seqs)
            if error:
                errors.append((pdb_id, error))
            if done % 500 == 0 or done == len(futures):
                print(
                    f"  {done}/{len(futures)} pdbs; "
                    f"protein={len(prot)} rna={len(rna)} errors={len(errors)}",
                    flush=True,
                )
    return prot, rna, errors


def first_existing_rna_key(rna_index: dict[str, Any], seq: str) -> tuple[str, Any] | None:
    for key in (seq, seq.replace("U", "T")):
        value = rna_index.get(key)
        if value:
            return key, value
    return None


def rna_cache_path(msa_root: Path, ids: Any) -> tuple[str, Path]:
    if isinstance(ids, str):
        entity_id = ids
    else:
        entity_id = str(ids[0])
    return entity_id, msa_root / entity_id / f"{entity_id}_all.a3m"


def detect_missing(
    protein_seqs: set[str], rna_seqs: set[str], dataset_root: Path
) -> dict[str, Any]:
    prot_index_path = dataset_root / "common" / "seq_to_pdb_index.json"
    rna_index_path = dataset_root / "rna_msa" / "rna_sequence_to_pdb_chains.json"
    prot_index = load_json(prot_index_path)
    rna_index = load_json(rna_index_path)
    prot_root = dataset_root / "mmcif_msa_template"
    rna_msa_root = dataset_root / "rna_msa" / "msas"

    missing_protein_msa: list[dict[str, Any]] = []
    missing_template: list[dict[str, Any]] = []
    for seq in sorted(protein_seqs):
        idx = prot_index.get(seq)
        if idx is None:
            missing_protein_msa.append({"sequence": seq, "reason": "missing_index"})
            missing_template.append({"sequence": seq, "reason": "missing_protein_msa"})
            continue

        msa_dir = prot_root / str(idx)
        has_pairing = valid_file(msa_dir / "pairing.a3m")
        has_non_pairing = valid_file(msa_dir / "non_pairing.a3m")
        if not (has_pairing and has_non_pairing):
            missing_protein_msa.append(
                {
                    "sequence": seq,
                    "index": idx,
                    "msa_dir": str(msa_dir),
                    "reason": "missing_msa_file",
                }
            )

        if not valid_file(msa_dir / "hmmsearch.a3m"):
            missing_template.append(
                {
                    "sequence": seq,
                    "index": idx,
                    "msa_dir": str(msa_dir),
                    "reason": "missing_hmmsearch",
                }
            )

    missing_rna_msa: list[dict[str, Any]] = []
    for seq in sorted(rna_seqs):
        hit = first_existing_rna_key(rna_index, seq)
        if hit is None:
            entity_id = sha_id("custom_rna", seq)
            missing_rna_msa.append(
                {
                    "sequence": seq,
                    "rna_key": seq,
                    "entity_id": entity_id,
                    "msa_path": str(rna_msa_root / entity_id / f"{entity_id}_all.a3m"),
                    "reason": "missing_index",
                }
            )
            continue

        key, ids = hit
        entity_id, msa_path = rna_cache_path(rna_msa_root, ids)
        if not valid_file(msa_path):
            missing_rna_msa.append(
                {
                    "sequence": seq,
                    "rna_key": key,
                    "entity_id": entity_id,
                    "msa_path": str(msa_path),
                    "reason": "missing_msa_file",
                }
            )

    def reason_counts(entries: list[dict[str, Any]]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for entry in entries:
            reason = str(entry.get("reason", "unknown"))
            counts[reason] = counts.get(reason, 0) + 1
        return counts

    return {
        "counts": {
            "unique_protein_sequences": len(protein_seqs),
            "unique_rna_sequences": len(rna_seqs),
            "missing_protein_msa": len(missing_protein_msa),
            "missing_protein_template": len(missing_template),
            "missing_rna_msa": len(missing_rna_msa),
            "missing_protein_msa_by_reason": reason_counts(missing_protein_msa),
            "missing_protein_template_by_reason": reason_counts(missing_template),
            "missing_rna_msa_by_reason": reason_counts(missing_rna_msa),
        },
        "missing_protein_msa": missing_protein_msa,
        "missing_protein_template": missing_template,
        "missing_rna_msa": missing_rna_msa,
    }


def run_rna_searches(
    entries: list[dict[str, Any]], dataset_root: Path, nhmmer_n_cpu: int, limit: int | None
) -> None:
    from runner.rna_msa_search import run_rna_msa_search

    search_db = dataset_root / "search_database"
    rna_root = dataset_root / "rna_msa"
    msa_root = rna_root / "msas"
    rna_index_path = rna_root / "rna_sequence_to_pdb_chains.json"
    rna_index = load_json(rna_index_path)
    backup = backup_once(rna_index_path, "fill_train_rna_before202606")
    print(f"Backed up RNA index to {backup}", flush=True)

    selected = entries[:limit] if limit is not None else entries
    for i, entry in enumerate(selected, 1):
        seq = entry["sequence"]
        entity_id = entry["entity_id"]
        final_path = msa_root / entity_id / f"{entity_id}_all.a3m"
        if valid_file(final_path):
            rna_index.setdefault(entry["rna_key"], [entity_id])
            continue

        print(f"[RNA {i}/{len(selected)}] searching {entity_id}", flush=True)
        with tempfile.TemporaryDirectory(prefix="rna_msa_search_") as tmp:
            tmp_path = Path(tmp)
            run_rna_msa_search(
                rna_seq_for_msa_search=seq,
                rna_result_path=str(tmp_path),
                rna_seq_id=entity_id,
                ntrna_database_path=str(
                    search_db / "nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta"
                ),
                rfam_database_path=str(
                    search_db / "rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta"
                ),
                rna_central_database_path=str(
                    search_db / "rnacentral_active_seq_id_90_cov_80_linclust.fasta"
                ),
                nhmmer_n_cpu=nhmmer_n_cpu,
            )
            produced = tmp_path / entity_id / "rna_msa.a3m"
            if not valid_file(produced):
                raise RuntimeError(f"RNA MSA search produced no usable file: {produced}")
            final_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(produced, final_path)

        rna_index[entry["rna_key"]] = [entity_id]
        write_json_atomic(rna_index_path, rna_index)


def msa_names_for_template(msa_dir: Path) -> str | None:
    names = []
    if valid_file(msa_dir / "pairing.a3m"):
        names.append("pairing")
    if valid_file(msa_dir / "non_pairing.a3m"):
        names.append("non_pairing")
    return ",".join(names) if names else None


def run_template_searches(
    entries: list[dict[str, Any]], dataset_root: Path, limit: int | None
) -> None:
    from runner.template_search import run_template_search

    selected = entries[:limit] if limit is not None else entries
    seqres = dataset_root / "search_database" / "pdb_seqres_2022_09_28.fasta"
    for i, entry in enumerate(selected, 1):
        if entry.get("reason") == "missing_protein_msa":
            print(
                f"[template {i}/{len(selected)}] skip sequence without protein MSA",
                flush=True,
            )
            continue
        msa_dir = Path(entry["msa_dir"])
        out_path = msa_dir / "hmmsearch.a3m"
        if valid_file(out_path):
            continue
        msa_names = msa_names_for_template(msa_dir)
        if not msa_names:
            print(f"[template {i}/{len(selected)}] skip {msa_dir}: no MSA", flush=True)
            continue
        print(f"[template {i}/{len(selected)}] searching {msa_dir}", flush=True)
        run_template_search(
            msa_for_template_search_dir=str(msa_dir),
            msa_for_template_search_name=msa_names,
            seqres_database_path=str(seqres),
        )


def write_fasta(path: Path, entries: list[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for i, entry in enumerate(entries):
            f.write(f">seq_{i}|{entry.get('reason', 'missing')}\n")
            f.write(f"{entry['sequence']}\n")


def run_protein_msa_searches(
    entries: list[dict[str, Any]],
    dataset_root: Path,
    work_dir: Path,
    colabfold_db_dir: Path,
    colabsearch: str,
    mmseqs: str,
    limit: int | None,
    search_templates: bool,
) -> None:
    prot_index_path = dataset_root / "common" / "seq_to_pdb_index.json"
    prot_index = load_json(prot_index_path)
    backup = backup_once(prot_index_path, "fill_train_rna_before202606")
    print(f"Backed up protein index to {backup}", flush=True)
    next_index = max(int(v) for v in prot_index.values()) + 1
    selected = entries[:limit] if limit is not None else entries

    for i, entry in enumerate(selected, 1):
        seq = entry["sequence"]
        idx = prot_index.get(seq)
        if idx is None:
            idx = next_index
            next_index += 1
        msa_dir = dataset_root / "mmcif_msa_template" / str(idx)
        if valid_file(msa_dir / "pairing.a3m") and valid_file(msa_dir / "non_pairing.a3m"):
            prot_index[seq] = idx
            continue

        print(f"[protein MSA {i}/{len(selected)}] searching index {idx}", flush=True)
        one_work = work_dir / "protein_msa" / str(idx)
        one_work.mkdir(parents=True, exist_ok=True)
        query = one_work / "query.fasta"
        query.write_text(f">query\n{seq}\n")
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "colabfold_msa.py"),
            str(query),
            str(colabfold_db_dir),
            str(one_work),
            "--colabsearch",
            colabsearch,
            "--mmseqs_path",
            mmseqs,
        ]
        subprocess.run(cmd, check=True)
        produced = one_work / "msa" / "0"
        if not (valid_file(produced / "pairing.a3m") and valid_file(produced / "non_pairing.a3m")):
            raise RuntimeError(f"Protein MSA search produced incomplete files in {produced}")
        msa_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(produced / "pairing.a3m", msa_dir / "pairing.a3m")
        shutil.copy2(produced / "non_pairing.a3m", msa_dir / "non_pairing.a3m")
        prot_index[seq] = idx
        write_json_atomic(prot_index_path, prot_index)

        if search_templates:
            run_template_searches(
                [{"sequence": seq, "index": idx, "msa_dir": str(msa_dir)}],
                dataset_root,
                limit=None,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan train_rna_before202606 and optionally fill missing RNA MSA, "
            "protein MSA, and protein template cache files."
        )
    )
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--bioassembly-dir", type=Path, default=DEFAULT_BIOASSEMBLY_DIR)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=REPO_ROOT / "tmp" / "fill_train_rna_before202606",
    )
    parser.add_argument("--scan-workers", type=int, default=12)
    parser.add_argument("--nhmmer-n-cpu", type=int, default=2)
    parser.add_argument("--apply", action="store_true", help="Write/search cache files.")
    parser.add_argument("--search-rna", action="store_true")
    parser.add_argument("--search-template", action="store_true")
    parser.add_argument("--search-protein-msa", action="store_true")
    parser.add_argument("--limit-rna", type=int, default=None)
    parser.add_argument("--limit-template", type=int, default=None)
    parser.add_argument("--limit-protein-msa", type=int, default=None)
    parser.add_argument("--colabfold-db-dir", type=Path, default=None)
    parser.add_argument("--colabsearch", default=shutil.which("colabfold_search") or "colabfold_search")
    parser.add_argument("--mmseqs", default=shutil.which("mmseqs") or "mmseqs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)

    protein_seqs, rna_seqs, errors = collect_unique_sequences(
        args.train_csv, args.bioassembly_dir, args.scan_workers
    )
    manifest = detect_missing(protein_seqs, rna_seqs, args.dataset_root)
    manifest["scan_errors"] = errors
    manifest_path = args.work_dir / "missing_manifest.json"
    write_json_atomic(manifest_path, manifest)
    write_fasta(args.work_dir / "missing_rna_msa.fasta", manifest["missing_rna_msa"])
    write_fasta(
        args.work_dir / "missing_protein_msa.fasta",
        manifest["missing_protein_msa"],
    )
    write_fasta(
        args.work_dir / "missing_protein_template.fasta",
        manifest["missing_protein_template"],
    )

    print(json.dumps(manifest["counts"], indent=2), flush=True)
    print(f"Wrote manifest and FASTA files under {args.work_dir}", flush=True)

    if not args.apply:
        print("Dry run only. Add --apply plus search flags to write cache files.", flush=True)
        return 0

    if args.search_protein_msa:
        if args.colabfold_db_dir is None:
            raise SystemExit("--search-protein-msa requires --colabfold-db-dir")
        run_protein_msa_searches(
            manifest["missing_protein_msa"],
            args.dataset_root,
            args.work_dir,
            args.colabfold_db_dir,
            args.colabsearch,
            args.mmseqs,
            args.limit_protein_msa,
            search_templates=args.search_template,
        )

    if args.search_template:
        run_template_searches(
            manifest["missing_protein_template"],
            args.dataset_root,
            args.limit_template,
        )

    if args.search_rna:
        run_rna_searches(
            manifest["missing_rna_msa"],
            args.dataset_root,
            args.nhmmer_n_cpu,
            args.limit_rna,
        )

    print("Done. Re-run without --apply to verify remaining missing entries.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
python -u scripts/fill_train_rna_before202606_msa_template.py \
    --apply \
    --search-template
"""