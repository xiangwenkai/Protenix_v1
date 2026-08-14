#!/usr/bin/env python3
"""Fill short train_rna_before202606 RNA MSA misses with rMSA.

This script scans the RNA entities used by ``train_rna_before202606`` and
selects RNA sequences shorter than ``--max-length`` whose Protenix RNA MSA cache
is not actually useful:

* no entry in ``rna_sequence_to_pdb_chains.json``;
* mapped entry exists, but ``msas/{id}/{id}_all.a3m`` is missing or empty;
* optionally, mapped A3M exists but has only the query sequence.

For selected sequences, it runs the user-provided ``rMSA.fast.pl`` command,
converts rMSA's ``.afa`` output to A3M, and imports the result into the public
Protenix RNA MSA cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.fill_train_rna_before202606_msa_template import (  # noqa: E402
    DEFAULT_BIOASSEMBLY_DIR,
    DEFAULT_DATASET_ROOT,
    DEFAULT_TRAIN_CSV,
    collect_unique_sequences,
    first_existing_rna_key,
    load_json,
    rna_cache_path,
    sha_id,
    valid_file,
    write_json_atomic,
)


DEFAULT_RMSA_DIR = Path(
    "/inspire/ssd/project/sais-bio/public/xiangwenkai/rna_pipline/rMSA"
)
DEFAULT_RMSA_DB_DIR = Path(
    "/inspire/ssd/project/sais-bio/public/ash_proj/code/rMSA/database"
)


def backup_once(path: Path, tag: str) -> Path:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    backup = path.with_name(f"{path.name}.bak.{tag}.{stamp}")
    shutil.copy2(path, backup)
    return backup


def normalize_rna(sequence: str) -> str:
    return re.sub(r"[^A-Za-z]", "", sequence).upper().replace("T", "U")


def read_fasta_records(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    name = ""
    chunks: list[str] = []
    with path.open(encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name:
                    records.append((name, "".join(chunks)))
                name = line[1:].strip() or "sequence"
                chunks = []
            else:
                chunks.append(line)
    if name:
        records.append((name, "".join(chunks)))
    return records


def afa_to_a3m_text(afa_path: Path) -> str:
    records = read_fasta_records(afa_path)
    if not records:
        raise ValueError(f"Empty rMSA alignment: {afa_path}")

    query = records[0][1].upper()
    aln_len = len(query)
    if aln_len == 0:
        raise ValueError(f"Empty query in rMSA alignment: {afa_path}")
    match_cols = [char != "-" for char in query]

    lines: list[str] = []
    for name, seq in records:
        seq = seq.upper()
        if len(seq) != aln_len:
            raise ValueError(
                f"Aligned length mismatch for {name}: {len(seq)} != {aln_len}"
            )
        a3m_chars: list[str] = []
        for i, char in enumerate(seq):
            if match_cols[i]:
                a3m_chars.append(char)
            elif char != "-":
                a3m_chars.append(char.lower())
        a3m_seq = "".join(a3m_chars).replace("T", "U").replace("t", "u")
        lines.append(f">{name}")
        lines.append(a3m_seq)
    return "\n".join(lines) + "\n"


def a3m_depth(a3m_text: str) -> int:
    return sum(1 for line in a3m_text.splitlines() if line.startswith(">"))


def existing_a3m_depth(path: Path) -> int:
    if not valid_file(path):
        return 0
    with path.open(encoding="utf-8", errors="ignore") as handle:
        return sum(1 for line in handle if line.startswith(">"))


def unique_rmsa_id(sequence: str, used_ids: set[str]) -> str:
    base = f"rmsa_{hashlib.sha256(sequence.encode()).hexdigest()[:16]}"
    candidate = base
    suffix = 1
    while candidate in used_ids:
        suffix += 1
        candidate = f"{base}_{suffix}"
    used_ids.add(candidate)
    return candidate


def collect_used_ids(mapping: dict[str, Any]) -> set[str]:
    used: set[str] = set()
    for value in mapping.values():
        if isinstance(value, str):
            used.add(value)
        elif value:
            used.update(str(item) for item in value)
    return used


def write_fasta(path: Path, sequence: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f">query\n{sequence}\n", encoding="utf-8")


def write_candidates_fasta(path: Path, entries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for i, entry in enumerate(entries, 1):
            handle.write(
                f">rna_{i}|len={len(entry['sequence'])}|reason={entry['reason']}\n"
            )
            handle.write(f"{entry['sequence']}\n")


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        handle.write(text)
        tmp_path = Path(handle.name)
    os.replace(tmp_path, path)


def find_candidates(
    rna_sequences: set[str],
    dataset_root: Path,
    max_length: int,
    include_query_only: bool,
    min_existing_depth: int,
) -> list[dict[str, Any]]:
    rna_root = dataset_root / "rna_msa"
    msa_root = rna_root / "msas"
    mapping = load_json(rna_root / "rna_sequence_to_pdb_chains.json")
    used_ids = collect_used_ids(mapping)

    entries: list[dict[str, Any]] = []
    for seq in sorted(rna_sequences, key=lambda item: (len(item), item)):
        seq = normalize_rna(seq)
        if not seq or len(seq) >= max_length:
            continue

        hit = first_existing_rna_key(mapping, seq)
        if hit is None:
            entity_id = sha_id("custom_rna", seq)
            if entity_id in used_ids:
                entity_id = unique_rmsa_id(seq, used_ids)
            entries.append(
                {
                    "sequence": seq,
                    "rna_key": seq,
                    "entity_id": entity_id,
                    "target_msa_path": str(msa_root / entity_id / f"{entity_id}_all.a3m"),
                    "reason": "missing_index",
                    "existing_depth": 0,
                }
            )
            continue

        key, ids = hit
        entity_id, msa_path = rna_cache_path(msa_root, ids)
        depth = existing_a3m_depth(msa_path)
        if depth == 0:
            entries.append(
                {
                    "sequence": seq,
                    "rna_key": key,
                    "entity_id": entity_id,
                    "target_msa_path": str(msa_path),
                    "reason": "missing_msa_file",
                    "existing_depth": 0,
                }
            )
        elif include_query_only and depth <= min_existing_depth:
            new_id = unique_rmsa_id(seq, used_ids)
            entries.append(
                {
                    "sequence": seq,
                    "rna_key": key,
                    "entity_id": new_id,
                    "target_msa_path": str(msa_root / new_id / f"{new_id}_all.a3m"),
                    "reason": f"existing_depth_le_{min_existing_depth}",
                    "existing_entity_id": entity_id,
                    "existing_msa_path": str(msa_path),
                    "existing_depth": depth,
                }
            )
    return entries


def validate_rmsa_inputs(args: argparse.Namespace) -> None:
    script = args.rmsa_dir / "rMSA.fast.pl"
    if not script.is_file():
        raise FileNotFoundError(f"rMSA.fast.pl not found: {script}")
    for label in ("db0", "db1", "db2", "db0to1", "db0to2"):
        path = Path(getattr(args, label))
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")


def run_rmsa_one(
    args: argparse.Namespace,
    entry: dict[str, Any],
    fasta_path: Path,
    log_path: Path,
) -> Path:
    script = args.rmsa_dir / "rMSA.fast.pl"
    prefix = fasta_path.with_suffix("")
    afa_path = prefix.with_suffix(".afa")
    tmp_dir = fasta_path.parent / "tmp"

    if afa_path.exists() and afa_path.stat().st_size > 0 and not args.overwrite:
        return afa_path

    write_fasta(fasta_path, entry["sequence"])
    cmd = [
        "perl",
        str(script),
        str(fasta_path),
        f"-db0={args.db0}",
        f"-db1={args.db1}",
        f"-db2={args.db2}",
        f"-db0to1={args.db0to1}",
        f"-db0to2={args.db0to2}",
        f"-cpu={args.cpu}",
        f"-fast={args.fast}",
        f"-tmpdir={tmp_dir}",
    ]
    if args.rmsa_timeout:
        cmd.append(f"-timeout={args.rmsa_timeout}")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="ignore") as log:
        result = subprocess.run(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=args.walltime_seconds,
            check=False,
        )
    if result.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="ignore")[-4000:]
        raise RuntimeError(f"rMSA failed with exit code {result.returncode}\n{tail}")
    if not valid_file(afa_path):
        raise FileNotFoundError(f"rMSA did not produce .afa output: {afa_path}")
    return afa_path


def should_import(entry: dict[str, Any], new_depth: int, import_query_only: bool) -> bool:
    reason = str(entry["reason"])
    if reason.startswith("existing_depth_le_"):
        return new_depth > int(entry.get("existing_depth", 0)) or import_query_only
    return new_depth > 0 and (new_depth > 1 or import_query_only)


def import_a3m(
    entry: dict[str, Any],
    a3m_text: str,
    mapping: dict[str, Any],
    mapping_path: Path,
) -> None:
    entity_id = str(entry["entity_id"])
    dst = Path(entry["target_msa_path"])
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(a3m_text, encoding="utf-8")

    key = str(entry["rna_key"])
    current = mapping.get(key)
    if current is None:
        mapping[key] = [entity_id]
    elif isinstance(current, str):
        mapping[key] = [entity_id] if current == entity_id else [entity_id, current]
    else:
        old_ids = [str(item) for item in current]
        mapping[key] = [entity_id] + [item for item in old_ids if item != entity_id]
    write_json_atomic(mapping_path, mapping)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--bioassembly-dir", type=Path, default=DEFAULT_BIOASSEMBLY_DIR)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=REPO_ROOT / "tmp" / "rmsa_short_rna_before202606",
    )
    parser.add_argument("--rmsa-dir", type=Path, default=DEFAULT_RMSA_DIR)
    parser.add_argument("--db0", default=str(DEFAULT_RMSA_DB_DIR / "Rfam.cm"))
    parser.add_argument("--db1", default=str(DEFAULT_RMSA_DB_DIR / "rnacentral.fasta"))
    parser.add_argument("--db2", default=str(DEFAULT_RMSA_DB_DIR / "nt"))
    parser.add_argument(
        "--db0to1", default=str(DEFAULT_RMSA_DB_DIR / "rfam_annotations.tsv.gz")
    )
    parser.add_argument(
        "--db0to2", default=str(DEFAULT_RMSA_DB_DIR / "Rfam.full_region.gz")
    )
    parser.add_argument("--max-length", type=int, default=150)
    parser.add_argument("--scan-workers", type=int, default=12)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--cpu", type=int, default=4, help="rMSA -cpu per sequence")
    parser.add_argument("--fast", type=int, default=2, help="rMSA -fast heuristic")
    parser.add_argument(
        "--rmsa-timeout",
        default=None,
        help="Passed to rMSA -timeout, e.g. 47h. Default: no rMSA timeout.",
    )
    parser.add_argument(
        "--walltime-seconds",
        type=int,
        default=None,
        help="Python subprocess timeout per sequence. Default: no timeout.",
    )
    parser.add_argument(
        "--min-existing-depth",
        type=int,
        default=1,
        help="Existing A3M depth <= this value is treated as no real MSA.",
    )
    parser.add_argument(
        "--missing-files-only",
        action="store_true",
        help="Only search missing index/file entries; ignore existing query-only A3Ms.",
    )
    parser.add_argument(
        "--import-query-only",
        action="store_true",
        help="Import rMSA output even when it contains only the query sequence.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Re-run existing work files.")
    parser.add_argument("--apply", action="store_true", help="Run rMSA and update cache.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)

    validate_rmsa_inputs(args)
    _, rna_sequences, errors = collect_unique_sequences(
        args.train_csv, args.bioassembly_dir, args.scan_workers
    )
    candidates = find_candidates(
        rna_sequences=rna_sequences,
        dataset_root=args.dataset_root,
        max_length=args.max_length,
        include_query_only=not args.missing_files_only,
        min_existing_depth=args.min_existing_depth,
    )
    if args.limit is not None:
        candidates = candidates[: args.limit]

    manifest = {
        "train_csv": str(args.train_csv),
        "bioassembly_dir": str(args.bioassembly_dir),
        "dataset_root": str(args.dataset_root),
        "max_length": args.max_length,
        "missing_files_only": bool(args.missing_files_only),
        "candidate_count": len(candidates),
        "scan_errors": errors,
        "entries": candidates,
    }
    atomic_write_text(
        args.work_dir / "rmsa_short_rna_candidates.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    write_candidates_fasta(args.work_dir / "rmsa_short_rna_candidates.fasta", candidates)
    print(
        f"Found {len(candidates)} candidate RNA sequences (<{args.max_length} nt). "
        f"Wrote candidate files under {args.work_dir}",
        flush=True,
    )

    if not args.apply:
        print("Dry run only. Add --apply to run rMSA and update the cache.", flush=True)
        return 0

    mapping_path = args.dataset_root / "rna_msa" / "rna_sequence_to_pdb_chains.json"
    mapping = load_json(mapping_path)
    backup = backup_once(mapping_path, "rmsa_short_rna_before202606")
    print(f"Backed up RNA MSA index to {backup}", flush=True)

    imported = 0
    skipped = 0
    failed = 0
    run_entries: list[dict[str, Any]] = []
    for i, entry in enumerate(candidates, 1):
        safe = hashlib.sha256(entry["sequence"].encode()).hexdigest()[:16]
        fasta_path = args.work_dir / "queries" / f"{i:05d}_{safe}.fasta"
        log_path = args.work_dir / "logs" / f"{i:05d}_{safe}.log"
        result_entry = dict(entry)
        try:
            afa_path = run_rmsa_one(args, entry, fasta_path, log_path)
            a3m_text = afa_to_a3m_text(afa_path)
            depth = a3m_depth(a3m_text)
            result_entry["rmsa_afa_path"] = str(afa_path)
            result_entry["rmsa_depth"] = depth
            result_entry["log_path"] = str(log_path)

            if should_import(entry, depth, args.import_query_only):
                import_a3m(entry, a3m_text, mapping, mapping_path)
                imported += 1
                result_entry["status"] = "imported"
            else:
                skipped += 1
                result_entry["status"] = "searched_not_imported"

            print(
                f"[{i}/{len(candidates)}] len={len(entry['sequence'])} "
                f"reason={entry['reason']} depth={depth} status={result_entry['status']}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - keep batch running per sequence
            failed += 1
            result_entry["status"] = "error"
            result_entry["error"] = str(exc)
            print(
                f"[{i}/{len(candidates)}] len={len(entry['sequence'])} "
                f"reason={entry['reason']} ERROR: {exc}",
                flush=True,
            )
        run_entries.append(result_entry)

    report = {
        "candidate_count": len(candidates),
        "imported": imported,
        "skipped": skipped,
        "failed": failed,
        "entries": run_entries,
    }
    atomic_write_text(
        args.work_dir / "rmsa_short_rna_run_report.json",
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )
    print(
        f"Done. imported={imported} skipped={skipped} failed={failed}. "
        f"Report: {args.work_dir / 'rmsa_short_rna_run_report.json'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
python -u scripts/msa/fill_train_rna_before202606_short_rna_with_rmsa.py \
    --apply \
    --missing-files-only \
    --cpu 4
"""