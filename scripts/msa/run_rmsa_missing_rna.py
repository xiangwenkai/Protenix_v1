#!/usr/bin/env python3
"""Run rMSA search for unmapped short RNA sequences and import results.

This script finds RNA sequences in ``train_rna_before202606`` that are shorter
than a configurable length and have no entry in the Protenix RNA MSA cache
(``rna_sequence_to_pdb_chains.json``), runs the rMSA pipeline for each of them,
converts the resulting aligned FASTA (.afa) into the A3M format used by the
Protenix RNA MSA cache, and (optionally) imports the result into the cache.

Default rMSA invocation (mirrors the user-provided command)::

    ./rMSA.fast.pl <seq>.fasta \
        -db0=.../Rfam.cm \
        -db1=.../rnacentral.fasta \
        -db2=.../nt \
        -db0to1=.../rfam_annotations.tsv.gz \
        -db0to2=.../Rfam.full_region.gz

The script is dry-run by default. Pass ``--apply`` to actually copy A3M files
into the cache and update the mapping JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

DEFAULT_INDICES = Path(
    "/inspire/ssd/project/sais-bio/public/xiangwenkai/GITHUB/"
    "Protenix_v1/data/protein_rna_train_filter.csv"
)
DEFAULT_MAPPING = Path(
    "/inspire/ssd/project/sais-bio/public/Protein/data/AI_Models/"
    "protenix_v1_dataset/rna_msa/rna_sequence_to_pdb_chains.json"
)
DEFAULT_MSA_DIR = Path(
    "/inspire/ssd/project/sais-bio/public/Protein/data/AI_Models/"
    "protenix_v1_dataset/rna_msa/msas"
)
DEFAULT_RMSA_DIR = Path(
    "/inspire/ssd/project/sais-bio/public/xiangwenkai/rna_pipline/rMSA"
)
DEFAULT_WORK_DIR = Path("output/rmsa_missing_rna")
DEFAULT_REPORT = Path("output/rmsa_missing_rna_report.json")

DEFAULT_DB0 = "/inspire/ssd/project/sais-bio/public/ash_proj/code/rMSA/database/Rfam.cm"
DEFAULT_DB1 = (
    "/inspire/ssd/project/sais-bio/public/ash_proj/code/rMSA/"
    "database/rnacentral.fasta"
)
DEFAULT_DB2 = "/inspire/ssd/project/sais-bio/public/ash_proj/code/rMSA/database/nt"
DEFAULT_DB0TO1 = (
    "/inspire/ssd/project/sais-bio/public/ash_proj/code/rMSA/"
    "database/rfam_annotations.tsv.gz"
)
DEFAULT_DB0TO2 = (
    "/inspire/ssd/project/sais-bio/public/ash_proj/code/rMSA/"
    "database/Rfam.full_region.gz"
)


def normalize_rna(sequence: str) -> str:
    """Strip non-letter characters and uppercase (keep T/U as-is)."""
    return re.sub(r"[^A-Za-z]", "", sequence).upper()


def _in_mapping(sequence: str, mapping: dict[str, Any]) -> bool:
    """Mirror the exact-match lookup used by MSASourceManager.fetch_msas."""
    if sequence in mapping:
        return True
    # RNA MSA lookups in the training pipeline are exact-match only; these two
    # fallbacks are kept for robustness when the CSV sequence uses T vs U.
    return sequence.replace("T", "U") in mapping or sequence.replace("U", "T") in mapping


def load_unmapped_short_rna(
    indices_path: Path,
    mapping: dict[str, Any],
    max_length: int,
    limit: Optional[int] = None,
) -> list[tuple[str, int]]:
    """Return (sequence, chain_count) for unmapped short RNA sequences."""
    import pandas as pd

    df = pd.read_csv(indices_path)
    rna = df[df["sub_mol_1_type"] == "rna"]
    counts = rna["cluster_1_id"].astype(str).value_counts()

    candidates: list[tuple[str, int]] = []
    for sequence, count in counts.items():
        if len(sequence) >= max_length:
            continue
        if _in_mapping(sequence, mapping):
            continue
        candidates.append((sequence, int(count)))

    candidates.sort(key=lambda item: (len(item[0]), item[0]))
    if limit is not None:
        candidates = candidates[:limit]
    return candidates


def read_fasta_records(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    name = ""
    seq_chunks: list[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name:
                    records.append((name, "".join(seq_chunks)))
                name = line[1:].strip()
                seq_chunks = []
            else:
                seq_chunks.append(line)
    if name:
        records.append((name, "".join(seq_chunks)))
    return records


def afa_to_a3m(afa_path: Path) -> str:
    """Convert an aligned FASTA (.afa) produced by rMSA to A3M.

    Follows the same rules as the bundled ``afa2a3m.pl``:
    columns where the query has a residue are match states (uppercase),
    columns where the query has a gap are insertion states (lowercase).
    T is converted back to U to match the Protenix RNA MSA cache convention.
    """
    records = read_fasta_records(afa_path)
    if not records:
        raise ValueError(f"Empty alignment: {afa_path}")

    query = records[0][1].upper()
    aligned_len = len(query)
    is_match = [char != "-" for char in query]

    out_lines: list[str] = []
    for name, seq in records:
        seq = seq.upper()
        if len(seq) != aligned_len:
            raise ValueError(
                f"Sequence {name} length {len(seq)} != alignment length {aligned_len}"
            )
        converted: list[str] = []
        for idx, char in enumerate(seq):
            if is_match[idx]:
                converted.append(char)
            elif char != "-":
                converted.append(char.lower())
        a3m_seq = "".join(converted).replace("T", "U").replace("t", "u")
        out_lines.append(f">{name}")
        out_lines.append(a3m_seq)
    return "\n".join(out_lines) + "\n"


def a3m_depth(a3m_text: str) -> int:
    return sum(1 for line in a3m_text.splitlines() if line.startswith(">"))


def msa_id_for_sequence(sequence: str, existing_ids: set[str]) -> str:
    digest = hashlib.sha1(sequence.encode("utf-8")).hexdigest()[:12]
    candidate = f"rmsa_{digest}"
    while candidate in existing_ids:
        candidate = f"{candidate}_x"
    return candidate


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)


def write_fasta(path: Path, sequence: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f">query\n{sequence}\n")


def backup_mapping(mapping_path: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = mapping_path.with_name(
        f"{mapping_path.stem}.bak.rmsa_missing_rna_{timestamp}.json"
    )
    shutil.copy2(mapping_path, backup)
    return backup


def collect_existing_msa_ids(mapping: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    for value in mapping.values():
        if isinstance(value, list):
            ids.update(str(item) for item in value)
        else:
            ids.add(str(value))
    return ids


def run_rmsa(
    rmsa_dir: Path,
    fasta_path: Path,
    db0: str,
    db1: str,
    db2: str,
    db0to1: str,
    db0to2: str,
    cpu: int,
    fast: int,
    timeout_seconds: Optional[int],
    log_path: Path,
) -> Path:
    """Run rMSA.fast.pl for one query FASTA and return the .afa output path."""
    script = rmsa_dir / "rMSA.fast.pl"
    if not script.exists():
        raise FileNotFoundError(f"rMSA script not found: {script}")

    prefix = fasta_path.parent / fasta_path.name.split(".")[0]
    afa_path = Path(f"{prefix}.afa")

    cmd = [
        "perl",
        str(script),
        str(fasta_path),
        f"-db0={db0}",
        f"-db1={db1}",
        f"-db2={db2}",
        f"-db0to1={db0to1}",
        f"-db0to2={db0to2}",
        f"-cpu={cpu}",
        f"-fast={fast}",
        f"-tmpdir={fasta_path.parent / 'tmp' / prefix.name}",
    ]

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="ignore") as handle:
        result = subprocess.run(
            cmd,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    if result.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="ignore")[-4000:]
        raise RuntimeError(
            f"rMSA failed with exit code {result.returncode}: {' '.join(cmd)}\n{tail}"
        )
    if not afa_path.exists():
        raise FileNotFoundError(f"rMSA did not produce expected output: {afa_path}")
    return afa_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--indices", type=Path, default=DEFAULT_INDICES)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--msa-dir", type=Path, default=DEFAULT_MSA_DIR)
    parser.add_argument("--rmsa-dir", type=Path, default=DEFAULT_RMSA_DIR)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-length", type=int, default=100)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--cpu", type=int, default=0, help="rMSA -cpu (0 = auto)")
    parser.add_argument("--fast", type=int, default=2, help="rMSA heuristic level")
    parser.add_argument("--timeout", type=int, default=None, help="per-sequence seconds")
    parser.add_argument("--db0", default=DEFAULT_DB0)
    parser.add_argument("--db1", default=DEFAULT_DB1)
    parser.add_argument("--db2", default=DEFAULT_DB2)
    parser.add_argument("--db0to1", default=DEFAULT_DB0TO1)
    parser.add_argument("--db0to2", default=DEFAULT_DB0TO2)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually copy A3M files and update the mapping. Default is dry-run.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run rMSA even if an output .afa already exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
    if not isinstance(mapping, dict):
        raise TypeError(f"Mapping is not a JSON object: {args.mapping}")

    candidates = load_unmapped_short_rna(
        args.indices,
        mapping,
        max_length=args.max_length,
        limit=args.limit,
    )
    print(f"Found {len(candidates)} unmapped RNA sequences (<{args.max_length}nt)")

    existing_ids = collect_existing_msa_ids(mapping)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, Any]] = []
    depth_counter: Counter = Counter()
    status_counter: Counter = Counter()
    new_mapping_entries: list[tuple[str, str, int]] = []

    for index, (sequence, chain_count) in enumerate(candidates, start=1):
        entry: dict[str, Any] = {
            "sequence_length": len(sequence),
            "chain_count": chain_count,
            "sequence": sequence,
        }
        safe_stem = f"{index:04d}_{hashlib.sha1(sequence.encode('utf-8')).hexdigest()[:10]}"
        fasta_path = args.work_dir / "fasta" / f"{safe_stem}.fasta"
        afa_path = args.work_dir / "fasta" / f"{safe_stem}.afa"
        log_path = args.work_dir / "logs" / f"{safe_stem}.log"

        try:
            if not args.overwrite and afa_path.exists():
                entry["status"] = "reused"
            else:
                write_fasta(fasta_path, normalize_rna(sequence))
                run_rmsa(
                    rmsa_dir=args.rmsa_dir,
                    fasta_path=fasta_path,
                    db0=args.db0,
                    db1=args.db1,
                    db2=args.db2,
                    db0to1=args.db0to1,
                    db0to2=args.db0to2,
                    cpu=args.cpu,
                    fast=args.fast,
                    timeout_seconds=args.timeout,
                    log_path=log_path,
                )
                entry["status"] = "searched"

            a3m_text = afa_to_a3m(afa_path)
            depth = a3m_depth(a3m_text)
            depth_counter[depth] += 1
            entry["msa_depth"] = depth
            entry["actual_hit"] = depth > 1

            new_id = msa_id_for_sequence(sequence, existing_ids)
            entry["new_msa_id"] = new_id
            entry["a3m_path"] = str(args.msa_dir / new_id / f"{new_id}_all.a3m")

            if args.apply:
                dst = args.msa_dir / new_id / f"{new_id}_all.a3m"
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_text(a3m_text, encoding="utf-8")
                mapping.setdefault(sequence, [])
                if new_id not in [str(item) for item in mapping[sequence]]:
                    mapping[sequence] = [new_id] + [
                        str(item) for item in mapping[sequence]
                    ]
                existing_ids.add(new_id)
                new_mapping_entries.append((sequence, new_id, depth))
                entry["status"] += "+imported"
            else:
                entry["status"] += "+dry_run"

            status_counter[entry["status"]] += 1
            print(
                f"[{index}/{len(candidates)}] len={len(sequence)} "
                f"depth={depth} chains={chain_count} status={entry['status']}"
            )
        except Exception as exc:  # noqa: BLE001 - per-sequence isolation
            entry["status"] = "error"
            entry["error"] = str(exc)
            status_counter["error"] += 1
            print(f"[{index}/{len(candidates)}] len={len(sequence)} ERROR: {exc}")

        entries.append(entry)

    if args.apply and new_mapping_entries:
        backup = backup_mapping(args.mapping)
        atomic_write_text(
            args.mapping,
            json.dumps(mapping, indent=2, sort_keys=True) + "\n",
        )
        print(f"Backup written to {backup}")
        print(f"Updated mapping with {len(new_mapping_entries)} new entries")

    total = len(entries)
    file_hit = sum(1 for e in entries if e.get("msa_depth", 0) >= 1)
    actual_hit = sum(1 for e in entries if e.get("actual_hit"))
    report = {
        "applied": bool(args.apply),
        "indices": str(args.indices),
        "mapping": str(args.mapping),
        "msa_dir": str(args.msa_dir),
        "max_length": args.max_length,
        "candidate_count": total,
        "file_hit": file_hit,
        "file_hit_rate": round(100 * file_hit / total, 2) if total else 0.0,
        "actual_hit": actual_hit,
        "actual_hit_rate": round(100 * actual_hit / total, 2) if total else 0.0,
        "depth_distribution": dict(sorted(depth_counter.items())),
        "status_distribution": dict(status_counter),
        "entries": entries,
    }
    atomic_write_text(args.report_path, json.dumps(report, indent=2) + "\n")
    print(f"Report written to {args.report_path}")
    print(
        f"file_hit={file_hit}/{total} ({report['file_hit_rate']}%)  "
        f"actual_hit={actual_hit}/{total} ({report['actual_hit_rate']}%)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
