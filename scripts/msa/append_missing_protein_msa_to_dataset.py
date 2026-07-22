#!/usr/bin/env python3
"""Append missing protein MSA entries into the dataset MSA namespace.

This script compares a source sequence->MSA-ID index against the target
protenix_v1_dataset index. For each protein sequence that exists in the source
index but not in the target index, it validates the source MSA query sequences,
copies the MSA directory into the target namespace with a new appended ID, and
updates the target ``seq_to_pdb_index.json``.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_TARGET_INDEX = Path(
    "/inspire/ssd/project/sais-bio/public/Protein/data/AI_Models/protenix_v1_dataset/common/seq_to_pdb_index.json"
)
DEFAULT_TARGET_MSA_DIR = Path(
    "/inspire/ssd/project/sais-bio/public/Protein/data/AI_Models/protenix_v1_dataset/mmcif_msa_template"
)
DEFAULT_SOURCE_INDEX = Path(
    "/inspire/ssd/project/sais-bio/public/xiangwenkai/GITHUB/Protenix_v1/data/pdb_seqs/seq_to_pdb_index.json"
)
DEFAULT_SOURCE_MSA_DIR = Path(
    "/inspire/ssd/project/sais-bio/public/xiangwenkai/GITHUB/Protenix_v1/data/mmcif_msa"
)
DEFAULT_REPORT = Path("output/append_missing_protein_msa_report.json")


def normalize_sequence(sequence: str) -> str:
    return re.sub(r"[^A-Za-z]", "", sequence).upper()


def read_first_fasta_sequence(path: Path) -> str | None:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            seen_header = False
            chunks: list[str] = []
            for line in handle:
                if line.startswith(">"):
                    if seen_header:
                        break
                    seen_header = True
                    continue
                if seen_header:
                    chunks.append(line.strip())
    except FileNotFoundError:
        return None
    if not chunks:
        return None
    return "".join(chunks)


def load_index(path: Path) -> dict[str, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Index is not a JSON object: {path}")
    return {str(sequence): int(msa_id) for sequence, msa_id in data.items()}


def atomic_write_text(path: Path, text: str) -> None:
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)


@dataclass(frozen=True)
class Candidate:
    sequence: str
    source_id: int
    paired_path: str
    unpaired_path: str
    new_id: int


def validate_source_entry(sequence: str, source_dir: Path) -> tuple[str, str]:
    paired_path = source_dir / "pairing.a3m"
    unpaired_path = source_dir / "non_pairing.a3m"
    if not paired_path.exists() or not unpaired_path.exists():
        raise FileNotFoundError(f"Missing MSA files under {source_dir}")

    expected = normalize_sequence(sequence)
    paired_query = normalize_sequence(read_first_fasta_sequence(paired_path) or "")
    unpaired_query = normalize_sequence(read_first_fasta_sequence(unpaired_path) or "")
    if paired_query != expected or unpaired_query != expected:
        raise ValueError(
            f"MSA query mismatch for {source_dir}: expected len={len(expected)}, "
            f"paired len={len(paired_query)}, unpaired len={len(unpaired_query)}"
        )
    return str(paired_path), str(unpaired_path)


def build_missing_candidates(
    target_index: dict[str, int],
    source_index: dict[str, int],
    source_msa_dir: Path,
) -> tuple[list[Candidate], dict[str, int]]:
    target_sequences = {normalize_sequence(sequence) for sequence in target_index}
    next_id = max(target_index.values()) + 1
    candidates: list[Candidate] = []
    stats = {
        "target_entries_before": len(target_index),
        "source_entries": len(source_index),
        "missing_sequences": 0,
        "validated_candidates": 0,
    }

    for sequence, source_id in sorted(source_index.items(), key=lambda item: item[1]):
        normalized = normalize_sequence(sequence)
        if normalized in target_sequences:
            continue
        paired_path, unpaired_path = validate_source_entry(sequence, source_msa_dir / str(source_id))
        candidates.append(
            Candidate(
                sequence=sequence,
                source_id=source_id,
                paired_path=paired_path,
                unpaired_path=unpaired_path,
                new_id=next_id,
            )
        )
        target_sequences.add(normalized)
        next_id += 1

    stats["missing_sequences"] = len(candidates)
    stats["validated_candidates"] = len(candidates)
    stats["target_entries_after"] = len(target_index) + len(candidates)
    stats["new_id_start"] = candidates[0].new_id if candidates else next_id
    stats["new_id_end"] = candidates[-1].new_id if candidates else next_id - 1
    return candidates, stats


def backup_index(path: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = path.with_name(f"{path.stem}.bak.append_missing_protein_msa_{timestamp}.json")
    shutil.copy2(path, backup_path)
    return backup_path


def write_report(report_path: Path, payload: dict) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(report_path, json.dumps(payload, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-index", type=Path, default=DEFAULT_TARGET_INDEX)
    parser.add_argument("--target-msa-dir", type=Path, default=DEFAULT_TARGET_MSA_DIR)
    parser.add_argument("--source-index", type=Path, default=DEFAULT_SOURCE_INDEX)
    parser.add_argument("--source-msa-dir", type=Path, default=DEFAULT_SOURCE_MSA_DIR)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually copy MSA directories and update the target index. Default is dry-run.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    target_index = load_index(args.target_index)
    source_index = load_index(args.source_index)
    candidates, stats = build_missing_candidates(target_index, source_index, args.source_msa_dir)

    backup_path: Path | None = None
    if args.apply and candidates:
        for candidate in candidates:
            dst_dir = args.target_msa_dir / str(candidate.new_id)
            if dst_dir.exists():
                raise FileExistsError(f"Target MSA directory already exists: {dst_dir}")
        backup_path = backup_index(args.target_index)
        for candidate in candidates:
            dst_dir = args.target_msa_dir / str(candidate.new_id)
            shutil.copytree(args.source_msa_dir / str(candidate.source_id), dst_dir)
            target_index[candidate.sequence] = candidate.new_id
        atomic_write_text(args.target_index, json.dumps(target_index, indent=2) + "\n")

    report = {
        "applied": bool(args.apply),
        "target_index": str(args.target_index),
        "target_msa_dir": str(args.target_msa_dir),
        "source_index": str(args.source_index),
        "source_msa_dir": str(args.source_msa_dir),
        "backup_index": str(backup_path) if backup_path else None,
        "stats": stats,
        "entries": [
            {
                "sequence_length": len(normalize_sequence(candidate.sequence)),
                "source_id": candidate.source_id,
                "new_id": candidate.new_id,
                "paired_path": candidate.paired_path,
                "unpaired_path": candidate.unpaired_path,
            }
            for candidate in candidates
        ],
    }
    write_report(args.report_path, report)

    print(json.dumps({"applied": bool(args.apply), **stats}, indent=2))
    print(f"Report written to {args.report_path}")
    if backup_path is not None:
        print(f"Backup written to {backup_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
