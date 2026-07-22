#!/usr/bin/env python3
"""Repair protein MSA paths in FoldBench PXMeter input JSON files.

This script scans local Protenix/PXMeter input JSONs for protein-chain entries
whose paired/unpaired MSA paths are both present on disk and whose first FASTA
record exactly matches the declared protein sequence. It then uses that
sequence-indexed lookup to repair generated FoldBench ``pxmeter_inputs`` JSONs
whose protein MSA paths are missing or point to mismatched queries.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEARCH_ROOTS = [REPO_ROOT / "examples", REPO_ROOT / "output"]
DEFAULT_TARGET_ROOT = REPO_ROOT / "output"
FOLDBENCH_INPUT_GLOB = "*/foldbench_validation/*/pxmeter_inputs/*/input_json/*.json"


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


def resolve_existing_path(path_text: object, base_dir: Path | None = None) -> Path | None:
    if not isinstance(path_text, str) or not path_text.strip():
        return None
    path = Path(path_text).expanduser()
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        if base_dir is not None:
            candidates.append(base_dir / path)
        candidates.append(REPO_ROOT / path)
        candidates.append(Path.cwd() / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


@dataclass(frozen=True)
class MsaPair:
    paired: str
    unpaired: str
    source_json: str


def validate_msa_pair(
    sequence: str,
    paired_path: Path | None,
    unpaired_path: Path | None,
) -> bool:
    if paired_path is None or unpaired_path is None:
        return False
    expected = normalize_sequence(sequence)
    paired_query = read_first_fasta_sequence(paired_path)
    unpaired_query = read_first_fasta_sequence(unpaired_path)
    if paired_query is None or unpaired_query is None:
        return False
    return normalize_sequence(paired_query) == expected and normalize_sequence(unpaired_query) == expected


def iter_json_tasks(path: Path) -> list[dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return []
    return [task for task in data if isinstance(task, dict)]


def candidate_rank(pair: MsaPair) -> tuple[int, int, str]:
    preferred_prefixes = [
        str((REPO_ROOT / "examples" / "sft" / "pxmeter_other_targets" / "output").resolve()),
        str(
            (
                REPO_ROOT
                / "examples"
                / "sft"
                / "pxmeter_other_targets"
                / "interface_protein_rna"
                / "output"
            ).resolve()
        ),
        str((REPO_ROOT / "output").resolve()),
    ]
    rank = len(preferred_prefixes)
    for idx, prefix in enumerate(preferred_prefixes):
        if pair.paired.startswith(prefix):
            rank = idx
            break
    return (rank, len(pair.paired), pair.paired)


def build_sequence_lookup(search_roots: list[Path]) -> dict[str, MsaPair]:
    candidates_by_seq: dict[str, dict[tuple[str, str], MsaPair]] = defaultdict(dict)
    for root in search_roots:
        if not root.exists():
            continue
        for input_dir in root.rglob("input_json"):
            if not input_dir.is_dir():
                continue
            for json_path in input_dir.glob("*.json"):
                for task in iter_json_tasks(json_path):
                    for sequence_entry in task.get("sequences", []):
                        protein_chain = sequence_entry.get("proteinChain")
                        if not isinstance(protein_chain, dict):
                            continue
                        sequence = normalize_sequence(str(protein_chain.get("sequence", "")))
                        if not sequence:
                            continue
                        paired_path = resolve_existing_path(
                            protein_chain.get("pairedMsaPath"), base_dir=json_path.parent
                        )
                        unpaired_path = resolve_existing_path(
                            protein_chain.get("unpairedMsaPath"), base_dir=json_path.parent
                        )
                        if not validate_msa_pair(sequence, paired_path, unpaired_path):
                            continue
                        pair = MsaPair(
                            paired=str(paired_path),
                            unpaired=str(unpaired_path),
                            source_json=str(json_path.resolve()),
                        )
                        candidates_by_seq[sequence][(pair.paired, pair.unpaired)] = pair

    resolved: dict[str, MsaPair] = {}
    for sequence, pair_map in candidates_by_seq.items():
        resolved[sequence] = sorted(pair_map.values(), key=candidate_rank)[0]
    return resolved


def repair_json_file(path: Path, lookup: dict[str, MsaPair]) -> dict[str, int]:
    stats = {
        "files_scanned": 1,
        "protein_entries": 0,
        "valid_unchanged": 0,
        "fixed": 0,
        "unresolved": 0,
    }
    tasks = iter_json_tasks(path)
    if not tasks:
        return stats

    changed = False
    for task in tasks:
        for sequence_entry in task.get("sequences", []):
            protein_chain = sequence_entry.get("proteinChain")
            if not isinstance(protein_chain, dict):
                continue
            stats["protein_entries"] += 1
            sequence = normalize_sequence(str(protein_chain.get("sequence", "")))
            if not sequence:
                stats["unresolved"] += 1
                continue
            paired_path = resolve_existing_path(protein_chain.get("pairedMsaPath"), base_dir=path.parent)
            unpaired_path = resolve_existing_path(
                protein_chain.get("unpairedMsaPath"), base_dir=path.parent
            )
            if validate_msa_pair(sequence, paired_path, unpaired_path):
                stats["valid_unchanged"] += 1
                continue
            replacement = lookup.get(sequence)
            if replacement is None:
                stats["unresolved"] += 1
                continue
            protein_chain["pairedMsaPath"] = replacement.paired
            protein_chain["unpairedMsaPath"] = replacement.unpaired
            stats["fixed"] += 1
            changed = True

    if changed:
        path.write_text(json.dumps(tasks, indent=4) + "\n", encoding="utf-8")
    return stats


def find_target_jsons(target_root: Path) -> list[Path]:
    return sorted(target_root.glob(FOLDBENCH_INPUT_GLOB))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair protein MSA paths in generated FoldBench PXMeter input JSONs."
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        default=DEFAULT_TARGET_ROOT,
        help="Root directory under which foldbench_validation pxmeter_inputs are searched.",
    )
    parser.add_argument(
        "--search-root",
        action="append",
        type=Path,
        default=None,
        help="Additional roots whose input_json directories are scanned for valid protein MSA paths.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Optional path to write the aggregate repair summary as JSON.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the lookup and report what would be fixed without modifying files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    search_roots = list(DEFAULT_SEARCH_ROOTS)
    if args.search_root:
        search_roots.extend(args.search_root)
    search_roots = [root.resolve() for root in search_roots]
    target_root = args.target_root.resolve()

    lookup = build_sequence_lookup(search_roots)
    target_jsons = find_target_jsons(target_root)

    summary = {
        "lookup_sequences": len(lookup),
        "target_files": len(target_jsons),
        "files_scanned": 0,
        "protein_entries": 0,
        "valid_unchanged": 0,
        "fixed": 0,
        "unresolved": 0,
        "dry_run": bool(args.dry_run),
    }

    for path in target_jsons:
        stats = repair_json_file(path, lookup) if not args.dry_run else repair_json_file_stats(path, lookup)
        for key in ("files_scanned", "protein_entries", "valid_unchanged", "fixed", "unresolved"):
            summary[key] += stats[key]

    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))


def repair_json_file_stats(path: Path, lookup: dict[str, MsaPair]) -> dict[str, int]:
    stats = {
        "files_scanned": 1,
        "protein_entries": 0,
        "valid_unchanged": 0,
        "fixed": 0,
        "unresolved": 0,
    }
    for task in iter_json_tasks(path):
        for sequence_entry in task.get("sequences", []):
            protein_chain = sequence_entry.get("proteinChain")
            if not isinstance(protein_chain, dict):
                continue
            stats["protein_entries"] += 1
            sequence = normalize_sequence(str(protein_chain.get("sequence", "")))
            if not sequence:
                stats["unresolved"] += 1
                continue
            paired_path = resolve_existing_path(protein_chain.get("pairedMsaPath"), base_dir=path.parent)
            unpaired_path = resolve_existing_path(
                protein_chain.get("unpairedMsaPath"), base_dir=path.parent
            )
            if validate_msa_pair(sequence, paired_path, unpaired_path):
                stats["valid_unchanged"] += 1
            elif sequence in lookup:
                stats["fixed"] += 1
            else:
                stats["unresolved"] += 1
    return stats


if __name__ == "__main__":
    main()
