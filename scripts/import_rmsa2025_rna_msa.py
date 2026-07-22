#!/usr/bin/env python3
"""Import precomputed RMSA_2025 RNA MSAs into a Protenix RNA MSA cache.

Protenix training/inference expects RNA MSAs in this layout:

    rna_msa/
      rna_sequence_to_pdb_chains.json
      msas/{msa_id}/{msa_id}_all.a3m

The RMSA_2025 files are aligned FASTA files named like
``6MTE_8.MSA.fasta``. Protenix parses FASTA and A3M with the same parser, so
the import mostly normalizes the records and writes them with the expected
``.a3m`` name. If the query row contains gaps, query-gap columns are removed so
the first MSA row matches the ungapped RNA sequence used as the JSON key.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from protenix.data.constants import RNA_CHAIN
from protenix.data.msa.msa_utils import RawMsa
from protenix.data.tools.common import parse_fasta


LOGGER = logging.getLogger("import_rmsa2025_rna_msa")

DEFAULT_SOURCE_ROOT = Path(
    "/inspire/ssd/project/sais-bio/public/ash_proj/data/RMSA_2025/"
    "MSA_combined_RNA"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/inspire/ssd/project/sais-bio/public/xiangwenkai/GITHUB/Protenix/"
    "rna_data/rna_msa"
)
RNA_MSA_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ-")


@dataclass(frozen=True)
class ImportRecord:
    src_path: Path
    msa_id: str
    query: str
    a3m: str
    depth: int
    removed_query_gap_columns: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert RMSA_2025 *.MSA.fasta files into the Protenix RNA MSA "
            "cache layout and update rna_sequence_to_pdb_chains.json."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="Directory containing RMSA_2025 *.MSA.fasta files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Protenix RNA MSA cache root to update.",
    )
    parser.add_argument(
        "--glob",
        default="*.MSA.fasta",
        help="Glob pattern under --source-root.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N input files after sorting.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing destination {msa_id}_all.a3m file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and report what would be imported without writing files.",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip RawMsa.from_a3m(...).featurize() validation.",
    )
    parser.add_argument(
        "--no-backup-json",
        action="store_true",
        help="Do not create a timestamped backup before updating the JSON map.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp_path, path)


def as_id_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        return [str(v) for v in value]
    return [str(value)]


def safe_msa_id(src_path: Path) -> str:
    name = src_path.name
    if name.endswith(".MSA.fasta"):
        name = name[: -len(".MSA.fasta")]
    else:
        name = src_path.stem
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or src_path.stem


def msa_file_for(msa_root: Path, msa_id: str) -> Path:
    return msa_root / msa_id / f"{msa_id}_all.a3m"


def has_usable_msa(msa_root: Path, msa_id: str) -> bool:
    path = msa_file_for(msa_root, msa_id)
    return path.exists() and path.stat().st_size > 0


def normalize_msa_sequence(sequence: str) -> str:
    normalized = "".join(sequence.split()).upper().replace("T", "U")
    return "".join(char if char in RNA_MSA_CHARS else "X" for char in normalized)


def drop_query_gap_columns(sequences: Sequence[str]) -> tuple[list[str], int]:
    if not sequences:
        return [], 0
    query = sequences[0]
    gap_positions = {idx for idx, char in enumerate(query) if char == "-"}
    if not gap_positions:
        return list(sequences), 0

    cleaned = []
    for sequence in sequences:
        cleaned.append(
            "".join(
                char
                for idx, char in enumerate(sequence)
                if idx not in gap_positions
            )
        )
    return cleaned, len(gap_positions)


def to_a3m(descriptions: Sequence[str], sequences: Sequence[str]) -> str:
    lines: list[str] = []
    for idx, (desc, seq) in enumerate(zip(descriptions, sequences)):
        header = desc.strip() or f"sequence_{idx + 1}"
        lines.append(f">{header}")
        lines.append(seq)
    return "\n".join(lines) + "\n"


def read_import_record(src_path: Path, validate: bool) -> ImportRecord:
    content = src_path.read_text()
    sequences, descriptions = parse_fasta(content)
    if not sequences:
        raise ValueError("no FASTA records found")
    if len(sequences) != len(descriptions):
        raise ValueError(
            f"sequence/header count mismatch: {len(sequences)} seqs, "
            f"{len(descriptions)} headers"
        )

    normalized_sequences = [normalize_msa_sequence(seq) for seq in sequences]
    normalized_sequences, removed_gap_columns = drop_query_gap_columns(
        normalized_sequences
    )
    query = normalized_sequences[0].replace("-", "")
    if not query:
        raise ValueError("empty query after normalization")

    a3m = to_a3m(descriptions, normalized_sequences)
    if validate:
        RawMsa.from_a3m(query, RNA_CHAIN, a3m, dedup=False).featurize()

    return ImportRecord(
        src_path=src_path,
        msa_id=safe_msa_id(src_path),
        query=query,
        a3m=a3m,
        depth=len(normalized_sequences),
        removed_query_gap_columns=removed_gap_columns,
    )


def update_mapping(
    mapping: dict[str, Any],
    sequence: str,
    msa_id: str,
    msa_root: Path,
) -> bool:
    """Merge msa_id into mapping[sequence].

    Existing usable first IDs keep priority. If the existing first ID is missing
    or empty, the imported ID is promoted to first so Protenix can use it.
    Returns True if the map changed.
    """
    existing_ids = as_id_list(mapping.get(sequence))
    if msa_id in existing_ids:
        if existing_ids[0] == msa_id or has_usable_msa(msa_root, existing_ids[0]):
            return False
        new_ids = [msa_id] + [item for item in existing_ids if item != msa_id]
        if new_ids == existing_ids:
            return False
        mapping[sequence] = new_ids
        return True

    deduped_existing = [item for item in existing_ids if item != msa_id]
    keep_existing_first = bool(
        deduped_existing and has_usable_msa(msa_root, deduped_existing[0])
    )
    if keep_existing_first:
        new_ids = deduped_existing + [msa_id]
    else:
        new_ids = [msa_id] + deduped_existing

    if new_ids == existing_ids:
        return False
    mapping[sequence] = new_ids
    return True


def backup_json(path: Path) -> Path | None:
    if not path.exists():
        return None
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_path = path.with_name(f"{path.name}.bak.{timestamp}")
    shutil.copy2(path, backup_path)
    return backup_path


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    map_path = output_root / "rna_sequence_to_pdb_chains.json"
    msa_root = output_root / "msas"
    validate = not args.no_validate

    if not source_root.exists():
        raise FileNotFoundError(f"source root not found: {source_root}")

    files = sorted(source_root.glob(args.glob))
    if args.limit is not None:
        files = files[: args.limit]
    LOGGER.info("source_root=%s", source_root)
    LOGGER.info("output_root=%s", output_root)
    LOGGER.info(
        "selected_files=%d validate=%s dry_run=%s",
        len(files),
        validate,
        args.dry_run,
    )

    mapping = load_json(map_path)
    imported = 0
    skipped_existing_file = 0
    map_updates = 0
    failed = 0
    query_gap_files = 0

    for index, src_path in enumerate(files, start=1):
        try:
            record = read_import_record(src_path, validate=validate)
        except Exception as exc:  # noqa: BLE001 - keep batch import resumable.
            failed += 1
            LOGGER.warning(
                "[%d/%d] skip invalid %s: %s",
                index,
                len(files),
                src_path.name,
                exc,
            )
            continue

        if record.removed_query_gap_columns:
            query_gap_files += 1

        dest_path = msa_file_for(msa_root, record.msa_id)
        file_exists = dest_path.exists() and dest_path.stat().st_size > 0
        if file_exists and not args.overwrite:
            action = "skip_existing_file"
            skipped_existing_file += 1
        elif not args.dry_run:
            action = "processed"
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            dest_path.write_text(record.a3m)
            imported += 1
        else:
            action = "would_import"
            imported += 1

        if update_mapping(mapping, record.query, record.msa_id, msa_root):
            map_updates += 1

        LOGGER.info(
            "[%d/%d] %s id=%s depth=%d query_len=%d removed_query_gap_cols=%d",
            index,
            len(files),
            action,
            record.msa_id,
            record.depth,
            len(record.query),
            record.removed_query_gap_columns,
        )

    backup_path = None
    if not args.dry_run and map_updates:
        if not args.no_backup_json:
            backup_path = backup_json(map_path)
        write_json_atomic(map_path, mapping)

    LOGGER.info(
        "done: selected=%d imported_or_would_import=%d skipped_existing_file=%d "
        "map_updates=%d query_gap_files=%d failed=%d",
        len(files),
        imported,
        skipped_existing_file,
        map_updates,
        query_gap_files,
        failed,
    )
    if backup_path is not None:
        LOGGER.info("backup_json=%s", backup_path)
    if args.dry_run:
        LOGGER.info("dry run only; no files or JSON were written")


if __name__ == "__main__":
    main()
