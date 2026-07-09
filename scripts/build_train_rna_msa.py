#!/usr/bin/env python3
"""Build missing RNA MSA cache for Protenix training datasets.

The training data loader does not read RNA MSA features from the pkl files.
Instead, it resolves RNA sequences through:

    rna_sequence_to_pdb_chains.json
    msas/{pdb_entity_id}/{pdb_entity_id}_all.a3m

This script scans a Protenix indices CSV, reuses any RNA MSA already present in
the configured cache, and searches missing RNA sequences with nhmmer.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.configs_data import RNA_DATA_ROOT_DIR, data_configs
from protenix.config.extend_types import ListValue
from runner.rna_msa_search import run_rna_msa_search


LOGGER = logging.getLogger("build_train_rna_msa")
RNA_ALPHABET = set("AUGCN")


@dataclass(frozen=True)
class RnaEntry:
    sequence: str
    search_sequence: str
    pdb_id: str
    entity_id: str
    chain_id: str
    row_index: int
    source_side: str


@dataclass(frozen=True)
class MsaSource:
    map_path: Path
    msa_root: Path
    mapping: dict[str, Any]


def unwrap_config_value(value: Any) -> Any:
    if isinstance(value, ListValue):
        return value.value
    return value


def default_train_csv() -> Path:
    return Path(data_configs["train_rna_before202606"]["base_info"]["indices_fpath"])


def default_existing_maps() -> list[Path]:
    return [
        Path(p)
        for p in unwrap_config_value(
            data_configs["msa"]["rna_seq_or_filename_to_msadir_jsons"]
        )
    ]


def default_existing_roots() -> list[Path]:
    return [
        Path(p)
        for p in unwrap_config_value(data_configs["msa"]["rna_msadir_raw_paths"])
    ]


def default_output_root() -> Path:
    return Path(RNA_DATA_ROOT_DIR) / "rna_msa_extra"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search missing RNA MSAs for a Protenix training indices CSV. "
            "Existing Protenix RNA MSAs are reused; only missing sequences are "
            "searched and written to an extra training-compatible RNA MSA cache."
        )
    )
    parser.add_argument(
        "--indices-csv",
        type=Path,
        default=default_train_csv(),
        help="Protenix indices CSV to scan. Defaults to train_rna_before202606.",
    )
    parser.add_argument(
        "--existing-map",
        type=Path,
        action="append",
        default=None,
        help=(
            "Existing rna_sequence_to_pdb_chains.json. May be repeated. "
            "Defaults to configs_data.py RNA MSA map."
        ),
    )
    parser.add_argument(
        "--existing-root",
        type=Path,
        action="append",
        default=None,
        help=(
            "Existing RNA MSA msas/ root paired with --existing-map. May be "
            "repeated. Defaults to configs_data.py RNA MSA root."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=default_output_root(),
        help=(
            "Output RNA MSA cache root. The script writes msas/ and "
            "rna_sequence_to_pdb_chains.json under this directory."
        ),
    )
    parser.add_argument(
        "--nhmmer-n-cpu",
        type=int,
        default=None,
        help="CPUs for each nhmmer search. runner/rna_msa_search.py defaults to <=8.",
    )
    parser.add_argument("--nhmmer-binary-path", type=str, default=None)
    parser.add_argument("--hmmalign-binary-path", type=str, default=None)
    parser.add_argument("--hmmbuild-binary-path", type=str, default=None)
    parser.add_argument("--ntrna-database-path", type=str, default=None)
    parser.add_argument("--rfam-database-path", type=str, default=None)
    parser.add_argument("--rna-central-database-path", type=str, default=None)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N unique RNA sequences after filtering.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-search sequences that already exist in the output cache.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report what would be reused or searched; do not write files.",
    )
    parser.add_argument(
        "--keep-tmp",
        action="store_true",
        help="Keep temporary rna_msa.a3m search outputs under output-root/_tmp_search.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def normalize_key_sequence(sequence: str) -> str:
    return sequence.strip().upper()


def make_search_sequence(sequence: str) -> str | None:
    """Return an nhmmer-friendly RNA query while preserving map keys elsewhere."""
    normalized = normalize_key_sequence(sequence).replace("T", "U")
    search_chars = []
    for char in normalized:
        if char in RNA_ALPHABET:
            search_chars.append(char)
        elif "A" <= char <= "Z":
            search_chars.append("N")
        else:
            return None
    return "".join(search_chars) if search_chars else None


def safe_id(raw_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_id.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "rna_seq"


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


def msa_file_for(root: Path, msa_id: str) -> Path:
    return root / str(msa_id) / f"{msa_id}_all.a3m"


def count_a3m_depth(path: Path) -> int:
    with path.open() as f:
        return sum(1 for line in f if line.startswith(">"))


def load_sources(map_paths: list[Path], roots: list[Path]) -> list[MsaSource]:
    if len(map_paths) != len(roots):
        raise ValueError(
            f"Expected the same number of existing maps and roots, got "
            f"{len(map_paths)} maps and {len(roots)} roots."
        )
    sources = []
    for map_path, root in zip(map_paths, roots):
        sources.append(
            MsaSource(
                map_path=map_path,
                msa_root=root,
                mapping=load_json(map_path),
            )
        )
    return sources


def find_cached_msa(sequence: str, sources: list[MsaSource]) -> tuple[str, Path] | None:
    for source in sources:
        ids = as_id_list(source.mapping.get(sequence))
        if not ids:
            continue
        # Match Protenix's RNA MSA loader behavior: it uses mapping[sequence][0].
        msa_id = ids[0]
        msa_path = msa_file_for(source.msa_root, msa_id)
        if msa_path.exists() and msa_path.stat().st_size > 0:
            return msa_id, msa_path
    return None


def extract_rna_entries(indices_csv: Path) -> list[RnaEntry]:
    df = pd.read_csv(indices_csv, dtype=str).fillna("")
    by_sequence: dict[str, RnaEntry] = {}
    replacement_count = 0
    for row_idx, row in df.iterrows():
        for side in ("1", "2"):
            if row.get(f"sub_mol_{side}_type", "").lower() != "rna":
                continue
            sequence = normalize_key_sequence(row.get(f"cluster_{side}_id", ""))
            search_sequence = make_search_sequence(sequence)
            if search_sequence is None:
                LOGGER.warning(
                    "Skip RNA sequence with non-letter characters at row %s side %s "
                    "(len=%d)",
                    row_idx,
                    side,
                    len(sequence),
                )
                continue
            if search_sequence != sequence.replace("T", "U"):
                replacement_count += 1
            if sequence in by_sequence:
                continue
            by_sequence[sequence] = RnaEntry(
                sequence=sequence,
                search_sequence=search_sequence,
                pdb_id=row.get("pdb_id", ""),
                entity_id=row.get(f"entity_{side}_id", ""),
                chain_id=row.get(f"chain_{side}_id", ""),
                row_index=int(row_idx),
                source_side=side,
            )
    if replacement_count:
        LOGGER.info(
            "Converted non-standard RNA letters to N for nhmmer search in %d rows. "
            "The output map keys still keep the original training sequences.",
            replacement_count,
        )
    return list(by_sequence.values())


def make_unique_msa_id(
    entry: RnaEntry,
    used_ids: set[str],
    output_map: dict[str, Any],
) -> str:
    base = safe_id(f"{entry.pdb_id.lower()}_{entry.entity_id}")
    candidate = base
    suffix = 2
    while candidate in used_ids:
        suffix += 1
        candidate = f"{base}_rna{suffix}"
    used_ids.add(candidate)

    existing_ids = as_id_list(output_map.get(entry.sequence))
    output_map[entry.sequence] = [candidate] + [
        msa_id for msa_id in existing_ids if msa_id != candidate
    ]
    return candidate


def init_status_log(path: Path, dry_run: bool) -> None:
    if dry_run or path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            [
                "timestamp",
                "status",
                "pdb_id",
                "entity_id",
                "chain_id",
                "msa_id",
                "seq_len",
                "depth",
                "msa_path",
                "message",
            ]
        )


def append_status(
    path: Path,
    dry_run: bool,
    entry: RnaEntry,
    status: str,
    msa_id: str = "",
    msa_path: Path | None = None,
    depth: int | None = None,
    message: str = "",
) -> None:
    if dry_run:
        return
    with path.open("a", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            [
                time.strftime("%Y-%m-%d %H:%M:%S"),
                status,
                entry.pdb_id,
                entry.entity_id,
                entry.chain_id,
                msa_id,
                len(entry.sequence),
                "" if depth is None else depth,
                "" if msa_path is None else str(msa_path),
                message,
            ]
        )


def search_and_store(
    entry: RnaEntry,
    msa_id: str,
    output_root: Path,
    args: argparse.Namespace,
) -> tuple[Path, int]:
    tmp_root = output_root / "_tmp_search"
    dest_dir = output_root / "msas" / msa_id
    dest_path = dest_dir / f"{msa_id}_all.a3m"

    run_rna_msa_search(
        rna_seq_for_msa_search=entry.search_sequence,
        rna_result_path=str(tmp_root),
        rna_seq_id=msa_id,
        nhmmer_binary_path=args.nhmmer_binary_path,
        hmmalign_binary_path=args.hmmalign_binary_path,
        hmmbuild_binary_path=args.hmmbuild_binary_path,
        ntrna_database_path=args.ntrna_database_path,
        rfam_database_path=args.rfam_database_path,
        rna_central_database_path=args.rna_central_database_path,
        nhmmer_n_cpu=args.nhmmer_n_cpu,
    )

    tmp_path = tmp_root / msa_id / "rna_msa.a3m"
    if not tmp_path.exists() or tmp_path.stat().st_size == 0:
        raise FileNotFoundError(f"RNA MSA search did not produce {tmp_path}")

    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(tmp_path, dest_path)
    if not args.keep_tmp:
        shutil.rmtree(tmp_root / msa_id, ignore_errors=True)

    return dest_path, count_a3m_depth(dest_path)


def collect_used_ids(
    sources: list[MsaSource],
    output_map: dict[str, Any],
) -> set[str]:
    used_ids: set[str] = set()
    for source in sources:
        for value in source.mapping.values():
            used_ids.update(as_id_list(value))
    for value in output_map.values():
        used_ids.update(as_id_list(value))
    return used_ids


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    existing_maps = args.existing_map or default_existing_maps()
    existing_roots = args.existing_root or default_existing_roots()
    sources = load_sources(existing_maps, existing_roots)

    output_map_path = args.output_root / "rna_sequence_to_pdb_chains.json"
    output_map = load_json(output_map_path)
    output_source = MsaSource(
        map_path=output_map_path,
        msa_root=args.output_root / "msas",
        mapping=output_map,
    )

    entries = extract_rna_entries(args.indices_csv)
    if args.limit is not None:
        entries = entries[: args.limit]

    status_path = args.output_root / "build_rna_msa_status.tsv"
    init_status_log(status_path, args.dry_run)

    used_ids = collect_used_ids(sources, output_map)
    n_existing = 0
    n_output = 0
    n_search = 0
    n_failed = 0

    LOGGER.info("indices_csv=%s", args.indices_csv)
    LOGGER.info("unique RNA sequences=%d", len(entries))
    LOGGER.info("output_root=%s", args.output_root)

    for i, entry in enumerate(entries, start=1):
        cached = find_cached_msa(entry.sequence, sources)
        if cached is not None:
            msa_id, msa_path = cached
            n_existing += 1
            LOGGER.info(
                "[%d/%d] reuse existing %s depth=%s seq_len=%d",
                i,
                len(entries),
                msa_id,
                count_a3m_depth(msa_path),
                len(entry.sequence),
            )
            append_status(
                status_path,
                args.dry_run,
                entry,
                "reuse_existing",
                msa_id=msa_id,
                msa_path=msa_path,
                depth=count_a3m_depth(msa_path),
            )
            continue

        cached = None if args.overwrite else find_cached_msa(
            entry.sequence, [output_source]
        )
        if cached is not None:
            msa_id, msa_path = cached
            n_output += 1
            LOGGER.info(
                "[%d/%d] reuse output %s depth=%s seq_len=%d",
                i,
                len(entries),
                msa_id,
                count_a3m_depth(msa_path),
                len(entry.sequence),
            )
            append_status(
                status_path,
                args.dry_run,
                entry,
                "reuse_output",
                msa_id=msa_id,
                msa_path=msa_path,
                depth=count_a3m_depth(msa_path),
            )
            continue

        msa_id = make_unique_msa_id(entry, used_ids, output_map)
        if args.dry_run:
            n_search += 1
            LOGGER.info(
                "[%d/%d] would search %s seq_len=%d",
                i,
                len(entries),
                msa_id,
                len(entry.sequence),
            )
            continue

        try:
            LOGGER.info(
                "[%d/%d] search missing RNA MSA %s seq_len=%d pdb=%s entity=%s",
                i,
                len(entries),
                msa_id,
                len(entry.sequence),
                entry.pdb_id,
                entry.entity_id,
            )
            msa_path, depth = search_and_store(entry, msa_id, args.output_root, args)
            write_json_atomic(output_map_path, output_map)
            output_source.mapping = output_map
            n_search += 1
            LOGGER.info("stored %s depth=%d", msa_path, depth)
            append_status(
                status_path,
                args.dry_run,
                entry,
                "searched",
                msa_id=msa_id,
                msa_path=msa_path,
                depth=depth,
            )
        except Exception as exc:  # noqa: BLE001 - keep batch search resumable.
            n_failed += 1
            LOGGER.exception("failed to search %s: %s", msa_id, exc)
            append_status(
                status_path,
                args.dry_run,
                entry,
                "failed",
                msa_id=msa_id,
                message=str(exc),
            )

    if not args.dry_run:
        write_json_atomic(output_map_path, output_map)

    LOGGER.info(
        "done: reuse_existing=%d reuse_output=%d searched=%d failed=%d",
        n_existing,
        n_output,
        n_search,
        n_failed,
    )
    LOGGER.info(
        "Add this extra RNA MSA cache to configs_data.py if not already present: "
        "map=%s root=%s",
        output_map_path,
        args.output_root / "msas",
    )


if __name__ == "__main__":
    main()


"""
python scripts/build_train_rna_msa.py \
    --output-root /inspire/ssd/project/sais-bio/public/xiangwenkai/GITHUB/Protenix/rna_data/rna_msa \
    --nhmmer-n-cpu 20
"""