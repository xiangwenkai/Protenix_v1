#!/usr/bin/env python3
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

"""Add structure-derived RNA binding signals to Protenix bioassembly pkl files.

The script processes every ``*.pkl.gz`` under ``--bioassembly-dir`` and writes
updated copies to ``--output-dir`` with the original file names. Binding signal
is stored as token-level annotations on ``bioassembly_dict["token_array"]`` so
that Protenix cropping keeps the signal aligned with cropped tokens.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from protenix.utils.file_io import dump_gzip_pickle, load_gzip_pickle

try:
    from scipy.spatial import cKDTree
except ImportError:  # pragma: no cover - depends on runtime environment.
    cKDTree = None


LOGGER = logging.getLogger(__name__)

@dataclass(frozen=True)
class SignalConfig:
    contact_radius: float = 8.0
    contact_midpoint: float = 6.0
    contact_temperature: float = 1.0
    protein_atom_chunk_size: int = 65536


def _is_heavy_atom_array(atom_array: Any) -> np.ndarray:
    n_atom = len(atom_array)
    atom_name = np.asarray(getattr(atom_array, "atom_name", [""] * n_atom)).astype(str)
    element = np.asarray(getattr(atom_array, "element", [""] * n_atom)).astype(str)
    element = np.char.upper(np.char.strip(element))
    atom_name = np.char.upper(np.char.strip(atom_name))

    has_element = element != ""
    heavy_by_element = (element != "H") & (element != "D")
    heavy_by_name = ~np.char.startswith(atom_name, "H") & ~np.char.startswith(atom_name, "D")
    return np.where(has_element, heavy_by_element, heavy_by_name)


def _soft_contact(distance: float, midpoint: float, temperature: float) -> float:
    x = (midpoint - distance) / max(float(temperature), 1e-6)
    if x >= 50:
        return 1.0
    if x <= -50:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def _distogram_rep_atom_mask(atom_array: Any, token_atoms: list[np.ndarray]) -> np.ndarray:
    """Return the AF3/Protenix distogram representative atom mask."""

    n_atom = len(atom_array)
    rep_mask = getattr(atom_array, "distogram_rep_atom_mask", None)
    if rep_mask is not None:
        return np.asarray(rep_mask).astype(bool)

    atom_names = np.asarray(atom_array.atom_name).astype(str)
    atom_names_clean = np.char.upper(np.char.strip(atom_names))
    res_names = np.asarray(getattr(atom_array, "res_name", [""] * n_atom)).astype(str)
    res_names = np.char.upper(np.char.strip(res_names))
    is_protein = np.asarray(atom_array.is_protein).astype(bool)
    is_rna = np.asarray(atom_array.is_rna).astype(bool)

    rep_mask = np.zeros(n_atom, dtype=np.bool_)
    rep_mask |= is_protein & (res_names != "GLY") & (atom_names_clean == "CB")
    rep_mask |= is_protein & (res_names == "GLY") & (atom_names_clean == "CA")
    rep_mask |= is_rna & np.isin(res_names, ["A", "G"]) & (atom_names_clean == "C4")
    rep_mask |= is_rna & np.isin(res_names, ["C", "U"]) & (atom_names_clean == "C2")
    rep_mask |= is_rna & (res_names == "N") & np.isin(atom_names_clean, ["C1'", "C1*"])

    heavy_mask = _is_heavy_atom_array(atom_array)
    for atom_indices in token_atoms:
        if atom_indices.size and not np.any(rep_mask[atom_indices]):
            fallback_atoms = atom_indices[heavy_mask[atom_indices]]
            if fallback_atoms.size:
                rep_mask[int(fallback_atoms[0])] = True
    return rep_mask


def _token_atom_arrays(token_array: Any) -> list[np.ndarray]:
    return [
        np.asarray(token.atom_indices, dtype=np.int64)
        for token in token_array
    ]


def _atom_to_token(token_atoms: list[np.ndarray], n_atom: int) -> np.ndarray:
    atom_to_token = np.full(n_atom, -1, dtype=np.int64)
    for token_idx, atom_indices in enumerate(token_atoms):
        atom_to_token[atom_indices] = token_idx
    return atom_to_token


def _query_candidate_protein_tokens(
    *,
    rna_coords: np.ndarray,
    protein_coords: np.ndarray,
    protein_atom_indices: np.ndarray,
    atom_to_token: np.ndarray,
    contact_radius: float,
    protein_tree: Any,
    chunk_size: int,
) -> np.ndarray:
    if rna_coords.size == 0 or protein_coords.size == 0:
        return np.empty((0,), dtype=np.int64)

    if protein_tree is not None:
        neighbors = protein_tree.query_ball_point(rna_coords, r=contact_radius)
        if len(neighbors) == 0:
            return np.empty((0,), dtype=np.int64)
        local_indices = sorted({idx for hit in neighbors for idx in hit})
        if not local_indices:
            return np.empty((0,), dtype=np.int64)
        global_indices = protein_atom_indices[np.asarray(local_indices, dtype=np.int64)]
        return np.unique(atom_to_token[global_indices])

    candidate_local: list[np.ndarray] = []
    for start in range(0, protein_coords.shape[0], chunk_size):
        chunk = protein_coords[start : start + chunk_size]
        distances = np.linalg.norm(rna_coords[:, None, :] - chunk[None, :, :], axis=-1)
        local = np.where(np.any(distances <= contact_radius, axis=0))[0]
        if local.size:
            candidate_local.append(local + start)
    if not candidate_local:
        return np.empty((0,), dtype=np.int64)
    global_indices = protein_atom_indices[np.concatenate(candidate_local)]
    return np.unique(atom_to_token[global_indices])


def compute_token_binding_signal(
    bioassembly_dict: dict[str, Any],
    config: SignalConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Compute token-level RNA protein-contact signal from a bioassembly dict."""

    atom_array = bioassembly_dict["atom_array"]
    token_array = bioassembly_dict["token_array"]
    n_atom = len(atom_array)
    n_token = len(token_array)

    token_signal = np.zeros(n_token, dtype=np.float32)
    token_signal_mask = np.zeros(n_token, dtype=np.bool_)
    token_resolved_mask = np.zeros(n_token, dtype=np.bool_)

    is_protein = np.asarray(atom_array.is_protein).astype(bool)
    is_rna = np.asarray(atom_array.is_rna).astype(bool)
    is_resolved = np.asarray(getattr(atom_array, "is_resolved", np.ones(n_atom))).astype(bool)
    is_heavy = _is_heavy_atom_array(atom_array)
    coords = np.asarray(atom_array.coord, dtype=np.float32)

    token_atoms = _token_atom_arrays(token_array)
    atom_to_token = _atom_to_token(token_atoms, n_atom)
    distogram_rep_atom_mask = _distogram_rep_atom_mask(atom_array, token_atoms)

    protein_mask = is_protein & is_heavy & is_resolved & distogram_rep_atom_mask
    protein_atom_indices = np.nonzero(protein_mask)[0].astype(np.int64)
    protein_coords = coords[protein_atom_indices]
    protein_tree = cKDTree(protein_coords) if cKDTree is not None and protein_coords.size else None

    stats = {
        "n_token": int(n_token),
        "n_atom": int(n_atom),
        "n_protein_atom": int(protein_mask.sum()),
        "n_rna_atom": int((is_rna & is_heavy & is_resolved & distogram_rep_atom_mask).sum()),
        "n_rna_token": 0,
        "n_valid_rna_token": 0,
        "n_positive_rna_token": 0,
        "signal_sum": 0.0,
        "signal_max": 0.0,
    }

    if protein_atom_indices.size == 0:
        return token_signal, token_signal_mask, token_resolved_mask, stats

    for token_idx, atom_indices in enumerate(token_atoms):
        rna_atom_indices = atom_indices[is_rna[atom_indices]]
        if rna_atom_indices.size == 0:
            continue
        stats["n_rna_token"] += 1

        rna_rep_atoms = rna_atom_indices[
            is_heavy[rna_atom_indices]
            & is_resolved[rna_atom_indices]
            & distogram_rep_atom_mask[rna_atom_indices]
        ]
        if rna_rep_atoms.size == 0:
            continue
        token_signal_mask[token_idx] = True
        token_resolved_mask[token_idx] = True
        stats["n_valid_rna_token"] += 1

        candidate_protein_tokens = _query_candidate_protein_tokens(
            rna_coords=coords[rna_rep_atoms],
            protein_coords=protein_coords,
            protein_atom_indices=protein_atom_indices,
            atom_to_token=atom_to_token,
            contact_radius=float(config.contact_radius),
            protein_tree=protein_tree,
            chunk_size=int(config.protein_atom_chunk_size),
        )
        candidate_protein_tokens = candidate_protein_tokens[candidate_protein_tokens >= 0]
        if candidate_protein_tokens.size == 0:
            continue

        score = 0.0
        rna_rep_coords = coords[rna_rep_atoms]
        for protein_token_idx in candidate_protein_tokens:
            protein_token_atoms = token_atoms[int(protein_token_idx)]
            protein_token_atoms = protein_token_atoms[protein_mask[protein_token_atoms]]
            if protein_token_atoms.size == 0:
                continue
            protein_token_coords = coords[protein_token_atoms]
            distances = np.linalg.norm(
                rna_rep_coords[:, None, :] - protein_token_coords[None, :, :],
                axis=-1,
            )
            distance = float(distances.min())
            if distance > config.contact_radius:
                continue
            score = max(
                score,
                _soft_contact(
                    distance=distance,
                    midpoint=config.contact_midpoint,
                    temperature=config.contact_temperature,
                ),
            )
        token_signal[token_idx] = np.float32(score)

    positive_mask = token_signal_mask & (token_signal > 0.0)
    stats["n_positive_rna_token"] = int(positive_mask.sum())
    stats["signal_sum"] = float(token_signal.sum(dtype=np.float32))
    stats["signal_max"] = float(token_signal.max(initial=0.0))
    return token_signal, token_signal_mask, token_resolved_mask, stats


def add_signal_annotations(
    bioassembly_dict: dict[str, Any],
    config: SignalConfig,
) -> dict[str, Any]:
    token_signal, token_signal_mask, token_resolved_mask, stats = compute_token_binding_signal(
        bioassembly_dict=bioassembly_dict,
        config=config,
    )
    token_array = bioassembly_dict["token_array"]
    token_array.set_annotation("rna_binding_signal", token_signal.astype(np.float32).tolist())
    token_array.set_annotation("rna_binding_signal_mask", token_signal_mask.astype(bool).tolist())
    token_array.set_annotation("rna_binding_resolved_mask", token_resolved_mask.astype(bool).tolist())
    bioassembly_dict["rna_binding_signal_meta"] = {
        "source": "structure_contact",
        "version": 1,
        "signal_level": "token",
        "signal_key": "token_array.rna_binding_signal",
        "signal_mask_key": "token_array.rna_binding_signal_mask",
        "resolved_mask_key": "token_array.rna_binding_resolved_mask",
        "distance_definition": "distogram_representative_atom_distance",
        "scoring": "max_over_candidate_protein_tokens_soft_contact",
        **asdict(config),
        "stats": stats,
    }
    return bioassembly_dict


def process_one_file(
    input_path: str,
    output_dir: str,
    config: SignalConfig,
    overwrite: bool,
) -> dict[str, Any]:
    input_file = Path(input_path)
    output_file = Path(output_dir) / input_file.name
    result = {
        "input": str(input_file),
        "output": str(output_file),
        "pdb_id": input_file.name.removesuffix(".pkl.gz"),
        "status": "unknown",
        "error": "",
        "stats": {},
    }

    if output_file.exists() and not overwrite:
        result["status"] = "skipped_exists"
        return result

    try:
        bioassembly_dict = load_gzip_pickle(input_file)
        bioassembly_dict = add_signal_annotations(bioassembly_dict, config)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        dump_gzip_pickle(bioassembly_dict, output_file)
        stats = dict(bioassembly_dict["rna_binding_signal_meta"]["stats"])
        result["stats"] = stats
        if stats.get("n_protein_atom", 0) <= 0:
            result["status"] = "no_protein"
        elif stats.get("n_rna_token", 0) <= 0:
            result["status"] = "no_rna"
        elif stats.get("n_positive_rna_token", 0) <= 0:
            result["status"] = "zero_signal"
        else:
            result["status"] = "ok"
        return result
    except Exception as exc:  # pragma: no cover - exercised through process pools.
        result["status"] = "failed"
        result["error"] = f"{exc}\n{traceback.format_exc()}"
        return result


def list_input_files(bioassembly_dir: Path, limit: int | None = None) -> list[Path]:
    files = sorted(bioassembly_dir.glob("*.pkl.gz"))
    if limit is not None:
        files = files[:limit]
    return files


def summarize_results(results: list[dict[str, Any]], config: SignalConfig, args: argparse.Namespace) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    total_rna_tokens = 0
    total_positive_rna_tokens = 0
    total_signal_sum = 0.0
    for result in results:
        status = str(result.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
        stats = result.get("stats") or {}
        total_rna_tokens += int(stats.get("n_valid_rna_token", 0))
        total_positive_rna_tokens += int(stats.get("n_positive_rna_token", 0))
        total_signal_sum += float(stats.get("signal_sum", 0.0))

    return {
        "bioassembly_dir": str(args.bioassembly_dir),
        "output_dir": str(args.output_dir),
        "num_files": len(results),
        "status_counts": status_counts,
        "total_valid_rna_tokens": total_rna_tokens,
        "total_positive_rna_tokens": total_positive_rna_tokens,
        "positive_rna_token_rate": (
            total_positive_rna_tokens / total_rna_tokens if total_rna_tokens > 0 else 0.0
        ),
        "total_signal_sum": total_signal_sum,
        "config": asdict(config),
        "failed": [
            {
                "input": result.get("input"),
                "error": result.get("error"),
            }
            for result in results
            if result.get("status") == "failed"
        ],
    }


def write_summary(output_dir: Path, summary: dict[str, Any], results: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "structure_rna_signal_pkl_summary.json"
    details_path = output_dir / "structure_rna_signal_pkl_details.jsonl"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    with details_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, sort_keys=True) + "\n")
    LOGGER.info("Wrote summary to %s", summary_path)
    LOGGER.info("Wrote per-file details to %s", details_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add structure-derived RNA binding signal annotations to Protenix pkl files."
    )
    parser.add_argument("--bioassembly-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contact-radius", type=float, default=8.0)
    parser.add_argument("--contact-midpoint", type=float, default=6.0)
    parser.add_argument("--contact-temperature", type=float, default=1.0)
    parser.add_argument(
        "--base-weight",
        type=float,
        default=1.0,
        help="Deprecated no-op kept for compatibility with older groupwise scoring commands.",
    )
    parser.add_argument(
        "--sugar-weight",
        type=float,
        default=0.7,
        help="Deprecated no-op kept for compatibility with older groupwise scoring commands.",
    )
    parser.add_argument(
        "--phosphate-weight",
        type=float,
        default=0.4,
        help="Deprecated no-op kept for compatibility with older groupwise scoring commands.",
    )
    parser.add_argument("--protein-atom-chunk-size", type=int, default=65536)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test cap.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    )
    if not args.bioassembly_dir.exists():
        raise FileNotFoundError(f"bioassembly dir not found: {args.bioassembly_dir}")

    config = SignalConfig(
        contact_radius=args.contact_radius,
        contact_midpoint=args.contact_midpoint,
        contact_temperature=args.contact_temperature,
        protein_atom_chunk_size=args.protein_atom_chunk_size,
    )
    files = list_input_files(args.bioassembly_dir, args.limit)
    if not files:
        raise FileNotFoundError(f"No *.pkl.gz files found under {args.bioassembly_dir}")
    LOGGER.info("Processing %d pkl file(s)", len(files))

    results: list[dict[str, Any]] = []
    if args.num_workers <= 1:
        iterator = (
            process_one_file(str(path), str(args.output_dir), config, args.overwrite)
            for path in files
        )
        for result in tqdm(iterator, total=len(files), desc="add-rna-signal"):
            if args.fail_fast and result["status"] == "failed":
                raise RuntimeError(result["error"])
            results.append(result)
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [
                executor.submit(
                    process_one_file,
                    str(path),
                    str(args.output_dir),
                    config,
                    args.overwrite,
                )
                for path in files
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc="add-rna-signal"):
                result = future.result()
                if args.fail_fast and result["status"] == "failed":
                    raise RuntimeError(result["error"])
                results.append(result)

    results = sorted(results, key=lambda item: str(item.get("input", "")))
    summary = summarize_results(results, config, args)
    write_summary(args.output_dir, summary, results)
    LOGGER.info("Status counts: %s", summary["status_counts"])
    LOGGER.info(
        "Positive RNA token rate: %.6f (%d/%d)",
        summary["positive_rna_token_rate"],
        summary["total_positive_rna_tokens"],
        summary["total_valid_rna_tokens"],
    )
    if summary["status_counts"].get("failed", 0) > 0:
        raise RuntimeError(f"{summary['status_counts']['failed']} file(s) failed. See summary JSON.")


if __name__ == "__main__":
    main()

"""
python scripts/add_structure_rna_signal_to_protenix_pkl.py \
    --bioassembly-dir /inspire/ssd/project/sais-bio/public/xiangwenkai/GITHUB/Protenix_v1/data/train \
    --output-dir /inspire/ssd/project/sais-bio/public/xiangwenkai/GITHUB/Protenix_v1/data/train_signal \
    --num-workers 16 \
    --overwrite
"""