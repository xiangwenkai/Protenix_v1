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

"""Precompute Protenix input features for eCLIP PPFT parquet data.

The output mirrors ``scripts/prepare_training_data.py``: an index CSV plus a
directory of gzip-pickled feature files. Each feature file contains no
structure labels, only Protenix sequence-derived input features and eCLIP
signal labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import random
import time
import traceback
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
from joblib import Parallel, delayed
from tqdm import tqdm

from protenix.data.eclip_ppft_dataset import (
    build_protenix_sample_dict,
    crop_eclip_sample_rna,
    load_protein_sequences,
)
from protenix.data.inference.json_to_feature import SampleDictToFeatures
from protenix.data.utils import data_type_transform, make_dummy_feature
from protenix.utils.file_io import dump_gzip_pickle


BASE_COLUMNS = ("rna_seq", "protein_symbol", "cell_line", "signal_vector")
OPTIONAL_COLUMNS = (
    "profile_label",
    "profile_loss_weight",
    "window_label",
    "quality_label",
    "is_highconf_pos",
    "is_highconf_neg",
    "signal_total",
    "signal_peak",
    "signal_sharpness",
    "signal_topk_ratio",
    "signal_peak_width",
    "signal_window_var_norm",
    "signal_nonzero_fraction",
)


def _to_float_list(value: Any) -> list[float]:
    if value is None:
        return []
    return [float(x) for x in value]


def _fit_signal_length(signal: list[float], length: int) -> list[float]:
    signal = signal[:length]
    if len(signal) < length:
        signal.extend([0.0] * (length - len(signal)))
    return signal


def _safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in text)


def _make_sample_id(split: str, source_file: Path, row_index: int, row: dict[str, Any]) -> str:
    digest = hashlib.sha1(
        (
            str(row.get("protein_symbol", ""))
            + "|"
            + str(row.get("cell_line", ""))
            + "|"
            + str(row.get("rna_seq", ""))
            + "|"
            + source_file.name
            + "|"
            + str(row_index)
        ).encode("utf-8")
    ).hexdigest()[:16]
    return f"{split}_{source_file.stem}_{row_index}_{digest}"


def _feature_path(output_dir: Path, split: str, sample_id: str) -> Path:
    digest = sample_id.rsplit("_", 1)[-1]
    return output_dir / split / digest[:2] / f"{sample_id}.pkl.gz"


def _row_to_sample(
    row: dict[str, Any],
    protein_sequences: dict[str, str],
    split: str,
    source_file: Path,
    row_index: int,
    max_rna_length: int | None,
    rna_crop_size: int | None,
    max_protein_length: int | None,
    crop_seed: int,
) -> dict[str, Any] | None:
    rna_seq = str(row["rna_seq"]).upper().replace("U", "T")
    protein_symbol = str(row["protein_symbol"])
    protein_sequence = protein_sequences.get(protein_symbol)
    if protein_sequence is None:
        return None
    if max_rna_length is not None and len(rna_seq) > max_rna_length:
        return None
    if max_protein_length is not None and len(protein_sequence) > max_protein_length:
        return None

    signal = _fit_signal_length(_to_float_list(row["signal_vector"]), len(rna_seq))
    sample_id = _make_sample_id(split, source_file, row_index, row)
    sample = {
        "sample_id": sample_id,
        "rna_seq": rna_seq,
        "protein_symbol": protein_symbol,
        "protein_sequence": protein_sequence,
        "cell_line": str(row["cell_line"]),
        "signal_vector": torch.tensor(signal, dtype=torch.float32),
    }
    if "profile_label" in row and row["profile_label"] is not None:
        profile = _fit_signal_length(_to_float_list(row["profile_label"]), len(rna_seq))
        sample["profile_label"] = torch.tensor(profile, dtype=torch.float32)
    for key in OPTIONAL_COLUMNS:
        if key in row and key != "profile_label":
            sample[key] = row[key]
    seed_material = f"{crop_seed}|{source_file.name}|{row_index}|{sample_id}"
    rng_seed = int(hashlib.sha1(seed_material.encode("utf-8")).hexdigest()[:16], 16)
    return crop_eclip_sample_rna(sample, rna_crop_size, random.Random(rng_seed))


def featurize_eclip_sample(sample: dict[str, Any]) -> dict[str, Any]:
    sample_dict = build_protenix_sample_dict(sample)
    sample2feat = SampleDictToFeatures(sample_dict)
    features_dict, atom_array, _ = sample2feat.get_feature_dict()
    features_dict["distogram_rep_atom_mask"] = torch.tensor(
        atom_array.distogram_rep_atom_mask
    ).long()
    features_dict = make_dummy_feature(features_dict=features_dict, dummy_feats=["msa", "template"])
    feat = data_type_transform(features_dict)

    eclip_label = {
        "signal_vector": sample["signal_vector"],
        "profile_label": sample.get("profile_label"),
        "profile_loss_weight": sample.get("profile_loss_weight", 1.0),
        "window_label": sample.get("window_label"),
        "quality_label": sample.get("quality_label"),
        "is_highconf_pos": sample.get("is_highconf_pos"),
        "is_highconf_neg": sample.get("is_highconf_neg"),
    }
    metadata = {
        "sample_id": sample["sample_id"],
        "protein_symbol": sample["protein_symbol"],
        "cell_line": sample["cell_line"],
        "rna_length": len(sample["rna_seq"]),
        "rna_original_length": sample.get("rna_original_length", len(sample["rna_seq"])),
        "rna_crop_start": sample.get("rna_crop_start", 0),
        "rna_crop_end": sample.get("rna_crop_end", len(sample["rna_seq"])),
        "protein_length": len(sample["protein_sequence"]),
        "signal_total": sample.get("signal_total"),
        "signal_peak": sample.get("signal_peak"),
        "signal_sharpness": sample.get("signal_sharpness"),
        "signal_topk_ratio": sample.get("signal_topk_ratio"),
        "signal_peak_width": sample.get("signal_peak_width"),
        "signal_window_var_norm": sample.get("signal_window_var_norm"),
        "signal_nonzero_fraction": sample.get("signal_nonzero_fraction"),
    }
    return {
        "input_feature_dict": feat,
        "eclip_label": eclip_label,
        "metadata": metadata,
    }


def process_one_parquet(
    parquet_path: Path,
    output_feature_dir: Path,
    protein_sequence_tsv: Path,
    split: str,
    max_rna_length: int | None,
    rna_crop_size: int | None,
    max_protein_length: int | None,
    parquet_batch_size: int,
    crop_seed: int,
    skip_existing: bool,
) -> list[dict[str, Any]]:
    protein_sequences = load_protein_sequences(protein_sequence_tsv)
    parquet_file = pq.ParquetFile(parquet_path)
    available = set(parquet_file.schema_arrow.names)
    missing = sorted(set(BASE_COLUMNS) - available)
    if missing:
        raise ValueError(f"{parquet_path} is missing required columns: {missing}")

    columns = list(BASE_COLUMNS) + [col for col in OPTIONAL_COLUMNS if col in available]
    index_rows: list[dict[str, Any]] = []
    row_offset = 0
    for batch in parquet_file.iter_batches(batch_size=parquet_batch_size, columns=columns):
        for local_idx, row in enumerate(batch.to_pylist()):
            row_index = row_offset + local_idx
            sample = _row_to_sample(
                row=row,
                protein_sequences=protein_sequences,
                split=split,
                source_file=parquet_path,
                row_index=row_index,
                max_rna_length=max_rna_length,
                rna_crop_size=rna_crop_size,
                max_protein_length=max_protein_length,
                crop_seed=crop_seed,
            )
            if sample is None:
                continue
            output_path = _feature_path(output_feature_dir, split, sample["sample_id"])
            rel_path = output_path.relative_to(output_feature_dir)
            try:
                if not (skip_existing and output_path.exists()):
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    feature_data = featurize_eclip_sample(sample)
                    dump_gzip_pickle(feature_data, output_path)
                index_rows.append(
                    {
                        "sample_id": sample["sample_id"],
                        "split": split,
                        "feature_path": str(rel_path),
                        "source_parquet": str(parquet_path),
                        "source_row": row_index,
                        "protein_symbol": sample["protein_symbol"],
                        "cell_line": sample["cell_line"],
                        "rna_length": len(sample["rna_seq"]),
                        "rna_original_length": sample.get("rna_original_length", len(sample["rna_seq"])),
                        "rna_crop_start": sample.get("rna_crop_start", 0),
                        "rna_crop_end": sample.get("rna_crop_end", len(sample["rna_seq"])),
                        "protein_length": len(sample["protein_sequence"]),
                        "signal_total": sample.get("signal_total", ""),
                        "signal_peak": sample.get("signal_peak", ""),
                        "quality_label": sample.get("quality_label", ""),
                    }
                )
            except Exception as exc:
                logging.warning(
                    "Failed to featurize %s row %d: %s\n%s",
                    parquet_path,
                    row_index,
                    exc,
                    traceback.format_exc(),
                )
        row_offset += batch.num_rows
    return index_rows


def list_parquet_files(input_path: Path, split: str) -> list[Path]:
    split_dir = input_path / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    files = sorted(split_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {split_dir}")
    return files


def write_index_csv(rows: list[dict[str, Any]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "split",
        "feature_path",
        "source_parquet",
        "source_row",
        "protein_symbol",
        "cell_line",
        "rna_length",
        "rna_original_length",
        "rna_crop_start",
        "rna_crop_end",
        "protein_length",
        "signal_total",
        "signal_peak",
        "quality_label",
    ]
    with open(output_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
        writer.writeheader()
        writer.writerows(rows)


def run_gen_eclip_features(
    input_path: Path,
    output_csv: Path,
    feature_output_dir: Path,
    protein_sequence_tsv: Path,
    splits: list[str],
    max_rna_length: int | None,
    rna_crop_size: int | None,
    max_protein_length: int | None,
    parquet_batch_size: int,
    crop_seed: int,
    num_workers: int,
    skip_existing: bool,
) -> None:
    input_path = Path(input_path)
    output_csv = Path(output_csv)
    feature_output_dir = Path(feature_output_dir)
    protein_sequence_tsv = Path(protein_sequence_tsv)
    feature_output_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for split in splits:
        for parquet_path in list_parquet_files(input_path, split):
            jobs.append((split, parquet_path))

    start = time.time()
    all_results = [
        result
        for result in tqdm(
            Parallel(n_jobs=num_workers, return_as="generator_unordered")(
                delayed(process_one_parquet)(
                    parquet_path=parquet_path,
                    output_feature_dir=feature_output_dir,
                    protein_sequence_tsv=protein_sequence_tsv,
                    split=split,
                    max_rna_length=max_rna_length,
                    rna_crop_size=rna_crop_size,
                    max_protein_length=max_protein_length,
                    parquet_batch_size=parquet_batch_size,
                    crop_seed=crop_seed,
                    skip_existing=skip_existing,
                )
                for split, parquet_path in jobs
            ),
            total=len(jobs),
        )
    ]
    rows = [row for result in all_results for row in result]
    write_index_csv(rows, output_csv)
    logging.info(
        "Finished eCLIP feature preprocessing: files=%d rows=%d output_csv=%s feature_dir=%s time=%.1fs",
        len(jobs),
        len(rows),
        output_csv,
        feature_output_dir,
        time.time() - start,
    )


def _parse_optional_int(value: str) -> int | None:
    if value.lower() in {"none", "null", "-1"}:
        return None
    return int(value)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_path",
        type=Path,
        default=Path(
            "/inspire/ssd/project/sais-bio/public/xiangwenkai/GITHUB/parnet/data_process/high_quality_positive"
        ),
        help="Path to eCLIP parquet root containing train/validation/test split directories.",
    )
    parser.add_argument(
        "-o",
        "--output_csv",
        type=Path,
        required=True,
        help="Path to output index CSV.",
    )
    parser.add_argument(
        "-b",
        "--feature_output_dir",
        type=Path,
        required=True,
        help="Directory where feature pkl.gz files will be written.",
    )
    parser.add_argument(
        "-p",
        "--protein_sequence_tsv",
        type=Path,
        default=Path(
            "/inspire/ssd/project/sais-bio/public/xiangwenkai/GITHUB/parnet/parnet/assets/ENCODE.protein_symbol2sequence.uniprot.tsv"
        ),
        help="TSV with protein_symbol and protein_sequence columns.",
    )
    parser.add_argument(
        "-s",
        "--splits",
        type=str,
        default="train,validation,test",
        help="Comma-separated split names to process.",
    )
    parser.add_argument(
        "--max_rna_length",
        type=_parse_optional_int,
        default=600,
        help="Drop samples longer than this RNA length. Use none/null/-1 to disable.",
    )
    parser.add_argument(
        "--max_protein_length",
        type=_parse_optional_int,
        default=1200,
        help="Drop samples longer than this protein length. Use none/null/-1 to disable.",
    )
    parser.add_argument(
        "--rna_crop_size",
        type=_parse_optional_int,
        default=256,
        help="Signal-weighted RNA crop length. Use none/null/-1 to disable cropping.",
    )
    parser.add_argument(
        "--crop_seed",
        type=int,
        default=42,
        help="Seed used for deterministic signal-weighted RNA crop sampling.",
    )
    parser.add_argument(
        "--parquet_batch_size",
        type=int,
        default=256,
        help="Number of rows read per parquet batch.",
    )
    parser.add_argument(
        "-n",
        "--n_cpu",
        type=int,
        default=1,
        help="Number of parallel parquet workers.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Reuse existing feature pkl.gz files and only rebuild the index.",
    )
    args = parser.parse_args()
    run_gen_eclip_features(
        input_path=args.input_path,
        output_csv=args.output_csv,
        feature_output_dir=args.feature_output_dir,
        protein_sequence_tsv=args.protein_sequence_tsv,
        splits=[split.strip() for split in args.splits.split(",") if split.strip()],
        max_rna_length=args.max_rna_length,
        rna_crop_size=args.rna_crop_size,
        max_protein_length=args.max_protein_length,
        parquet_batch_size=args.parquet_batch_size,
        crop_seed=args.crop_seed,
        num_workers=args.n_cpu,
        skip_existing=args.skip_existing,
    )
