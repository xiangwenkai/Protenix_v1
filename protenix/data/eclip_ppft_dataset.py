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

"""Streaming eCLIP parquet dataset for Protenix PPFT."""

from __future__ import annotations

import hashlib
import math
import random
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset, get_worker_info


BASE_COLUMNS = ("rna_seq", "protein_symbol", "cell_line", "signal_vector")


def _to_float_values(value: Any) -> list[float]:
    if isinstance(value, torch.Tensor):
        return [float(x) for x in value.detach().cpu().flatten().tolist()]
    return [float(x) for x in value]


def sample_signal_weighted_crop_start(
    signal: Any,
    seq_length: int,
    crop_size: int,
    rng: random.Random,
) -> int:
    """Sample an RNA crop start with probability proportional to signal mass."""

    if crop_size <= 0:
        raise ValueError(f"crop_size must be positive, got {crop_size}")
    if seq_length <= crop_size:
        return 0

    signal_values = _to_float_values(signal)
    signal_values = signal_values[:seq_length]
    if len(signal_values) < seq_length:
        signal_values.extend([0.0] * (seq_length - len(signal_values)))
    weights = [math.log1p(max(value, 0.0)) for value in signal_values]

    window_count = seq_length - crop_size + 1
    current = sum(weights[:crop_size])
    window_weights = [current]
    for start in range(1, window_count):
        current += weights[start + crop_size - 1] - weights[start - 1]
        window_weights.append(current)

    total = sum(window_weights)
    if total <= 0.0:
        return rng.randrange(window_count)

    threshold = rng.random() * total
    cumulative = 0.0
    for start, weight in enumerate(window_weights):
        cumulative += weight
        if cumulative >= threshold:
            return start
    return window_count - 1


def _crop_vector(value: Any, start: int, end: int, original_length: int) -> Any:
    if isinstance(value, torch.Tensor):
        if value.ndim > 0 and value.shape[0] == original_length:
            return value[start:end].clone()
        return value
    if isinstance(value, list) and len(value) == original_length:
        return value[start:end]
    if isinstance(value, tuple) and len(value) == original_length:
        return value[start:end]
    return value


def crop_eclip_sample_rna(
    sample: dict[str, Any],
    crop_size: int | None,
    rng: random.Random,
    *,
    signal_key: str = "signal_vector",
    vector_keys: tuple[str, ...] = ("signal_vector", "profile_label"),
) -> dict[str, Any]:
    """Crop RNA sequence and aligned per-RNA vectors using signal-weighted sampling."""

    if crop_size is None:
        return sample
    rna_seq = sample["rna_seq"]
    original_length = len(rna_seq)
    if original_length <= crop_size:
        sample["rna_crop_start"] = 0
        sample["rna_crop_end"] = original_length
        sample["rna_original_length"] = original_length
        return sample

    start = sample_signal_weighted_crop_start(
        signal=sample[signal_key],
        seq_length=original_length,
        crop_size=crop_size,
        rng=rng,
    )
    end = start + crop_size
    sample["rna_seq"] = rna_seq[start:end]
    for key in vector_keys:
        if key in sample:
            sample[key] = _crop_vector(sample[key], start, end, original_length)
    sample["rna_crop_start"] = start
    sample["rna_crop_end"] = end
    sample["rna_original_length"] = original_length
    sample["sample_id"] = f"{sample['sample_id']}:rna{start}-{end}"
    return sample


def load_protein_sequences(path: str | Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        if "protein_symbol" not in header or "protein_sequence" not in header:
            raise ValueError(f"{path} must contain protein_symbol and protein_sequence columns")
        symbol_idx = header.index("protein_symbol")
        sequence_idx = header.index("protein_sequence")
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) <= max(symbol_idx, sequence_idx):
                continue
            symbol = fields[symbol_idx].strip()
            sequence = fields[sequence_idx].strip().upper()
            if symbol and sequence:
                mapping[symbol] = sequence
    return mapping


def list_parquet_files(data_dir: str | Path, split: str) -> list[Path]:
    split_dir = Path(data_dir) / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    files = sorted(split_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {split_dir}")
    return files


def collect_cell_vocab(data_dir: str | Path, splits: Iterable[str] = ("train", "validation", "test")) -> dict[str, int]:
    cell_lines: set[str] = set()
    for split in splits:
        split_dir = Path(data_dir) / split
        if not split_dir.exists():
            continue
        for file_path in sorted(split_dir.glob("*.parquet")):
            table = pq.read_table(file_path, columns=["cell_line"])
            cell_lines.update(str(value.as_py()) for value in table["cell_line"] if value.as_py())
    return {cell_line: idx for idx, cell_line in enumerate(sorted(cell_lines))}


class EclipPPFTDataset(IterableDataset):
    """Stream eCLIP rows and attach protein sequences."""

    def __init__(
        self,
        data_dir: str | Path,
        split: str,
        protein_sequence_tsv: str | Path,
        *,
        cell_vocab: dict[str, int] | None = None,
        max_rna_length: int | None = 600,
        rna_crop_size: int | None = 256,
        max_protein_length: int | None = 1200,
        min_signal_max: float | None = None,
        zero_signal_keep_prob: float = 1.0,
        limit: int | None = None,
        shuffle_files: bool = False,
        shuffle_buffer: int = 0,
        seed: int = 42,
        parquet_batch_size: int = 1024,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split = split
        self.files = list_parquet_files(self.data_dir, split)
        self.protein_sequences = load_protein_sequences(protein_sequence_tsv)
        self.cell_vocab = dict(cell_vocab or {})
        self.unknown_cell_id = len(self.cell_vocab)
        self.max_rna_length = max_rna_length
        self.rna_crop_size = rna_crop_size
        self.max_protein_length = max_protein_length
        self.min_signal_max = min_signal_max
        self.zero_signal_keep_prob = zero_signal_keep_prob
        self.limit = limit
        self.shuffle_files = shuffle_files
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.parquet_batch_size = parquet_batch_size
        self.rank = int(rank)
        self.world_size = int(world_size)
        if self.rank < 0 or self.rank >= self.world_size:
            raise ValueError(
                f"Invalid distributed rank/world_size: rank={rank}, world_size={world_size}"
            )

    def _iter_files_for_worker(self) -> list[Path]:
        files = list(self.files)
        if self.shuffle_files:
            random.Random(self.seed).shuffle(files)
        return files

    def _global_worker_info(self) -> tuple[int, int]:
        worker = get_worker_info()
        if worker is None:
            return self.rank, self.world_size
        global_worker_id = self.rank * worker.num_workers + worker.id
        global_num_workers = self.world_size * worker.num_workers
        return global_worker_id, global_num_workers

    def _worker_seed_offset(self) -> int:
        global_worker_id, _ = self._global_worker_info()
        return global_worker_id

    def _row_rng(self, source_file: Path, row_index: int) -> random.Random:
        digest = hashlib.sha1(
            f"{self.seed}:{source_file.name}:{row_index}".encode("utf-8")
        ).hexdigest()
        return random.Random(int(digest[:16], 16))

    def _row_to_sample(self, row: dict[str, Any], source_file: Path, row_index: int, rng: random.Random) -> dict[str, Any] | None:
        rna_seq = str(row["rna_seq"]).upper().replace("U", "T")
        protein_symbol = str(row["protein_symbol"])
        protein_sequence = self.protein_sequences.get(protein_symbol)
        if protein_sequence is None:
            return None
        if self.max_rna_length is not None and len(rna_seq) > self.max_rna_length:
            return None
        if self.max_protein_length is not None and len(protein_sequence) > self.max_protein_length:
            return None

        signal = [float(value) for value in row["signal_vector"]]
        if len(signal) != len(rna_seq):
            signal = signal[: len(rna_seq)]
            if len(signal) < len(rna_seq):
                signal.extend([0.0] * (len(rna_seq) - len(signal)))

        signal_max = max(signal) if signal else 0.0
        if self.min_signal_max is not None and signal_max < self.min_signal_max:
            return None
        if signal_max <= 0.0 and self.zero_signal_keep_prob < 1.0:
            if rng.random() > self.zero_signal_keep_prob:
                return None

        cell_line = str(row["cell_line"])
        sample = {
            "sample_id": f"{source_file.stem}:{row_index}",
            "rna_seq": rna_seq,
            "protein_symbol": protein_symbol,
            "protein_sequence": protein_sequence,
            "cell_line": cell_line,
            "cell_id": self.cell_vocab.get(cell_line, self.unknown_cell_id),
            "signal_vector": torch.tensor(signal, dtype=torch.float32),
        }
        return crop_eclip_sample_rna(sample, self.rna_crop_size, rng)

    def _yield_from_files(self) -> Iterable[dict[str, Any]]:
        global_worker_id, global_num_workers = self._global_worker_info()
        valid_sample_index = 0
        emitted = 0
        for file_path in self._iter_files_for_worker():
            parquet_file = pq.ParquetFile(file_path)
            available = set(parquet_file.schema_arrow.names)
            columns = [col for col in BASE_COLUMNS if col in available]
            missing = sorted(set(BASE_COLUMNS) - set(columns))
            if missing:
                raise ValueError(f"{file_path} is missing required columns: {missing}")
            row_offset = 0
            for batch in parquet_file.iter_batches(batch_size=self.parquet_batch_size, columns=columns):
                rows = batch.to_pylist()
                for local_idx, row in enumerate(rows):
                    sample = self._row_to_sample(
                        row,
                        file_path,
                        row_offset + local_idx,
                        self._row_rng(file_path, row_offset + local_idx),
                    )
                    if sample is None:
                        continue
                    shard_id = valid_sample_index % global_num_workers
                    valid_sample_index += 1
                    if shard_id != global_worker_id:
                        continue
                    yield sample
                    emitted += 1
                    if self.limit is not None and emitted >= self.limit:
                        return
                row_offset += len(rows)

    def __iter__(self) -> Iterable[dict[str, Any]]:
        if self.shuffle_buffer <= 1:
            yield from self._yield_from_files()
            return
        rng = random.Random(self.seed + 9176 * self._worker_seed_offset())
        buffer: list[dict[str, Any]] = []
        for sample in self._yield_from_files():
            buffer.append(sample)
            if len(buffer) >= self.shuffle_buffer:
                idx = rng.randrange(len(buffer))
                yield buffer.pop(idx)
        while buffer:
            idx = rng.randrange(len(buffer))
            yield buffer.pop(idx)


def collate_eclip_ppft_samples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return samples


def build_protenix_sample_dict(sample: dict[str, Any]) -> dict[str, Any]:
    """Build an inference-style Protenix JSON sample from an eCLIP row."""

    name = str(sample.get("sample_id") or f"{sample['protein_symbol']}_{hash(sample['rna_seq'])}")
    return {
        "name": name.replace("/", "_"),
        "sequences": [
            {
                "proteinChain": {
                    "sequence": sample["protein_sequence"],
                    "count": 1,
                }
            },
            {
                "rnaSequence": {
                    "sequence": sample["rna_seq"].replace("T", "U"),
                    "count": 1,
                }
            },
        ],
    }
