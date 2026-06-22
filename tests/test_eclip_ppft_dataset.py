from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from protenix.data.eclip_ppft_dataset import (
    EclipPPFTDataset,
    build_protenix_sample_dict,
    crop_eclip_sample_rna,
)


def test_eclip_dataset_streams_rows_and_builds_protenix_sample(tmp_path: Path):
    data_dir = tmp_path / "data"
    train_dir = data_dir / "train"
    train_dir.mkdir(parents=True)
    table = pa.table(
        {
            "rna_seq": ["ATUT"],
            "protein_symbol": ["RBP1"],
            "cell_line": ["K562"],
            "signal_vector": [[0.0, 1.0]],
        }
    )
    pq.write_table(table, train_dir / "train-000000.parquet")
    protein_tsv = tmp_path / "proteins.tsv"
    protein_tsv.write_text(
        "protein_symbol\tprotein_sequence\nRBP1\tMKT\n",
        encoding="utf-8",
    )

    dataset = EclipPPFTDataset(
        data_dir=data_dir,
        split="train",
        protein_sequence_tsv=protein_tsv,
        max_rna_length=None,
        max_protein_length=None,
    )
    sample = next(iter(dataset))

    assert sample["rna_seq"] == "ATTT"
    assert sample["protein_sequence"] == "MKT"
    assert torch.equal(sample["signal_vector"], torch.tensor([0.0, 1.0, 0.0, 0.0]))

    protenix_sample = build_protenix_sample_dict(sample)
    assert protenix_sample["sequences"][0]["proteinChain"]["sequence"] == "MKT"
    assert protenix_sample["sequences"][1]["rnaSequence"]["sequence"] == "AUUU"


def test_signal_weighted_rna_crop_keeps_sequence_and_signal_aligned():
    import random

    sample = {
        "sample_id": "crop",
        "rna_seq": "A" * 10,
        "signal_vector": torch.tensor([0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 0.0, 0.0, 0.0, 0.0]),
        "profile_label": torch.arange(10, dtype=torch.float32),
    }

    cropped = crop_eclip_sample_rna(sample, crop_size=4, rng=random.Random(0))

    assert len(cropped["rna_seq"]) == 4
    assert cropped["signal_vector"].shape == (4,)
    assert cropped["profile_label"].shape == (4,)
    assert 0 <= cropped["rna_crop_start"] < cropped["rna_crop_end"] <= 10
    assert cropped["rna_original_length"] == 10
    assert torch.equal(
        cropped["profile_label"],
        torch.arange(10, dtype=torch.float32)[cropped["rna_crop_start"] : cropped["rna_crop_end"]],
    )
