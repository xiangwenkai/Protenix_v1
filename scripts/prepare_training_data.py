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

import argparse
import csv
from pathlib import Path
from typing import Optional

import pandas as pd
from joblib import delayed, Parallel

from protenix.data.pipeline.data_pipeline import DataPipeline
from protenix.utils.file_io import dump_gzip_pickle
from tqdm import tqdm


def gen_a_bioassembly_data(
    mmcif: Path,
    bioassembly_output_dir: Path,
    cluster_file: Optional[Path],
    distillation: bool = False,
) -> Optional[list[dict]]:
    """
    Generates bioassembly data from an mmCIF file and saves it to the specified output directory.

    Args:
        mmcif (Path): Path to the mmCIF file.
        bioassembly_output_dir (Path): Directory where the bioassembly data will be saved.
        cluster_file (Optional[Path]): Path to the cluster file, if available.
        distillation (bool, optional): Flag indicating whether to use the 'Distillation' setting. Defaults to False.

    Returns:
        Optional[list[dict]]: A list of sample indices if data is successfully generated, otherwise None.
    """
    if distillation:
        dataset = "Distillation"
    else:
        dataset = "WeightedPDB"

    sample_indices_list, bioassembly_dict = DataPipeline.get_data_from_mmcif(
        mmcif, cluster_file, dataset
    )

    if sample_indices_list and bioassembly_dict:
        pdb_id = bioassembly_dict["pdb_id"]
        # save to output dir
        dump_gzip_pickle(bioassembly_dict, bioassembly_output_dir / f"{pdb_id}.pkl.gz")
        return sample_indices_list


def gen_data_from_mmcifs(
    mmcif_list: list[Path],
    output_indices_csv: Path,
    bioassembly_output_dir: Path,
    cluster_file: Optional[Path],
    distillation: bool = False,
    num_workers: int = 1,
):
    """
    Generates training data from a list of mmCIF files and saves the results to a CSV file.

    Args:
        mmcif_list (list[Path]): List of paths to mmCIF files.
        output_indices_csv (Path): Path to the output CSV file where the indices will be saved.
        bioassembly_output_dir (Path): Directory where the bioassembly output will be stored.
        cluster_file (Optional[Path]): Path to the cluster file. If None, clustering is not performed.
        distillation (bool, optional): Flag indicating whether to use the 'Distillation' setting. Defaults to False.
        num_workers (int, optional): Number of parallel workers to use. Defaults to 1.
    """

    all_sample_indices_list = [
        r
        for r in tqdm(
            Parallel(n_jobs=num_workers, return_as="generator_unordered")(
                delayed(gen_a_bioassembly_data)(
                    mmcif, bioassembly_output_dir, cluster_file, distillation
                )
                for mmcif in mmcif_list
            ),
            total=len(mmcif_list),
        )
    ]

    merged_results = []
    for sample_indices_list in all_sample_indices_list:
        if sample_indices_list:
            merged_results += sample_indices_list
    df = pd.DataFrame(merged_results)

    df.to_csv(output_indices_csv, index=False, quoting=csv.QUOTE_NONNUMERIC)


def run_gen_data(
    input_path: Path,
    output_indices_csv: Path,
    bioassembly_output_dir: Path,
    cluster_file: Optional[Path],
    distillation: bool = False,
    num_workers: int = 1,
    max_file_size_mb: Optional[float] = None,
    skipped_files_txt: Optional[Path] = None,
):
    """
    Generates data from MMCIF files and saves the output to specified locations.

    Args:
        input_path (str): Path to the input directory containing MMCIF files or a text file listing MMCIF file paths.
        output_indices_csv (str): Path to the output CSV file where indices will be saved.
        bioassembly_output_dir (str): Directory where bioassembly outputs will be saved.
        cluster_file (Optional[str]): Path to the cluster file, if any.
        distillation (bool, optional): Flag indicating whether to use the 'Distillation' setting. Defaults to False.
        num_workers (int, optional): Number of worker processes to use. Defaults to 1.
        max_file_size_mb (float, optional): Skip mmCIF files larger than this size in MiB.
        skipped_files_txt (Path, optional): Path to save skipped mmCIF file paths.

    Raises:
        NotImplementedError: If the input path is not a directory or a text file.
    """

    input_path = Path(input_path)
    bioassembly_output_dir = Path(bioassembly_output_dir)
    output_indices_csv = Path(output_indices_csv)

    # create directory for output
    output_indices_csv.parent.mkdir(parents=True, exist_ok=True)
    bioassembly_output_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_dir():
        mmcif_list = list(input_path.glob("*.cif")) + list(input_path.glob("*.cif.gz"))
    elif input_path.suffix == ".txt":
        with open(input_path) as f:
            mmcif_list = [Path(i.strip()) for i in f.readlines() if i.strip()]
    else:
        raise NotImplementedError(f"Unsupported input path: {input_path}")

    if max_file_size_mb is not None:
        max_file_size_bytes = max_file_size_mb * 1024 * 1024
        kept_mmcifs = []
        skipped_mmcifs = []
        for mmcif in mmcif_list:
            file_size = Path(mmcif).stat().st_size
            if file_size > max_file_size_bytes:
                skipped_mmcifs.append(Path(mmcif))
            else:
                kept_mmcifs.append(Path(mmcif))
        mmcif_list = kept_mmcifs
        print(
            f"Skip {len(skipped_mmcifs)} mmCIF files larger than "
            f"{max_file_size_mb:g} MiB; keep {len(mmcif_list)} files.",
            flush=True,
        )
        if skipped_files_txt is not None:
            skipped_files_txt = Path(skipped_files_txt)
            skipped_files_txt.parent.mkdir(parents=True, exist_ok=True)
            skipped_files_txt.write_text(
                "\n".join(str(i) for i in skipped_mmcifs) + ("\n" if skipped_mmcifs else "")
            )
            print(f"Skipped file list written to {skipped_files_txt}", flush=True)
        elif skipped_mmcifs:
            preview = ", ".join(str(i) for i in skipped_mmcifs[:20])
            suffix = " ..." if len(skipped_mmcifs) > 20 else ""
            print(f"Skipped files: {preview}{suffix}", flush=True)

    gen_data_from_mmcifs(
        mmcif_list,
        output_indices_csv,
        bioassembly_output_dir,
        cluster_file,
        distillation,
        num_workers,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_path",
        type=Path,
        default=None,
        help="Path to the input directory containing MMCIF files or a .txt file listing MMCIF file paths.",
    )
    parser.add_argument(
        "-o",
        "--output_csv",
        type=Path,
        default=None,
        help="Path to the output CSV file where indices will be saved.",
    )
    parser.add_argument(
        "-b",
        "--bio_output_dir",
        type=Path,
        default=None,
        help="Directory where bioassembly outputs will be saved.",
    )
    parser.add_argument(
        "-c",
        "--cluster_file",
        type=Path,
        default=None,
        help="Path to the cluster txt file, if any",
    )

    parser.add_argument(
        "-d",
        "--distillation",
        action="store_true",
        help="Whether to use the 'Distillation' setting",
    )

    parser.add_argument(
        "-n",
        "--n_cpu",
        type=int,
        default=1,
        help="Number of worker processes to use. Defaults to 1.",
    )
    parser.add_argument(
        "--max_file_size_mb",
        type=float,
        default=None,
        help="Skip mmCIF files larger than this size in MiB. Defaults to no size filter.",
    )
    parser.add_argument(
        "--skipped_files_txt",
        type=Path,
        default=None,
        help="Optional path to save skipped mmCIF file paths.",
    )

    args = parser.parse_args()

    run_gen_data(
        input_path=args.input_path,
        output_indices_csv=args.output_csv,
        bioassembly_output_dir=args.bio_output_dir,
        cluster_file=args.cluster_file,
        distillation=args.distillation,
        num_workers=args.n_cpu,
        max_file_size_mb=args.max_file_size_mb,
        skipped_files_txt=args.skipped_files_txt,
    )
