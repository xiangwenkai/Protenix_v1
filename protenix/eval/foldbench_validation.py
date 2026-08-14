"""Distributed PXMeter/FoldBench validation for training-time structure checks."""

from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.distributed as dist

from protenix.data.inference.infer_dataloader import get_inference_dataloader
from protenix.utils.distributed import DIST_WRAPPER
from protenix.utils.seed import seed_everything
from protenix.utils.torch_utils import to_device
from runner.dumper import DataDumper
from runner.inference import update_inference_configs

logger = logging.getLogger(__name__)

TARGET_TYPE = "interface_protein_rna"


def _cfg(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _cfg_path(config: Any, keys: list[str], default: Any = None) -> Any:
    value = config
    for key in keys:
        value = _cfg(value, key, None)
        if value is None:
            return default
    return value


def _barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        try:
            if dst.resolve() == src.resolve():
                return
        except FileNotFoundError:
            pass
        dst.unlink()
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _safe_id(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_.-]+", "_", value.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "rna_msa"


def _normalize_rna_sequence(sequence: str) -> str:
    return re.sub(r"[^A-Za-z]", "", sequence).upper()


def _rna_cache_keys(sequence: str) -> list[str]:
    normalized = _normalize_rna_sequence(sequence)
    keys = []
    for key in (
        normalized,
        normalized.replace("T", "U"),
        normalized.replace("U", "T"),
    ):
        if key and key not in keys:
            keys.append(key)
    return keys


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return data


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp_path, path)


def _foldbench_rna_msa_cache_root(fold_cfg: Any) -> Path:
    configured = _cfg(fold_cfg, "foldbench_rna_msa_cache_root", None)
    if configured:
        return Path(configured).resolve()
    foldbench_repo = _cfg(fold_cfg, "foldbench_repo", None)
    if foldbench_repo:
        return Path(foldbench_repo).resolve() / "targets" / f"{TARGET_TYPE}_rna_msa"
    targets_dir = Path(
        _cfg(
            fold_cfg,
            "targets_dir",
            "/inspire/ssd/project/sais-bio/public/xiangwenkai/GITHUB/PXMeter/targets",
        )
    ).resolve()
    return targets_dir / f"{TARGET_TYPE}_rna_msa"


def _rna_msa_path(cache_root: Path, msa_id: str) -> Path:
    return cache_root / "msas" / msa_id / f"{msa_id}_all.a3m"


def _lookup_foldbench_rna_msa(
    sequence: str,
    cache_root: Path,
) -> tuple[str, Path] | None:
    mapping = _load_json_object(cache_root / "rna_sequence_to_pdb_chains.json")
    for key in _rna_cache_keys(sequence):
        msa_ids = _as_list(mapping.get(key))
        if not msa_ids:
            continue
        msa_id = msa_ids[0]
        msa_path = _rna_msa_path(cache_root, msa_id)
        if msa_path.exists() and msa_path.stat().st_size > 0:
            return msa_id, msa_path
    return None


def _path_exists(path_text: str) -> bool:
    if not path_text:
        return False
    path = Path(path_text)
    if path.exists():
        return True
    if not path.is_absolute():
        return (Path.cwd() / path).exists()
    return False


def _resolve_existing_path(path_text: str) -> Path | None:
    if not path_text:
        return None
    path = Path(path_text)
    if path.exists():
        return path.resolve()
    if not path.is_absolute():
        candidate = Path.cwd() / path
        if candidate.exists():
            return candidate.resolve()
    return None


def _attach_foldbench_rna_msa_cache(fold_cfg: Any, input_json: Path) -> Path:
    if not bool(_cfg(fold_cfg, "use_rna_msa", True)):
        return input_json
    cache_root = _foldbench_rna_msa_cache_root(fold_cfg)
    index_path = cache_root / "rna_sequence_to_pdb_chains.json"
    if not index_path.exists():
        return input_json

    data = json.loads(input_json.read_text(encoding="utf-8"))
    updated = 0
    for task in data:
        for item in task.get("sequences", []):
            rna_chain = item.get("rnaSequence")
            if not rna_chain:
                continue
            if _path_exists(str(rna_chain.get("unpairedMsaPath", ""))):
                continue
            hit = _lookup_foldbench_rna_msa(
                str(rna_chain.get("sequence", "")),
                cache_root,
            )
            if hit is None:
                continue
            _, msa_path = hit
            rna_chain["unpairedMsaPath"] = str(msa_path)
            updated += 1

    if updated == 0:
        return input_json

    output_path = input_json.with_name(
        f"{input_json.stem}-foldbench-rna-msa{input_json.suffix}"
    )
    output_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    logger.info(
        "Attached %d FoldBench RNA MSA cache hit(s) from %s",
        updated,
        cache_root,
    )
    return output_path


def _import_foldbench_rna_msa_cache(
    fold_cfg: Any,
    input_json: Path,
    searched_root: Path,
) -> None:
    if not bool(_cfg(fold_cfg, "use_rna_msa", True)):
        return

    cache_root = _foldbench_rna_msa_cache_root(fold_cfg)
    map_path = cache_root / "rna_sequence_to_pdb_chains.json"
    mapping = _load_json_object(map_path)
    searched_root = searched_root.resolve()
    data = json.loads(input_json.read_text(encoding="utf-8"))

    imported = 0
    reused = 0
    for task_idx, task in enumerate(data):
        task_name = str(task.get("name", f"task_{task_idx}"))
        for seq_idx, item in enumerate(task.get("sequences", [])):
            rna_chain = item.get("rnaSequence")
            if not rna_chain:
                continue
            sequence = _normalize_rna_sequence(str(rna_chain.get("sequence", "")))
            if not sequence:
                continue
            src = _resolve_existing_path(str(rna_chain.get("unpairedMsaPath", "")))
            if src is None or src.stat().st_size == 0:
                continue
            try:
                src.relative_to(searched_root)
            except ValueError:
                continue

            existing = _lookup_foldbench_rna_msa(sequence, cache_root)
            if existing is not None:
                reused += 1
                continue

            digest = hashlib.sha1(sequence.encode("utf-8")).hexdigest()[:10]
            msa_id = _safe_id(f"{task_name}_rna_{seq_idx}_{digest}")
            dest = _rna_msa_path(cache_root, msa_id)
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
            ids = _as_list(mapping.get(sequence))
            mapping[sequence] = [msa_id] + [item for item in ids if item != msa_id]
            imported += 1

    if imported:
        _write_json_atomic(map_path, mapping)
    logger.info(
        "FoldBench RNA MSA cache import: imported=%d reused=%d root=%s",
        imported,
        reused,
        cache_root,
    )


def _fallback_chains_from_name(sample_name: str) -> tuple[list[str], list[str]]:
    parts = sample_name.rsplit("_", 2)
    if len(parts) == 3:
        return [parts[1]], [parts[2]]
    return [], []


def _base_pdb_id(sample_name: str) -> str:
    match = re.match(r"interface_protein_rna_\d{6}_(.+?)_[^_]+_[^_]+$", sample_name)
    if match:
        return match.group(1).split("-assembly", 1)[0].split("-", 1)[0].upper()
    parts = sample_name.split("_")
    if len(parts) >= 2:
        return parts[-2].split("-assembly", 1)[0].split("-", 1)[0].upper()
    return sample_name.split("-assembly", 1)[0].split("-", 1)[0].upper()


def _base_pdb_id_lower(sample_name: str) -> str:
    return _base_pdb_id(sample_name).lower()


def _sample_name_suffix(sample_name: str) -> str:
    match = re.match(rf"{TARGET_TYPE}_\d{{6}}_(.+)$", sample_name)
    if match:
        return match.group(1)
    return sample_name


def _ground_truth_candidates(gt_dir: Path, sample_name: str) -> list[Path]:
    suffix = _sample_name_suffix(sample_name)
    candidates = [gt_dir / f"{sample_name}.cif"]
    candidates.extend(sorted(gt_dir.glob(f"{TARGET_TYPE}_*_{suffix}.cif")))
    deduped = []
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        deduped.append(candidate)
    return deduped


def _read_pdb_filter_list(pdb_list: Any) -> list[str] | None:
    if pdb_list is None:
        return None
    if isinstance(pdb_list, (list, tuple)):
        return [str(item).strip() for item in pdb_list if str(item).strip()]
    pdb_list = str(pdb_list)
    if not pdb_list:
        return None
    with Path(pdb_list).open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _training_hit_pdbs(train_configs: Any, fold_cfg: Any) -> tuple[set[str], dict[str, Any]]:
    train_set = _cfg(fold_cfg, "filter_train_set", None)
    if not train_set:
        train_sets = _cfg_path(train_configs, ["data", "train_sets"], [])
        train_set = train_sets[0] if train_sets else None
    if not train_set:
        raise ValueError("foldbench_eval.filter_to_train_pdb_hits requires a train set")

    base_info = _cfg_path(train_configs, ["data", str(train_set), "base_info"], None)
    if base_info is None:
        raise ValueError(f"Could not find data.{train_set}.base_info for FoldBench filter")

    indices_fpath = _cfg(base_info, "indices_fpath", None)
    if not indices_fpath:
        raise ValueError(f"data.{train_set}.base_info.indices_fpath is required")

    indices = pd.read_csv(indices_fpath)
    total_rows = int(len(indices))
    pdb_values = indices["pdb_id"].astype(str)

    pdb_filter = _read_pdb_filter_list(_cfg(base_info, "pdb_list", None))
    if pdb_filter is not None:
        pdb_filter_set = set(pdb_filter)
        indices = indices[pdb_values.isin(pdb_filter_set)].copy()
        pdb_values = indices["pdb_id"].astype(str)

    hit_pdbs = {value.lower() for value in pdb_values.unique()}
    report = {
        "enabled": True,
        "train_set": str(train_set),
        "indices_fpath": str(indices_fpath),
        "pdb_list": str(_cfg(base_info, "pdb_list", "")),
        "indices_rows_before_filter": total_rows,
        "indices_rows_after_filter": int(len(indices)),
        "hit_pdb_count": len(hit_pdbs),
    }
    if pdb_filter is not None:
        matched = {value.lower() for value in pdb_values.unique()}
        requested = {value.lower() for value in pdb_filter}
        report["pdb_list_count"] = len(pdb_filter)
        report["pdb_list_missing_from_indices"] = sorted(requested - matched)
    return hit_pdbs, report


def _normalize_chain_id(value: str) -> str:
    return re.sub(r"(?<=[A-Za-z])\d+$", "", str(value))


def _write_chain_normalized_cif(
    src: Path,
    dst: Path,
    target_chain_ids: list[str] | None = None,
) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if (
        target_chain_ids is None
        and dst.exists()
        and dst.stat().st_mtime >= src.stat().st_mtime
    ):
        return
    try:
        import biotite.structure.io.pdbx as pdbx

        cif_file = pdbx.CIFFile.read(str(src))
        atom_site = cif_file.block.get("atom_site")
        chain_map: dict[str, str] = {}
        if target_chain_ids and "label_asym_id" in atom_site:
            source_chain_ids = []
            for value in atom_site["label_asym_id"].as_array().tolist():
                chain_id = _normalize_chain_id(value)
                if chain_id not in source_chain_ids:
                    source_chain_ids.append(chain_id)
            if len(source_chain_ids) == len(target_chain_ids):
                chain_map = {
                    source_chain: target_chain
                    for source_chain, target_chain in zip(
                        source_chain_ids,
                        target_chain_ids,
                    )
                }
        for column_name in ("label_asym_id", "auth_asym_id"):
            if column_name in atom_site:
                values = atom_site[column_name].as_array().tolist()
                atom_site[column_name] = pdbx.CIFColumn(
                    pdbx.CIFData(
                        [
                            chain_map.get(
                                _normalize_chain_id(value),
                                _normalize_chain_id(value),
                            )
                            for value in values
                        ]
                    )
                )
        entity_poly = cif_file.block.get("entity_poly")
        if entity_poly is not None and "pdbx_strand_id" in entity_poly:
            values = entity_poly["pdbx_strand_id"].as_array().tolist()
            entity_poly["pdbx_strand_id"] = pdbx.CIFColumn(
                pdbx.CIFData(
                    [
                        ",".join(
                            chain_map.get(
                                _normalize_chain_id(chain_id),
                                _normalize_chain_id(chain_id),
                            )
                            for chain_id in str(value).split(",")
                        )
                        for value in values
                    ]
                )
            )
        cif_file.write(str(dst))
    except Exception:
        _link_or_copy(src, dst)


def _import_pxmeter_builder(repo_root: Path):
    script = (
        repo_root
        / "foldbench/protein_rna/code/run_pxmeter_other_targets_inference.py"
    )
    return _load_module(script, "pxmeter_other_targets_inference_for_foldbench")


def _prepare_targets_dir(
    fold_cfg: Any,
    train_configs: Any,
    work_dir: Path,
) -> Path:
    source_dir = Path(
        _cfg(
            fold_cfg,
            "targets_dir",
            "/inspire/ssd/project/sais-bio/public/xiangwenkai/GITHUB/PXMeter/targets",
        )
    ).resolve()
    source_csv = source_dir / f"{TARGET_TYPE}.csv"
    target_dir = work_dir / "targets"
    target_dir.mkdir(parents=True, exist_ok=True)

    target_df = pd.read_csv(source_csv)
    report: dict[str, Any] = {
        "enabled": False,
        "source_csv": str(source_csv),
        "target_rows_before_filter": int(len(target_df)),
    }
    if bool(_cfg(fold_cfg, "filter_to_train_pdb_hits", False)):
        hit_pdbs, train_report = _training_hit_pdbs(train_configs, fold_cfg)
        before = int(len(target_df))
        target_df["_base_pdb_id"] = (
            target_df["pdb_id"].astype(str).map(_base_pdb_id_lower)
        )
        removed_df = target_df[~target_df["_base_pdb_id"].isin(hit_pdbs)].copy()
        target_df = target_df[target_df["_base_pdb_id"].isin(hit_pdbs)].copy()
        target_df = target_df.drop(columns=["_base_pdb_id"])
        report.update(train_report)
        report.update(
            {
                "source_csv": str(source_csv),
                "target_rows_before_filter": before,
                "target_rows_after_filter": int(len(target_df)),
                "removed_target_rows": int(len(removed_df)),
                "removed_target_pdb_ids": removed_df["pdb_id"].astype(str).tolist(),
            }
        )
        if target_df.empty:
            raise ValueError("FoldBench target filter removed all rows")

    target_df.to_csv(target_dir / f"{TARGET_TYPE}.csv", index=False)
    report["target_csv"] = str(target_dir / f"{TARGET_TYPE}.csv")
    (work_dir / "filter_to_train_pdb_hits_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    return target_dir


def _generate_pxmeter_inputs(
    fold_cfg: Any,
    work_dir: Path,
    repo_root: Path,
    train_configs: Any,
) -> Path:
    px = _import_pxmeter_builder(repo_root)
    targets_dir = _prepare_targets_dir(fold_cfg, train_configs, work_dir)
    args = type("Args", (), {})()
    args.targets_dir = str(targets_dir)
    args.target_csv = ["interface_protein_rna.csv"]
    args.mmcif_dir = _cfg(
        fold_cfg,
        "mmcif_dir",
        "/inspire/ssd/project/sais-bio/public/Protein/data/AI_Models/protenix_v1_dataset/mmcif",
    )
    args.seq_to_pdb_index = _cfg(
        fold_cfg,
        "seq_to_pdb_index",
        "/inspire/ssd/project/sais-bio/public/Protein/data/AI_Models/protenix_v1_dataset/common/seq_to_pdb_index.json",
    )
    args.msa_template_dir = _cfg(
        fold_cfg,
        "msa_template_dir",
        "/inspire/ssd/project/sais-bio/public/Protein/data/AI_Models/protenix_v1_dataset/mmcif_msa_template",
    )
    args.work_dir = str(work_dir)
    args.max_targets = None
    args.start_index = _cfg(fold_cfg, "start_index", 0)
    args.fail_fast = bool(_cfg(fold_cfg, "fail_fast", False))
    args.allow_msa_search_fallback = True
    args.use_rna_msa = bool(_cfg(fold_cfg, "use_rna_msa", True))
    args.rna_msa_cache_root = _cfg(
        fold_cfg,
        "rna_msa_cache_root",
        "/inspire/ssd/project/sais-bio/public/xiangwenkai/GITHUB/Protenix/rna_data/rna_msa",
    )

    seq_to_msa_index = px.load_seq_to_msa_index(Path(args.seq_to_pdb_index).resolve())
    rna_msa_cache_index = None
    if args.use_rna_msa:
        rna_msa_cache_index = px.load_rna_msa_cache_index(
            Path(args.rna_msa_cache_root).resolve()
        )

    csv_path = Path(args.targets_dir).resolve() / "interface_protein_rna.csv"
    json_paths = px.generate_for_csv(csv_path, args, seq_to_msa_index, rna_msa_cache_index)

    combined = []
    for path in sorted(json_paths):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        combined.extend(data)

    combined_path = work_dir / TARGET_TYPE / "foldbench_input.json"
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    combined_path.write_text(json.dumps(combined, indent=2) + "\n", encoding="utf-8")
    return combined_path


def _preprocess_input_json(fold_cfg: Any, input_json: Path, output_dir: Path) -> Path:
    from runner.batch_inference import preprocess_input

    updated = preprocess_input(
        str(input_json),
        out_dir=str(output_dir),
        use_msa=bool(_cfg(fold_cfg, "use_msa", True)),
        use_template=False,
        use_rna_msa=bool(_cfg(fold_cfg, "use_rna_msa", True)),
        msa_server_mode=str(_cfg(fold_cfg, "msa_server_mode", "protenix")),
        ntrna_database_path=_cfg(
            fold_cfg,
            "ntrna_database_path",
            "/inspire/ssd/project/sais-bio/public/Protein/data/AI_Models/protenix_v1_dataset/search_database/nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta",
        ),
        rfam_database_path=_cfg(
            fold_cfg,
            "rfam_database_path",
            "/inspire/ssd/project/sais-bio/public/Protein/data/AI_Models/protenix_v1_dataset/search_database/rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta",
        ),
        rna_central_database_path=_cfg(
            fold_cfg,
            "rna_central_database_path",
            "/inspire/ssd/project/sais-bio/public/Protein/data/AI_Models/protenix_v1_dataset/search_database/rnacentral_active_seq_id_90_cov_80_linclust.fasta",
        ),
        nhmmer_n_cpu=int(_cfg(fold_cfg, "nhmmer_n_cpu", 2)),
    )
    return Path(updated)


def prepare_validation_input(
    fold_cfg: Any,
    eval_dir: Path,
    repo_root: Path,
    train_configs: Any,
) -> Path:
    marker = eval_dir / "input_json_path.txt"
    if DIST_WRAPPER.rank == 0:
        input_json = _generate_pxmeter_inputs(
            fold_cfg,
            eval_dir / "pxmeter_inputs",
            repo_root,
            train_configs,
        )
        input_json = _attach_foldbench_rna_msa_cache(fold_cfg, input_json)
        input_json = _preprocess_input_json(
            fold_cfg,
            input_json,
            eval_dir / "input_prep",
        )
        _import_foldbench_rna_msa_cache(fold_cfg, input_json, eval_dir / "input_prep")
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(str(input_json) + "\n", encoding="utf-8")
    _barrier()
    return Path(marker.read_text(encoding="utf-8").strip())


def _make_inference_configs(train_configs: Any, fold_cfg: Any, input_json: Path, output_dir: Path) -> Any:
    configs = copy.deepcopy(train_configs)
    configs.input_json_path = str(input_json)
    configs.dump_dir = str(output_dir)
    configs.num_workers = int(_cfg(fold_cfg, "num_workers", 0))
    configs.use_msa = bool(_cfg(fold_cfg, "use_msa", True))
    configs.use_rna_msa = bool(_cfg(fold_cfg, "use_rna_msa", True))
    configs.use_template = False
    configs.use_seeds_in_json = False
    configs.msa_pair_as_unpair = bool(_cfg(fold_cfg, "msa_pair_as_unpair", True))
    configs.sorted_by_ranking_score = True
    configs.need_atom_confidence = False
    configs.dtype = str(_cfg(fold_cfg, "dtype", configs.dtype))
    configs.seeds = [
        int(item.strip())
        for item in str(_cfg(fold_cfg, "seeds", "102")).split(",")
        if item.strip()
    ]
    configs.model.N_cycle = int(_cfg(fold_cfg, "cycle", configs.model.N_cycle))
    configs.sample_diffusion.N_sample = int(
        _cfg(fold_cfg, "sample", configs.sample_diffusion.N_sample)
    )
    configs.sample_diffusion.N_step = int(
        _cfg(fold_cfg, "step", configs.sample_diffusion.N_step)
    )
    return configs


@torch.no_grad()
def distributed_sample(
    model: torch.nn.Module,
    train_configs: Any,
    fold_cfg: Any,
    input_json: Path,
    output_dir: Path,
    device: torch.device,
) -> None:
    configs = _make_inference_configs(train_configs, fold_cfg, input_json, output_dir)
    dataloader = get_inference_dataloader(configs=configs)
    dumper = DataDumper(
        base_dir=str(output_dir),
        need_atom_confidence=False,
        sorted_by_ranking_score=True,
    )
    eval_precision = {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[configs.dtype]
    enable_amp = (
        torch.autocast(device_type="cuda", dtype=eval_precision)
        if torch.cuda.is_available()
        else nullcontext()
    )

    was_training = model.training
    original_model_configs = getattr(model, "configs", None)
    model.eval()
    try:
        for seed in configs.seeds:
            seed_everything(seed=seed, deterministic=configs.deterministic)
            for batch in dataloader:
                sample_name = "unknown"
                try:
                    data, atom_array, data_error_message = batch[0]
                    sample_name = data.get("sample_name", sample_name)
                    if data_error_message:
                        raise RuntimeError(data_error_message)
                    data = to_device(data, device)
                    new_configs = update_inference_configs(
                        configs, data["N_token"].item()
                    )
                    model.configs = new_configs
                    with enable_amp:
                        prediction, _, _ = model(
                            input_feature_dict=data["input_feature_dict"],
                            label_full_dict=None,
                            label_dict=None,
                            mode="inference",
                            mc_dropout_apply_rate=configs.mc_dropout_apply_rate,
                        )
                    dumper.dump(
                        dataset_name="",
                        pdb_id=sample_name,
                        seed=seed,
                        pred_dict=prediction,
                        atom_array=atom_array,
                        entity_poly_type={
                            k: v
                            for k, v in data["entity_poly_type"].items()
                            if v != "non-polymer"
                        },
                    )
                    logger.info(
                        "[FoldBench val][rank %s] sampled %s seed=%s",
                        DIST_WRAPPER.rank,
                        sample_name,
                        seed,
                    )
                except Exception:
                    error_dir = output_dir / "ERR"
                    error_dir.mkdir(parents=True, exist_ok=True)
                    (error_dir / f"{sample_name}.txt").write_text(
                        traceback.format_exc(), encoding="utf-8"
                    )
                    logger.exception(
                        "[FoldBench val][rank %s] sampling failed for %s",
                        DIST_WRAPPER.rank,
                        sample_name,
                    )
                finally:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
    finally:
        if original_model_configs is not None:
            model.configs = original_model_configs
        model.train(was_training)
    _barrier()


def _prediction_rows(
    input_json: Path,
    prediction_dir: Path,
    eval_root: Path,
    fold_cfg: Any,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tasks = json.loads(input_json.read_text(encoding="utf-8"))
    target_rows = []
    prediction_rows = []
    gt_dir = Path(_cfg(fold_cfg, "ground_truth_dir")).resolve()
    dockq_gt_dir = Path(_cfg(fold_cfg, "dockq_ground_truth_dir", gt_dir)).resolve()
    fold_gt_dir = eval_root / "ground_truths"
    dockq_eval_gt_dir = eval_root / "dockq_ground_truths"
    normalized_dir = eval_root / "chain_normalized_predictions"
    sample_pattern = re.compile(r"_sample_(\d+)(?:_postprocessed)?\.cif$")

    for task in tasks:
        sample_name = str(task["name"])
        proteins: list[str] = []
        rnas: list[str] = []
        target_chain_ids: list[str] = []
        for item in task.get("sequences", []):
            if "proteinChain" in item:
                chain_ids = _as_list(item["proteinChain"].get("label_asym_id"))
                proteins.extend(chain_ids)
                target_chain_ids.extend(chain_ids)
            elif "rnaSequence" in item:
                chain_ids = _as_list(item["rnaSequence"].get("label_asym_id"))
                rnas.extend(chain_ids)
                target_chain_ids.extend(chain_ids)
        if not proteins or not rnas:
            proteins, rnas = _fallback_chains_from_name(sample_name)
            target_chain_ids = proteins + rnas
        if not proteins or not rnas:
            continue

        for gt_path in _ground_truth_candidates(gt_dir, sample_name):
            if gt_path.exists():
                _link_or_copy(gt_path, fold_gt_dir / f"{sample_name}.cif")
                break

        base_id = _base_pdb_id(sample_name)
        dockq_candidates = _ground_truth_candidates(dockq_gt_dir, sample_name)
        dockq_candidates.extend(
            [
                dockq_gt_dir / f"{base_id}.cif",
                dockq_gt_dir / f"{base_id.lower()}.cif",
                dockq_gt_dir / f"{base_id.upper()}.cif",
            ]
        )
        for candidate in dockq_candidates:
            if candidate.exists():
                _write_chain_normalized_cif(
                    candidate,
                    dockq_eval_gt_dir / f"{sample_name}.cif",
                    target_chain_ids,
                )
                break

        for protein_chain in proteins:
            for rna_chain in rnas:
                target_rows.append(
                    {
                        "pdb_id": sample_name,
                        "interface_chain_id_1": protein_chain,
                        "interface_chain_id_2": rna_chain,
                        "interface_chain_type_1": "protein",
                        "interface_chain_type_2": "rna",
                    }
                )

        task_dir = prediction_dir / sample_name
        seen = set()
        for cif_path in sorted(task_dir.glob("seed_*/predictions/*_sample_*.cif")):
            if cif_path.name.endswith("_wounresol.cif"):
                continue
            match = sample_pattern.search(cif_path.name)
            if not match:
                continue
            postprocessed_peer = cif_path.with_name(cif_path.stem + "_postprocessed.cif")
            if not cif_path.name.endswith("_postprocessed.cif") and postprocessed_peer.exists():
                continue
            sample_idx = int(match.group(1))
            seed = cif_path.parents[1].name.removeprefix("seed_")
            key = (sample_name, seed, sample_idx)
            if key in seen:
                continue
            seen.add(key)
            confidence_path = cif_path.with_name(
                f"{sample_name}_summary_confidence_sample_{sample_idx}.json"
            )
            ranking_score = 0.0
            if confidence_path.exists():
                ranking_score = float(
                    json.loads(confidence_path.read_text(encoding="utf-8")).get(
                        "ranking_score", 0.0
                    )
                )
            normalized_path = normalized_dir / cif_path.relative_to(prediction_dir)
            _write_chain_normalized_cif(cif_path, normalized_path, target_chain_ids)
            prediction_rows.append(
                {
                    "pdb_id": sample_name,
                    "seed": seed,
                    "sample": sample_idx,
                    "ranking_score": ranking_score,
                    "prediction_path": str(normalized_path),
                }
            )

    return pd.DataFrame(target_rows), pd.DataFrame(prediction_rows)


def _run_dockqv2_case(args: tuple[dict[str, Any], str, str, int]) -> dict[str, Any] | None:
    row, native_dir, detail_dir, allowed_mismatches = args
    try:
        from evaluation.eval_by_dockqv2 import NumpyEncoder, dockq

        pdb_id = row["pdb_id"]
        chain1 = row["interface_chain_id_1"]
        chain2 = row["interface_chain_id_2"]
        seed = row["seed"]
        sample = row["sample"]
        prediction_path = row["prediction_path"]
        native_path = os.path.join(native_dir, f"{pdb_id}.cif")
        if not os.path.exists(prediction_path) or not os.path.exists(native_path):
            return None
        info = dockq(
            model_path=prediction_path,
            native_path=native_path,
            model_chains=[chain1, chain2],
            native_chains=[chain1, chain2],
            small_molecule=False,
            allowed_mismatches=allowed_mismatches,
        )
        if info is None or not info.get("best_result"):
            return None
        os.makedirs(detail_dir, exist_ok=True)
        output_path = os.path.join(
            detail_dir, f"{pdb_id}_{seed}_{sample}_{chain1}_{chain2}_dockqv2.json"
        )
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(info, handle, cls=NumpyEncoder)
        best_result = info["best_result"][list(info["best_result"].keys())[0]]
        result = dict(row)
        result.update(
            {
                "lrmsd": best_result["LRMSD"],
                "irmsd": best_result["iRMSD"],
                "dockq_score": best_result["DockQ"],
            }
        )
        return result
    except Exception:
        logger.error("DockQv2 failed:\n%s", traceback.format_exc())
        return None


def _conda_executable(fold_cfg: Any) -> str:
    configured = _cfg(fold_cfg, "foldbench_conda_executable", None)
    if configured:
        return str(configured)
    conda = shutil.which("conda")
    if conda:
        return conda
    fallback = Path(
        "/inspire/ssd/project/sais-bio/public/xiangwenkai/anaconda3/bin/conda"
    )
    if fallback.exists():
        return str(fallback)
    raise FileNotFoundError("Could not find conda executable for FoldBench metrics")


def _run_rank_evaluation(
    rank_df: pd.DataFrame,
    rank_eval_dir: Path,
    eval_root: Path,
    fold_cfg: Any,
) -> None:
    foldbench_repo = Path(
        _cfg(
            fold_cfg,
            "foldbench_repo",
            "/inspire/ssd/project/sais-bio/public/xiangwenkai/GITHUB/FoldBench",
        )
    ).resolve()

    raw_dir = rank_eval_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    if rank_df.empty:
        pd.DataFrame().to_csv(raw_dir / f"{TARGET_TYPE}_ost.csv", index=False)
        pd.DataFrame().to_csv(raw_dir / f"{TARGET_TYPE}_dockqv2.csv", index=False)
        return

    rank_eval_dir.mkdir(parents=True, exist_ok=True)
    rank_csv = rank_eval_dir / "rank_eval_input.csv"
    rank_df.to_csv(rank_csv, index=False)

    worker = Path(__file__).with_name("foldbench_metric_worker.py").resolve()
    cmd = [
        _conda_executable(fold_cfg),
        "run",
        "-n",
        str(_cfg(fold_cfg, "foldbench_conda_env", "foldbench")),
        "python",
        str(worker),
        "--rank-csv",
        str(rank_csv),
        "--rank-eval-dir",
        str(rank_eval_dir),
        "--eval-root",
        str(eval_root),
        "--foldbench-repo",
        str(foldbench_repo),
        "--target-type",
        TARGET_TYPE,
        "--dockq-allowed-mismatches",
        str(int(_cfg(fold_cfg, "dockq_allowed_mismatches", 8))),
    ]
    if bool(_cfg(fold_cfg, "skip_dockqv2", False)):
        cmd.append("--skip-dockqv2")

    stdout_path = rank_eval_dir / "foldbench_metric_worker.stdout.log"
    stderr_path = rank_eval_dir / "foldbench_metric_worker.stderr.log"
    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
        "w",
        encoding="utf-8",
    ) as stderr_handle:
        result = subprocess.run(
            cmd,
            check=False,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
        )
    if result.returncode != 0:
        stderr_tail = stderr_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        raise RuntimeError(
            "FoldBench metric worker failed with exit code "
            f"{result.returncode}: {' '.join(cmd)}\n{stderr_tail}"
        )


def _concat_rank_csvs(eval_root: Path, algorithm_name: str, world_size: int) -> None:
    final_raw = eval_root / "evaluation" / algorithm_name / "raw"
    final_raw.mkdir(parents=True, exist_ok=True)
    for suffix in ("ost", "dockqv2"):
        frames = []
        for rank in range(world_size):
            path = eval_root / "rank_eval" / f"rank_{rank}" / "raw" / f"{TARGET_TYPE}_{suffix}.csv"
            if path.exists() and path.stat().st_size > 1:
                try:
                    frames.append(pd.read_csv(path))
                except pd.errors.EmptyDataError:
                    pass
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(
                final_raw / f"{TARGET_TYPE}_{suffix}.csv", index=False
            )


def _best_rows(df: pd.DataFrame, metric: str, metric_type: str) -> pd.DataFrame:
    group_cols = ["pdb_id", "interface_chain_id_1", "interface_chain_id_2"]
    if df.empty:
        return df
    if metric_type == "best" and metric in {"irmsd", "lrmsd", "rmsd"}:
        idx = df.groupby(group_cols)[metric].idxmin()
    elif metric_type == "best":
        idx = df.groupby(group_cols)[metric].idxmax()
    else:
        idx = df.groupby(group_cols)["ranking_score"].idxmax()
    return df.loc[idx]


def _summarize(eval_root: Path, algorithm_name: str, metric_type: str) -> dict[str, float]:
    raw_dir = eval_root / "evaluation" / algorithm_name / "raw"
    metrics: dict[str, float] = {}
    ost_path = raw_dir / f"{TARGET_TYPE}_ost.csv"
    if ost_path.exists() and ost_path.stat().st_size > 1:
        ost_df = pd.read_csv(ost_path)
        if "lddt" in ost_df.columns:
            df = ost_df[ost_df["lddt"].notna()].copy()
            if not df.empty:
                metrics["foldbench/lddt"] = float(
                    _best_rows(df, "lddt", metric_type)["lddt"].mean()
                )
    dockq_path = raw_dir / f"{TARGET_TYPE}_dockqv2.csv"
    if dockq_path.exists() and dockq_path.stat().st_size > 1:
        dockq_df = pd.read_csv(dockq_path)
        for metric in ("dockq_score", "irmsd", "lrmsd"):
            if metric not in dockq_df.columns:
                continue
            df = dockq_df[dockq_df[metric].notna()].copy()
            if df.empty:
                continue
            best = _best_rows(df, metric, metric_type)
            if metric == "dockq_score":
                metrics["foldbench/dockq_score_success_rate"] = float(
                    (best[metric] >= 0.23).mean() * 100.0
                )
            else:
                metrics[f"foldbench/{metric}"] = float(best[metric].mean())

    summary_path = eval_root / "summary_table.csv"
    if DIST_WRAPPER.rank == 0:
        with summary_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["target", "metric", algorithm_name])
            writer.writeheader()
            for key, value in sorted(metrics.items()):
                writer.writerow(
                    {
                        "target": TARGET_TYPE,
                        "metric": key.removeprefix("foldbench/"),
                        algorithm_name: round(value, 4),
                    }
                )
    return metrics


def distributed_evaluate(input_json: Path, prediction_dir: Path, eval_root: Path, fold_cfg: Any, step: int) -> dict[str, float]:
    algorithm_name = f"foldbench_step_{step}"
    if DIST_WRAPPER.rank == 0:
        target_df, pred_df = _prediction_rows(input_json, prediction_dir, eval_root, fold_cfg)
        targets_dir = eval_root / "targets"
        algorithm_dir = eval_root / "evaluation" / algorithm_name
        targets_dir.mkdir(parents=True, exist_ok=True)
        algorithm_dir.mkdir(parents=True, exist_ok=True)
        target_df.to_csv(targets_dir / f"{TARGET_TYPE}.csv", index=False)
        pred_df.to_csv(algorithm_dir / "prediction_reference.csv", index=False)
        if pred_df.empty:
            merged = target_df.assign(prediction_path=None)
        else:
            merged = pd.merge(target_df, pred_df, on="pdb_id", how="left")
        merged = merged[merged["prediction_path"].notna()].copy()
        merged.to_csv(eval_root / "merged_eval_rows.csv", index=False)
    _barrier()

    merged_path = eval_root / "merged_eval_rows.csv"
    merged = pd.read_csv(merged_path) if merged_path.exists() else pd.DataFrame()
    rank_df = merged.iloc[DIST_WRAPPER.rank :: DIST_WRAPPER.world_size].copy()
    rank_eval_dir = eval_root / "rank_eval" / f"rank_{DIST_WRAPPER.rank}"
    _run_rank_evaluation(rank_df, rank_eval_dir, eval_root, fold_cfg)
    _barrier()

    metrics: dict[str, float] = {}
    if DIST_WRAPPER.rank == 0:
        _concat_rank_csvs(eval_root, algorithm_name, DIST_WRAPPER.world_size)
        metrics = _summarize(
            eval_root,
            algorithm_name,
            str(_cfg(fold_cfg, "metric_type", "rank")),
        )
        (eval_root / "metrics.json").write_text(
            json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
        )
    gathered = DIST_WRAPPER.all_gather_object(metrics if DIST_WRAPPER.rank == 0 else {})
    for item in gathered:
        if item:
            return item
    return {}


def run_foldbench_validation(
    model: torch.nn.Module,
    train_configs: Any,
    fold_cfg: Any,
    run_dir: Path,
    step: int,
    device: torch.device,
) -> dict[str, float]:
    eval_dir = run_dir / "foldbench_validation" / f"step_{step:08d}"
    repo_root = Path(__file__).resolve().parents[2]
    input_json = prepare_validation_input(
        fold_cfg,
        eval_dir,
        repo_root,
        train_configs,
    )
    prediction_dir = eval_dir / "predictions"
    t0 = time.time()
    distributed_sample(model, train_configs, fold_cfg, input_json, prediction_dir, device)
    _barrier()
    metrics = distributed_evaluate(input_json, prediction_dir, eval_dir, fold_cfg, step)
    if DIST_WRAPPER.rank == 0:
        metrics["foldbench/elapsed_sec"] = time.time() - t0
        logger.info("FoldBench validation metrics at step %s: %s", step, metrics)
    return metrics
