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


import os
import re
import shutil
import subprocess
import tempfile
from typing import Any, Optional

from torch.utils.cpp_extension import load


def _cuda_include_paths(cuda_home: str | None) -> list[str]:
    paths = []
    candidates = []
    if cuda_home:
        candidates.extend(
            [
                os.path.join(cuda_home, "include"),
                os.path.join(cuda_home, "targets", "x86_64-linux", "include"),
            ]
        )
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.extend(
            [
                os.path.join(conda_prefix, "include"),
                os.path.join(conda_prefix, "targets", "x86_64-linux", "include"),
                os.path.join(
                    conda_prefix,
                    "lib",
                    f"python{os.sys.version_info.major}.{os.sys.version_info.minor}",
                    "site-packages",
                    "nvidia",
                    "cuda_runtime",
                    "include",
                ),
            ]
        )

    for path in candidates:
        if path not in paths and os.path.isfile(os.path.join(path, "cuda_runtime.h")):
            paths.append(path)
    return paths


def _supported_arches(nvcc: str | None) -> set[str]:
    try:
        out = subprocess.check_output(
            [nvcc, "--list-gpu-arch"], text=True, stderr=subprocess.STDOUT
        )
        return set(re.findall(r"compute_(\d+)", out))
    except Exception:
        return {"70", "80", "86", "90"}


def _torch_accepts_arch(arch: str) -> bool:
    from torch.utils.cpp_extension import _get_cuda_arch_flags

    old_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")
    try:
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{int(arch) // 10}.{int(arch) % 10}"
        _get_cuda_arch_flags()
        return True
    except Exception:
        return False
    finally:
        if old_arch_list is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = old_arch_list


def _normalize_arch(arch: str) -> str:
    return arch.strip().replace(".", "").replace("sm_", "").replace("compute_", "")


def _requested_arches() -> list[str]:
    override = os.environ.get("PROTENIX_CUDA_ARCH_LIST")
    if override:
        return [_normalize_arch(arch) for arch in re.split(r"[;, ]+", override) if arch]

    try:
        import torch

        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            return [f"{major}{minor}"]
    except Exception:
        pass

    return ["70", "75", "80", "86", "89", "90", "100", "103", "120", "121"]


def _extra_ldflags() -> list[str]:
    flags = []
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        conda_lib = os.path.join(conda_prefix, "lib")
        if os.path.isdir(conda_lib):
            flags.extend(
                [
                    f"-L{conda_lib}",
                    f"-Wl,-rpath,{conda_lib}",
                    "-static-libstdc++",
                    "-static-libgcc",
                ]
            )
    return flags


def compile(
    name: str,
    sources: list[str],
    extra_include_paths: list[str],
    build_directory: Optional[str] = None,
) -> Any:
    # Query supported architectures from nvcc (resolved via PyTorch's
    # CUDA_HOME so we use the same toolchain as cpp_extension.load).
    from torch.utils.cpp_extension import CUDA_HOME

    _nvcc = shutil.which("nvcc")
    if CUDA_HOME:
        _candidate = os.path.join(CUDA_HOME, "bin", "nvcc")
        if os.path.isfile(_candidate):
            _nvcc = _candidate

    _supported = _supported_arches(_nvcc)

    # Compile for the requested or current GPU by default. This allows
    # Blackwell/sm120 when the local CUDA and PyTorch toolchain support it.
    _wanted = _requested_arches()
    _enabled = [arch for arch in _wanted if arch in _supported and _torch_accepts_arch(arch)]
    if not _enabled:
        _enabled = [
            arch
            for arch in ["70", "75", "80", "86", "89", "90"]
            if arch in _supported and _torch_accepts_arch(arch)
        ]
    gencode_flags = []
    for arch in _enabled:
        gencode_flags += ["-gencode", f"arch=compute_{arch},code=sm_{arch}"]
    if not gencode_flags:
        gencode_flags = ["-gencode", "arch=compute_80,code=sm_80"]

    # Build TORCH_CUDA_ARCH_LIST dynamically from supported architectures.
    _arch_list = [f"{int(c) // 10}.{int(c) % 10}" for c in _enabled]
    os.environ["TORCH_CUDA_ARCH_LIST"] = ";".join(_arch_list) if _arch_list else "8.0"
    extra_include_paths = extra_include_paths + _cuda_include_paths(CUDA_HOME)
    if build_directory is None:
        build_root = os.environ.get(
            "PROTENIX_TORCH_EXTENSIONS_DIR",
            os.path.join(tempfile.gettempdir(), "protenix_torch_extensions"),
        )
        build_directory = os.path.join(build_root, name)
    os.makedirs(build_directory, exist_ok=True)

    return load(
        name=name,
        sources=sources,
        extra_include_paths=extra_include_paths,
        extra_cflags=[
            "-O3",
            "-DVERSION_GE_1_1",
            "-DVERSION_GE_1_3",
            "-DVERSION_GE_1_5",
        ],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-DVERSION_GE_1_1",
            "-DVERSION_GE_1_3",
            "-DVERSION_GE_1_5",
            "-std=c++17",
            "-maxrregcount=32",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
        ]
        + gencode_flags,
        extra_ldflags=_extra_ldflags(),
        verbose=True,
        build_directory=build_directory,
    )
