#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" ]]; then
  echo 'Usage: bash install.sh [NEW_ENVIRONMENT_PATH]'
  echo 'Default: ~/.local/share/gsedit-scd. Requires Conda, Linux x86_64 and an NVIDIA GPU.'
  exit 0
fi
if (( $# > 1 )); then
  echo 'Usage: bash install.sh [NEW_ENVIRONMENT_PATH]' >&2
  exit 1
fi
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
  echo 'This installer supports Linux x86_64.' >&2
  exit 1
fi

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
env_prefix="$(realpath -m -- "${1:-$HOME/.local/share/gsedit-scd}")"
if [[ -e "$env_prefix" || -L "$env_prefix" ]]; then
  echo "Environment path already exists: $env_prefix. Choose a new path; nothing will be overwritten." >&2
  exit 1
fi
conda_executable="${CONDA_EXE:-conda}"
if ! command -v "$conda_executable" >/dev/null 2>&1; then
  echo 'Conda is required. Open a terminal with Conda available, then rerun this command.' >&2
  exit 1
fi

cd -- "$repo_dir"
mkdir -p .tmp .cache
export TMPDIR="$repo_dir/.tmp"
export XDG_CACHE_HOME="$repo_dir/.cache"

"$conda_executable" create --prefix "$env_prefix" --override-channels -c conda-forge -y \
  python=3.12 pip cuda-nvcc=12.8 cuda-cudart-dev=12.8 \
  cuda-driver-dev_linux-64=12.8 gcc_linux-64=13 gxx_linux-64=13

# Conda applies these settings on activation and restores the previous values on deactivation.
"$conda_executable" env config vars set --prefix "$env_prefix" \
  "CUDA_HOME=$env_prefix" \
  "CC=$env_prefix/bin/x86_64-conda-linux-gnu-gcc" \
  "CXX=$env_prefix/bin/x86_64-conda-linux-gnu-g++" \
  "CUDAHOSTCXX=$env_prefix/bin/x86_64-conda-linux-gnu-g++" \
  "CPATH=$env_prefix/targets/x86_64-linux/include" \
  "LIBRARY_PATH=$env_prefix/targets/x86_64-linux/lib/stubs" \
  "TMPDIR=$TMPDIR" "XDG_CACHE_HOME=$XDG_CACHE_HOME"

run=("$conda_executable" run --no-capture-output --prefix "$env_prefix")
"${run[@]}" python -m pip install torch==2.11.0 torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cu128
"${run[@]}" python -m pip install -r requirements.txt
"${run[@]}" python -m pip install setuptools wheel ninja

gpu_arches="$("${run[@]}" python -c '
import torch
if not torch.cuda.is_available():
    raise SystemExit("No CUDA GPU is visible. Check the NVIDIA driver before compiling the extensions.")
print(";".join(sorted({"%d.%d" % torch.cuda.get_device_capability(i)
                       for i in range(torch.cuda.device_count())})))
')"
"$conda_executable" env config vars set --prefix "$env_prefix" "TORCH_CUDA_ARCH_LIST=$gpu_arches"

"${run[@]}" python -m pip install --no-build-isolation --no-deps ./submodules/diff-gaussian-rasterization_fastgs
"${run[@]}" python -m pip install --no-build-isolation --no-deps ./submodules/simple-knn

echo 'Installation complete. Activate the environment before running project commands:'
printf 'conda activate %q\n' "$env_prefix"
