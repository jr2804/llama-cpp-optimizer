# Portable Setup (prebuilt binaries)

Run llama.cpp without compiling or a system install: download official prebuilt
binaries into a project folder (`./bin` by default) and switch backends (cpu /
cuda / vulkan / …) by re-downloading. All binaries live next to your project, so
versions and backends are trivially swappable; nothing touches PATH or Program Files.

## Install / Update

```bash
# print available backends for the latest release (no download)
uv run scripts/install-llama.py

# install latest, cpu backend -> ./bin
uv run scripts/install-llama.py latest cpu

# install a specific backend + pinned version
uv run scripts/install-llama.py b10520 cuda-12.4

# install into a custom folder
LLAMA_BIN=./tools/bin uv run scripts/install-llama.py latest vulkan
```

`uv run scripts/install-llama.py [VERSION] [BACKEND]`:

| Arg       | Meaning                                             |
|-----------|-----------------------------------------------------|
| `VERSION` | release tag, default `latest` (e.g. `b10520`)       |
| `BACKEND` | `cpu` (default), `cuda-12.4`, `cuda-13.3`, `vulkan`, `rocm-7.14`, `openvino-2026.3`, `sycl` |

Backends query the GitHub release asset list, so the exact names follow what
`ggml-org/llama.cpp` actually publishes. Run with no `BACKEND` to see the current
list. It resolves `latest` via the GitHub API, downloads the matching
`llama-<ver>-bin-<os>-<backend>-<arch>.zip`/`.tar.gz`, extracts into `./bin`, and
flattens the versioned subfolder llama zips produce.

## Windows CUDA: runtime DLLs

On Windows, CUDA backend binaries do **not** ship with the CUDA runtime
(`cublas*.dll`, `cudart*.dll`). These live in a separate `cudart-llama-*` asset
published alongside every release. The installer handles this automatically:
when a CUDA backend is selected on Windows, it downloads and extracts the
matching runtime asset into the same `./bin` folder.

On Linux, CUDA is statically linked — no extra download needed.

If you install manually (without the script), grab both:

```bash
# main binaries
curl -L -o llama.zip "https://huggingface.co/ggml-org/llama.cpp/releases/download/b10520/llama-b10520-bin-win-cuda-12.4-x64.zip"
# CUDA runtime (Windows only)
curl -L -o cudart.zip "https://huggingface.co/ggml-org/llama.cpp/releases/download/b10520/cudart-llama-bin-win-cuda-12.4-x64.zip"
# extract both into the same folder
```

## Switching backend / bumping version

Re-running the script rewrites `./bin` (old files removed first), so switching is
just another invocation:

```bash
# cpu -> vulkan
uv run scripts/install-llama.py b10520 vulkan
# vulkan -> newest cuda
uv run scripts/install-llama.py latest cuda-12.4
```

`./bin` holds every llama tool (`llama-cli`, `llama-server`, `llama-bench`, …)
plus the runtime DLLs, so point any runner at that folder. To serve differently
configured instances, keep separate folders and pass `LLAMA_BIN`.

## Automatic backend choice

`scripts/detect-system.py` already reports a GPU backend per device (`cuda` / `vulkan`).
Wire it in — map detect labels to published assets (`cuda` -> `cuda-12.4`):

```bash
# picks the first non-CPU GPU backend, falls back to cpu
BACKEND="$(uv run scripts/detect-system.py | uv run python -c '
import json,sys
g=json.load(sys.stdin).get("gpus",[])
b=next((x["backend"] for x in g if x["backend"]!="unknown"),"cpu")
print("cuda-12.4" if b=="cuda" else b)')"
uv run scripts/install-llama.py latest "${BACKEND:-cpu}"
```
