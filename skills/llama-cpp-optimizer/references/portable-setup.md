# Portable Setup (prebuilt binaries)

Run llama.cpp without compiling or a system install: download official prebuilt
binaries into a project folder (`./bin` by default) and switch backends (cpu /
cuda / vulkan / …) by re-downloading. All binaries live next to your project, so
versions and backends are trivially swappable; nothing touches PATH or Program Files.

## Install / Update

```bash
# print available backends for the latest release (no download)
scripts/install-llama.sh latest

# install latest, cpu backend -> ./bin
scripts/install-llama.sh latest cpu

# install a specific backend + pinned version
scripts/install-llama.sh b10520 cuda-12.4

# install into a custom folder
LLAMA_BIN=./tools/bin scripts/install-llama.sh latest vulkan
```

`scripts/install-llama.sh [VERSION] [BACKEND]`:

| Arg       | Meaning                                             |
|-----------|-----------------------------------------------------|
| `VERSION` | release tag, default `latest` (e.g. `b10520`)       |
| `BACKEND` | `cpu` (default), `cuda-12.4`, `cuda-13.3`, `vulkan`, `rocm-7.14`, `openvino-2026.3`, `sycl` |

Backends query the GitHub release asset list, so the exact names follow what
`ggml-org/llama.cpp` actually publishes. Run with no `BACKEND` to see the current
list. It resolves `latest` via the GitHub API, downloads the matching
`llama-<ver>-bin-<os>-<backend>-<arch>.zip`/`.tar.gz`, extracts into `./bin`, and
flattens the versioned subfolder llama zips produce.

## Switching backend / bumping version

Re-running the script rewrites `./bin` (old files removed first), so switching is
just another invocation:

```bash
# cpu -> vulkan
scripts/install-llama.sh b10520 vulkan
# vulkan -> newest cuda
scripts/install-llama.sh latest cuda-12.4
```

`./bin` holds every llama tool (`llama-cli`, `llama-server`, `llama-bench`, …)
plus the runtime DLLs, so point any runner at that folder. To serve differently
configured instances, keep separate folders and pass `LLAMA_BIN`.

## Automatic backend choice

`scripts/detect-system.py` already reports a GPU backend per device (`cuda` / `vulkan`).
Wire it in — map detect labels to published assets (`cuda` -> `cuda-12.4`):

```bash
# picks the first non-CPU GPU backend, falls back to cpu
BACKEND="$(uv run scripts/detect-system.py | python3 -c '
import json,sys
g=json.load(sys.stdin).get("gpus",[])
b=next((x["backend"] for x in g if x["backend"]!="unknown"),"cpu")
print("cuda-12.4" if b=="cuda" else b)')"
scripts/install-llama.sh latest "${BACKEND:-cpu}"
```