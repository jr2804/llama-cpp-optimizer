# Portable Setup (prebuilt binaries)

Run llama.cpp without compiling or a system install: download official prebuilt
binaries into a project folder and switch backend (cpu / cuda / vulkan / …) by
re-downloading. Nothing touches PATH, Program Files, or the registry.

Works for **upstream** `ggml-org/llama.cpp` and for **forks** that ship specialized
quantizations or tensor types — see [Forks and specialized builds](#forks-and-specialized-builds).

## Where the binaries go

| Build | Default folder |
|-------|----------------|
| upstream `ggml-org/llama.cpp` | `./bin` |
| a fork, e.g. `PrismML-Eng/llama.cpp` | `./bin-<owner>` → `./bin-prismml-eng` |

**One folder per build variant. Never mix variants in one folder** — fork builds
carry different tensor/quant support and their DLLs overwrite each other's. Each
folder is a complete, self-contained, swappable install; switching variants means
pointing the runner at another folder. Re-running an install *rewrites* its target
folder, so old files never linger next to new ones.

Override the location with `--bin DIR` or `LLAMA_BIN`. The installer **refuses to
write inside the skill directory** — builds are ~100 MB each and belong to your
project. Add them to `.gitignore`:

```gitignore
bin/
bin-*/
```

## Install / update

Run from **your project directory** so the binaries land in your project, not in
the skill:

```bash
# <skill> is this skill's folder
S=<skill>/scripts/install-llama.py

# upstream, Vulkan, into ./bin
uv run "$S" --bin ./bin latest vulkan

# pinned version
uv run "$S" --bin ./bin b11165 vulkan

# a fork — installs into ./bin-prismml-eng automatically
uv run "$S" --repo PrismML-Eng/llama.cpp latest vulkan

# list available backends for your platform (no download, no destination needed)
uv run "$S"
uv run "$S" --repo PrismML-Eng/llama.cpp
```

| Arg / flag | Meaning |
|------------|---------|
| `VERSION` | release tag, default `latest` (e.g. `b11165`) |
| `BACKEND` | `cpu` (default), `cuda-12.4`, `cuda-13.4`, `vulkan`, `rocm-10.0`, `openvino-2026.4`, `sycl` |
| `--repo` | GitHub `OWNER/REPO` to install from (env `LLAMA_REPO`) |
| `--bin` | destination folder (env `LLAMA_BIN`) |

Run with no `BACKEND` to print what the repo actually publishes for your OS/arch.

### Why `latest` scans releases

`latest` picks the newest release that **really publishes a build for your
platform**, rather than trusting `releases/latest`. Two real traps this avoids:

- Upstream tags stable milestones (e.g. `v0.5.0`) that carry **no** binaries; the
  builds live in prerelease `bXXXXX` tags.
- Release assets upload incrementally, so the newest tag can briefly lack your
  OS/backend while an older one has it.

Forks differ here too: `PrismML-Eng/llama.cpp` marks its binary release as the
stable one, while upstream never does.

## Windows CUDA: runtime DLLs

On Windows, CUDA backend binaries do **not** ship with the CUDA runtime
(`cublas*.dll`, `cudart*.dll`). These live in a separate `cudart-*` asset published
alongside each release, and the installer downloads it automatically into the same
folder.

On Linux, CUDA is statically linked — no extra download. Note that upstream
publishes no generic Linux CUDA build at all (only Vulkan/openvino/sycl); some
forks do.

For a manual install, extract both archives into the *same* folder:

```bash
unzip llama-b11165-bin-win-cuda-12.4-x64.zip -d ./bin
unzip cudart-llama-bin-win-cuda-12.4-x64.zip -d ./bin
```

## Switching backend / version

Just run the installer again — it rewrites the target folder:

```bash
uv run "$S" --bin ./bin b11165 vulkan       # cpu -> vulkan
uv run "$S" --bin ./bin latest cuda-12.4    # vulkan -> newest cuda
```

`./bin` then holds every llama tool (`llama-cli`, `llama-server`, `llama-bench`, …)
plus its runtime DLLs. Running more than one build at the same time is covered in
[Endpoints](#endpoints-one-server-instance-per-build).

## Endpoints: one server instance per build

A `llama-server` process **is** one binary — the tensor/quant formats it understands
are compiled in. A fork build therefore cannot serve a model that needs a different
build, and router mode (one instance, many models) only works *within one build*.

| What you run | Endpoints |
|--------------|-----------|
| Several models, same build | **one** `llama-server`, one port, router mode (`--models-preset`) |
| Models needing different builds/forks | **one instance per build**, each on its own port |

Specialized forks usually **lag upstream** — they carry the quant kernels but trail on
new architectures and fixes. So running upstream *and* a fork side by side is the normal
case, not an edge case: latest architectures from upstream, the specialized quant from
the fork. Keep each build's folder and its `llama-server` on a distinct port:

| Build | Folder | Port |
|-------|--------|------|
| upstream `ggml-org/llama.cpp` | `./bin` | 8080 |
| Prism fork | `./bin-prismml-eng` | 8081 |

```bash
./bin/llama-server             --models-preset presets-main.ini  --port 8080
./bin-prismml-eng/llama-server --models-preset presets-prism.ini --port 8081
```

Each build gets **its own** `--models-preset` INI. The format is identical, but the
models and their flags differ — never point one build's INI at models that belong to
another build. Pin the fork's `VERSION` rather than tracking `latest`, so a fork update
cannot silently change what an existing endpoint serves.

## Forks and specialized builds

Some models need a llama.cpp **fork**: the quantization or tensor type is not in
upstream (yet). Examples:

| Model | Quant | Build needed |
|-------|-------|--------------|
| [Ternary-Bonsai](https://huggingface.co/prism-ml/Ternary-Bonsai-27B-gguf) `*-Q2_0.gguf` | ternary, group size 128 (fork packing) | **Prism fork** — `--repo PrismML-Eng/llama.cpp` |
| Ternary-Bonsai `*-Q2_0_g64.gguf` | ternary, group size 64 (official packing) | upstream |
| [Spark-X2.5](https://huggingface.co/XHToken/Spark-X2.5-4B-GGUF) | `spark2_5` arch | upstream (merged — no fork needed) |

Install a fork side by side with upstream, then run whichever binary matches the model:

```bash
uv run "$S" --repo PrismML-Eng/llama.cpp --bin ./bin-prism latest vulkan
./bin-prismml-eng/llama-cli -m models/prism-ml/Ternary-Bonsai-27B-Q2_0.gguf ...

uv run "$S" --bin ./bin latest vulkan
./bin/llama-cli -m models/XHToken/Spark-X2.5-4B-Q4_K_M.gguf ...
```

A fork build typically keeps upstream's asset naming
(`llama-<tag>-bin-<os>-<backend>-<arch>`), so the installer finds assets by
filename suffix and the fork's version prefix is irrelevant. Where a fork differs
— e.g. it publishes `bin-linux-*` instead of `bin-ubuntu-*`, or extra backends like
`hip-radeon` — the same lookup still resolves it. If a build is not found, the
installer prints every asset that *does* exist for your OS/arch.

A fork build only knows **its own** formats. If a model fails to load with a
tensor-offset error, check [caveats.md](caveats.md) before blaming the download.

## Automatic backend choice

`scripts/detect-system.py` reports a GPU backend per device (`cuda` / `vulkan`).
Wire it in — map detect labels to published assets (`cuda` → `cuda-12.4`):

```bash
BACKEND="$(uv run scripts/detect-system.py | uv run python -c '
import json,sys
g=json.load(sys.stdin).get("gpus",[])
b=next((x["backend"] for x in g if x["backend"]!="unknown"),"cpu")
print("cuda-12.4" if b=="cuda" else b)')"
uv run "$S" --bin ./bin latest "${BACKEND:-cpu}"
```
