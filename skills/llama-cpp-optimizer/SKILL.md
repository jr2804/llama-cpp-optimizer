---
name: llama-cpp-optimizer
description: Launch optimized LLMs via llama.cpp with faster inference and lower VRAM consumption. Use when the user wants to run a local LLM via llama.cpp, tune inference parameters (context size, batch size, GPU layers, KV cache, MoE offload), determine the best model configuration for their hardware, or apply VRAM/RAM reduction hacks. Covers model selection, quantization, parameter derivation from model metadata + system capabilities, and execution via llama-cli/llama-server.
---

# llama.cpp Optimizer

Launch optimized llama.cpp models with faster inference and lower VRAM/RAM consumption, using parameters derived from model metadata and local system capabilities.

## When to Use

- User wants to run a local LLM via llama.cpp
- User needs help tuning inference parameters (context size, batch size, GPU layers, KV cache, etc.)
- User wants to reduce VRAM/RAM usage or speed up inference
- User wants to determine the best model configuration for their hardware
- User asks about model selection, quantization levels, or MoE optimization
- User provides a Hugging Face model URL and wants to run it locally

## Quick Start

```bash
# 0. (optional) install portable llama.cpp binaries into ./bin
uv run scripts/install-llama.py latest vulkan  # or cpu / cuda-12.4; see references/portable-setup.md

# 1. Detect system capabilities
uv run scripts/detect-system.py

# 2. Get model metadata from Hugging Face
uv run scripts/model-info.py Qwen/Qwen3.6-35B-A3B

# 3. Derive optimal parameters (auto-detect system + model)
uv run scripts/model-info.py Qwen/Qwen3.6-35B-A3B | uv run scripts/derive-params.py --model -

# 4. Run the model with derived parameters
llama-cli --hf-repo <user>/<model> --hf-file <file.gguf> \
  $(uv run scripts/model-info.py <model> | uv run scripts/derive-params.py --model - --cli)
```

## Core Workflow

### 1. Detect System Capabilities

Run system detection to determine available hardware:

```bash
# Using Python script (recommended)
uv run scripts/detect-system.py

# Or using raw commands
nvidia-smi --query-gpu=name,memory.total,memory.free,compute_cap --format=csv,noheader
powershell -Command "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"
powershell -Command "[Environment]::ProcessorCount"
```

See [references/system-capabilities.md](references/system-capabilities.md) for complete detection commands.

### 2. Retrieve Model Metadata

Given a Hugging Face model URL or identifier, retrieve model architecture metadata:

```bash
# Using Python script (recommended)
uv run scripts/model-info.py Nanbeige/Nanbeige4.2-3B
uv run scripts/model-info.py https://huggingface.co/Qwen/Qwen3.6-35B-A3B
uv run scripts/model-info.py ggml-org/Qwen3.6-35B-A3B-GGUF --list-files

# Or using uvx hf.
uvx hf. -- model-info Nanbeige/Nanbeige4.2-3B
```

See [references/hf-model-info.md](references/hf-model-info.md) for complete instructions.

### 3. Derive Optimal Parameters

Combine system capabilities with model metadata to derive optimal llama.cpp parameters:

```bash
# Auto-detect system + model (pipe model info)
uv run scripts/model-info.py Qwen/Qwen3.6-35B-A3B | uv run scripts/derive-params.py --model -

# With explicit system info (from file)
uv run scripts/detect-system.py > system.json
uv run scripts/model-info.py Qwen/Qwen3.6-35B-A3B > model.json
uv run scripts/derive-params.py --system system.json --model model.json

# Get CLI argument string directly
uv run scripts/derive-params.py --model model.json --cli

# Pipe model info directly to CLI args
uv run scripts/model-info.py Qwen/Qwen3.6-35B-A3B | uv run scripts/derive-params.py --model - --cli
```

The script outputs structured JSON with all derived parameters plus a `_cli` field with the complete argument string.

See [references/parameter-tuning.md](references/parameter-tuning.md) for detailed derivation logic.

### 4. Run the Model

**Interactive chat:**

```bash
llama-cli --hf-repo <user>/<model> --hf-file <file.gguf> \
  --ctx-size 64000 --flash-attn on \
  --n-gpu-layers 99 --temp 0.80 --top-p 0.95 --min-p 0.05 \
  --conversation --color auto --multiline-input
```

**OpenAI-compatible server:**

```bash
llama-server --hf-repo <user>/<model> --hf-file <file.gguf> \
  --host 127.0.0.1 --port 8080 \
  --ctx-size 64000 --flash-attn on \
  --n-gpu-layers 99
```

**MoE model (VRAM constrained):**

```bash
llama-server --hf-repo <user>/<model> --hf-file <file.gguf> \
  --ctx-size 64000 --flash-attn on \
  --n-gpu-layers 20 --cpu-moe \
  --load-mode mmap \
  --cache-type-k q4_0 --cache-type-v q4_0
```

## Python Scripts

The `scripts/` directory contains three Python scripts that automate the parameter derivation workflow:

| Script             | Purpose                                    | Usage                                       |
| ------------------ | ------------------------------------------ | ------------------------------------------- |
| `detect-system.py` | Detect system capabilities (GPU, RAM, CPU) | `uv run scripts/detect-system.py`           |
| `model-info.py`    | Fetch model metadata from Hugging Face     | `uv run scripts/model-info.py <model_id>`   |
| `derive-params.py` | Derive optimal llama.cpp parameters        | `uv run scripts/derive-params.py --model -` |

All scripts use inline dependencies (`# /// script` header) and run via `uv run` — no manual dependency management needed.

## Optimization Techniques

The derived parameters and run commands combine several techniques for faster inference and lower memory consumption:

| Technique                     | Effect                                                     |
| ----------------------------- | ---------------------------------------------------------- |
| `--flash-attn on`             | Faster attention, lower memory (esp. long contexts)        |
| `--cache-type-k/v q8_0/q4_0`  | Quantize KV cache → lower VRAM, slight quality cost        |
| `--cpu-moe` / `--n-cpu-moe N` | Keep MoE expert weights in CPU RAM → fit larger MoE models |
| `--load-mode mmap`            | Memory-map model file → lower RAM footprint, faster load   |
| `--n-gpu-layers N`            | Offload the right number of layers to GPU                  |
| `--tensor-split N0,N1,...`    | Distribute across multiple GPUs                            |

See [references/moe-optimization.md](references/moe-optimization.md) for MoE-specific tuning and the derivation script for KV cache / layer-offload logic.

## llama.cpp vs. Alternatives

| Framework        | Best For                                          | When to Choose Instead |
|------------------|---------------------------------------------------|------------------------|
| **llama.cpp**    | CPU, Apple Silicon, AMD/Intel GPUs, edge devices  | You have NVIDIA A100/H100 → use TensorRT-LLM |
|                  | GGUF quantization (1.5–8 bit)                     | You need 100K+ tok/s throughput → use TensorRT-LLM |
|                  | Simple deployment without Docker/Python           | You need PagedAttention + Python API → use vLLM |
| **TensorRT-LLM** | NVIDIA datacenter GPUs (A100, H100)               | You're on CPU/Apple Silicon → use llama.cpp |
| **vLLM**         | NVIDIA GPUs with Python-first API                 | You need maximum throughput → use TensorRT-LLM |

## References

- [quantization-guide.md](references/quantization-guide.md) — GGUF formats, model size scaling, imatrix calibration
- [optimization-guide.md](references/optimization-guide.md) — Thread tuning, GPU offload strategy, context memory
- [server-tuning.md](references/server-tuning.md) — Concurrency, continuous batching, metrics, load balancing
- [llama-cli-reference.md](references/llama-cli-reference.md) — Comprehensive CLI flag reference
- [hf-model-info.md](references/hf-model-info.md) — Retrieving model metadata from Hugging Face
- [system-capabilities.md](references/system-capabilities.md) — Detecting local system capabilities
- [parameter-tuning.md](references/parameter-tuning.md) — Deriving optimal parameters from model + system
- [moe-optimization.md](references/moe-optimization.md) — MoE-specific optimization guide
- [portable-setup.md](references/portable-setup.md) — Install/update/switch backends via prebuilt binaries (no build)

## Model Download

```bash
# Auto-download via HF (built into llama-cli/llama-server)
llama-cli --hf-repo <user>/<model> --hf-file <file.gguf> --prompt "test" --predict 1

# Manual download with resume support
curl -L -C - -o models/model.gguf "https://huggingface.co/<user>/<model>/resolve/main/<file.gguf>"

# Verify SHA256 from LFS pointer
curl -sL "https://huggingface.co/<user>/<model>/raw/main/<file.gguf>"
# Returns: oid sha256:<hash> / size <bytes>
```

## Reference Documents

- [llama-cli-reference.md](references/llama-cli-reference.md) — Comprehensive CLI reference for all llama.cpp tools
- [hf-model-info.md](references/hf-model-info.md) — Retrieving model metadata from Hugging Face
- [system-capabilities.md](references/system-capabilities.md) — Detecting local system capabilities
- [parameter-tuning.md](references/parameter-tuning.md) — Deriving optimal parameters from model + system
- [moe-optimization.md](references/moe-optimization.md) — MoE-specific optimization guide
- [caveats.md](references/caveats.md) — Fork-format GGUFs, load failures, sidecar confusion
