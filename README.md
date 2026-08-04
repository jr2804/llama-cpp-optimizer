# llama-cpp-optimizer

Launch optimized LLMs via [llama.cpp](https://github.com/ggml-org/llama.cpp) with faster inference and lower VRAM/RAM consumption. This repository hosts the **`llama-cpp-optimizer`** Agent Skill: parameter derivation from model metadata + system capabilities, plus memory-reduction and speed hacks (flash attention, KV-cache quantization, MoE CPU offload, memory-mapping, layer offloading).

## Install

Add the skill directly from this repository with `bun`:

```bash
bun x skills add. jr2804/llama-cpp-optimizer
```

The layout follows the [Agent Skills](https://agentskills.io) spec so tooling can point straight at this repo:

```
skills/llama-cpp-optimizer/
├── SKILL.md            # Skill definition + usage guide
├── scripts/
│   ├── detect-system.py  # Detect GPU/RAM/CPU capabilities
│   ├── model-info.py     # Fetch model metadata from Hugging Face
│   └── derive-params.py  # Derive optimal llama.cpp parameters
└── references/
    ├── llama-cli-reference.md  # CLI reference for all llama.cpp tools
    ├── hf-model-info.md        # Retrieving model metadata from HF
    ├── system-capabilities.md  # Detecting local system capabilities
    ├── parameter-tuning.md     # Derivation logic
    └── moe-optimization.md     # MoE-specific optimization guide
```

## Quick Start

Prerequisites: [uv](https://docs.astral.sh/uv/) and a llama.cpp build on `PATH`.

```bash
# 1. Detect system capabilities
uv run skills/llama-cpp-optimizer/scripts/detect-system.py

# 2. Get model metadata from Hugging Face
uv run skills/llama-cpp-optimizer/scripts/model-info.py Qwen/Qwen3.6-35B-A3B

# 3. Derive optimal parameters and emit ready-to-use CLI args
uv run skills/llama-cpp-optimizer/scripts/model-info.py Qwen/Qwen3.6-35B-A3B \
  | uv run skills/llama-cpp-optimizer/scripts/derive-params.py --model - --cli
```

## Optimizations Covered

| Technique | Effect |
|-----------|--------|
| `--flash-attn on` | Faster attention, lower memory (esp. long contexts) |
| `--cache-type-k/v q8_0/q4_0` | Quantized KV cache → lower VRAM |
| `--cpu-moe` / `--n-cpu-moe N` | MoE expert weights in CPU RAM → fit larger models |
| `--load-mode mmap` | Memory-mapped model file → lower RAM, faster load |
| `--n-gpu-layers N` / `--tensor-split` | Optimal GPU offloading / multi-GPU distribution |

See `skills/llama-cpp-optimizer/SKILL.md` for the full workflow, flag reference, and run examples.

## License

MIT — see [LICENSE](LICENSE).
