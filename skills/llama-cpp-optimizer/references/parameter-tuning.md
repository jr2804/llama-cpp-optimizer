# Parameter Tuning

How to derive optimal llama.cpp parameters from model metadata and system capabilities.

## Overview

The goal is to maximize throughput (tokens/second) while staying within hardware limits. The key constraints are:

1. **VRAM** — model weights + KV cache + overhead must fit in GPU memory
2. **RAM** — model file + KV cache + overhead must fit in system memory
3. **CPU/GPU bandwidth** — determines token generation speed

## Step 1: Gather Inputs

### From System (see [system-capabilities.md](system-capabilities.md))

| Value | Source |
|-------|--------|
| Total VRAM | `nvidia-smi --query-gpu=memory.total` |
| Free VRAM | `nvidia-smi --query-gpu=memory.free` |
| Total RAM | `(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory` |
| Free RAM | `(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory * 1024` |
| CPU cores | `[Environment]::ProcessorCount` |
| GPU compute capability | `nvidia-smi --query-gpu=compute_cap` |

### From Model (see [hf-model-info.md](hf-model-info.md))

| Value | Source |
|-------|--------|
| `num_hidden_layers` | `config.json` → `text_config.num_hidden_layers` |
| `num_key_value_heads` | `config.json` → `text_config.num_key_value_heads` |
| `head_dim` | `config.json` → `text_config.head_dim` |
| `hidden_size` | `config.json` → `text_config.hidden_size` |
| `max_position_embeddings` | `config.json` → `text_config.max_position_embeddings` |
| `num_experts` | `config.json` → `text_config.num_experts` (MoE only) |
| `num_experts_per_tok` | `config.json` → `text_config.num_experts_per_tok` (MoE only) |
| Model file size | LFS pointer or `ls -lh` |
| Quantization | From filename (Q4_K_M, Q5_K_M, etc.) |

## Step 2: Calculate Memory Requirements

### Model Weight Size

```text
model_weights = file_size  (for GGUF, this is exact)
```

For estimation without downloading:

```text
model_weights ≈ total_parameters * bytes_per_param
```

Where `bytes_per_param` by quantization:

| Quant | Bytes/param |
|-------|-------------|
| BF16/FP16 | 2.0 |
| Q8_0 | 1.0 |
| Q6_K | 0.75 |
| Q5_K_M | 0.625 |
| Q5_0 | 0.625 |
| Q4_K_M | 0.5 |
| Q4_0 | 0.5 |
| Q3_K_M | 0.375 |
| Q2_K | 0.25 |
| IQ4_NL | 0.5 |
| MXFP4 | ~0.5 |

### KV Cache Size

```text
KV_cache_per_token = 2 * num_hidden_layers * num_key_value_heads * head_dim * bytes_per_value
Total_KV_cache = KV_cache_per_token * ctx_size
```

Where `bytes_per_value`:

| Cache type | Bytes/value |
|------------|-------------|
| `f16` | 2 |
| `q8_0` | 1 |
| `q4_0` | 0.5 |
| `q4_1` | 0.5 |
| `iq4_nl` | 0.5 |

### Overhead

Add ~10% overhead for buffers, activations, and temporary tensors.

### Total VRAM Required (GPU offload)

```text
VRAM_needed = model_weights_on_gpu + KV_cache + overhead
```

Where `model_weights_on_gpu` depends on `--n-gpu-layers`:

```text
model_weights_on_gpu = (n_gpu_layers / total_layers) * model_weights
```

For MoE with `--cpu-moe`:

```text
model_weights_on_gpu = dense_weights + (num_experts_per_tok / num_experts) * expert_weights
```

This is much smaller than the full model file size.

## Step 3: Derive Parameters

### Context Size (`--ctx-size`)

```text
max_ctx = min(
    model_max_position_embeddings,
    floor((free_RAM * 0.75) / KV_cache_per_token)
)
```

**Guidelines:**

- Start with 64000 for most models (good balance)
- Reduce to 32000 if VRAM constrained
- Increase to 128000+ if you have 24 GB+ VRAM and the model supports it
- Never exceed `max_position_embeddings` from model config

### GPU Layers (`--n-gpu-layers`)

```text
max_layers = floor(total_layers * (free_VRAM - KV_cache - overhead) / model_weights)
```

**Practical approach:**

1. Start with `--n-gpu-layers -1` (all layers on GPU)
2. If it crashes (OOM), reduce by 10 layers
3. Repeat until it loads
4. For MoE with `--cpu-moe`, start with `--n-gpu-layers 20` and tune

**Rule of thumb:**

| Model size | 6 GB VRAM | 8 GB VRAM | 12 GB VRAM | 24 GB VRAM |
|------------|-----------|-----------|------------|------------|
| 3B dense | -1 (all) | -1 (all) | -1 (all) | -1 (all) |
| 7B dense | 0 (CPU) | 20-30 | -1 (all) | -1 (all) |
| 8B dense | 0 (CPU) | 20-30 | -1 (all) | -1 (all) |
| 14B dense | 0 (CPU) | 0 (CPU) | 20-30 | -1 (all) |
| 35B MoE (Q4) | 20 + `--cpu-moe` | 28 + `--cpu-moe` | 35 + `--cpu-moe` | -1 (all) |
| 70B MoE (Q4) | 0 (CPU) | 0 (CPU) | 20 + `--cpu-moe` | 30 + `--cpu-moe` |

### KV Cache Type (`--cache-type-k`, `--cache-type-v`)

| VRAM available | Recommended cache type |
|----------------|----------------------|
| < 4 GB free | `q4_0` (saves ~75% vs f16) |
| 4-8 GB free | `q4_0` or `q8_0` |
| 8-16 GB free | `f16` |
| > 16 GB free | `f16` |

### Batch Size (`--batch-size`, `--ubatch-size`)

| VRAM available | `--batch-size` | `--ubatch-size` |
|----------------|----------------|-----------------|
| < 4 GB free | 1024 | 256 |
| 4-8 GB free | 2048 | 512 |
| 8-16 GB free | 2048 | 512 |
| > 16 GB free | 4096 | 1024 |

### Threads (`--threads`)

```text
--threads = CPU_physical_cores  (not logical cores with hyperthreading)
--threads-batch = CPU_physical_cores  (or slightly less for prompt processing)
```

**Rule of thumb:** Use `[Environment]::ProcessorCount` for simplicity. For CPUs with hyperthreading, try `physical_cores` for generation and `physical_cores * 2` for batch processing.

### Flash Attention (`--flash-attn`)

| GPU | Support |
|-----|---------|
| NVIDIA 2xxx+ (Turing) | Yes |
| NVIDIA 3xxx+ (Ampere) | Yes |
| NVIDIA 4xxx+ (Ada) | Yes |
| AMD RX 6xxx+ (Vulkan) | Yes |
| Intel Arc (Vulkan) | Yes |
| Apple M1+ (Metal) | Yes |
| Older GPUs | No (use `off`) |

Set to `on` if supported, `off` otherwise. `auto` detects at runtime.

### Sampling Parameters

| Use case | `--temp` | `--top-p` | `--min-p` |
|----------|----------|-----------|-----------|
| Code generation | 0.20-0.40 | 0.90 | 0.05 |
| Factual Q&A | 0.30-0.50 | 0.90 | 0.05 |
| General chat | 0.70-0.90 | 0.95 | 0.05 |
| Creative writing | 0.90-1.20 | 0.95 | 0.05 |
| Brainstorming | 1.00-1.50 | 0.98 | 0.02 |

## Step 4: MoE-Specific Tuning

See [moe-optimization.md](moe-optimization.md) for detailed MoE tuning.

## Step 5: Benchmark and Iterate

```bash
# Run benchmark
llama-bench --model model.gguf --n-gpu-layers N --flash-attn on --ctx-size 64000

# Test generation speed
llama-cli --model model.gguf --n-gpu-layers N --flash-attn on \
  --ctx-size 64000 --temp 0.80 --prompt "Hello" --predict 100
```

**Target speeds:**

- Prompt processing: > 1000 tokens/s (GPU), > 100 tokens/s (CPU)
- Text generation: > 10 tokens/s (acceptable), > 30 tokens/s (good), > 50 tokens/s (excellent)

## Quick Reference Card

**Default: let `--fit` auto-tune offload** (it's on by default). Only set `--n-gpu-layers` explicitly for MoE CPU-expert splits or when `--fit` picks poorly.

```text
Any VRAM:    --fit on (default) --cache-type-k q4_0 --cache-type-v q4_0 --ctx-size 32768
             (omit --n-gpu-layers so --fit can tune it)

# Fallback: explicit offload (disables --fit)
VRAM < 4 GB:  --n-gpu-layers 0  --cache-type-k q4_0 --cache-type-v q4_0 --ctx-size 8192
VRAM 4-8 GB:  --n-gpu-layers 20 --cache-type-k q4_0 --cache-type-v q4_0 --ctx-size 32768
VRAM 8-16 GB: --n-gpu-layers 30 --cache-type-k f16  --cache-type-v f16  --ctx-size 64000
VRAM > 16 GB: --n-gpu-layers -1 --cache-type-k f16  --cache-type-v f16  --ctx-size 128000

MoE + VRAM < 8 GB: add --cpu-moe --load-mode mmap
MoE + VRAM 8-16 GB: try --n-cpu-moe 30 (reduce until it fits)
```
