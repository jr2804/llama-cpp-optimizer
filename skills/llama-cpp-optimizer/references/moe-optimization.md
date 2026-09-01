# MoE Optimization Guide

How to run Mixture-of-Experts (MoE) models efficiently on consumer GPUs with limited VRAM.

## Why MoE Models Are Different

MoE models have two types of parameters:

1. **Dense parameters** — always active: embeddings, attention layers, shared experts
2. **Expert parameters** — routed per token: only a subset is active at any time

For Qwen3.6-35B-A3B:

- 256 total experts, only **8 active per token** (~3% of expert params used per step)
- ~35B total parameters, but only ~3B active per token
- The model file is ~19 GB (Q4_K_M) or ~21 GB (MXFP4), but only ~3B params need GPU compute

This means: **you can run a 35B MoE on a 6 GB GPU** by keeping inactive experts in CPU RAM.

## Key Flags

### `--cpu-moe` (keep ALL experts in CPU)

```bash
llama-server --model model.gguf --n-gpu-layers 99 --cpu-moe
```

- All expert weight matrices stay in CPU RAM
- Attention/shared/dense layers go to GPU (use `-ngl 99` together with `--cpu-moe`)
- VRAM usage: dense params (~2-3 GB) + KV cache + overhead
- **Best for:** VRAM < 8 GB
- Measured (Gemma-4-26B-A4B, 128 experts, RTX 2070 8GB): `-ngl 99 --cpu-moe` =
  **10.0 tok/s** vs 8.7 tok/s CPU-only (+15%) — always try this before falling
  back to pure CPU

### `--n-cpu-moe N` (keep N layers' experts in CPU)

```bash
llama-server --model model.gguf --n-gpu-layers 30 --n-cpu-moe 35
```

- Keeps experts of the first N layers (counting from the **highest-numbered** layers) in CPU
- Remaining layers' experts stay on GPU
- Start with a high N and **reduce** until the model fits in VRAM
- **Best for:** VRAM 8-16 GB, want to maximize GPU usage

### `--load-mode mmap` (memory-map the model file)

```bash
llama-server --model model.gguf --load-mode mmap
```

- The model file is memory-mapped, not fully loaded into RAM
- Pages are loaded on demand as the CPU accesses expert weights
- Saves RAM at the cost of slightly higher latency for expert switching
- **Essential** for large MoE models on systems with limited RAM

### `--load-mode mlock` (lock model in RAM)

```bash
llama-server --model model.gguf --load-mode mlock
```

- Forces the OS to keep the entire model in physical RAM
- Prevents swapping/paging during inference
- Uses more RAM but gives more consistent performance
- **Use if:** you have enough RAM and want maximum speed

## VRAM Budget Calculation

For Qwen3.6-35B-A3B (MXFP4, ~21 GB file):

### With `--cpu-moe`

| Component | Size | Location |
|-----------|------|----------|
| Dense weights (attention, embeddings) | ~2.5 GB | GPU |
| Active expert weights (8/256) | ~0.7 GB | GPU |
| KV cache (64K ctx, q4_0) | ~1.2 GB | GPU |
| Overhead + buffers | ~0.5 GB | GPU |
| **Total GPU** | **~4.9 GB** | **Fits in 6 GB** |
| Inactive expert weights (248/256) | ~17 GB | CPU (mmap) |

### Without `--cpu-moe` (all on GPU)

| Component | Size | Location |
|-----------|------|----------|
| All weights | ~21 GB | GPU |
| KV cache (64K ctx, q4_0) | ~1.2 GB | GPU |
| **Total GPU** | **~22.2 GB** | **Does NOT fit in 6 GB** |

## Performance Tuning

### CPU RAM Speed Matters

With `--cpu-moe`, expert weights are fetched from CPU RAM on demand. The speed depends on:

| RAM type | Bandwidth | Impact on MoE |
|----------|-----------|---------------|
| DDR4-3200 dual channel | ~50 GB/s | Acceptable |
| DDR5-4800 dual channel | ~70 GB/s | Good |
| DDR5-6000 dual channel | ~90 GB/s | Good |
| DDR4-3200 quad channel | ~100 GB/s | Excellent |
| DDR5-5600 quad channel | ~180 GB/s | Excellent |

**Rule of thumb:** Each expert fetch moves ~2 MB of weights. At 8 experts per token, that's ~16 MB per token. At DDR4-3200 speeds, this takes ~0.3 ms — acceptable for generation.

### Thread Tuning

```bash
# For MoE with CPU offload, more threads help
--threads 16 --threads-batch 16
```

The CPU needs to:

1. Fetch expert weights from RAM
2. Run the expert FFN computation
3. Route the next token's experts

More threads reduce the CPU bottleneck.

### Batch Size

```bash
# Smaller batches reduce VRAM pressure
--batch-size 1024 --ubatch-size 256
```

For MoE with CPU offload, smaller batches also reduce the number of expert weights that need to be in flight simultaneously.

## Step-by-Step Tuning Process

### For 6 GB VRAM (e.g., RTX 3060, GTX 1060)

```bash
# Step 1: Start safe
llama-server --model model.gguf \
  --n-gpu-layers 20 --cpu-moe \
  --load-mode mmap \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --ctx-size 64000 --flash-attn on

# Step 2: If it fits, try more GPU layers
llama-server --model model.gguf \
  --n-gpu-layers 25 --cpu-moe \
  --load-mode mmap \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --ctx-size 64000 --flash-attn on

# Step 3: If still room, try reducing --n-cpu-moe
llama-server --model model.gguf \
  --n-gpu-layers 25 --n-cpu-moe 35 \
  --load-mode mmap \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --ctx-size 64000 --flash-attn on
```

### For 8-12 GB VRAM (e.g., RTX 3070, RTX 3080)

```bash
# More GPU layers, fewer CPU experts
llama-server --model model.gguf \
  --n-gpu-layers 30 --n-cpu-moe 30 \
  --load-mode mmap \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --ctx-size 64000 --flash-attn on
```

### For 24 GB VRAM (e.g., RTX 3090, RTX 4090)

```bash
# Full GPU offload possible
llama-server --model model.gguf \
  --n-gpu-layers -1 \
  --cache-type-k f16 --cache-type-v f16 \
  --ctx-size 128000 --flash-attn on
```

## MoE-Specific Warnings

1. **`--n-cpu-moe` counts from the top layers.** Models like DeepSeek V3 have dense FFN layers at the start (lowest-numbered layers). `--n-cpu-moe 3` would keep the last 3 layers' experts on CPU, not the first 3.

2. **Linear attention layers** (Qwen3.6) may show warnings about fused ops not being supported on Vulkan. This is non-fatal — the unfused path is used instead.

3. **Tensor overrides with mmap** may show a warning about performance. This is expected with `--cpu-moe` + `--load-mode mmap`. If performance is poor, try `--load-mode mlock` (uses more RAM).

4. **Draft model (speculative decoding)** with MoE: if using `--spec-draft-hf`, also pass `--spec-draft-cpu-moe` to keep the draft model's experts in CPU.

## MoE Model Comparison

| Model | Total Params | Active Params | Experts | Active/Token | File Size (Q4) |
|-------|-------------|---------------|---------|-------------|-----------------|
| Qwen3.6-35B-A3B | 35B | ~3B | 256 | 8 | ~19 GB |
| Qwen3.5-35B-A3B | 35B | ~3B | 256 | 8 | ~19 GB |
| DeepSeek V3 | 671B | ~37B | 256 | 8 | ~350 GB (Q4) |
| DeepSeek V2 | 236B | ~21B | 64 | 6 | ~130 GB (Q4) |
| Mixtral 8x7B | 47B | ~13B | 8 | 2 | ~26 GB (Q4) |
| Mixtral 8x22B | 141B | ~39B | 8 | 2 | ~78 GB (Q4) |
| DBRX | 132B | ~36B | 16 | 4 | ~73 GB (Q4) |
