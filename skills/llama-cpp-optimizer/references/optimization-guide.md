# Performance Optimization Guide

Maximize llama.cpp inference speed and efficiency.

## CPU Thread Tuning

llama.cpp uses one thread per CPU core. **Physical cores outperform hyperthreaded (logical) cores** for matrix operations.

```bash
# Set threads — use physical cores, not logical
./llama-cli -m model.gguf -t 16   # AMD Ryzen 9 7950X: 16 physical cores, 32 logical
                                   # Use -t 16, not -t 32
```

```bash
# Avoid hyperthreading — it slows matrix ops on most CPUs
# Ryzen 9 7950X: -t 16 (physical), not -t 32 (logical)
```

**BLAS acceleration** gives a 2–3× speedup for CPU inference. Build with:

```bash
make LLAMA_OPENBLAS=1
```

## GPU Layer Offloading

Offload transformer layers to GPU for maximum throughput. The strategy:

```bash
# Start with full offload
./llama-cli -m model.gguf -ngl 999

# If OOM, reduce by 5 until it fits
./llama-cli -m model.gguf -ngl 995
# ...
./llama-cli -m model.gguf -ngl 980
```

**Hybrid mode** (partial offload):

```bash
# Offload 20 of 40 layers — GPU + CPU split
./llama-cli -m llama-70b.Q4_K_M.gguf -ngl 20
```

Monitor VRAM in real time:

```bash
# NVIDIA
nvidia-smi dmon

# Apple Silicon (Metal)
# VRAM is unified with system RAM — watch sys Monitor
```

## Batch Processing

```bash
# Increase batch size for throughput (default: 512)
./llama-cli -m model.gguf --batch-size 512

# Physical batch — process N tokens at once on GPU
./llama-cli -m model.gguf --ubatch-size 128
```

## Context Size Management

Context size directly affects memory usage. The relationship is approximately linear:

```bash
# Default context (512 tokens) — minimal memory
-c 512

# Standard context (4K)
-c 4096

# Long context — more memory, slower initial prompt processing
-c 32768
```

**Context memory estimate** (Q4_K_M model, 7B, 32K context):

| Context | Approx. KV cache memory |
|---------|------------------------|
| 4K | ~2 GB |
| 16K | ~8 GB |
| 32K | ~16 GB |
| 64K | ~32 GB |

Adjust `--cache-type-k` and `--cache-type-v` to `q4_0` to halve KV cache memory at a small quality cost.

## Performance Benchmarks (Reference)

### CPU — Llama-2-7B Q4_K_M

| Setup | Speed | Notes |
|-------|-------|-------|
| Apple M3 Max (Metal) | 50 tok/s | 16 threads |
| AMD Ryzen 9 7950X (OpenBLAS) | 35 tok/s | 16 threads |
| Intel i9-13900K (AVX2) | 30 tok/s | 24 threads |

### GPU Offload — Llama-2-7B Q4_K_M (RTX 4090)

| Layers on GPU | Speed | VRAM used |
|---------------|-------|-----------|
| 0 (CPU only) | 30 tok/s | 0 GB |
| 20 (hybrid) | 80 tok/s | ~8 GB |
| 35 (all layers) | 120 tok/s | ~12 GB |
