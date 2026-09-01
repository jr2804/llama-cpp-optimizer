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

Offload transformer layers to GPU for maximum throughput.

### Let `--fit` do it (default, recommended)

`llama-server` and `llama-cli` ship with `--fit on` **enabled by default**.
It auto-tunes `n-gpu-layers` (and can shrink `--ctx-size` down to `--fit-ctx`,
default 4096) to fit device memory, leaving a safety margin
(`--fit-target`, default 1024 MiB).

```bash
# Just set ctx and let fit pick the offload — works for any model/card combo
llama-server -m model.gguf -c 32768          # --fit on is the default
```

**Setting `-ngl` explicitly disables `--fit` for that run** — you'll see:

```text
W common_fit_params: failed to fit params to free device memory: n_gpu_layers already set by user to 99, abort
```

Only hardcode `-ngl` when you need a specific split (e.g. MoE experts in RAM via `--cpu-moe`).

### The VRAM budget: weights + KV cache + overhead

OOM at load is almost always the **KV cache**, not the weights. A 7 GiB model on an 8 GiB card can still OOM if the context window is large — KV cache scales linearly with `--ctx-size` and with `layers × kv_heads × head_dim`. Full offload may fit at 4K ctx but OOM at 32K. Reduce ctx before reducing offload.

### Empirical: more GPU is almost always faster, even with unsupported ops

Hybrid-attention models (e.g. Ternary-Bonsai 27B, ~75% linear/delta-net attention) emit warnings on some backends:

```text
W resolve_fused_ops: fused Gated Delta Net (chunked) not supported, set to disabled
```

It's tempting to force those layers to CPU where the kernel works. **Don't** — measured on RTX A2000 8GB (Vulkan), Ternary-Bonsai-27B Q2_0_g64:

| offload | pp64 (t/s) | tg32 (t/s) |
|---------|-----------|-----------|
| ngl=0 (all CPU) | 7.6 | 1.4 |
| ngl=50 | 78 | 3.5 |
| `--fit` (auto) | 57 | **4.6** |

The unfused GPU fallback path still beats CPU by 2–3× on token generation,
10× on prompt processing. The one wrinkle: **maxing out offload can slow
generation** if it leaves no VRAM headroom for activation buffers — `--fit`'s
conservative split won on tg (4.6 vs 3.5) even though it offloaded fewer
layers. When in doubt, bench both.

### Manual tuning (when `--fit` is wrong)

```bash
# Binary search for the max ngl that fits at your ctx
for ngl in 99 80 60 40 20; do
  llama-server -m model.gguf -c 32768 --n-gpu-layers $ngl --port 8012 --host 127.0.0.1 \
    && echo "ngl=$ngl OK" || echo "ngl=$ngl OOM"
  pkill -f "port 8012"
done
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

## Measuring: `llama-bench`

Don't guess — bench. `llama-bench` loads the model once per config and reports prompt-processing (`pp`) and token-generation (`tg`) throughput. Compare offload levels, ctx sizes, or cache types in one run:

```bash
# Compare offload levels for a model that won't fully fit
llama-bench -m model.gguf -p 64 -n 32 -ngl 99 -ngl 50 -ngl 0 -t 16

# Compare KV cache quantization
llama-bench -m model.gguf -p 512 -n 128 -ctk f16 -ctk q4_0
```

`pp` (prompt processing) is compute-bound and matters for long inputs. `tg` (token generation) is memory-bandwidth-bound and is what you feel in interactive chat — **optimize tg for chat, pp for batch/RAG**. Numbers are noisy (±10%); run with multiple `-r` repeats if you need stable comparisons.

Each `-ngl` value triggers a full model reload (~30–60s for a 7 GB model), so bench a few targeted values, not a sweep.

### Context window vs token speed: measure the trade-off

A model that *loads* at 256K ctx does not necessarily *run well* at 256K ctx. Larger context = larger KV cache = more memory bandwidth pressure = lower tok/s. Use `llama-bench` to find the largest ctx where token speed stays acceptable:

```bash
# Compare tg (tokens/sec) across context sizes
llama-bench -m model.gguf -p 1024 -n 128 -c 32768 -c 65536 -c 131072 -c 262144 -ngl 99 -r 3
```

- If `tg` drops below ~15 tok/s at a given ctx, the model is bandwidth-starved — consider reducing ctx or quantizing KV (`--cache-type-k q4_0`).
- For interactive chat, target `tg` ≥ 20 tok/s. For batch/RAG, `pp` (prompt processing) matters more.
- **Always report the measured tok/s alongside the chosen context size** — "131K at 45 tok/s" is actionable; "131K context" alone is not.

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
