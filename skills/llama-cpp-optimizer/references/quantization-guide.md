# GGUF Quantization Guide

Complete guide to GGUF quantization formats, model size estimation, and quantization calibration.

## Quantization Format Comparison

| Format | Bits | Size (7B) | Perplexity Delta | Tokens/sec (Llama-2-7B) | Use Case |
|--------|------|-----------|------------------|--------------------------|----------|
| FP16 | 16.0 | 13.0 GB | baseline | 15 | Original quality |
| Q8_0 | 8.0 | 7.0 GB | +0.03% | 25 | Nearly lossless |
| Q6_K | 6.5 | 5.5 GB | +0.13% | 30 | Best quality/size |
| Q5_K_M | 5.5 | 4.8 GB | +0.39% | 35 | Balanced |
| **Q4_K_M** | 4.5 | 4.1 GB | +1.68% | 40 | **Recommended default** |
| Q4_K_S | 4.3 | 3.9 GB | +2.62% | 42 | Speed-critical |
| Q3_K_M | 3.7 | 3.3 GB | +6.07% | 45 | Small models only |
| Q2_K | 2.5 | 2.7 GB | +15.3% | 50 | Not recommended |

**Recommendation**: Use **Q4_K_M** for best balance of quality and speed.

## K-Quantization Method

K-quants use mixed precision: attention weights stay at higher precision while feed-forward weights use lower precision.

**Variants:**
- `_S` (Small): Faster, lower quality
- `_M` (Medium): Balanced (recommended)
- `_L` (Large): Better quality, larger size

**Example**: `Q4_K_M` = 4-bit quantization, K-mixed method, medium quality setting.

## Model Size Scaling (estimated at Q4_K_M)

### 7B models

| Format | Size | Min RAM |
|--------|------|---------|
| Q2_K | 2.7 GB | 5 GB |
| Q3_K_M | 3.3 GB | 6 GB |
| Q4_K_M | 4.1 GB | 7 GB |
| Q5_K_M | 4.8 GB | 8 GB |
| Q6_K | 5.5 GB | 9 GB |
| Q8_0 | 7.0 GB | 11 GB |

### 13B models

| Format | Size | Min RAM |
|--------|------|---------|
| Q2_K | 5.1 GB | 8 GB |
| Q3_K_M | 6.2 GB | 10 GB |
| Q4_K_M | 7.9 GB | 12 GB |
| Q5_K_M | 9.2 GB | 14 GB |
| Q6_K | 10.7 GB | 16 GB |

### 70B models

| Format | Size | Min RAM |
|--------|------|---------|
| Q2_K | 26 GB | 32 GB |
| Q3_K_M | 32 GB | 40 GB |
| Q4_K_M | 41 GB | 48 GB |
| Q4_K_S | 39 GB | 46 GB |
| Q5_K_M | 48 GB | 56 GB |

**Recommendation for 70B**: Use Q3_K_M or Q4_K_S to fit in consumer hardware.

## Use Case Guide

| Use Case | Recommended Format |
|----------|-------------------|
| General chatbots / assistants | Q4_K_M |
| Code generation | Q5_K_M or Q6_K |
| Creative writing | Q4_K_M (Q3_K_M for draft) |
| Technical / medical / factual | Q6_K or Q8_0 |
| Edge devices (Raspberry Pi) | Q2_K or Q3_K_S |

## Quality Testing with Perplexity

```bash
# Calculate perplexity (lower = better quality)
uv run llama-perplexity \
    -m model.gguf \
    -f wikitext-2-raw/wiki.test.raw \
    -c 512

# Baseline: ~5.96 (FP16)
# Q4_K_M: ~6.06 (+1.7%)
# Q2_K: ~6.87 (+15.3% — too much degradation)
```

## Importance Matrices (imatrix)

Importance matrices are calibration data that improve quantization quality by ~10–20% perplexity on Q4, and are essential for Q3 and below.

```bash
# 1. Generate importance matrix from domain-specific text
uv run llama-imatrix \
    -m model-f16.gguf \
    -f calibration-data.txt \
    -o model.imatrix

# 2. Quantize using the matrix
uv run llama-quantize \
    --imatrix model.imatrix \
    model-f16.gguf \
    model-Q4_K_M.gguf \
    Q4_K_M
```

**Calibration data**: ~100 MB of representative text from the target domain (e.g., code for code models). Higher quality data = better quantization.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Model outputs gibberish | Quantization too aggressive (Q2_K) | Try Q4_K_M or Q5_K_M |
| Out of memory | Model too large for available VRAM/RAM | Use lower quantization (Q4_K_S instead of Q5_K_M), or reduce `--ctx-size` |
| Slow inference | High quantization = more compute | Q8_0 is much slower than Q4_K_M; trade speed vs. quality |
