# Hugging Face Model Info Retrieval

How to retrieve model metadata from Hugging Face without downloading the model.

## Prerequisites

`uv` is installed system-wide. Use `uvx hf.` to run Hugging Face Hub CLI commands.

```bash
# Verify availability
uvx hf. -- --help
```

## Retrieving Model Metadata

### From a Model ID (short form)

```bash
uvx hf. -- model-info Qwen/Qwen3.6-35B-A3B
uvx hf. -- model-info Nanbeige/Nanbeige4.2-3B
```

### From a Full URL

```bash
uvx hf. -- model-info https://huggingface.co/Qwen/Qwen3.6-35B-A3B
uvx hf. -- model-info https://huggingface.co/Nanbeige/Nanbeige4.2-3B
```

### Key Metadata to Extract

From the model's `config.json` (accessible via the HF API), extract:

```json
{
  "text_config": {
    "hidden_size": 2048,           // Model dimension
    "num_hidden_layers": 40,       // Total transformer layers
    "num_attention_heads": 16,     // Attention heads
    "num_key_value_heads": 2,      // KV heads (GQA)
    "intermediate_size": 512,      // FFN intermediate size (MoE)
    "num_experts": 256,            // MoE: total experts
    "num_experts_per_tok": 8,      // MoE: active experts per token
    "head_dim": 256,               // Head dimension
    "max_position_embeddings": 262144,  // Max context length
    "model_type": "qwen3_5_moe",   // Architecture type
    "rope_theta": 10000000,        // RoPE base frequency
    "partial_rotary_factor": 0.25  // Partial rotary embeddings
  }
}
```

### Direct API Access (without uvx)

```bash
# Get model metadata
curl -sL "https://huggingface.co/api/models/Qwen/Qwen3.6-35B-A3B"

# Get config.json directly
curl -sL "https://huggingface.co/Qwen/Qwen3.6-35B-A3B/raw/main/config.json"

# Get GGUF metadata (if available)
curl -sL "https://huggingface.co/api/models/ggml-org/Qwen3.6-35B-A3B-GGUF"
```

## Listing Available GGUF Files

```bash
# List all GGUF files in a repo
uvx hf. -- list-files ggml-org/Qwen3.6-35B-A3B-GGUF

# Or via the HF API
curl -sL "https://huggingface.co/api/models/ggml-org/Qwen3.6-35B-A3B-GGUF" | \
  python3 -c "import sys,json; [print(s['rfilename']) for s in json.load(sys.stdin)['siblings'] if s['rfilename'].endswith('.gguf')]"
```

## Getting File Sizes and Hashes

### LFS Pointer File (authoritative)

The LFS pointer file contains the exact SHA256 and size:

```bash
curl -sL "https://huggingface.co/<user>/<model>/raw/main/<file.gguf>"
```

Returns:

```text
version https://git-lfs.github.com/spec/v1
oid sha256:a3a730920068d8c102238364c3fd415d89bc1c4a2f2d03960249b299d67522c5
size 20261569888
```

### HF API (file listing with sizes)

```bash
curl -sL "https://huggingface.co/api/models/<user>/<model>" | \
  python3 -c "import sys,json; d=json.load(sys.stdin); [print(s['rfilename'], s.get('size','?')) for s in d['siblings']]"
```

## Downloading Models

### Via llama.cpp built-in (recommended)

```bash
llama-cli --hf-repo <user>/<model> --hf-file <file.gguf> --prompt "test" --predict 1
```

This auto-downloads to the Hugging Face cache (`~/.cache/huggingface/hub/`).

### Via curl (manual, with resume)

```bash
curl -L -C - -o models/model.gguf \
  "https://huggingface.co/<user>/<model>/resolve/main/<file.gguf>"
```

### Via Hugging Face Hub CLI

```bash
uvx hf. -- download <user>/<model> --include "*.gguf" --local-dir models/
```

## Model Architecture Detection

Determine the model architecture from `config.json`:

| `model_type` | Architecture | MoE? | Notes |
|-------------|-------------|------|-------|
| `llama` | LLaMA dense | No | Standard transformer |
| `qwen2` | Qwen2 dense | No | |
| `qwen3_5_moe` | Qwen3.5/3.6 MoE | Yes | 256 experts, 8 active |
| `deepseek_v3` | DeepSeek V3 MoE | Yes | 256 experts, 8 active |
| `deepseek_v2` | DeepSeek V2 MoE | Yes | 64 experts, 6 active |
| `mixtral` | Mixtral MoE | Yes | 8 experts, 2 active |
| `dbrx` | DBRX MoE | Yes | 16 experts, 4 active |
| `gemma2` | Gemma 2 dense | No | |
| `command-r` | Cohere Command-R | No | |
| `exaone3` | Exaone 3 dense | No | |
| `phi3` | Phi-3 dense | No | |
| `phi4` | Phi-4 dense | No | |

## Estimating Model Size

### Dense Model Size

```text
model_size_bytes = vocab_size * hidden_size * 2  (embedding)
                 + num_hidden_layers * (
                     hidden_size * intermediate_size * 4  (FFN gate/up/down)
                     + hidden_size * hidden_size * 4  (attention Q/K/V/O)
                   )
                 + hidden_size * vocab_size * 2  (lm_head)
```

Rough estimate: `parameters * bytes_per_param * quantization_factor`

| Quant | Bytes/param |
|-------|-------------|
| BF16/FP16 | 2 |
| Q8_0 | 1 |
| Q4_K_M | ~0.5 |
| Q5_K_M | ~0.625 |
| Q3_K_M | ~0.375 |
| Q2_K | ~0.25 |

### MoE Model Size

MoE models have two components:

- **Dense params** (attention, embeddings, shared experts): always loaded
- **Expert params** (routed experts): only active experts per token need compute, but all experts must be in RAM/VRAM

```text
total_size = dense_params + num_experts * expert_params_per_expert
active_per_token = dense_params + num_experts_per_tok * expert_params_per_expert
```

The key insight: with `--cpu-moe`, only the dense params and active expert weights need GPU VRAM. The inactive expert weights stay in CPU RAM.
