# Server Tuning Guide

Production deployment patterns for `llama-server` with OpenAI-compatible API.

> **Windows service:** see [windows-service.md](windows-service.md) to run `llama-server` as a Windows service via Servy (auto-start at boot).

## Multiple Models: ONE instance, ONE port (router mode)

`llama-server` has a built-in **router mode** that serves multiple models from a single process on a single port. **Never spawn one instance per model** — use `--models-preset` with an INI file instead:

```bash
llama-server --models-preset presets.ini --host 127.0.0.1 --port 8080
```

The INI file (`presets.ini`) defines each model as a section; any `llama-server` CLI flag is a valid key. A `[*]` section applies to all models:

```ini
[*]
# Global defaults inherited by every model below
ctx-size = 131072
flash-attn = on
cache-type-k = q4_0      # almost lossless; halves KV cache VRAM
cache-type-v = q4_0      # ditto — set once here, not per-model

[ornith-9b]
# Dense 9B: leave n-gpu-layers unset so --fit (global default) tunes offload
# for the requested ctx. Hardcoding -ngl is reserved for MoE cpu-moe splits.
model = models/Ornith-1.5-9B-Q4_K_M.gguf

[ornith-35b]
model = models/Ornith-1.5-35B-A3B-Q4_K_M.gguf
n-gpu-layers = 20
cpu-moe = true
ctx-size = 262144          # overrides the global 131072
```

### VRAM sharing in router mode

Models load lazily (first request loads the first model's weights into VRAM), but
**once loaded, weights stay resident** — the server does not swap models out.
Queried two models with `-ngl 99` each and total weights > GPU VRAM → the second
model's load succeeds but performance collapses (memory thrashing).

**Rule of thumb**: the sum of all models' working-set sizes (weights + KV cache at
their configured context) must fit in VRAM if you plan to switch between them in
the same session. For single-model-per-session use, this is fine — each session
only loads the model it needs.

To control VRAM sharing:

- Set `--models-max 1` (only one model stays loaded; least-recently-used is evicted)
- Run separate `llama-server` instances on different ports for models that must
  always be GPU-ready simultaneously (see [windows-service.md](windows-service.md)).

Router-mode behavior:

- Models load lazily on first request (`--models-autoload` is on by default) → fast service start.
- `--models-max N` caps how many stay loaded simultaneously (default 4; LRU eviction).
- Clients select the model per request: `"model": "ornith-9b"` in the OpenAI payload, or list via `GET /models`.
- This replaces the `--hf-repo`/`--model` flags — in router mode the server loads no model at startup.

> Do not confuse this file with Servy's service configuration. Servy is a Windows service wrapper; `presets.ini` is consumed by `llama-server` itself via `--models-preset`. Servy only forwards the argument (see [windows-service.md](windows-service.md)).

## Basic Server

```bash
uv run llama-server \
    --hf-repo Qwen/Qwen3.6-35B-A3B \
    --hf-file Qwen3.6-35B-A3B-Q4_K_M.gguf \
    --host 0.0.0.0 \
    --port 8080 \
    --ctx-size 32768
```

## Concurrency Tuning

```bash
# Parallel request slots (default: 1 — serial processing)
uv run llama-server --parallel-requests 4

# Continuous batching — overlap prompt processing across requests
uv run llama-server --cont-batching

# Prompt caching — reuse processed KV cache for repeated prompts
uv run llama-server --cache-prompt
```

## OpenAI-Compatible API

### Chat completions

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-2",
    "messages": [
      {"role": "system", "content": "You are helpful"},
      {"role": "user", "content": "Hello"}
    ],
    "temperature": 0.7,
    "max_tokens": 100
  }'
```

### Streaming

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-2",
    "messages": [{"role": "user", "content": "Count to 10"}],
    "stream": true
  }'
```

## Health & Metrics

```bash
# Health check
curl http://localhost:8080/health

# Server metrics
curl http://localhost:8080/metrics
```

**Metrics exposed:**

- `requests_total` — total requests received
- `tokens_generated` — tokens produced
- `prompt_tokens` — tokens in prompts
- `completion_tokens` — tokens in completions
- `kv_cache_tokens` — tokens in KV cache

## Load Balancing (NGINX)

```nginx
upstream llama_cpp {
    server llama1:8080;
    server llama2:8080;
}

server {
    location / {
        proxy_pass http://llama_cpp;
        proxy_read_timeout 300s;
    }
}
```
