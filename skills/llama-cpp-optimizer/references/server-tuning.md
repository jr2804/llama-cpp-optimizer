# Server Tuning Guide

Production deployment patterns for `llama-server` with OpenAI-compatible API.

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
