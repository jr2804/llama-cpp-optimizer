# llama.cpp CLI Reference

Comprehensive reference for llama.cpp command-line tools. Covers all flags grouped by category.

## Common Flags (all tools)

These flags work across `llama-cli`, `llama-server`, `llama-bench`, and most other tools.

### Model Loading
| Flag | Description | Default |
|------|-------------|---------|
| `-m, --model <path>` | Path to GGUF model file | (required) |
| `-mu, --model-url <url>` | Model download URL | unused |
| `-hfr, --hf-repo <user>/<model>[:quant]` | Hugging Face repo (auto-download) | — |
| `-hff, --hf-file <file>` | Specific GGUF file in HF repo | — |
| `-hft, --hf-token <token>` | HF access token | `$HF_TOKEN` |
| `-lm, --load-mode <mode>` | Loading mode: `mmap`, `mlock`, `mmap+mlock`, `none`, `dio` | `mmap` |
| `--mlock` | DEPRECATED: use `--load-mode mlock` | — |
| `--no-mmap` | DEPRECATED: use `--load-mode none` | — |

### Context & Generation
| Flag | Description | Default |
|------|-------------|---------|
| `-c, --ctx-size N` | Prompt context size in tokens | model default |
| `-n, --predict N` | Number of tokens to predict (-1 = infinity) | -1 |
| `-b, --batch-size N` | Logical max batch size | 2048 |
| `-ub, --ubatch-size N` | Physical max batch size | 512 |
| `-t, --threads N` | CPU threads for generation | CPU count |
| `-tb, --threads-batch N` | CPU threads for batch/prompt processing | same as `--threads` |
| `-fa, --flash-attn <on\|off\|auto>` | Flash Attention | `auto` |
| `-np, --parallel N` | Parallel sequences to decode | 1 |
| `--keep N` | Tokens to keep from initial prompt | 0 |
| `--swa-full` | Use full-size SWA cache | false |

### GPU Offloading
| Flag | Description | Default |
|------|-------------|---------|
| `-ngl, --n-gpu-layers N` | Layers to offload to GPU (-1 = all) | 0 |
| `-dev, --device <dev1,dev2,...>` | Devices for offloading (`none` = CPU only) | all available |
| `-ts, --tensor-split N0,N1,...` | Fraction per GPU (comma-separated) | 0 |
| `-cmoe, --cpu-moe` | Keep ALL MoE expert weights in CPU | off |
| `-ncmoe, --n-cpu-moe N` | Keep first N layers' MoE experts in CPU | — |
| `-ctk, --cache-type-k <type>` | KV cache type for K | `f16` |
| `-ctv, --cache-type-v <type>` | KV cache type for V | `f16` |
| `--kv-offload, --no-kv-offload` | Enable KV cache offloading | enabled |
| `--op-offload, --no-op-offload` | Offload host tensor ops to device | enabled |
| `--repack, --no-repack` | Enable weight repacking | enabled |

**KV cache types:** `f32`, `f16`, `bf16`, `q8_0`, `q4_0`, `q4_1`, `iq4_nl`, `q5_0`, `q5_1`

### Sampling
| Flag | Description | Default |
|------|-------------|---------|
| `--temp N` | Temperature | 0.80 |
| `--top-p N` | Nucleus sampling cutoff | 0.95 |
| `--min-p N` | Minimum probability relative to top token | 0.05 |
| `--top-k N` | Top-K sampling (0 = disabled) | 40 |
| `--repeat-penalty N` | Repeat penalty | 1.00 |
| `--repeat-last-n N` | Tokens to consider for repeat penalty | 64 |
| `--dry-allowed-length N` | DRY sampling allowed length | 2 |
| `--dry-penalty-last-n N` | DRY penalty lookback | -1 (ctx_size) |
| `--xtc-probability N` | XTC probability | 0.00 |
| `--xtc-threshold N` | XTC threshold | 0.10 |
| `--typical-p N` | Typical sampling | 1.00 |
| `--penalize-nl` | Penalize newline tokens | off |
| `--ignore-eos` | Ignore EOS token | off |

### RoPE / Context Extension
| Flag | Description | Default |
|------|-------------|---------|
| `--rope-scaling <type>` | RoPE scaling type | model default |
| `--rope-scale N` | RoPE scale factor | model default |
| `--rope-freq-base N` | RoPE base frequency | model default |
| `--rope-freq-scale N` | RoPE frequency scale | model default |
| `--yarn-orig-ctx N` | YaRN original context size | 0 (model training) |
| `--yarn-ext-factor N` | YaRN extrapolation mix factor | -1.00 |
| `--yarn-attn-factor N` | YaRN attention magnitude | -1.00 |
| `--yarn-beta-slow N` | YaRN high correction dim | -1.00 |
| `--yarn-beta-fast N` | YaRN low correction dim | -1.00 |

### Chat & Interaction
| Flag | Description | Default |
|------|-------------|---------|
| `-cnv, --conversation` | Conversation mode (auto if chat template available) | auto |
| `-st, --single-turn` | Single turn only, then exit | off |
| `-sys, --system-prompt <text>` | System prompt | — |
| `-mli, --multiline-input` | Allow multi-line input without `\` | off |
| `-r, --reverse-prompt <text>` | Halt generation at prompt, return control | — |
| `-co, --color <on\|off\|auto>` | Colorize output | auto |
| `-rea, --reasoning <on\|off\|auto>` | Use reasoning/thinking in chat | auto |
| `--reasoning-format <format>` | Thought tag handling format | auto |
| `--reasoning-budget N` | Token budget for thinking | -1 (unrestricted) |
| `--chat-template <template>` | Custom Jinja chat template | model metadata |
| `--jinja, --no-jinja` | Use Jinja template engine | enabled |
| `--simple-io` | Basic IO for subprocess compatibility | off |

### Logging & Debug
| Flag | Description | Default |
|------|-------------|---------|
| `-lv, --verbosity N` | Verbosity threshold | 3 |
| `--perf, --no-perf` | Internal performance timings | off |
| `--log-colors <on\|off\|auto>` | Colored logging | auto |
| `--log-prompts-dir <path>` | Log prompts to directory | — |

## llama-cli Specific

| Flag | Description | Default |
|------|-------------|---------|
| `-f, --file <path>` | File containing prompt | — |
| `-bf, --binary-file <path>` | Binary file containing prompt | — |
| `-o, --output-file <path>` | Output file | — |
| `--prompt <text>` | Prompt text | — |
| `--interactive` | Interactive mode (DEPRECATED: use `--conversation`) | — |
| `--in-prefix <text>` | Prefix to prepend to user input | — |
| `--in-suffix <text>` | Suffix to append to user input | — |
| `--no-display-prompt` | Don't display prompt | off |
| `--spec-default` | Enable default speculative decoding | off |
| `--spec-draft-hf <repo>` | Draft model HF repo | — |
| `--spec-draft-n-gpu-layers N` | Draft model GPU layers | 0 |
| `--spec-draft-cpu-moe` | Draft model CPU MoE | off |

## llama-server Specific

| Flag | Description | Default |
|------|-------------|---------|
| `--host <addr>` | Bind address | `127.0.0.1` |
| `--port N` | Port | 8080 |
| `--timeout N` | Server timeout in seconds | 0 (no timeout) |
| `--api-key <key>` | API key for authentication | none |
| `--cors-origins <origins>` | CORS allowed origins | `*` |
| `--cors-methods <methods>` | CORS allowed methods | `GET, POST, PUT, DELETE, PATCH` |
| `--cors-headers <headers>` | CORS allowed headers | `*` |
| `--reuse-port` | Allow multiple sockets on same port | off |
| `--slots` | Show slot info in completion response | off |
| `--slot-save-path <path>` | Path to save slot states | — |
| `--embeddings, --no-embeddings` | Enable embedding endpoint | off |
| `--rerank, --no-rerank` | Enable reranking endpoint | off |
| `--cont-batching, --no-cont-batching` | Enable continuous batching | on |
| `--metrics, --no-metrics` | Enable Prometheus metrics | off |
| `--chat-template-endpoint, --no-chat-template-endpoint` | Enable chat template endpoint | on |

### Endpoints (when running)
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | OpenAI-compatible chat completions |
| `/v1/completions` | POST | Text completions |
| `/v1/embeddings` | POST | Embeddings (if `--embeddings`) |
| `/v1/rerank` | POST | Reranking (if `--rerank`) |
| `/v1/models` | GET | List loaded models |
| `/health` | GET | Health check |
| `/metrics` | GET | Prometheus metrics (if `--metrics`) |
| `/slots` | GET | Slot information |

## llama-bench Specific

| Flag | Description | Default |
|------|-------------|---------|
| `-r, --repetitions N` | Times to repeat each test | 5 |
| `--delay N` | Delay between tests (seconds) | 0 |
| `-o, --output <format>` | Output format: `csv`, `json`, `jsonl`, `md`, `sql` | `md` |
| `-oe, --output-err <format>` | Stderr output format | none |
| `--numa <mode>` | NUMA mode: `distribute`, `isolate`, `numactl` | disabled |
| `--prio N` | Process/thread priority | 0 |

Multiple values can be given per parameter by separating with `,` or by specifying the parameter multiple times. Ranges: `first-last` or `first-last+step` or `first-last*mult`.

## llama-perplexity Specific

| Flag | Description | Default |
|------|-------------|---------|
| `-f, --file <path>` | Test corpus file | (required) |
| `--ppl-stride N` | Perplexity stride | 0 |
| `--ppl-output-type <type>` | Output type | 0 |

## llama-tokenize Specific

| Flag | Description | Default |
|------|-------------|---------|
| `--prompt <text>` | Text to tokenize | — |
| `--show-count` | Show token count | off |
| `--detokenize` | Detokenize (reverse) | off |
| `--special` | Allow special tokens | off |

## Environment Variables

| Variable | Affects | Description |
|----------|---------|-------------|
| `LLAMA_ARG_THREADS` | All | CPU threads |
| `LLAMA_ARG_CTX_SIZE` | All | Context size |
| `LLAMA_ARG_N_PREDICT` | All | Tokens to predict |
| `LLAMA_ARG_BATCH` | All | Batch size |
| `LLAMA_ARG_UBATCH` | All | Physical batch size |
| `LLAMA_ARG_FLASH_ATTN` | All | Flash Attention |
| `LLAMA_ARG_CPU_MOE` | All | CPU MoE offloading |
| `LLAMA_ARG_N_CPU_MOE` | All | N layers CPU MoE |
| `LLAMA_ARG_CACHE_TYPE_K` | All | KV cache type K |
| `LLAMA_ARG_CACHE_TYPE_V` | All | KV cache type V |
| `LLAMA_ARG_KV_OFFLOAD` | All | KV cache offloading |
| `LLAMA_ARG_MMAP` | All | Memory-mapping |
| `LLAMA_ARG_MLOCK` | All | Mlock |
| `LLAMA_ARG_DIO` | All | Direct I/O |
| `LLAMA_ARG_REPACK` | All | Weight repacking |
| `LLAMA_ARG_NO_HOST` | All | Host buffer bypass |
| `LLAMA_ARG_SYSTEM_PROMPT` | cli | System prompt |
| `LLAMA_ARG_CHAT_TEMPLATE` | cli, server | Chat template |
| `LLAMA_ARG_SKIP_CHAT_PARSING` | cli, server | Skip chat parsing |
| `LLAMA_ARG_REASONING` | cli, server | Reasoning mode |
| `LLAMA_ARG_THINK` | cli, server | Think mode |
| `LLAMA_ARG_THINK_BUDGET` | cli, server | Think budget |
| `LLAMA_ARG_JINJA` | cli, server | Jinja template engine |
| `LLAMA_ARG_YARN_*` | All | YaRN parameters |
| `LLAMA_ARG_ROPE_*` | All | RoPE parameters |
| `LLAMA_ARG_N_PARALLEL` | server | Parallel sequences |
| `LLAMA_ARG_N_SLOTS` | server | Slot count |
| `LLAMA_ARG_PORT` | server | Port |
| `LLAMA_ARG_HOST` | server | Host |
| `LLAMA_ARG_API_KEY` | server | API key |
| `HF_TOKEN` | All | Hugging Face access token |
