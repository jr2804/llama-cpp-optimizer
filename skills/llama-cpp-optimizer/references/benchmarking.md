# Benchmarking llama-server presets

`scripts/bench-model.py` measures real-world chat performance of a preset
served by a running `llama-server` in router mode. It answers the question
the presets.ini comments care about: *"what tok/s will a user actually see,
and how much of the first-request wall time is model load?"*

## Why not just llama-bench?

`llama-bench` measures raw prefill (`pp`) and decode (`tg`) throughput with
no chat template, no EOS handling, no router, no KV-cache reuse. It is the
right tool for comparing quantizations or offload levels in isolation.

`bench-model.py` measures the full path a real client hits:
`/v1/chat/completions` → router → model load (if cold) → template → prefill
→ decode → EOS. That is the number users feel. Run both when you need both;
the script's `--llama-bench` flag shells out to `llama-bench` with the same
prompt/output sizes so the two are directly comparable.

## Methodology

1. **Prompt**: a fixed ~500-token system+user prompt (self-referencing
   capability list — no external knowledge required, so small models answer
   properly). The prompt length is the point: prefill cost scales with it.
2. **Forced output**: `min_tokens=200` (default) + `max_tokens=400` in the
   request body. `min_tokens` is a sampler constraint, not a guarantee —
   the model can still emit EOS early, but on this prompt it reliably runs
   to `length`.
3. **Cold run**: first request after server start. The router loads the
   model into VRAM on demand, so this request's wall time includes model
   load + graph compile + prefill + decode.
4. **Warm runs** (default 3): identical requests with `cache_prompt: false`
   so each pays a fresh prefill (no KV-cache reuse from the previous run).
   Median of the warm runs is the steady-state number.
5. **Warmup penalty** = cold wall − warm median wall. This is the per-session
   startup cost beyond what warm requests pay. It is a *lower bound* on
   load+compile time: cold prefill may also suffer page faults on first
   weight reads.

## Usage

```bash
# Server must be running in router mode first:
llama-server --models-preset presets.ini --no-warmup

# Bench one preset (cold + 3 warm runs):
uv run scripts/bench-model.py --preset gemma-4-12b-coder

# More warm runs, plus llama-bench for raw numbers:
uv run scripts/bench-model.py --preset qwen3.6-35b-a3b --warm 5 --llama-bench

# Non-default port:
uv run scripts/bench-model.py --preset ornith-9b --port 8080
```

Output: markdown table to stdout, full JSON (per-run timings) to
`./bench-results/<preset>-<timestamp>.json`.

## Interpreting the numbers

| Metric | What it means |
|--------|---------------|
| `decode tok/s` | Steady-state generation speed — the number users feel in chat. |
| `prefill tok/s` | Prompt-processing speed — matters for long inputs / RAG. |
| `warmup_penalty_s` | One-time cost per server session (load + graph compile). |
| `completion tokens` | Should hit `max_tokens` (400) — if lower, the model EOS'd early and the run is not a full decode test. |

### Known caveats

- **Cold run is only cold if the model isn't already loaded.** If a previous
  request loaded the model, the "cold" run is actually warm. Restart the
  server (`servy/restart.ps1`) for a true cold measurement.
- **`cache_prompt: false` is essential for warm runs.** Without it, the
  router reuses the slot's KV cache (LCP similarity match) and the warm
  prefill number collapses to near-zero — you'd be measuring cache hits, not
  prefill.
- **Speculative decoding (MTP draft) inflates decode tok/s.** A preset with
  `model-draft` + `spec-type = draft-mtp` reports the *effective* rate
  including accepted draft tokens. Check the server log for
  `draft acceptance` to know how much of the speedup is speculative.
- **`--fit` and Gemma-4 MTP drafts**: `--fit` tries to measure the draft
  model's memory standalone, which fails for `Gemma4Assistant` models
  (`requires ctx_other`). This is a warning — fit falls back to fitting
  without the draft — but if the draft then fails to load with
  `invalid vector subscript`, reduce `ctx-size` (e.g. 32768) or drop the
  draft. The 12B gemma preset documents this exact pattern.
- **Slot erase is unreliable in router mode** (this build returns 500 on
  `POST /slots/{id}?action=erase`). The script uses `cache_prompt: false`
  instead, which achieves the same fresh-prefill effect.
- **Resident models distort numbers.** The router keeps loaded models in
  VRAM until evicted (`--models-max`, default 4). Benching a model while
  other presets are resident fights for 8 GB of bandwidth: measured 15%+
  lower decode and 2× cold-load time on an 8 GB GPU. For clean numbers,
  bench each preset on a freshly restarted router, or unload other models
  first (`DELETE /models/{id}`, cache models only — presets cannot be
  unloaded in this build). Cross-run comparisons are only valid at the
  same residency state.
- **Warm-run variance is a real signal.** stdev across warm runs reflects
  draft-model acceptance (speculative decoding) and system noise. A stdev
  >2 tok/s on an MTP-draft preset usually means draft acceptance varies,
  not measurement error.

## Recording results

Update the preset's comment in `presets.ini` with the measured numbers:

```ini
; Speed @96K warm: ~8.1 tok/s decode (bench-model.py 2026-08-31: 3 warm runs,
;   400 tok each, stdev 0.19; prefill ~270 tps). Cold wall 67s incl ~17s load.
```

Include: date, warm-run count, output length, decode stdev, prefill rate,
and cold wall time. The JSON in `bench-results/` is the raw evidence; the
comment is the summary.
