# Model Format Caveats

Gotchas that produce confusing "model won't load" failures. The failure mode is usually a GGUF tensor offset error:

```text
gguf_init_from_reader: tensor 'output_norm.weight' has offset X, expected Y
llama_model_load: error loading model
```

**That error means the file's tensor layout doesn't match what your llama.cpp build expects — not a corrupted download and not a missing sidecar file.**

## 1. Fork-format GGUFs vs mainline llama.cpp

Some model repos (e.g. [prism-ml/Ternary-Bonsai-27B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-27B-gguf)) ship weights in a **fork-specific GGUF packing** that mainline llama.cpp cannot load, even after the kernels merge upstream. The file may share a type id with a mainline format but use a different block size → offset mismatch on load.

Example — Ternary-Bonsai (ternary Q2_0), from the [upstream status table](https://github.com/PrismML-Eng/Bonsai-demo#upstream-status-for-ternary):

| File | Format | Runs on |
|------|--------|---------|
| `*-Q2_0.gguf` | Group size 128 (fork format) | Fork only — **will not load on mainline** (same type id, different block size) |
| `*-Q2_0_g64.gguf` | Group size 64 (official llama.cpp format) | Mainline (CPU, Metal, Vulkan, CUDA) |
| `*-PQ2_0.gguf` | Experimental migration format | Do not use yet |

Checklist before diagnosing a load failure:

1. **Verify the download** — compare SHA256 against the repo's LFS pointer:

   ```bash
   curl -sL "https://huggingface.co/<user>/<repo>/raw/main/<file>" | grep oid
   certutil -hashfile models/<file> SHA256
   ```

2. **Check the repo README for an upstream-status / compatibility section** — it usually states which build the file needs.
3. **Prefer the mainline-named variant** — `*_g64`, plain `Q2_0` on mainline builds; the fork's `Q2_0` (g128) is a different format with the same name.

## 2. "Missing sidecar" is rarely the cause

Model repos often ship companion files — vision projectors (`*mmproj*`) and speculative-decoding drafters (`*draft*`, `*dspark*`). These are **optional**:

- mmproj loads only for image input; text-only serving never needs it
- drafter only accelerates speculative decoding; the model runs fine without it

A load failure is almost never a missing sidecar. Check the tensor layout first (section 1).

## 3. Architecture support

Check whether your build knows the model's architecture before debugging further:

```bash
# Does the build contain the arch string? (Windows)
grep -ac "<arch>" <llama-dir>/llama.dll
```

Presence of the arch string in the binary is necessary but **not sufficient** — the custom tensor packing may still be unsupported (section 1).

## Quick triage

1. Load fails with offset error → fork-format GGUF (section 1), not corruption, not sidecar
2. File hash matches but still fails → format incompatibility, get the mainline variant
3. `unknown architecture` → your build is too old; update llama.cpp
4. Works in `llama-cli` but not `llama-server` → check server args, not the file
5. Hybrid-attention model unexpectedly slow or OOM at full offload → try CUDA (section 4)

## 4. Backend choice: CUDA vs Vulkan for hybrid-attention models

Vulkan is the portable default (works on AMD/Intel/NVIDIA), but it lags CUDA on **fused kernels for new attention types**. Hybrid-attention models (linear/delta-net, gated-delta-net, mamba, etc.) ship fused CUDA/Metal kernels that Vulkan may not have yet.

Symptom on Vulkan — a warning at load:
```
W resolve_fused_ops: fused Gated Delta Net (chunked) not supported, set to disabled
```
The model still runs (unfused fallback), but **5–6× slower** at generation.

Measured — Ternary-Bonsai-27B Q2_0_g64 on RTX A2000 8GB:

| Backend | ngl=99 tg (t/s) | ngl=99 at 32K ctx |
|---------|-----------------|-------------------|
| Vulkan | 2.64 | ❌ OOM |
| CUDA | 15.57 | ✅ fits (7.8 GB) |

CUDA both unlocks the fused kernel AND has lower allocation overhead (fits full offload where Vulkan OOM'd at the same ctx). For an NVIDIA card, prefer the CUDA build for these model families. Vulkan stays the right call for portability/sharing across GPU vendors.

### Installing both via mise

```toml
# mise.toml — two entries won't both resolve; pick one per project, or
# install the CUDA build ad-hoc for local use and keep Vulkan as the shared default.
[tools]
"github:ggml-org/llama.cpp" = {version = "latest", asset_pattern = "llama-*-cuda-12.4-x64-*"}   # NVIDIA local
# "github:ggml-org/llama.cpp" = {version = "latest", asset_pattern = "llama-*-vulkan-*"}        # portable
```

CUDA builds ship as two zips on the GitHub release: `llama-*-bin-win-cuda-12.4-x64.zip` (the binaries) **plus** `cudart-llama-bin-win-cuda-12.4-x64.zip` (the ~370 MB CUDA runtime DLLs). Both must be extracted to the same directory.
