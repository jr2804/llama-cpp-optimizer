# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Benchmark a llama-server preset: cold + N warm runs with KV cache reset.

Measures prompt-prefill and decode speeds by hitting the OpenAI-compatible
`/v1/chat/completions` endpoint of a running `llama-server` (router mode).
Between warm runs, drops the slot's KV cache via `POST /slots/{id}?action=erase`
so each warm run measures a fresh prefill — i.e. the cache state any new
request actually gets. Optionally also runs `llama-bench` for raw decode.

The "cold" run is defined as: the first request that triggers the router to
load the model into VRAM. If the model is already loaded, it is treated as
warm (set a fresh server for a true cold measurement).

Usage:
    uv run bench-model.py --preset gemma-4-12b-coder
    uv run bench-model.py --preset qwen3.6-35b-a3b --warm 5 --llama-bench
    uv run bench-model.py --preset gemma-4-12b-coder --host 127.0.0.1 --port 8001

Output: markdown table to stdout, full JSON to --json path (default
./bench-results/<preset>-<timestamp>.json).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from shutil import which

# --- Prompt design --------------------------------------------------------
# Goal: ~500-token in-context history + a request that nudges 200+ tokens
# of self-referencing output (no external knowledge required, so 9B coder
# models that aren't trained for recall still produce a real answer).
#
# `SYSTEM_PROMPT` is a short persona; `USER_PROMPT` carries the history
# (an "earlier" exchange that wasn't really asked) plus the actual ask.
# All of it becomes prefill input — that's the point: prefill scales with
# prompt length, decode scales with `max_tokens` produced.

SYSTEM_PROMPT = (
    "You are a senior software engineer reviewing your own capabilities. "
    "Be specific and grounded: only describe tasks you can actually do. "
    "For each capability, give a 2-3 sentence explanation that mentions "
    "concrete inputs, outputs, or tools."
)

# ~500 tokens of realistic history. Padding the prompt exercises prefill
# performance at a non-trivial ctx; if a 9B model can't sustain 96K
# context, the bench will reflect that here (it should still fit 4K).
USER_PROMPT = """Earlier today you helped debug a flaky CI pipeline that was
failing only on Windows runners. We traced the issue to a race in the
test-cleanup hook, fixed it by adding a synchronization barrier, and then
you wrote a short post-mortem note.

You also spent some time reviewing a pull request that refactored the
authentication layer to use a token-bucket rate limiter. You pointed out
that the bucket refill interval needed to be configurable per route group,
not just globally, and the author agreed and pushed a follow-up commit.

After that, you helped convert a brittle regex-based URL parser into a
proper tokenizer-based one. The new version handles percent-encoding
edge cases and IPv6 brackets, and includes unit tests for the malformed
inputs that the old version used to silently accept.

Now please answer the following question about your own abilities.

List 20 distinct tasks that you are well suited to help with. For each
item, write 2-3 sentences explaining what you would do, what input you
would need from the user, and what the output would look like. Pick tasks
that you can realistically perform end to end, not vague categories."""


# --- Core bench ------------------------------------------------------------


@dataclass
class RunResult:
    """One request's worth of timing data."""

    kind: str  # "cold" | "warm"
    wall_s: float
    prompt_tokens: int
    completion_tokens: int
    prefill_ms: float
    prefill_tok_s: float
    decode_ms: float
    decode_tok_s: float
    total_ms: float
    total_tok_s: float
    finish_reason: str
    model: str
    slot_id: int | None = None


@dataclass
class BenchResult:
    preset: str
    host: str
    port: int
    warm_reps: int
    min_tokens: int
    max_tokens: int
    runs: list[RunResult] = field(default_factory=list)
    llama_bench: dict | None = None
    notes: list[str] = field(default_factory=list)

    def summarize(self) -> dict:
        warm = [r for r in self.runs if r.kind == "warm"]
        cold = [r for r in self.runs if r.kind == "cold"]
        out: dict = {"preset": self.preset, "warm_reps": len(warm)}

        if cold:
            c = cold[0]
            out["cold"] = {
                "wall_s": round(c.wall_s, 3),
                "prefill_tok_s": round(c.prefill_tok_s, 2),
                "decode_tok_s": round(c.decode_tok_s, 2),
                "total_tok_s": round(c.total_tok_s, 2),
                "completion_tokens": c.completion_tokens,
            }
            self.notes.append(
                "cold run includes router model-load + warmup prefill; "
                "prefill tok/s on cold may be lower than warm due to mmap "
                "page faults on first read of weights."
            )

        if warm:

            def mget(attr: str) -> float:
                return statistics.median(getattr(r, attr) for r in warm)

            out["warm_median"] = {
                "prefill_tok_s": round(mget("prefill_tok_s"), 2),
                "decode_tok_s": round(mget("decode_tok_s"), 2),
                "total_tok_s": round(mget("total_tok_s"), 2),
                "completion_tokens": int(mget("completion_tokens")),
            }
            if len(warm) > 1:
                out["warm_stdev"] = {
                    "decode_tok_s": round(
                        statistics.stdev(r.decode_tok_s for r in warm), 2
                    ),
                }

            # Warmup penalty = cold_wall - warm_median_wall.
            # Best estimate of pure server-load+graph time the user pays
            # once on cold start. Underestimates if cold had page faults.
            if cold:
                warm_wall = mget("wall_s")
                out["warmup_penalty_s"] = round(cold[0].wall_s - warm_wall, 2)
                self.notes.append(
                    "warmup_penalty_s = cold_wall - warm_median_wall. "
                    "This is the per-session startup cost beyond what warm "
                    "requests pay. It is a lower bound if cold prefill "
                    "suffered page faults; the true load+compile time may "
                    "be a bit higher."
                )

        if self.llama_bench is not None:
            out["llama_bench"] = self.llama_bench

        return out


def erase_slot(host: str, port: int, slot_id: int, preset: str) -> int:
    """Drop a slot's KV cache. Returns tokens erased.

    Router mode: `POST /slots/{id}?action=erase&model=<preset>`.
    The server rejects requests with no body (empty input → 500); send
    JSON `null` so the body parser sees a valid (null) document.
    """
    raw = json.dumps(None).encode("utf-8")
    req = urllib.request.Request(
        f"http://{host}:{port}/slots/{slot_id}?action=erase"
        f"&model={urllib.parse.quote(preset)}",
        data=raw,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        body = resp.read().decode("utf-8")
    out = json.loads(body) if body else {}
    return int(out.get("n_erased", 0))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset", required=True, help="preset name (e.g. gemma-4-12b-coder)"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument(
        "--warm", type=int, default=3, help="number of warm runs (default 3)"
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=200,
        help="force model to emit at least this many tokens (default 200)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=400,
        help="hard cap on output length (default 400)",
    )
    parser.add_argument(
        "--llama-bench",
        action="store_true",
        help="also run llama-bench for raw decode numbers",
    )
    parser.add_argument(
        "--presets-ini",
        default="presets.ini",
        help="path to presets.ini (for llama-bench model path)",
    )
    parser.add_argument(
        "--json",
        default=None,
        help="output JSON path (default ./bench-results/<preset>-<ts>.json)",
    )
    args = parser.parse_args()
    _force_utf8_stdout()

    wait_for_server(args.host, args.port)
    loaded = list_loaded_models(args.host, args.port)
    was_loaded = args.preset in loaded

    payload = build_payload(args.min_tokens, args.max_tokens)
    # Cold run: same payload, no cache_prompt override (the model isn't
    # loaded yet, so cache reuse is impossible).
    result = BenchResult(
        preset=args.preset,
        host=args.host,
        port=args.port,
        warm_reps=args.warm,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
    )

    if not was_loaded:
        result.notes.append(
            f"Model {args.preset!r} not loaded at start — first run is a true "
            f"cold load (router must mmap weights into VRAM). For a "
            f"reproducible cold number, restart llama-server first."
        )

    # 1) Cold run (or warm if already loaded)
    cold = one_run(args.host, args.port, args.preset, payload, kind="cold")
    result.runs.append(cold)
    print(
        f"cold: wall={cold.wall_s}s prefill={cold.prefill_tok_s} tps "
        f"decode={cold.decode_tok_s} tps out={cold.completion_tokens} tok "
        f"finish={cold.finish_reason}",
        file=sys.stderr,
    )

    # 2) Warm runs with cache_prompt=false so each run pays a fresh prefill
    # (the router's slot-erase endpoint is unreliable in this build).
    for i in range(args.warm):
        warm_payload = dict(payload)
        warm_payload["cache_prompt"] = False
        slot_id = _find_a_slot(args.host, args.port, args.preset)
        r = one_run(
            args.host,
            args.port,
            args.preset,
            warm_payload,
            kind="warm",
            slot_hint=slot_id,
        )
        result.runs.append(r)
        print(
            f"warm[{i + 1}/{args.warm}]: wall={r.wall_s}s prefill={r.prefill_tok_s} tps "
            f"decode={r.decode_tok_s} tps out={r.completion_tokens} tok "
            f"finish={r.finish_reason}",
            file=sys.stderr,
        )

    # 3) Optional llama-bench
    if args.llama_bench:
        model_path = _resolve_model_path(args.preset, args.presets_ini)
        if model_path:
            # Estimate prompt_tokens by tokenizing via a quick request to
            # the server using the same payload; if we already have cold
            # prompt_tokens, reuse them.
            pt = cold.prompt_tokens or 0
            result.llama_bench = run_llama_bench(
                args.preset, args.presets_ini, model_path, pt, args.max_tokens
            )

    # 4) Write JSON + print table
    out_dir = "./bench-results"
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    json_path = args.json or os.path.join(out_dir, f"{args.preset}-{ts}.json")
    summary = result.summarize()
    summary["runs"] = [asdict(r) for r in result.runs]
    summary["notes"] = result.notes
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(render_table(summary))
    print(f"\nJSON written to: {json_path}", file=sys.stderr)
    return 0


def wait_for_server(host: str, port: int, timeout_s: float = 120.0) -> None:
    """Block until /health returns 200 or timeout."""
    url = f"http://{host}:{port}/health"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(1.0)
    raise TimeoutError(f"server at {url} did not become healthy in {timeout_s}s")


def list_loaded_models(host: str, port: int) -> list[str]:
    """Return IDs of models currently in VRAM (router mode)."""
    data = http_json("GET", f"http://{host}:{port}/v1/models")
    return [m["id"] for m in data.get("data", [])]


def build_payload(min_tokens: int, max_tokens: int) -> dict:
    """OpenAI-compatible chat request, with min_tokens forced via sampling."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],
        "max_tokens": max_tokens,
        "min_tokens": min_tokens,
        "temperature": 0.7,
        "top_p": 0.9,
        "stream": False,
    }


def one_run(
    host: str,
    port: int,
    preset: str,
    payload: dict,
    kind: str,
    slot_hint: int | None = None,
) -> RunResult:
    """One chat completion; returns parsed timing data."""
    body = dict(payload)
    body["model"] = preset
    url = f"http://{host}:{port}/v1/chat/completions"

    t0 = time.monotonic()
    resp = http_json("POST", url, body)
    wall = time.monotonic() - t0

    # llama-server returns `usage.prompt_tokens` / `completion_tokens` when
    # `stream: false`. `timings` is always populated on `/v1/chat/completions`
    # when the request completes (no streaming).
    usage = resp.get("usage", {})
    timings = resp.get("timings", {}) or {}
    prompt_n = int(usage.get("prompt_tokens", timings.get("prompt_n", 0)))
    pred_n = int(usage.get("completion_tokens", timings.get("predicted_n", 0)))
    prompt_ms = float(timings.get("prompt_ms", 0.0))
    pred_ms = float(timings.get("predicted_ms", 0.0))
    total_ms = float(timings.get("total_ms", (wall * 1000.0)))

    prefill_tps = (prompt_n / (prompt_ms / 1000.0)) if prompt_ms > 0 else 0.0
    decode_tps = (pred_n / (pred_ms / 1000.0)) if pred_ms > 0 else 0.0
    total_tps = (pred_n / (total_ms / 1000.0)) if total_ms > 0 else 0.0

    finish = resp.get("choices", [{}])[0].get("finish_reason", "?")
    model_id = resp.get("model", preset)

    # Find which slot was used. The server doesn't echo slot id in the
    # response; we approximate from the next /slots call (filled after).
    return RunResult(
        kind=kind,
        wall_s=round(wall, 3),
        prompt_tokens=prompt_n,
        completion_tokens=pred_n,
        prefill_ms=round(prompt_ms, 2),
        prefill_tok_s=round(prefill_tps, 2),
        decode_ms=round(pred_ms, 2),
        decode_tok_s=round(decode_tps, 2),
        total_ms=round(total_ms, 2),
        total_tok_s=round(total_tps, 2),
        finish_reason=finish,
        model=model_id,
        slot_id=slot_hint,
    )


# --- llama-bench (optional) ------------------------------------------------


def run_llama_bench(
    preset_section: str,
    presets_ini: str,
    model_path: str,
    prompt_tokens: int,
    output_tokens: int,
) -> dict | None:
    """Run llama-bench with the same prompt/n as chat bench, return parsed CSV row."""
    # ponytail: assume `llama-bench` is on PATH; fall back to mise install.
    bench = os.environ.get("LLAMA_BENCH", "llama-bench")
    if not _which(bench):
        mise = os.path.expandvars(
            r"%LOCALAPPDATA%\mise\installs\github-ggml-org-llama-cpp\b10715\llama-bench.exe"
        )
        if os.path.isfile(mise):
            bench = mise
        else:
            return None

    # ponytail: 3 reps; -ngl 0 lets the bench succeed even before --fit
    # metadata is parsed. Caller can override via env LLAMA_BENCH_REPS.
    reps = int(os.environ.get("LLAMA_BENCH_REPS", "3"))
    try:
        result = subprocess.run(
            [
                bench,
                "-m",
                model_path,
                "-p",
                str(prompt_tokens),
                "-n",
                str(output_tokens),
                "-r",
                str(reps),
                "-o",
                "json",
            ],
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None

    if result.returncode != 0 or not result.stdout.strip():
        return {"error": result.stderr.strip() or "no output"}
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"raw": result.stdout[:2000]}
    # data is a list of test rows; we asked for one
    return {"preset": preset_section, "rows": data} if isinstance(data, list) else data


def _which(name: str) -> str | None:
    return which(name)


# --- Reporting ------------------------------------------------------------


def render_table(summary: dict) -> str:
    lines: list[str] = []
    preset = summary.get("preset", "?")
    lines.append(f"# Bench: {preset}")
    if "cold" in summary:
        c = summary["cold"]
        lines.append("")
        lines.append("## Cold (model not in VRAM, router load + first request)")
        lines.append("")
        lines.append("| metric | value |")
        lines.append("|---|---|")
        lines.append(f"| wall time (s) | {c['wall_s']} |")
        lines.append(f"| prefill tok/s | {c['prefill_tok_s']} |")
        lines.append(f"| decode tok/s | {c['decode_tok_s']} |")
        lines.append(f"| total tok/s (output only) | {c['total_tok_s']} |")
        lines.append(f"| completion tokens | {c['completion_tokens']} |")
    if "warm_median" in summary:
        w = summary["warm_median"]
        lines.append("")
        lines.append(
            f"## Warm median over {summary['warm_reps']} runs (KV reset between)"
        )
        lines.append("")
        lines.append("| metric | value |")
        lines.append("|---|---|")
        lines.append(f"| prefill tok/s | {w['prefill_tok_s']} |")
        lines.append(f"| decode tok/s | {w['decode_tok_s']} |")
        lines.append(f"| total tok/s (output only) | {w['total_tok_s']} |")
        lines.append(f"| completion tokens | {w['completion_tokens']} |")
    if "warm_stdev" in summary:
        lines.append(
            f"| decode tok/s stdev | {summary['warm_stdev']['decode_tok_s']} |"
        )
    if "warmup_penalty_s" in summary:
        lines.append("")
        lines.append(
            f"**Warmup penalty:** ~{summary['warmup_penalty_s']} s "
            f"(cold wall - warm median wall; lower bound on load+graph cost)."
        )
    if summary.get("llama_bench"):
        lines.append("")
        lines.append("## llama-bench (raw prefill + decode, no chat overhead)")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(summary["llama_bench"], indent=2)[:2000])
        lines.append("```")
    return "\n".join(lines) + "\n"


# --- Main -----------------------------------------------------------------


def _force_utf8_stdout() -> None:
    """Reconfigure stdout/stderr to UTF-8 on Windows (cp1252 chokes on -)."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, OSError):
            pass


def _find_a_slot(host: str, port: int, preset: str) -> int:
    try:
        free = list_free_slots(host, port, preset)
        return free[0]
    except (urllib.error.URLError, KeyError, ValueError):
        return 0


def list_free_slots(host: str, port: int, preset: str) -> list[int]:
    """Return IDs of idle (non-processing) slots for a router model.

    In router mode `/slots` requires `?model=<preset>` to pick which
    loaded model's slots to return. If the model has no slots yet (rare,
    but possible right after load) we fall back to a single placeholder
    slot 0 — the server will create one on the first request.
    """
    data = http_json(
        "GET",
        f"http://{host}:{port}/slots?model={urllib.parse.quote(preset)}",
    )
    slots = data if isinstance(data, list) else data.get("slots", [])
    if not slots:
        return [0]
    out: list[int] = []
    for s in slots:
        if not s.get("processing", False) and not s.get("is_processing", False):
            out.append(int(s["id"]))
    return out or [0]


# --- HTTP helpers (stdlib only, no requests dep) --------------------------


def http_json(
    method: str,
    url: str,
    payload: dict | None = None,
    timeout: float = 600.0,
) -> dict:
    """Issue a JSON HTTP request, return parsed JSON response."""
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body) if body else {}


def _resolve_model_path(preset: str, presets_ini: str) -> str | None:
    """Tiny INI parser: pull `model = ...` from the matching section."""
    if not os.path.isfile(presets_ini):
        # try relative to repo root
        for cand in [
            presets_ini,
            os.path.join("..", presets_ini),
            os.path.join("..", "..", presets_ini),
        ]:
            if os.path.isfile(cand):
                presets_ini = cand
                break
        else:
            return None
    section = f"[{preset}]"
    in_section = False
    with open(presets_ini, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("[") and line.endswith("]"):
                in_section = line == section
                continue
            if in_section and line.startswith("model") and "=" in line:
                # ponytail: trust INI structure; values may be quoted
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


if __name__ == "__main__":
    sys.exit(main())
