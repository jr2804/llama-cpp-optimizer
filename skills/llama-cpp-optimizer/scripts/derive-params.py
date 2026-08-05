# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

"""Derive optimal llama.cpp parameters from system capabilities and model metadata.

Usage:
    # From JSON files
    uv run derive-params.py --system system.json --model model.json

    # From piped JSON
    uv run model-info.py Qwen/Qwen3.6-35B-A3B | uv run derive-params.py --model -

    # Interactive mode (auto-detect system)
    uv run derive-params.py --model model.json

Outputs a JSON object with recommended llama.cpp command-line arguments.
"""

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Quantization bytes per parameter
QUANT_BYTES: dict[str, float] = {
    "bf16": 2.0,
    "fp16": 2.0,
    "f16": 2.0,
    "q8_0": 1.0,
    "q6_k": 0.75,
    "q5_k_m": 0.625,
    "q5_k": 0.625,
    "q5_0": 0.625,
    "q4_k_m": 0.5,
    "q4_k": 0.5,
    "q4_0": 0.5,
    "q3_k_m": 0.375,
    "q3_k": 0.375,
    "q2_k": 0.25,
    "iq4_nl": 0.5,
    "mxfp4": 0.5,
}

# Cache type bytes per value
CACHE_BYTES: dict[str, float] = {
    "f16": 2.0,
    "q8_0": 1.0,
    "q4_0": 0.5,
    "q4_1": 0.5,
    "iq4_nl": 0.5,
}

# Flash attention support by compute capability
FLASH_ATTN_MIN_CC = 7.0  # Turing (RTX 2xxx)


@dataclass
class GpuDetail:
    """GPU detail from system detection."""

    name: str = ""
    vram_total_mb: int = 0
    vram_free_mb: int = 0
    compute_cap: str = ""


@dataclass
class SystemInfo:
    """System capabilities (subset of detect-system.py output)."""

    gpus: list[GpuDetail] = field(default_factory=list)
    ram_total_gb: float = 0.0
    ram_free_gb: float = 0.0
    cpu_cores_logical: int = 0
    cpu_cores_physical: int = 0
    cpu_name: str = ""
    ram_speed_mhz: int = 0


@dataclass
class ModelInfo:
    """Model metadata (subset of model-info.py output)."""

    model_id: str = ""
    model_type: str = ""
    is_moe: bool = False
    num_hidden_layers: int = 0
    num_attention_heads: int = 0
    num_key_value_heads: int = 0
    hidden_size: int = 0
    intermediate_size: int = 0
    head_dim: int = 0
    max_position_embeddings: int = 0
    vocab_size: int = 0
    num_experts: int = 0
    num_experts_per_tok: int = 0
    gguf_files: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class DerivedParams:
    """Optimal llama.cpp parameters."""

    ctx_size: int = 64000
    n_gpu_layers: int = 0
    flash_attn: str = "on"
    cache_type_k: str = "f16"
    cache_type_v: str = "f16"
    cpu_moe: bool = False
    n_cpu_moe: int = 0
    load_mode: str = "mmap"
    batch_size: int = 2048
    ubatch_size: int = 512
    threads: int = 0
    temp: float = 0.80
    top_p: float = 0.95
    min_p: float = 0.05
    reasoning: bool = False
    warnings: list[str] = field(default_factory=list)


def main() -> None:
    """Main entry point."""

    parser = argparse.ArgumentParser(
        description="Derive optimal llama.cpp parameters from system + model info"
    )
    parser.add_argument(
        "--system",
        help="System info JSON file (or '-' for stdin). Auto-detects if omitted.",
        default=None,
    )
    parser.add_argument(
        "--model",
        help="Model info JSON file (or '-' for stdin). Required.",
        required=True,
    )
    parser.add_argument(
        "--model-path",
        help="Local path to GGUF model file (for CLI output).",
        default="",
    )
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Output as CLI argument string instead of JSON.",
    )

    args = parser.parse_args()

    # Load system info
    if args.system:
        system_data = load_json_input(args.system)
        system = SystemInfo(
            **{k: system_data.get(k, v) for k, v in asdict(SystemInfo()).items()}
        )
    else:
        system = detect_system()

    # Load model info
    model_data = load_json_input(args.model)
    model = ModelInfo(
        **{k: model_data.get(k, v) for k, v in asdict(ModelInfo()).items()}
    )

    # Derive parameters
    params = derive_params(system, model)

    if args.cli:
        cli_args = format_cli_args(params, args.model_path)
        print(" ".join(cli_args))
    else:
        output = asdict(params)
        output["_cli"] = " ".join(format_cli_args(params, args.model_path))
        print(json.dumps(output, indent=2))


def detect_system() -> SystemInfo:
    """Auto-detect system by running detect-system.py."""
    script_dir = Path(__file__).parent
    detect_script = script_dir / "detect-system.py"

    if not detect_script.exists():
        return SystemInfo()

    try:
        result = subprocess.run(
            ["uv", "run", str(detect_script)],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            return SystemInfo(
                gpus=data.get("gpus", []),
                ram_total_gb=data.get("ram_total_gb", 0.0),
                ram_free_gb=data.get("ram_free_gb", 0.0),
                cpu_cores_logical=data.get("cpu_cores_logical", 0),
                cpu_cores_physical=data.get("cpu_cores_physical", 0),
                cpu_name=data.get("cpu_name", ""),
                ram_speed_mhz=data.get("ram_speed_mhz", 0),
            )
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
        pass

    return SystemInfo()


def load_json_input(arg: str) -> dict[str, Any]:
    """Load JSON from a file path or stdin ('-')."""
    if arg == "-":
        return json.loads(sys.stdin.read())
    path = Path(arg)
    if path.exists() and path.is_file():
        return json.loads(path.read_text())
    # Try as raw JSON string
    try:
        return json.loads(arg)
    except json.JSONDecodeError:
        print(
            f"Error: file not found and not valid JSON: {arg[:80]}...",
            file=sys.stderr,
        )
        print(f"  CWD: {Path.cwd()}", file=sys.stderr)
        print(f"  Tried path: {path.absolute()}", file=sys.stderr)
        sys.exit(1)


def derive_params(system: SystemInfo, model: ModelInfo) -> DerivedParams:
    """Derive optimal llama.cpp parameters from system + model info."""
    params = DerivedParams()
    params.threads = system.cpu_cores_physical or system.cpu_cores_logical or 8

    # ── VRAM analysis ──────────────────────────────────────────────
    vram_total_mb = 0
    vram_free_mb = 0
    gpu_cc = 0.0
    gpu_name = ""

    for gpu in system.gpus:
        vram_total_mb = max(vram_total_mb, gpu.vram_total_mb)
        vram_free_mb = max(vram_free_mb, gpu.vram_free_mb)
        if gpu.compute_cap:
            try:
                gpu_cc = max(gpu_cc, float(gpu.compute_cap))
            except ValueError:
                pass
        if gpu.name:
            gpu_name = gpu.name

    # Fallback: assume 75% of total VRAM is free
    if vram_free_mb == 0 and vram_total_mb > 0:
        vram_free_mb = int(vram_total_mb * 0.75)

    vram_free_gb = vram_free_mb / 1024

    # ── Flash Attention ────────────────────────────────────────────
    if gpu_cc >= FLASH_ATTN_MIN_CC:
        params.flash_attn = "on"
    else:
        params.flash_attn = "off"
        params.warnings.append(
            f"GPU {gpu_name} (CC {gpu_cc}) does not support Flash Attention"
        )

    # ── Context Size ───────────────────────────────────────────────
    max_ctx = model.max_position_embeddings or 131072
    # Cap at reasonable default
    if vram_free_gb < 4:
        params.ctx_size = min(32000, max_ctx)
    elif vram_free_gb < 8:
        params.ctx_size = min(64000, max_ctx)
    elif vram_free_gb < 16:
        params.ctx_size = min(128000, max_ctx)
    else:
        params.ctx_size = min(128000, max_ctx)

    # ── KV Cache Type ──────────────────────────────────────────────
    # Use q4_0 for tight VRAM, f16 when there is headroom
    if vram_free_gb < 8:
        params.cache_type_k = "q4_0"
        params.cache_type_v = "q4_0"
    else:
        params.cache_type_k = "f16"
        params.cache_type_v = "f16"

    # ── KV Cache Size Estimate ─────────────────────────────────────
    kv_heads = model.num_key_value_heads or (model.num_attention_heads or 16)
    hd = model.head_dim or (
        model.hidden_size // (model.num_attention_heads or 16)
        if model.hidden_size
        else 128
    )
    cache_bytes_per_value = CACHE_BYTES.get(params.cache_type_k, 2.0)

    kv_cache_per_token_gb = (
        2 * model.num_hidden_layers * kv_heads * hd * cache_bytes_per_value / (1024**3)
    )
    kv_cache_gb = kv_cache_per_token_gb * params.ctx_size

    # ── Model Size Estimate ────────────────────────────────────────
    model_file_gb = estimate_model_file_size(model)

    # ── GPU Layers ─────────────────────────────────────────────────
    if model.is_moe:
        # MoE: with --cpu-moe, only dense + active experts need VRAM
        # Dense is roughly 15-20% of total for large MoE
        dense_ratio = 0.2 if model.num_experts > 8 else 0.5
        dense_gb = model_file_gb * dense_ratio
        active_expert_gb = (
            model_file_gb
            * (1 - dense_ratio)
            * (model.num_experts_per_tok / model.num_experts)
            if model.num_experts > 0
            else 0
        )
        moe_gpu_gb = dense_gb + active_expert_gb

        vram_needed_moe = moe_gpu_gb + kv_cache_gb + 0.5  # overhead

        if vram_needed_moe < vram_free_gb:
            # Full MoE offload possible
            params.n_gpu_layers = -1
            params.cpu_moe = False
        else:
            # Need CPU offloading
            params.cpu_moe = True
            params.load_mode = "mmap"

            # Calculate how many layers can fit
            # With --cpu-moe, each layer's dense part is ~dense_gb / layers
            layer_gb = (
                dense_gb / model.num_hidden_layers if model.num_hidden_layers else 0.5
            )
            available = vram_free_gb - kv_cache_gb - 0.5  # overhead
            if layer_gb > 0:
                max_layers = int(available / layer_gb)
                params.n_gpu_layers = min(max_layers, model.num_hidden_layers)
            else:
                params.n_gpu_layers = 20

            params.n_gpu_layers = max(0, params.n_gpu_layers)

            # Suggest --n-cpu-moe if there's room for some experts on GPU
            if vram_free_gb >= 8:
                params.n_cpu_moe = model.num_hidden_layers - 5
            else:
                params.n_cpu_moe = 0  # all experts on CPU

            params.warnings.append(
                f"MoE model: using --cpu-moe with {params.n_gpu_layers} GPU layers "
                f"(VRAM: {vram_free_gb:.1f} GB free, model needs ~{moe_gpu_gb:.1f} GB)"
            )
    else:
        # Dense model
        layer_gb = (
            model_file_gb / model.num_hidden_layers if model.num_hidden_layers else 0.5
        )
        available = vram_free_gb - kv_cache_gb - 0.5  # overhead
        if layer_gb > 0 and available > 0:
            max_layers = int(available / layer_gb)
            params.n_gpu_layers = min(max_layers, model.num_hidden_layers)
        else:
            params.n_gpu_layers = 0

        params.n_gpu_layers = max(0, params.n_gpu_layers)

        if params.n_gpu_layers >= model.num_hidden_layers:
            params.n_gpu_layers = -1  # all layers

    # ── Batch Size ─────────────────────────────────────────────────
    if vram_free_gb < 4:
        params.batch_size = 1024
        params.ubatch_size = 256
    elif vram_free_gb < 8:
        params.batch_size = 2048
        params.ubatch_size = 512
    else:
        params.batch_size = 2048
        params.ubatch_size = 512

    # ── Reasoning (Qwen models) ────────────────────────────────────
    if "qwen" in model.model_type.lower():
        params.reasoning = True

    return params


def estimate_model_file_size(info: ModelInfo) -> float:
    """Estimate model file size in GB from architecture metadata."""
    if info.gguf_files and info.gguf_files[0].get("size"):
        return info.gguf_files[0]["size"] / (1024**3)

    hs = info.hidden_size or 2048
    layers = info.num_hidden_layers or 40
    vocab = info.vocab_size or 248320
    intermediate = info.intermediate_size or hs * 4
    kv_heads = info.num_key_value_heads or (info.num_attention_heads or 16)
    num_heads = info.num_attention_heads or 16
    hd = info.head_dim or (hs // num_heads)

    # Embedding + LM head (2 bytes per param for f16 base)
    embedding_size = vocab * hs * 2
    lm_head_size = hs * vocab * 2

    # Per layer: attention (Q/K/V/O) + FFN (gate/up/down)
    # Q projection
    q_size = hs * num_heads * hd * 2
    # K projection (GQA)
    k_size = hs * kv_heads * hd * 2
    # V projection (GQA)
    v_size = hs * kv_heads * hd * 2
    # O projection
    o_size = num_heads * hd * hs * 2
    attn_size = q_size + k_size + v_size + o_size

    if info.is_moe and info.num_experts > 0:
        # MoE: shared expert + routed experts
        shared_expert = hs * intermediate * 3 * 2  # gate/up/down
        expert_per_expert = hs * intermediate * 3 * 2  # gate/up/down
        expert_total = expert_per_expert * info.num_experts
        ffn_size = shared_expert + expert_total
    else:
        # Dense: standard FFN
        ffn_size = hs * intermediate * 3 * 2  # gate/up/down

    per_layer = attn_size + ffn_size
    total_bytes = embedding_size + layers * per_layer + lm_head_size

    # Apply quantization factor (assume Q4_K_M ~0.5 bytes/param)
    # For GGUF, the file is already quantized, so we estimate at ~0.5 bytes/param
    total_gb = total_bytes / (1024**3) * 0.5
    return round(max(total_gb, 1.0), 1)


def format_cli_args(params: DerivedParams, model_path: str = "") -> list[str]:
    """Format derived parameters as a list of CLI arguments."""
    args: list[str] = []

    if model_path:
        args.extend(["--model", model_path])

    args.extend(["--ctx-size", str(params.ctx_size)])
    args.extend(["--flash-attn", params.flash_attn])

    if params.n_gpu_layers == -1:
        args.extend(["--n-gpu-layers", "-1"])
    elif params.n_gpu_layers > 0:
        args.extend(["--n-gpu-layers", str(params.n_gpu_layers)])

    if params.cpu_moe:
        args.append("--cpu-moe")
    if params.n_cpu_moe > 0:
        args.extend(["--n-cpu-moe", str(params.n_cpu_moe)])

    args.extend(["--load-mode", params.load_mode])
    args.extend(["--cache-type-k", params.cache_type_k])
    args.extend(["--cache-type-v", params.cache_type_v])
    args.extend(["--batch-size", str(params.batch_size)])
    args.extend(["--ubatch-size", str(params.ubatch_size)])
    args.extend(["--threads", str(params.threads)])
    args.extend(["--temp", str(params.temp)])
    args.extend(["--top-p", str(params.top_p)])
    args.extend(["--min-p", str(params.min_p)])

    if params.reasoning:
        args.append("--reasoning")

    return args


if __name__ == "__main__":
    main()
