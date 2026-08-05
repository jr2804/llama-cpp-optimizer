# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx"]
# ///

"""Fetch model metadata from Hugging Face and output as JSON.

Usage:
    uv run model-info.py Qwen/Qwen3.6-35B-A3B
    uv run model-info.py https://huggingface.co/Qwen/Qwen3.6-35B-A3B
    uv run model-info.py ggml-org/Qwen3.6-35B-A3B-GGUF --list-files

Outputs structured JSON with model architecture metadata.
"""

import json
import re
import sys
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx  # type: ignore


@dataclass
class ModelInfo:
    """Model architecture metadata extracted from Hugging Face."""

    model_id: str = ""
    model_type: str = ""
    architecture: str = ""
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
    rope_theta: float = 0.0
    partial_rotary_factor: float = 0.0
    gguf_files: list[dict[str, Any]] = field(default_factory=list)


def main() -> None:
    """Main entry point."""
    if len(sys.argv) < 2:
        print(
            "Usage: uv run model-info.py <model_id_or_url> [--list-files]",
            file=sys.stderr,
        )
        sys.exit(1)

    input_str = sys.argv[1]
    list_files = "--list-files" in sys.argv

    try:
        model_id = parse_model_id(input_str)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    with httpx.Client() as client:
        info = extract_model_info(model_id, client)

        if list_files:
            info.gguf_files = list_gguf_files(model_id, client)

        print(json.dumps(asdict(info), indent=2, default=str))


def parse_model_id(input_str: str) -> str:
    """Parse a model ID from a URL or short form."""
    # Full URL: https://huggingface.co/org/model
    match = re.search(r"huggingface\.co/([^/\s]+/[^/\s]+)", input_str)
    if match:
        return match.group(1)
    # Short form: org/model
    if "/" in input_str and not input_str.startswith("http"):
        return input_str
    raise ValueError(f"Could not parse model ID from: {input_str}")


def extract_model_info(model_id: str, client: httpx.Client) -> ModelInfo:
    """Fetch and parse model metadata from Hugging Face."""
    info = ModelInfo(model_id=model_id)

    # Fetch config.json
    config_url = f"https://huggingface.co/{model_id}/raw/main/config.json"
    config = fetch_json(config_url, client)
    if config:
        tc = extract_text_config(config)
        info.model_type = tc.get("model_type", "")
        info.architecture = str(config.get("architectures", [""])[0])
        info.num_hidden_layers = tc.get("num_hidden_layers", 0)
        info.num_attention_heads = tc.get("num_attention_heads", 0)
        info.num_key_value_heads = tc.get("num_key_value_heads", 0)
        info.hidden_size = tc.get("hidden_size", 0)
        info.intermediate_size = (
            tc.get("intermediate_size", 0)
            or tc.get("moe_intermediate_size", 0)
            or tc.get("shared_expert_intermediate_size", 0)
        )
        info.head_dim = tc.get("head_dim", 0)
        info.max_position_embeddings = tc.get("max_position_embeddings", 0)
        info.vocab_size = tc.get("vocab_size", 0)
        info.num_experts = tc.get("num_experts", 0)
        info.num_experts_per_tok = tc.get("num_experts_per_tok", 0)

        # Handle nested rope_parameters (Qwen3.6 style)
        rope_params = tc.get("rope_parameters", {})
        if rope_params:
            info.rope_theta = float(rope_params.get("rope_theta", 0))
        else:
            info.rope_theta = float(tc.get("rope_theta", 0))
        info.partial_rotary_factor = float(
            tc.get("partial_rotary_factor", 0)
            or rope_params.get("partial_rotary_factor", 0)
        )

        # Detect MoE
        mt = info.model_type.lower()
        if any(k in mt for k in ("moe", "mixtral", "dbrx")):
            info.is_moe = True
        if info.num_experts > 0:
            info.is_moe = True

    return info


def extract_text_config(config: dict[str, Any]) -> dict[str, Any]:
    """Extract the text config (handles vision-language models with nested text_config)."""
    tc = config.get("text_config", config)
    if isinstance(tc, str):
        return config
    return tc


def list_gguf_files(model_id: str, client: httpx.Client) -> list[dict[str, Any]]:
    """List GGUF files in a model repository."""
    api_url = f"https://huggingface.co/api/models/{model_id}"
    data = fetch_json(api_url, client)
    if not data:
        return []

    files: list[dict[str, Any]] = []
    for sibling in data.get("siblings", []):
        rfilename: str = sibling.get("rfilename", "")
        if rfilename.endswith(".gguf"):
            entry: dict[str, Any] = {"filename": rfilename}
            lfs = sibling.get("lfs", {})
            if lfs:
                entry["size"] = lfs.get("size", 0)
                entry["sha256"] = lfs.get("sha256", "")
            files.append(entry)
    return files


def fetch_json(url: str, client: httpx.Client) -> dict[str, Any]:
    """Fetch JSON from a URL; raises on failure."""
    resp = client.get(url, timeout=15, follow_redirects=True)
    resp.raise_for_status()
    return resp.json()


if __name__ == "__main__":
    main()
