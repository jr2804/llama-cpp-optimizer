# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Portable llama.cpp installer: download prebuilt binaries into ./bin (override: LLAMA_BIN env).

No system install, no build. Switch backend by re-running with a different one.

Usage:
    uv run scripts/install-llama.py [VERSION] [BACKEND]
      VERSION  release tag, default: latest  (e.g. b10520)
      BACKEND  cpu cuda-12.4 cuda-13.3 vulkan rocm-7.14 openvino-2026.3 sycl
               Omit BACKEND to print available backends for VERSION.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

REPO = "ggml-org/llama.cpp"
REPO_URL = f"https://api.github.com/repos/{REPO}"

# ---------------------------------------------------------------------------
# OS / arch helpers
# ---------------------------------------------------------------------------

def _os_tag() -> str:
    s = sys.platform
    if s == "win32":
        return "win"
    if s == "darwin":
        return "macos"
    return "ubuntu"  # linux


def _arch_tag() -> str:
    if os.environ.get("LLAMA_ARCH"):
        a = os.environ["LLAMA_ARCH"]
    else:
        a = platform.machine()
    return {"x86_64": "x64", "amd64": "x64", "AMD64": "x64", "aarch64": "arm64", "arm64": "arm64"}.get(a, a)


def _is_windows() -> bool:
    return sys.platform == "win32"


# ---------------------------------------------------------------------------
# GitHub helpers  (urllib + gh CLI fallback, no third-party deps)
# ---------------------------------------------------------------------------

def _gh_available() -> bool:
    return shutil.which("gh") is not None


def _api_get(path: str) -> dict | list:
    """GET a GitHub API path and return parsed JSON."""
    if _gh_available() and (os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")):
        r = subprocess.run(
            ["gh", "api", path],
            capture_output=True,
            text=True,
        )
        r.check_returncode()
        return json.loads(r.stdout)
    url = f"https://api.github.com{path}"
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def _download(url: str, dest: Path) -> None:
    urllib.request.urlretrieve(url, dest)  # noqa: S310


# ---------------------------------------------------------------------------
# Asset resolution
# ---------------------------------------------------------------------------

def _latest_tag() -> str:
    info = _api_get(f"/repos/{REPO}/releases/latest")
    return info["tag_name"]


def _release_assets(tag: str) -> list[dict]:
    return _api_get(f"/repos/{REPO}/releases/tags/{tag}")["assets"]


def _candidate_names(tag: str, os_tag: str, backend: str, arch: str) -> list[str]:
    """Return plausible asset filenames for this OS/backend/arch, best guess first."""
    if os_tag == "macos":
        # macos: cpu-only build, no backend token
        return [f"llama-{tag}-bin-macos-{arch}.tar.gz"]
    if os_tag == "ubuntu" and backend == "cpu":
        return [f"llama-{tag}-bin-ubuntu-{arch}.tar.gz"]
    # win or linux+gpu: backend is always part of the name
    ext = "zip" if os_tag == "win" else "tar.gz"
    return [f"llama-{tag}-bin-{os_tag}-{backend}-{arch}.{ext}"]


def _pick_asset(tag: str, os_tag: str, backend: str, arch: str) -> tuple[str, str]:
    """Return (backend_label, download_url) or exit with error."""
    assets = _release_assets(tag)
    names = {a["name"]: a["browser_download_url"] for a in assets}
    for name in _candidate_names(tag, os_tag, backend, arch):
        if name in names:
            return backend, names[name]

    # No match — show what IS available for this OS + arch so user can pick.
    available = sorted(
        n for n in names
        if f"bin-{os_tag}-" in n and arch in n
    )
    print(f"Backend '{backend}' not available for {tag} ({os_tag} {arch}).\n", file=sys.stderr)
    if available:
        print("Available assets:", file=sys.stderr)
        for n in available:
            print(f"  {n}", file=sys.stderr)
    sys.exit(1)


def _list_backends(tag: str, os_tag: str, arch: str) -> list[str]:
    """Derive human-friendly backend labels from published asset names."""
    assets = _release_assets(tag)
    backends: dict[str, bool] = {}
    for a in assets:
        name = a["name"]
        if f"bin-{os_tag}-" not in name or arch not in name:
            continue
        # strip  llama-<tag>-bin-<os>-  and  -<arch>.<ext>
        suffix = name.split(f"bin-{os_tag}-", 1)[-1]
        suffix = suffix.rsplit(f"-{arch}.", 1)[0]
        label = suffix if suffix else "cpu"  # bare = cpu
        if label not in backends:
            backends[label] = True
    return sorted(backends)


# ---------------------------------------------------------------------------
# CUDA runtime (Windows-only: not bundled in main asset)
# ---------------------------------------------------------------------------

def _is_cuda(backend: str) -> bool:
    return backend.startswith("cuda")


def _cuda_runtime_url(tag: str, backend: str, arch: str) -> str | None:
    """Return the download URL for the CUDA runtime asset, or None.

    On Windows the CUDA backend binaries ship without runtime DLLs
    (cublas, cudart, ...). They live in a separate `cudart-llama-*` asset.
    Linux/macOS static-link CUDA, so no extra download is needed.
    """
    if not (_is_windows() and _is_cuda(backend)):
        return None
    assets = _release_assets(tag)
    names = {a["name"]: a["browser_download_url"] for a in assets}
    candidate = f"cudart-llama-bin-win-{backend}-{arch}.zip"
    return names.get(candidate)


# ---------------------------------------------------------------------------
# Extract + flatten
# ---------------------------------------------------------------------------

def _extract(archive: Path, dest: Path) -> None:
    """Extract zip/tar.gz into dest."""
    if archive.suffix == ".zip" or (str(archive).endswith(".zip")):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
    elif str(archive).endswith(".tar.gz"):
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(dest)
    else:
        sys.exit(f"install-llama: unsupported archive format {archive.name}")


def _flatten(dest: Path) -> None:
    """If every top-level entry in dest is a single directory with no files, move its contents up."""
    entries = list(dest.iterdir())
    dirs = [e for e in entries if e.is_dir()]
    files = [e for e in entries if e.is_file()]
    if len(dirs) == 1 and not files:
        child = dirs[0]
        for item in child.iterdir():
            shutil.move(str(item), str(dest / item.name))
        child.rmdir()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os_tag = _os_tag()
    arch_tag = _arch_tag()
    default_backend = "cpu" if os_tag != "macos" else "cpu"

    version = sys.argv[1] if len(sys.argv) > 1 else "latest"
    backend = sys.argv[2] if len(sys.argv) > 2 else default_backend

    dest = Path(os.environ.get("LLAMA_BIN", str(Path.cwd() / "bin")))

    if version == "latest":
        version = _latest_tag()

    # No backend arg → list and exit.
    if len(sys.argv) <= 2:
        backends = _list_backends(version, os_tag, arch_tag)
        print(f"Available backends for {version} ({os_tag} {arch_tag}):")
        for b in backends:
            print(f"  {b}")
        return

    backend_label, url = _pick_asset(version, os_tag, backend, arch_tag)
    ext = "zip" if os_tag == "win" else "tar.gz"

    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / f"pkg.{ext}"
        print(f"Downloading llama.cpp {version} ({backend_label}) ...")
        _download(url, archive)

        # Wipe dest and re-extract.
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True, exist_ok=True)
        _extract(archive, dest)
        _flatten(dest)

        # Windows CUDA: download and extract the runtime DLLs alongside.
        cudart_url = _cuda_runtime_url(version, backend, arch_tag)
        if cudart_url:
            cudart_archive = Path(tmp) / "cudart.zip"
            print(f"Downloading CUDA runtime for {backend_label} ...")
            _download(cudart_url, cudart_archive)
            _extract(cudart_archive, dest)
            _flatten(dest)

    # Quick smoke: run version check if llama-cli exists.
    llama_cli = dest / ("llama-cli.exe" if _is_windows() else "llama-cli")
    if llama_cli.exists():
        r = subprocess.run([str(llama_cli), "--version"], capture_output=True, text=True)
        version_line = (r.stdout or r.stderr).splitlines()[0] if r.returncode == 0 else "(version unknown)"
        print(f"Installed llama.cpp {version} ({backend_label}) -> {dest}")
        print(f"  {version_line}")
    else:
        print(f"Installed llama.cpp {version} ({backend_label}) -> {dest}")


if __name__ == "__main__":
    main()
