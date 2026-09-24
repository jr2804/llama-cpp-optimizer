# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Portable llama.cpp installer: download prebuilt binaries into ./bin.

No system install, no build. Switch backend by re-running with a different one.
Works for upstream ggml-org/llama.cpp and for forks that ship specialized
quantizations/tensor types (e.g. PrismML-Eng/llama.cpp for ternary Q2_0).

Usage:
    uv run scripts/install-llama.py [VERSION] [BACKEND] [--repo OWNER/REPO] [--bin DIR]
      VERSION  release tag, default: latest  (e.g. b10520)
      BACKEND  cpu cuda-12.4 cuda-13.3 vulkan rocm-7.14 openvino-2026.3 sycl
               Omit BACKEND to print available backends for VERSION.
      --repo   GitHub repo to install from (default: ggml-org/llama.cpp).
               Env: LLAMA_REPO
      --bin    destination folder. Env: LLAMA_BIN.
               Default: ./bin for upstream; ./bin-<owner> for any fork, so a
               fork install never overwrites the upstream build.

Examples:
    # upstream, Vulkan
    uv run scripts/install-llama.py latest vulkan

    # Prism fork (ternary quants), own folder
    uv run scripts/install-llama.py --repo PrismML-Eng/llama.cpp latest vulkan
"""

from __future__ import annotations

import argparse
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

DEFAULT_REPO = "ggml-org/llama.cpp"
REPO = DEFAULT_REPO  # set from --repo/LLAMA_REPO in main(); read by API helpers


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="install-llama.py",
        description="Download prebuilt llama.cpp binaries into a portable folder.",
    )
    p.add_argument(
        "version", nargs="?", default="latest", help="release tag (default: latest)"
    )
    p.add_argument(
        "backend", nargs="?", help="cpu, cuda-12.4, vulkan, ... (omit to list available)"
    )
    p.add_argument(
        "--repo",
        default=os.environ.get("LLAMA_REPO", DEFAULT_REPO),
        help=f"GitHub repo OWNER/REPO (default: {DEFAULT_REPO})",
    )
    p.add_argument(
        "--bin",
        dest="bin_dir",
        default=None,
        help="destination folder (default: ./bin, or ./bin-<owner> for forks)",
    )
    return p.parse_args()


def _default_bin_dir(repo: str) -> Path:
    """`./bin` for upstream; `./bin-<owner>` for a fork, so builds never collide."""
    if repo == DEFAULT_REPO:
        return Path.cwd() / "bin"
    return Path.cwd() / f"bin-{repo.split('/', 1)[0].lower()}"


def main() -> None:
    global REPO

    args = _parse_args()
    REPO = args.repo

    os_tag = _os_tag()
    arch_tag = _arch_tag()
    backend = args.backend or "cpu"
    dest = Path(
        args.bin_dir or os.environ.get("LLAMA_BIN") or str(_default_bin_dir(REPO))
    )

    # No backend arg -> list and exit (no destination needed).
    if args.backend is None:
        tag = args.version
        if tag == "latest":
            tag = _latest_tag_with_binaries(os_tag, arch_tag)
        backends = _list_backends(tag, os_tag, arch_tag)
        print(f"Available backends for {REPO} {tag} ({os_tag} {arch_tag}):")
        for b in backends:
            print(f"  {b}")
        return

    # Never install into the skill itself: builds are ~100 MB each and belong in
    # the user's project. Catches `cd <skill-dir> && uv run scripts/...`.
    skill_dir = Path(__file__).resolve().parent.parent
    dest = dest.resolve()
    if dest == skill_dir or skill_dir in dest.parents:
        sys.exit(
            f"Refusing to install into the skill directory ({skill_dir}).\n"
            "Run this from your own project, or pass --bin <your-project>/bin"
        )

    version, url = _resolve(args.version, os_tag, backend, arch_tag)
    ext = "zip" if os_tag == "win" else "tar.gz"

    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / f"pkg.{ext}"
        print(f"Downloading {REPO} {version} ({backend}) -> {dest}")
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
            print(f"Downloading CUDA runtime for {backend} ...")
            _download(cudart_url, cudart_archive)
            _extract(cudart_archive, dest)
            _flatten(dest)

    # Quick smoke: run version check if llama-cli exists.
    llama_cli = dest / ("llama-cli.exe" if _is_windows() else "llama-cli")
    if llama_cli.exists():
        r = subprocess.run(
            [str(llama_cli), "--version"], capture_output=True, text=True, check=False
        )
        version_line = (
            (r.stdout or r.stderr).splitlines()[0]
            if r.returncode == 0
            else "(version unknown)"
        )
        print(f"Installed llama.cpp {version} ({backend}) -> {dest}")
        print(f"  {version_line}")
    else:
        print(f"Installed llama.cpp {version} ({backend}) -> {dest}")


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
    return {
        "x86_64": "x64",
        "amd64": "x64",
        "AMD64": "x64",
        "aarch64": "arm64",
        "arm64": "arm64",
    }.get(a, a)


def _download(url: str, dest: Path) -> None:
    urllib.request.urlretrieve(url, dest)


# ---------------------------------------------------------------------------
# Asset resolution
# ---------------------------------------------------------------------------


def _fetch_releases(limit: int = 30) -> list[dict]:
    """Newest releases first, prereleases included - that is where the binaries live."""
    return _api_get(f"/repos/{REPO}/releases?per_page={limit}")


def _is_build_asset(name: str) -> bool:
    """A real binary build, not a companion runtime or an SDK bundle.

    `cudart-*` archives carry only runtime DLLs and would otherwise match the
    same `bin-<os>-<backend>-<arch>` suffix as the build itself.
    """
    return "bin-" in name and not name.startswith("cudart-")


def _match_asset(
    assets: list[dict], os_tag: str, backend: str, arch: str
) -> tuple[str, str] | None:
    """Return (asset_name, download_url) for the first matching suffix, else None."""
    by_name = {
        a["name"]: a["browser_download_url"]
        for a in assets
        if _is_build_asset(a["name"])
    }
    for suffix in _asset_suffixes(os_tag, backend, arch):
        name = next((n for n in by_name if n.endswith(suffix)), None)
        if name:
            return name, by_name[name]
    return None


def _latest_tag_with_binaries(os_tag: str, arch: str) -> str:
    """Newest release that actually ships a build for this OS/arch.

    Upstream tags stable milestones (e.g. v0.5.0) that carry no assets; the
    builds live in prerelease `bXXXXX` tags. A release can also be partially
    uploaded, so scan until one has a build for this platform.
    """
    for rel in _fetch_releases():
        for a in rel.get("assets", []):
            name = a["name"]
            if (
                _is_build_asset(name)
                and _asset_matches_os(name, os_tag)
                and f"-{arch}" in name
            ):
                return rel["tag_name"]
    sys.exit(f"No release in {REPO} ships builds for {os_tag} {arch}.")


def _fail_no_asset(
    version: str, os_tag: str, backend: str, arch: str, assets: list[dict]
) -> None:
    """Print the assets that DO exist for this OS/arch, then exit."""
    available = sorted(
        a["name"]
        for a in assets
        if _is_build_asset(a["name"])
        and _asset_matches_os(a["name"], os_tag)
        and arch in a["name"]
    )
    print(
        f"Backend '{backend}' not available for {REPO} {version} ({os_tag} {arch}).\n",
        file=sys.stderr,
    )
    if available:
        print("Available assets:", file=sys.stderr)
        for n in available:
            print(f"  {n}", file=sys.stderr)
    sys.exit(1)


def _resolve(version: str, os_tag: str, backend: str, arch: str) -> tuple[str, str]:
    """Resolve (tag, download_url) for the requested build.

    `latest` scans newest releases for one that carries the wanted asset, which
    also handles the fork case where the stable tag differs from the binary tag.
    """
    if version != "latest":
        assets = _release_assets(version)
        hit = _match_asset(assets, os_tag, backend, arch)
        if hit:
            return version, hit[1]
        _fail_no_asset(version, os_tag, backend, arch, assets)

    for rel in _fetch_releases():
        hit = _match_asset(rel.get("assets", []), os_tag, backend, arch)
        if hit:
            return rel["tag_name"], hit[1]
    sys.exit(f"No release in {REPO} publishes a '{backend}' build for {os_tag} {arch}.")


def _os_tokens(os_tag: str) -> tuple[str, ...]:
    """OS tokens used in asset names. Linux appears as both `ubuntu` and `linux`."""
    if os_tag == "ubuntu":
        return ("ubuntu", "linux")
    return (os_tag,)


def _asset_matches_os(name: str, os_tag: str) -> bool:
    return any(f"bin-{t}-" in name for t in _os_tokens(os_tag))


def _asset_suffixes(os_tag: str, backend: str, arch: str) -> list[str]:
    """Asset filename suffixes to try, best guess first.

    Matched with str.endswith, so a fork's version prefix (e.g. `prism-b10735-`)
    is irrelevant. Linux builds are published under both `ubuntu-` and `linux-`.
    """
    if os_tag == "macos":
        return [f"bin-macos-{arch}.tar.gz"]
    if os_tag == "win":
        return [f"bin-win-{backend}-{arch}.zip"]
    if backend == "cpu":
        return [f"bin-ubuntu-{arch}.tar.gz", f"bin-linux-cpu-{arch}.tar.gz"]
    return [
        f"bin-ubuntu-{backend}-{arch}.tar.gz",
        f"bin-linux-{backend}-{arch}.tar.gz",
    ]


def _list_backends(tag: str, os_tag: str, arch: str) -> list[str]:
    """Derive human-friendly backend labels from published asset names."""
    backends = set()
    for a in _release_assets(tag):
        name = a["name"]
        if (
            not _is_build_asset(name)
            or not _asset_matches_os(name, os_tag)
            or f"-{arch}." not in name
        ):
            continue
        stem = name.rsplit(f"-{arch}.", 1)[0]  # drop -<arch>.<ext>
        for token in _os_tokens(os_tag):
            marker = f"bin-{token}-"
            if marker in stem:
                backends.add(stem.split(marker, 1)[1] or "cpu")  # bare = cpu
                break
    return sorted(backends)


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


def _is_windows() -> bool:
    return sys.platform == "win32"


def _release_assets(tag: str) -> list[dict]:
    return _api_get(f"/repos/{REPO}/releases/tags/{tag}")["assets"]


def _api_get(path: str) -> dict | list:
    """GET a GitHub API path and return parsed JSON."""
    if _gh_available() and (
        os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    ):
        r = subprocess.run(
            ["gh", "api", path],
            capture_output=True,
            text=True,
            check=False,
        )
        r.check_returncode()
        return json.loads(r.stdout)
    url = f"https://api.github.com{path}"
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


# ---------------------------------------------------------------------------
# GitHub helpers  (urllib + gh CLI fallback, no third-party deps)
# ---------------------------------------------------------------------------


def _gh_available() -> bool:
    return shutil.which("gh") is not None


# ---------------------------------------------------------------------------
# CUDA runtime (Windows-only: not bundled in main asset)
# ---------------------------------------------------------------------------


def _is_cuda(backend: str) -> bool:
    return backend.startswith("cuda")


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


if __name__ == "__main__":
    main()
