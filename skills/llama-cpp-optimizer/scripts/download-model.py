# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Download a GGUF from Hugging Face into models/<owner>/, named as published.

Keeps the provider (the HF repo owner) visible in the path, so a file is still
identifiable after download instead of being renamed to something generic.
Resumes partial downloads and verifies SHA256 against the repo's LFS metadata.

Usage:
    uv run download-model.py <owner/repo>                  # list the .gguf files
    uv run download-model.py <owner/repo> --file X.gguf    # download it
    uv run download-model.py <owner/repo> --file X.gguf --dir models
    uv run download-model.py <owner/repo> --file X.gguf --force
    uv run download-model.py <owner/repo> --revision <sha>

Resulting layout:
    models/ornith-ai/Ornith-1.5-9B-Q4_K_M.gguf
    models/bartowski/Ornith-1.5-9B-Q4_K_M.gguf
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

API = "https://huggingface.co/api/models"
RESOLVE = "https://huggingface.co"
CHUNK = 8 * 1024 * 1024


def _parse_repo(arg: str) -> tuple[str, str]:
    """Accept `owner/repo` or a huggingface.co URL; return (owner, repo)."""
    if "huggingface.co" in arg:
        arg = arg.split("huggingface.co/", 1)[1]
    parts = arg.strip("/").split("/")
    if len(parts) < 2 or not parts[0] or not parts[1]:
        sys.exit(f"Expected <owner>/<repo>, got '{arg}'")
    return parts[0], parts[1]


def _list_gguf(owner: str, repo: str, revision: str) -> list[dict]:
    """Return [{'path', 'size', 'sha256'}] for the *.gguf files in the repo."""
    url = f"{API}/{owner}/{repo}/tree/{revision}?recursive=true"
    with urllib.request.urlopen(url) as r:
        entries = json.loads(r.read())
    out = []
    for e in entries:
        if e.get("type") != "file" or not e["path"].endswith(".gguf"):
            continue
        lfs = e.get("lfs") or {}
        out.append({"path": e["path"], "size": e.get("size") or 0, "sha256": lfs.get("oid")})
    return sorted(out, key=lambda x: x["path"])


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def _download(url: str, dest: Path, total: int, force: bool) -> None:
    have = 0 if force else (dest.stat().st_size if dest.exists() else 0)

    if total and have >= total:
        print(f"already downloaded: {dest}")
        return

    headers = {"Range": f"bytes={have}-"} if have else {}
    req = urllib.request.Request(url, headers=headers)
    dest.parent.mkdir(parents=True, exist_ok=True)

    with urllib.request.urlopen(req) as r:
        # A server that ignores Range answers 200, so append would corrupt the file.
        resume = have > 0 and r.status == 206
        if have and not resume:
            print("server ignored Range; restarting download")
            have = 0
        elif resume:
            print(f"resuming at {have / 1e9:.2f} GB")

        mode = "ab" if resume else "wb"
        written = have
        step = max(CHUNK, total // 50) if total else CHUNK
        next_report = have + step

        with dest.open(mode) as f:
            while chunk := r.read(CHUNK):
                f.write(chunk)
                written += len(chunk)
                if written >= next_report:
                    next_report = written + step
                    pct = f" ({written * 100 / total:.0f}%)" if total else ""
                    print(f"  {written / 1e9:.2f} GB{pct}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Download a GGUF into models/<owner>/.")
    p.add_argument("repo", help="<owner>/<repo> or a huggingface.co URL")
    p.add_argument("--file", help="GGUF filename; omit to list what is available")
    p.add_argument("--dir", default="models", help="base directory (default: models)")
    p.add_argument("--revision", default="main", help="branch, tag, or commit (default: main)")
    p.add_argument("--force", action="store_true", help="re-download even if complete")
    args = p.parse_args()

    owner, repo = _parse_repo(args.repo)
    files = _list_gguf(owner, repo, args.revision)
    if not files:
        sys.exit(f"No .gguf files in {owner}/{repo}@{args.revision}")

    if not args.file:
        print(f"{owner}/{repo}@{args.revision} - {len(files)} GGUF file(s):")
        for f in files:
            print(f"  {f['path']}  ({f['size'] / 1e9:.2f} GB)")
        print(f"\nDownload with: --file <name>   ->  {args.dir}/{owner}/<name>")
        return

    entry = next((f for f in files if f["path"] == args.file or Path(f["path"]).name == args.file), None)
    if not entry:
        sys.exit(f"'{args.file}' not in {owner}/{repo}; run without --file to list")

    dest = Path(args.dir) / owner / Path(entry["path"]).name
    url = f"{RESOLVE}/{owner}/{repo}/resolve/{args.revision}/{entry['path']}"
    print(f"{owner}/{repo}@{args.revision}: {entry['path']} -> {dest}")
    _download(url, dest, entry["size"], args.force)

    if entry["sha256"]:
        actual = _sha256(dest)
        if actual != entry["sha256"]:
            sys.exit(f"SHA256 mismatch\n  expected {entry['sha256']}\n  actual   {actual}")
        print(f"sha256 ok: {actual[:16]}...")
    else:
        print("note: file is not LFS-stored; no SHA256 to verify against")

    print(f"ready: {dest}")


if __name__ == "__main__":
    main()
