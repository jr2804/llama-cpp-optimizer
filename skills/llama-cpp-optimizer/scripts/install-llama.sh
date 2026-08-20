#!/usr/bin/env bash
# Portable llama.cpp installer: download prebuilt binaries into ./bin (override: LLAMA_BIN).
# No system install, no build. Switch backend by re-running with a different one.
#
# Usage: install-llama.sh [VERSION] [BACKEND]
#   VERSION  release tag, default: latest  (e.g. b10520, b8448)
#   BACKEND  cpu cuda-12.4 cuda-13.3 vulkan rocm openvino  (default: cpu)
#            Omit BACKEND to only print available backends for VERSION.
set -euo pipefail

VER="${1:-latest}"
BACKEND="${2:-}"
DEST="${LLAMA_BIN:-$(pwd)/bin}"
REPO="ggml-org/llama.cpp"
ARCH="${LLAMA_ARCH:-$(uname -m)}"
case "${ARCH}" in
  x86_64|amd64)  ARCH=x64 ;;
  aarch64|arm64) ARCH=arm64 ;;
esac

case "$(uname -s)" in
  Linux*)  OS=linux ;;
  Darwin*) OS=macos ;;
  MINGW*|MSYS*|CYGWIN*) OS=win ;;
  *) echo "install-llama: unsupported OS" >&2; exit 2 ;;
esac

if command -v gh >/dev/null && [ -n "${GH_TOKEN:-}${GITHUB_TOKEN:-}" ]; then
  API() { gh api "$1"; }
else
  API() { curl -sfL "https://api.github.com$1"; }
fi

if [ "$VER" = latest ]; then
  VER="$(API /repos/$REPO/releases/latest | grep -o '"tag_name": *"[^"]*"' | cut -d'"' -f4)"
fi
ASSETS="$(API /repos/$REPO/releases/tags/$VER)"

if [ -z "${BACKEND:-}" ]; then
  echo "Available backends for $VER ($OS $ARCH):"
  echo "$ASSETS" | grep -o '"name": *"[^"]*"' | cut -d'"' -f4 \
    | grep -F "bin-$OS" | grep -- "$ARCH" | while read -r n; do
        b="${n#*bin-}"; b="${b#${OS}-}"; b="${b%-${ARCH}.[a-z]*}"
        echo "${b:-cpu}"
      done | sort -u
  exit 0
fi

# Asset filename patterns, first existing wins.
case "$OS/$BACKEND" in
  win/*)     CAND=("llama-$VER-bin-$OS-$BACKEND-$ARCH.zip") ;;
  linux/cpu) CAND=("llama-$VER-bin-ubuntu-$ARCH.tar.gz") ;;       # linux cpu drops token
  macos/*)   CAND=("llama-$VER-bin-macos-$ARCH.tar.gz"); BACKEND=cpu ;;  # macos: cpu-only
  *)         CAND=("llama-$VER-bin-$OS-$BACKEND-$ARCH.tar.gz") ;;
esac

URL=""
for p in "${CAND[@]}"; do
  if echo "$ASSETS" | grep -q "\"name\": *\"$p\""; then
    URL="$(echo "$ASSETS" | grep -o '"browser_download_url": *"[^"]*"' \
      | grep -F "/$p" | cut -d'"' -f4)"
    break
  fi
done
[ -n "$URL" ] || { echo "install-llama: backend '$BACKEND' not published for $VER ($OS $ARCH)" >&2; exit 1; }

TMP="$(mktemp -d)"
curl -sfL -o "$TMP/pkg" "$URL"
mkdir -p "$DEST"
find "$DEST" -mindepth 1 -delete 2>/dev/null || true
if [ "$OS" = win ]; then
  unzip -q "$TMP/pkg" -d "$DEST"
else
  tar -xzf "$TMP/pkg" -C "$DEST"
fi
# Replace the on-disk folder for this version (llama creates a stale empty bin/ on Windows besides the versioned dir)
# Flatten the single nested dir that llama zips produce, if present.
if [ "$(find "$DEST" -mindepth 1 -maxdepth 1 -type d | wc -l)" -eq 1 ] \
   && [ -z "$(find "$DEST" -mindepth 1 -maxdepth 1 -type f | head -1)" ]; then
  find "$DEST" -mindepth 1 -maxdepth 1 -type f -o -type d -exec mv -t "$DEST" {} + 2>/dev/null || true
fi
rm -rf "$TMP"
echo "llama.cpp $VER ($BACKEND) -> $DEST"
"$DEST/llama-cli" --version | head -1