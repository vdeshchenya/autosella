#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  cat <<'EOF'
Build the disposable autoresearch runtime image on the chosen Docker host.
Required explicit environment variables (no model or tool-version defaults):
  BASE_IMAGE       Node Debian image pinned with @sha256:<64 hex characters>
  BUN_VERSION      Exact Bun version, e.g. 1.2.15
  RALPH_COMMIT     Approved full 40-character open-ralph-wiggum commit
  CODEX_VERSION    Exact @openai/codex npm version
  CLAUDE_VERSION   Exact @anthropic-ai/claude-code npm version
  IMAGE_TAG        Local name for the resulting image, e.g. ar-runtime:reviewed
The script builds only the runtime, then prints its immutable sha256 image ID.
It does not install host agent CLIs, import host homes, or launch a research run.
EOF
  exit 0
fi
: "${BASE_IMAGE:?Set a reviewed Node Debian base image pinned by digest}"
: "${BUN_VERSION:?Set an exact Bun version}"
: "${RALPH_COMMIT:?Set the approved full Ralph commit}"
: "${CODEX_VERSION:?Set an exact Codex CLI version}"
: "${CLAUDE_VERSION:?Set an exact Claude Code CLI version}"
: "${IMAGE_TAG:?Set the local runtime image tag}"
[[ "$BASE_IMAGE" =~ @sha256:[a-f0-9]{64}$ ]] || { echo 'BASE_IMAGE must be pinned by digest' >&2; exit 2; }
[[ "$RALPH_COMMIT" =~ ^[a-f0-9]{40}$ ]] || { echo 'RALPH_COMMIT must be a full commit ID' >&2; exit 2; }
for version in "$BUN_VERSION" "$CODEX_VERSION" "$CLAUDE_VERSION"; do
  [[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+([.-][A-Za-z0-9.-]+)?$ ]] || { echo 'Tool versions must be exact' >&2; exit 2; }
done
# Only explicitly selected runtime code is sent to the Docker daemon.
build_context="$(mktemp -d "${TMPDIR:-/tmp}/autoresearch-image.XXXXXX")"
trap 'rm -rf "$build_context"' EXIT
mkdir -p "$build_context/isolated_runs" "$build_context/evaluation_access"
cp "$ROOT/isolated_runs/Dockerfile" "$build_context/Dockerfile"
for module in __init__.py network.py patch_ralph.py runner.py repository_access_policy.md; do
  cp "$ROOT/isolated_runs/$module" "$build_context/isolated_runs/$module"
done
for module in __init__.py client.py; do
  cp "$ROOT/evaluation_access/$module" "$build_context/evaluation_access/$module"
done
docker build --tag "$IMAGE_TAG" \
  --build-arg "BASE_IMAGE=$BASE_IMAGE" --build-arg "BUN_VERSION=$BUN_VERSION" \
  --build-arg "RALPH_COMMIT=$RALPH_COMMIT" --build-arg "CODEX_VERSION=$CODEX_VERSION" \
  --build-arg "CLAUDE_VERSION=$CLAUDE_VERSION" "$build_context"
docker image inspect --format '{{.Id}}' "$IMAGE_TAG"
