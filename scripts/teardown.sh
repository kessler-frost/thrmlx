#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "$script_dir/.." && pwd -P)"

if [[ "$(basename -- "$repo_root")" != "thrmlx" || ! -f "$repo_root/pyproject.toml" ]]; then
  echo "Refusing to clean an unexpected directory: $repo_root" >&2
  exit 1
fi

targets=(
  "$repo_root/.venv"
  "$repo_root/.pytest_cache"
  "$repo_root/.ruff_cache"
  "$repo_root/.ty"
  "$repo_root/build"
  "$repo_root/dist"
  "$repo_root/src/thrmlx.egg-info"
)

for target in "${targets[@]}"; do
  rm -rf -- "$target"
done

find "$repo_root" -type d -name __pycache__ -prune -exec rm -rf -- {} +
echo "Removed thrmlx's project-local environment, build artifacts, and tool caches."
