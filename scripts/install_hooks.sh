#!/bin/sh
# Run from the repository root after the existing UV dev environment is ready.
set -eu
export CUDA_VISIBLE_DEVICES=-1
export PYTHONDONTWRITEBYTECODE=1
root=$(git rev-parse --show-toplevel)
cd "$root"
if [ -n "$(git config --get core.hooksPath || true)" ]; then
    printf '%s\n' 'Explicit core.hooksPath is configured; refusing to replace another hook.' >&2
    exit 1
fi
hook=$(git rev-parse --git-path hooks/pre-commit)
# No dependency installation and no alternate/bypassed hook path.
uv run --offline --no-sync python -B -m pre_commit validate-config
cp .githooks/pre-commit "$hook"
chmod +x "$hook"
printf '%s\n' 'Installed UV-only check-mode pre-commit hook.'
