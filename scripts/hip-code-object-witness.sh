#!/usr/bin/env bash
# Enter the GPU-denied boundary for HIP code-object compilation and inspection.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
RUNNER="$ROOT/scripts/gpu-denied-runner.sh"

if [[ "$#" -ne 0 ]]; then
    echo "usage: scripts/hip-code-object-witness.sh" >&2
    exit 64
fi

exec "$RUNNER" -- /usr/bin/bash "$ROOT/scripts/hip-code-object-witness-inner.sh"
