#!/usr/bin/env bash
# Verify kernel build modes without invoking a HIP compiler or device runtime.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
RUNNER="$ROOT/scripts/gpu-denied-runner.sh"
OUT="$ROOT/target/hip-build-mode-witness"

mkdir -p "$OUT"

"$RUNNER" -- /bin/sh -ceu '
    root=$1
    out=$2
    rustc "$root/crates/kernels/build.rs" -o "$out/kernels-build"

    mkdir -p "$out/cpu"
    LOGISMOS_HIP_BUILD=cpu-only OUT_DIR="$out/cpu" "$out/kernels-build" >"$out/cpu.log"
    grep -F "cargo:rustc-cfg=logismos_no_gpu_kernels" "$out/cpu.log"

    mkdir -p "$out/required"
    if LOGISMOS_HIP_BUILD=required HIPCC=/not-a-hipcc OUT_DIR="$out/required" "$out/kernels-build" >"$out/required.log" 2>&1; then
        echo "required HIP mode accepted a missing compiler" >&2
        exit 1
    fi
    grep -F "hipcc not found on PATH while LOGISMOS_HIP_BUILD=required" "$out/required.log"

    mkdir -p "$out/default"
    if env -u LOGISMOS_HIP_BUILD HIPCC=/not-a-hipcc OUT_DIR="$out/default" "$out/kernels-build" >"$out/default.log" 2>&1; then
        echo "unset HIP mode accepted a missing compiler" >&2
        exit 1
    fi
    grep -F "hipcc not found on PATH while LOGISMOS_HIP_BUILD=required" "$out/default.log"

    mkdir -p "$out/retired"
    if LOGISMOS_SKIP_HIP_BUILD=1 OUT_DIR="$out/retired" "$out/kernels-build" >"$out/retired.log" 2>&1; then
        echo "retired HIP skip variable was accepted" >&2
        exit 1
    fi
    grep -F "LOGISMOS_SKIP_HIP_BUILD is retired" "$out/retired.log"
' /bin/sh "$ROOT" "$OUT"

echo "HIP build-mode witness: PASS"
