#!/usr/bin/env bash
# WHY: This witness verifies build-mode selection without invoking a HIP compiler or device runtime.
set -euo pipefail
PATH=/usr/bin:/bin

SCRIPT_DIR=${BASH_SOURCE[0]%/*}
if [[ "$SCRIPT_DIR" == "${BASH_SOURCE[0]}" ]]; then
    SCRIPT_DIR=.
fi
ROOT=$(builtin cd -- "$SCRIPT_DIR/.." && builtin pwd -P)
RUNNER="$ROOT/scripts/gpu-denied-runner.sh"
OUT="$ROOT/target/hip-build-mode-witness"

{
    # WHY: `$1`, `$2`, and the derived paths intentionally expand in the
    # child `/bin/sh` inside the denied boundary, not in this wrapper.
    # shellcheck disable=SC2016
    "$RUNNER" -- /bin/sh -ceu '
        root=$1
        out=$2
        mkdir -p "$out"
        rustc "$root/crates/kernels/build.rs" -o "$out/kernels-build"

        mkdir -p "$out/cpu"
        LOGISMOS_HIP_BUILD=cpu-only OUT_DIR="$out/cpu" "$out/kernels-build" >"$out/cpu.log"
        grep -F "cargo:rustc-cfg=logismos_no_gpu_kernels" "$out/cpu.log"

        mkdir -p "$out/empty-cpu"
        (
            cd "$out/empty-cpu"
            LOGISMOS_HIP_BUILD=cpu-only OUT_DIR="$out/empty-cpu/out" "$out/kernels-build"
        ) >"$out/empty-cpu.log"
        grep -F "cargo:rustc-cfg=logismos_no_gpu_kernels" "$out/empty-cpu.log"
        if [[ -e "$out/empty-cpu/out/liblogismos_kernels.a" ]]; then
            echo "cpu-only empty source fixture produced a kernel archive" >&2
            exit 1
        fi

        mkdir -p "$out/empty-required"
        if (
            cd "$out/empty-required"
            LOGISMOS_HIP_BUILD=required HIPCC=/bin/true OUT_DIR="$out/empty-required/out" \
                "$out/kernels-build"
        ) >"$out/empty-required.log" 2>&1; then
            echo "required HIP mode accepted an empty HIP/CPP source tree" >&2
            exit 1
        fi
        grep -F "no HIP/CPP sources under src/ while LOGISMOS_HIP_BUILD=required" \
            "$out/empty-required.log"

        mkdir -p "$out/required"
        if LOGISMOS_HIP_BUILD=required HIPCC=/not-a-hipcc OUT_DIR="$out/required" \
            "$out/kernels-build" >"$out/required.log" 2>&1; then
            echo "required HIP mode accepted a missing compiler" >&2
            exit 1
        fi
        grep -F "hipcc not found on PATH while LOGISMOS_HIP_BUILD=required" "$out/required.log"

        mkdir -p "$out/default"
        if env -u LOGISMOS_HIP_BUILD HIPCC=/not-a-hipcc OUT_DIR="$out/default" \
            "$out/kernels-build" >"$out/default.log" 2>&1; then
            echo "unset HIP mode accepted a missing compiler" >&2
            exit 1
        fi
        grep -F "hipcc not found on PATH while LOGISMOS_HIP_BUILD=required" "$out/default.log"

        mkdir -p "$out/retired"
        if LOGISMOS_SKIP_HIP_BUILD=1 OUT_DIR="$out/retired" \
            "$out/kernels-build" >"$out/retired.log" 2>&1; then
            echo "retired HIP skip variable was accepted" >&2
            exit 1
        fi
        grep -F "LOGISMOS_SKIP_HIP_BUILD is retired" "$out/retired.log"
    ' /bin/sh "$ROOT" "$OUT" </dev/null
} 2>&1 | /usr/bin/cat

echo "HIP build-mode witness: PASS"
