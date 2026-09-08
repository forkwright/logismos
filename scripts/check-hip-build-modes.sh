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
        cd "$root"
        env -u CARGO_FEATURE_GPU -u LOGISMOS_HIP_BUILD -u LOGISMOS_SKIP_HIP_BUILD \
            HIPCC=/not-a-hipcc \
            cargo build --offline --locked --no-default-features -p kernels --target-dir "$out/cargo"
        kernels_build=$(/usr/bin/bash "$root/scripts/find-kernels-build-script.sh" "$out/cargo")

        mkdir -p "$out/cpu"
        CARGO_FEATURE_GPU=1 LOGISMOS_HIP_BUILD=cpu-only OUT_DIR="$out/cpu" "$kernels_build" >"$out/cpu.log"
        grep -F "cargo:rustc-cfg=logismos_no_gpu_kernels" "$out/cpu.log"

        mkdir -p "$out/empty-cpu"
        (
            cd "$out/empty-cpu"
            CARGO_FEATURE_GPU=1 LOGISMOS_HIP_BUILD=cpu-only OUT_DIR="$out/empty-cpu/out" "$kernels_build"
        ) >"$out/empty-cpu.log"
        grep -F "cargo:rustc-cfg=logismos_no_gpu_kernels" "$out/empty-cpu.log"
        if [ -e "$out/empty-cpu/out/liblogismos_kernels.a" ]; then
            echo "cpu-only empty source fixture produced a kernel archive" >&2
            exit 1
        fi

        mkdir -p "$out/empty-required"
        if (
            cd "$out/empty-required"
            CARGO_FEATURE_GPU=1 LOGISMOS_HIP_BUILD=required HIPCC=/bin/true OUT_DIR="$out/empty-required/out" \
                "$kernels_build"
        ) >"$out/empty-required.log" 2>&1; then
            echo "required HIP mode accepted an empty HIP/CPP source tree" >&2
            exit 1
        fi
        grep -F "no HIP/CPP sources under src/ while LOGISMOS_HIP_BUILD=required" \
            "$out/empty-required.log"

        mkdir -p "$out/required"
        if CARGO_FEATURE_GPU=1 LOGISMOS_HIP_BUILD=required HIPCC=/not-a-hipcc OUT_DIR="$out/required" \
            "$kernels_build" >"$out/required.log" 2>&1; then
            echo "required HIP mode accepted a missing compiler" >&2
            exit 1
        fi
        grep -F "hipcc not found on PATH while LOGISMOS_HIP_BUILD=required" "$out/required.log"

        mkdir -p "$out/default"
        if env -u LOGISMOS_HIP_BUILD CARGO_FEATURE_GPU=1 HIPCC=/not-a-hipcc OUT_DIR="$out/default" \
            "$kernels_build" >"$out/default.log" 2>&1; then
            echo "unset HIP mode accepted a missing compiler" >&2
            exit 1
        fi
        grep -F "hipcc not found on PATH while LOGISMOS_HIP_BUILD=required" "$out/default.log"

        mkdir -p "$out/retired"
        if CARGO_FEATURE_GPU=1 LOGISMOS_SKIP_HIP_BUILD=1 OUT_DIR="$out/retired" \
            "$kernels_build" >"$out/retired.log" 2>&1; then
            echo "retired HIP skip variable was accepted" >&2
            exit 1
        fi
        grep -F "LOGISMOS_SKIP_HIP_BUILD is retired" "$out/retired.log"

        mkdir -p "$out/no-gpu-cpu"
        if env -u CARGO_FEATURE_GPU LOGISMOS_HIP_BUILD=cpu-only OUT_DIR="$out/no-gpu-cpu" \
            "$kernels_build" >"$out/no-gpu-cpu.log" 2>&1; then
            if grep -Fq "cargo:rustc-cfg=logismos_no_gpu_kernels" "$out/no-gpu-cpu.log"; then
                echo "GPU-disabled build emitted a GPU launcher cfg" >&2
                exit 1
            fi
        else
            echo "GPU-disabled cpu-only build was rejected" >&2
            exit 1
        fi

        mkdir -p "$out/no-gpu-default"
        env -u CARGO_FEATURE_GPU -u LOGISMOS_HIP_BUILD HIPCC=/not-a-hipcc \
            OUT_DIR="$out/no-gpu-default" "$kernels_build" >"$out/no-gpu-default.log" 2>&1
        if grep -Eq "cargo:rustc-(cfg=logismos_no_gpu_kernels|link-lib=)" "$out/no-gpu-default.log"; then
            echo "GPU-disabled default build emitted GPU compilation or linkage" >&2
            exit 1
        fi

        mkdir -p "$out/no-gpu-required"
        if env -u CARGO_FEATURE_GPU LOGISMOS_HIP_BUILD=required OUT_DIR="$out/no-gpu-required" \
            "$kernels_build" >"$out/no-gpu-required.log" 2>&1; then
            echo "GPU-disabled build accepted an explicit required HIP request" >&2
            exit 1
        fi
        grep -F "kernels/gpu is disabled but LOGISMOS_HIP_BUILD=required explicitly requests HIP compilation" \
            "$out/no-gpu-required.log"

        mkdir -p "$out/no-gpu-invalid"
        if env -u CARGO_FEATURE_GPU LOGISMOS_HIP_BUILD=unsupported OUT_DIR="$out/no-gpu-invalid" \
            "$kernels_build" >"$out/no-gpu-invalid.log" 2>&1; then
            echo "GPU-disabled build accepted an unsupported HIP mode" >&2
            exit 1
        fi
        grep -F "invalid LOGISMOS_HIP_BUILD=" "$out/no-gpu-invalid.log"

        mkdir -p "$out/no-gpu-retired"
        if env -u CARGO_FEATURE_GPU LOGISMOS_SKIP_HIP_BUILD=1 OUT_DIR="$out/no-gpu-retired" \
            "$kernels_build" >"$out/no-gpu-retired.log" 2>&1; then
            echo "GPU-disabled build accepted the retired HIP skip variable" >&2
            exit 1
        fi
        grep -F "LOGISMOS_SKIP_HIP_BUILD is retired" "$out/no-gpu-retired.log"

        # WHY: Build-script fixtures alone cannot detect a consumer re-enabling
        # the GPU feature through Cargo feature unification.
        cd "$root"
        cargo tree --offline --locked -p kernels --no-default-features --features gpu \
            --edges normal,build --prefix none --format "{p}" >"$out/gpu-dependencies.log"
        if ! grep -Eq "^(hipcore|taxis) " "$out/gpu-dependencies.log"; then
            echo "GPU dependency witness failed to observe its forbidden case" >&2
            exit 1
        fi
        # WHY: One package selection owns both graph and compiler witnesses.
        set -- -p kernels -p transformers -p decoders -p text -p decode -p embed -p rerank -p templates
        cargo tree --offline --locked --no-default-features "$@" \
            --edges normal,build --prefix none --format "{p}" >"$out/cpu-dependencies.log"
        if grep -Eq "^(hipcore|taxis) " "$out/cpu-dependencies.log"; then
            echo "CPU consumers acquired a GPU runtime dependency" >&2
            exit 1
        fi
        env -u LOGISMOS_HIP_BUILD HIPCC=/not-a-hipcc \
            cargo check --offline --locked --no-default-features "$@" --lib --jobs 4
        # WHY: Workspace feature unification can hide broken CPU-only test paths;
        # exercise the same minimal consumers in both numerical build profiles.
        env -u LOGISMOS_HIP_BUILD HIPCC=/not-a-hipcc \
            cargo test --offline --locked --no-default-features "$@" --lib --jobs 4
        env -u LOGISMOS_HIP_BUILD HIPCC=/not-a-hipcc \
            cargo test --offline --locked --release --no-default-features "$@" --lib --jobs 4
        # WHY: `--lib` deliberately omits the CPU-only CLI binary and its
        # end-to-end fixture. Keep that proof distinct from the shared library
        # package selection above instead of widening it with an ineffective
        # target flag.
        env -u LOGISMOS_HIP_BUILD HIPCC=/not-a-hipcc \
            cargo check --offline --locked --no-default-features -p bin --bin logismos --jobs 4
        env -u LOGISMOS_HIP_BUILD HIPCC=/not-a-hipcc \
            cargo test --offline --locked --no-default-features -p bin --test prepare_text_cli --jobs 4
    ' /bin/sh "$ROOT" "$OUT" </dev/null
} 2>&1 | /usr/bin/cat

echo "HIP build-mode witness: PASS"
