#!/usr/bin/env bash
# Compile the in-tree HIP kernels and inspect their embedded AMDGPU code objects.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
RUNNER="$ROOT/scripts/gpu-denied-runner.sh"

fail() {
    echo "HIP code-object witness: $*" >&2
    exit 1
}

if [[ "${HIP_CODE_OBJECT_WITNESS_INNER:-}" != "1" ]]; then
    command=("$RUNNER" -- env HIP_CODE_OBJECT_WITNESS_INNER=1)
    for variable in HIPCC HIP_PATH BINDGEN_EXTRA_CLANG_ARGS CARGO_BUILD_JOBS; do
        if [[ -v "$variable" ]]; then
            command+=("$variable=${!variable}")
        fi
    done
    command+=("$0")
    exec "${command[@]}"
fi

if [[ "$#" -ne 0 ]]; then
    fail "usage: scripts/hip-code-object-witness.sh"
fi

target=$(<"$ROOT/contracts/gpu-target.txt")
if [[ ! "$target" =~ ^gfx[0-9]+$ ]]; then
    fail "contracts/gpu-target.txt must contain one gfx target"
fi

if [[ -n "${HIPCC:-}" ]]; then
    hipcc=$HIPCC
elif [[ -x /opt/rocm/bin/hipcc ]]; then
    hipcc=/opt/rocm/bin/hipcc
else
    hipcc=hipcc
fi
if ! hipcc=$(command -v "$hipcc"); then
    fail "required HIP compiler is unavailable"
fi

resource_dir=$("$hipcc" --print-resource-dir) || fail "HIP compiler did not report its resource directory"
llvm_root=$(dirname -- "$(dirname -- "$(dirname -- "$resource_dir")")")
llvm_objdump="$llvm_root/bin/llvm-objdump"
if [[ ! -x "$llvm_objdump" ]]; then
    fail "ROCm llvm-objdump is unavailable beside the selected HIP compiler"
fi
if ! command -v ar >/dev/null; then
    fail "ar is unavailable for archive inspection"
fi

scratch=$(mktemp -d "${TMPDIR:-/tmp}/logismos-hip-code-object.XXXXXX")
trap 'rm -rf -- "$scratch"' EXIT
mkdir "$scratch/inspect"

cargo_args=(
    env
    LOGISMOS_HIP_BUILD=required
    HIPCC="$hipcc"
    CARGO_TARGET_DIR="$scratch/cargo-target"
)
if [[ -n "${HIP_PATH:-}" ]]; then
    cargo_args+=(HIP_PATH="$HIP_PATH")
elif [[ "$hipcc" == /opt/rocm/bin/hipcc ]]; then
    cargo_args+=(HIP_PATH=/opt/rocm)
fi
if [[ -n "${BINDGEN_EXTRA_CLANG_ARGS:-}" ]]; then
    cargo_args+=(BINDGEN_EXTRA_CLANG_ARGS="$BINDGEN_EXTRA_CLANG_ARGS")
else
    cargo_args+=(BINDGEN_EXTRA_CLANG_ARGS="-resource-dir $resource_dir")
fi
cargo_args+=(cargo build --offline --locked -p kernels --jobs "${CARGO_BUILD_JOBS:-8}")
"${cargo_args[@]}"

mapfile -t archives < <(find "$scratch/cargo-target/debug/build" -type f -path '*/out/liblogismos_kernels.a')
if [[ "${#archives[@]}" -ne 1 ]]; then
    fail "expected exactly one kernel archive, found ${#archives[@]}"
fi
archive=${archives[0]}
if [[ ! -s "$archive" ]]; then
    fail "kernel archive is empty"
fi

mapfile -t hip_sources < <(find "$ROOT/crates/kernels/src" -type f -name '*.hip' | sort)
if [[ "${#hip_sources[@]}" -eq 0 ]]; then
    fail "no in-tree HIP sources were found"
fi
members=$(ar t "$archive")
for source in "${hip_sources[@]}"; do
    member="$(basename -- "$source").o"
    if ! grep -Fqx -- "$member" <<<"$members"; then
        fail "kernel archive omits HIP object $member"
    fi
done

metadata=$(cd "$scratch/inspect" && "$llvm_objdump" --offloading "$archive")
amdgpu_bundles=$(grep -E 'hipv4-amdgcn-amd-amdhsa--' <<<"$metadata" || true)
expected_bundle="hipv4-amdgcn-amd-amdhsa--$target"
expected_count=$(grep -Fc -- "$expected_bundle" <<<"$amdgpu_bundles" || true)
unexpected_bundles=$(grep -Fv -- "$expected_bundle" <<<"$amdgpu_bundles" || true)
if [[ "$expected_count" -ne "${#hip_sources[@]}" || -n "$unexpected_bundles" ]]; then
    fail "expected ${#hip_sources[@]} $target code objects; inspection was: $amdgpu_bundles"
fi

printf 'HIP code-object evidence: target=%s archive-members=%s code-objects=%s\n' \
    "$target" "${#hip_sources[@]}" "$expected_count"
