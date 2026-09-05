#!/usr/bin/env bash
# Compile kernels and inspect their embedded AMDGPU code objects inside the boundary.
set -euo pipefail

readonly BUNDLE_PREFIX='hipv4-amdgcn-amd-amdhsa--'
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)

fail() {
    echo "HIP code-object witness: $*" >&2
    exit 1
}

bundle_target_from_line() {
    local line=$1
    local target
    if [[ "$line" != *"$BUNDLE_PREFIX"* ]]; then
        return 1
    fi
    target=${line##*"$BUNDLE_PREFIX"}
    if [[ ! "$target" =~ ^gfx[0-9]+(:[a-z][a-z0-9_]*[+-])*$ ]]; then
        return 1
    fi
    printf '%s\n' "$target"
}

bundle_base_target() {
    local target=$1
    if [[ "$target" =~ ^(gfx[0-9]+)(:[a-z][a-z0-9_]*[+-])*$ ]]; then
        printf '%s\n' "${BASH_REMATCH[1]}"
        return 0
    fi
    return 1
}

verify_code_object_metadata() {
    local metadata=$1
    local expected_target=$2
    local expected_count=$3
    local line
    local bundle_target
    local base_target
    local code_objects=0

    while IFS= read -r line; do
        [[ "$line" == *"$BUNDLE_PREFIX"* ]] || continue
        if ! bundle_target=$(bundle_target_from_line "$line"); then
            echo "malformed AMDGPU code-object bundle: $line" >&2
            return 1
        fi
        if ! base_target=$(bundle_base_target "$bundle_target"); then
            echo "unparseable AMDGPU code-object target: $bundle_target" >&2
            return 1
        fi
        if [[ "$base_target" != "$expected_target" ]]; then
            echo "unexpected AMDGPU code-object target: $bundle_target" >&2
            return 1
        fi
        ((code_objects += 1))
    done <<<"$metadata"

    if [[ "$code_objects" -ne "$expected_count" ]]; then
        echo "expected $expected_count code objects, found $code_objects" >&2
        return 1
    fi
    printf '%s\n' "$code_objects"
}

main() {
    if [[ "$#" -ne 0 ]]; then
        fail "inner helper accepts no arguments"
    fi

    target=$(<"$ROOT/contracts/gpu-target.txt")
    if [[ ! "$target" =~ ^gfx[0-9]+$ ]]; then
        fail "contracts/gpu-target.txt must contain one gfx target"
    fi

    if [[ -x /opt/rocm/bin/hipcc ]]; then
        hipcc=/opt/rocm/bin/hipcc
    elif [[ -x /usr/bin/hipcc ]]; then
        hipcc=/usr/bin/hipcc
    else
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
    compiler_version=$("$hipcc" --version | head -n 1) || fail "HIP compiler did not report its version"
    if [[ -z "$compiler_version" ]]; then
        fail "HIP compiler reported an empty version"
    fi

    witness_target="${CARGO_TARGET_DIR:?GPU-denied runner must set CARGO_TARGET_DIR}/hip-code-object-witness"
    cargo_target="$witness_target/cargo"
    mkdir -p "$witness_target"
    scratch=$(mktemp -d "${TMPDIR:-/tmp}/logismos-hip-code-object.XXXXXX")
    trap 'rm -rf -- "$scratch"' EXIT
    mkdir "$scratch/inspect"

    env \
        LOGISMOS_HIP_BUILD=required \
        HIPCC="$hipcc" \
        CARGO_TARGET_DIR="$cargo_target" \
        BINDGEN_EXTRA_CLANG_ARGS="-resource-dir $resource_dir" \
        cargo build --offline --locked -p kernels --jobs 8

    mapfile -t archives < <(find "$cargo_target/debug/build" -type f -path '*/out/liblogismos_kernels.a')
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
    if ! code_objects=$(verify_code_object_metadata "$metadata" "$target" "${#hip_sources[@]}"); then
        fail "code-object inspection did not prove target $target"
    fi

    receipt="$witness_target/receipt.txt"
    printf 'target=%s\ncompiler=%s\nresource_dir=%s\narchive=%s\narchive_members=%s\ncode_objects=%s\n' \
        "$target" "$compiler_version" "$resource_dir" "$archive" "${#hip_sources[@]}" "$code_objects" >"$receipt"
    printf 'HIP code-object evidence: target=%s archive-members=%s code-objects=%s compiler=%s receipt=%s\n' \
        "$target" "${#hip_sources[@]}" "$code_objects" "$compiler_version" "$receipt"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
