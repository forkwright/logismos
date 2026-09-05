#!/usr/bin/env bash
# Pure parser fixtures for the HIP code-object witness; no compiler or GPU access.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
# shellcheck source=hip-code-object-witness-inner.sh
source "$ROOT/scripts/hip-code-object-witness-inner.sh"

target=$(<"$ROOT/contracts/gpu-target.txt")
bundle="Extracting offload bundle: kernel.hip.o.0.hipv4-amdgcn-amd-amdhsa--"

metadata_for() {
    local object_count=$1
    local code_target=$2
    local output=
    local index
    for ((index = 1; index <= object_count; index += 1)); do
        output+="$bundle$code_target"
        output+=$'\n'
    done
    printf '%s' "$output"
}

assert_rejected() {
    local label=$1
    local metadata=$2
    local expected_count=$3
    if verify_code_object_metadata "$metadata" "$target" "$expected_count" >/dev/null 2>&1; then
        echo "fixture $label unexpectedly passed" >&2
        exit 1
    fi
}

if ! verify_code_object_metadata "$(metadata_for 5 "$target:xnack+")" "$target" 5 >/dev/null; then
    echo "valid feature suffix fixture failed" >&2
    exit 1
fi
assert_rejected wrong_isa "$(metadata_for 5 gfx1101)" 5
assert_rejected target_prefix "$(metadata_for 5 "${target}0")" 5
assert_rejected empty_target "$(metadata_for 5 '')" 5
assert_rejected missing_object "$(metadata_for 4 "$target")" 5
printf 'HIP code-object metadata fixtures: PASS\n'
