#!/usr/bin/env bash
# Locate the one Cargo-resolved kernels build script for witness fixtures.
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
    echo "usage: $0 CARGO_TARGET_DIR" >&2
    exit 64
fi

target_dir=$1
mapfile -t build_scripts < <(
    find "$target_dir/debug/build" -type f \
        -path '*/kernels-*/build-script-build' -perm -111 | sort
)
if [[ "${#build_scripts[@]}" -ne 1 ]]; then
    echo "expected exactly one Cargo-built kernels build script under $target_dir, found ${#build_scripts[@]}" >&2
    exit 1
fi
printf '%s\n' "${build_scripts[0]}"
