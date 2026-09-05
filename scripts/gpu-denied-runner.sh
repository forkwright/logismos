#!/usr/bin/env bash
# Run a command in the GPU-denied test boundary.
set -euo pipefail

if [[ "${1:-}" != "--" || "$#" -eq 1 ]]; then
    echo "usage: $0 -- COMMAND [ARG...]" >&2
    exit 64
fi
shift

if ! BWRAP=$(command -v bwrap); then
    echo "gpu-denied runner requires bwrap; refusing to execute outside the boundary" >&2
    exit 69
fi

if ! SETPRIV=$(command -v setpriv); then
    echo "gpu-denied runner requires setpriv; refusing to execute outside the boundary" >&2
    exit 69
fi

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
SUPERVISOR="$ROOT/scripts/gpu-denied-exec.py"
if [[ ! -x /usr/bin/python3 || ! -f "$SUPERVISOR" ]]; then
    echo "gpu-denied runner requires its Python descriptor-closing supervisor" >&2
    exit 69
fi

HOST_HOME=${HOME:-}
SANDBOX_HOME=/tmp/home
LIBCLANG_PATH=/usr/lib64/rocm/llvm/lib

mount_args=(
    --tmpfs /
    --ro-bind /usr /usr
    --dir /etc
    --ro-bind /etc/ld.so.cache /etc/ld.so.cache
    --ro-bind-try /etc/ld.so.conf /etc/ld.so.conf
    --ro-bind-try /etc/ld.so.conf.d /etc/ld.so.conf.d
    --symlink usr/bin /bin
    --symlink usr/sbin /sbin
    --symlink usr/lib /lib
    --symlink usr/lib64 /lib64
    --bind "$ROOT" "$ROOT"
    --dev /dev
    --tmpfs /sys
    --tmpfs /run
    --tmpfs /tmp
    --proc /proc
)

# ROCm packages use /opt/rocm on the official Ubuntu toolchain image. Bind
# only that read-only compiler tree when it exists; the sandbox still has a
# synthetic /dev, /sys, /run, and its own network namespace.
if [[ -d /opt/rocm ]]; then
    mount_args+=(
        --dir /opt
        --ro-bind /opt/rocm /opt/rocm
    )
    LIBCLANG_PATH=/opt/rocm/llvm/lib
fi

if [[ -n "$HOST_HOME" && -d "$HOST_HOME/.cargo/bin" && -d "$HOST_HOME/.rustup" ]]; then
    mount_args+=(
        --dir "$(dirname -- "$HOST_HOME")"
        --dir "$HOST_HOME"
        --dir "$HOST_HOME/.cargo"
        --ro-bind "$HOST_HOME/.cargo/bin" "$HOST_HOME/.cargo/bin"
        --ro-bind-try "$HOST_HOME/.cargo/registry" "$HOST_HOME/.cargo/registry"
        --ro-bind-try "$HOST_HOME/.cargo/git" "$HOST_HOME/.cargo/git"
        --ro-bind "$HOST_HOME/.rustup" "$HOST_HOME/.rustup"
    )
    SANDBOX_HOME=$HOST_HOME
fi

# WHY: descriptor inheritance can bypass a pathname-only device policy, so the
# supervisor closes every descriptor except stdin/stdout/stderr before bwrap.
exec /usr/bin/python3 "$SUPERVISOR" "$BWRAP" "$SETPRIV" "$ROOT" "$SANDBOX_HOME" "$LIBCLANG_PATH" "${mount_args[@]}" -- "$@"
