#!/usr/bin/bash -p
# WHY: A fixed interpreter and privileged mode prevent caller-controlled PATH,
# BASH_ENV, exported functions, and shell options from running code before the boundary.
set -euo pipefail
PATH=/usr/bin:/bin
unset CDPATH LD_LIBRARY_PATH LD_PRELOAD PYTHONHOME PYTHONPATH

usage() {
    builtin printf 'usage: %s [--ro-input-file FILE | --ro-model-file FILE --ro-tokenizer-file FILE] -- COMMAND [ARG...]\n' "$0" >&2
}

read_only_inputs=()
while [[ "$#" -gt 0 && "${1:-}" != '--' ]]; do
    case "$1" in
        --ro-input-file|--ro-model-file|--ro-tokenizer-file)
            if [[ "$#" -lt 2 || -z "${2:-}" ]]; then
                usage
                exit 64
            fi
            read_only_inputs+=("$1" "$2")
            shift 2
            ;;
        *)
            usage
            exit 64
            ;;
    esac
done
if [[ "${1:-}" != '--' || "$#" -eq 1 ]]; then
    usage
    exit 64
fi
shift

if [[ ! -x /usr/bin/bwrap ]]; then
    builtin printf '%s\n' \
        'gpu-denied runner requires /usr/bin/bwrap; refusing to execute outside the boundary' >&2
    exit 69
fi

if [[ ! -x /usr/bin/unshare ]]; then
    builtin printf '%s\n' \
        'gpu-denied runner requires /usr/bin/unshare; refusing to execute outside the boundary' >&2
    exit 69
fi

if [[ ! -x /usr/bin/setpriv ]]; then
    builtin printf '%s\n' \
        'gpu-denied runner requires /usr/bin/setpriv; refusing to execute outside the boundary' >&2
    exit 69
fi

SCRIPT_DIR=${BASH_SOURCE[0]%/*}
if [[ "$SCRIPT_DIR" == "${BASH_SOURCE[0]}" ]]; then
    SCRIPT_DIR=.
fi
ROOT=$(builtin cd -- "$SCRIPT_DIR/.." && builtin pwd -P)
SUPERVISOR="$ROOT/scripts/gpu-denied-exec.py"
if [[ ! -x /usr/bin/python3 || ! -f "$SUPERVISOR" ]]; then
    builtin printf '%s\n' \
        'gpu-denied runner requires /usr/bin/python3 and its descriptor supervisor' >&2
    exit 69
fi

builtin exec /usr/bin/python3 -I "$SUPERVISOR" "$ROOT" "${read_only_inputs[@]}" -- "$@"
