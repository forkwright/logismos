#!/usr/bin/bash -p
# WHY: This executes as the image's unprivileged account. Trusted provisioning
# obtains only lockfile sources; workspace compilation begins inside the runner.
set -euo pipefail
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
unset CDPATH LD_LIBRARY_PATH LD_PRELOAD PYTHONHOME PYTHONPATH

if [[ ${1:-} != /workspace || ! -r "$1/Cargo.lock" || ! -d "$1/target" ]]; then
    builtin printf '%s\n' 'usage: ci-hip-code-object.sh /workspace' >&2
    exit 64
fi
if [[ ! ${PINNED_TOOLCHAIN:-} =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    builtin printf '%s\n' 'ci-hip-code-object requires an exact PINNED_TOOLCHAIN' >&2
    exit 69
fi

readonly WORKSPACE=$1
readonly CI_HOME=/home/gpu-ci
readonly HOSTED_RUST_ROOT=/opt/ci-rust
readonly HOSTED_CARGO_BIN=$HOSTED_RUST_ROOT/cargo/bin
readonly HOSTED_RUSTUP=$HOSTED_RUST_ROOT/rustup
if [[ $(/usr/bin/id -u) -eq 0 || ! -O "$CI_HOME" ]]; then
    builtin printf '%s\n' 'ci-hip-code-object requires the image non-root gpu-ci account' >&2
    exit 69
fi
if [[ ! -d "$HOSTED_CARGO_BIN" || ! -d "$HOSTED_RUSTUP" ]]; then
    builtin printf '%s\n' 'ci-hip-code-object requires mounted pinned Rust sources' >&2
    exit 69
fi

install -d -m 0755 "$CI_HOME/.cargo" "$CI_HOME/.rustup" "$WORKSPACE/target"
cp -a -- "$HOSTED_CARGO_BIN" "$CI_HOME/.cargo/bin"
cp -a -- "$HOSTED_RUSTUP/." "$CI_HOME/.rustup/"

python3 "$WORKSPACE/scripts/ci-locked-crates.py" "$WORKSPACE/Cargo.lock"

cd /tmp
/usr/bin/setpriv \
    --no-new-privs \
    --inh-caps=-all --ambient-caps=-all -- \
    /usr/bin/env -i \
        HOME="$CI_HOME" USER=gpu-ci LOGNAME=gpu-ci \
        PATH="$CI_HOME/.cargo/bin:/usr/bin:/bin" \
        CARGO_HOME="$CI_HOME/.cargo" RUSTUP_HOME="$CI_HOME/.rustup" \
        RUSTUP_TOOLCHAIN="$PINNED_TOOLCHAIN" \
        "$CI_HOME/.cargo/bin/cargo" fetch --locked --manifest-path "$WORKSPACE/Cargo.toml"

{
    /usr/bin/setpriv \
        --no-new-privs \
        --inh-caps=-all --ambient-caps=-all -- \
        /usr/bin/env -i \
            HOME="$CI_HOME" USER=gpu-ci LOGNAME=gpu-ci PATH=/usr/bin:/bin \
            /usr/bin/bash --noprofile --norc \
                "$WORKSPACE/scripts/hip-code-object-witness.sh" </dev/null
} 2>&1 | /usr/bin/cat
