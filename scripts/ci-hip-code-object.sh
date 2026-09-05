#!/usr/bin/bash -p
# WHY: Trusted provisioning obtains only lockfile sources; workspace compilation
# begins only after the GPU-denied runner has entered its boundary.
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
export HOME=/root
export CARGO_HOME=/root/.cargo
export RUSTUP_HOME=/root/.rustup

apt-get update
apt-get install --yes --no-install-recommends bubblewrap ca-certificates curl python3 util-linux
curl --proto '=https' --tlsv1.2 --retry 10 --retry-connrefused --location --silent --show-error --fail \
    https://sh.rustup.rs | sh -s -- --profile minimal --default-toolchain "$PINNED_TOOLCHAIN" -y

useradd --create-home --user-group --shell /bin/bash gpu-ci
test -d "$CARGO_HOME/bin"
test -d "$RUSTUP_HOME"
install -d -m 0755 -o gpu-ci -g gpu-ci \
    "$CI_HOME/.cargo" "$CI_HOME/.rustup" "$WORKSPACE/target"
cp -a -- "$CARGO_HOME/bin" "$CI_HOME/.cargo/bin"
cp -a -- "$RUSTUP_HOME/." "$CI_HOME/.rustup/"

for cache in registry git; do
    if [[ -d "$CARGO_HOME/$cache" ]]; then
        cp -a -- "$CARGO_HOME/$cache" "$CI_HOME/.cargo/$cache"
    fi
done
chown -R gpu-ci:gpu-ci -- "$CI_HOME/.cargo" "$CI_HOME/.rustup" "$WORKSPACE/target"

python3 - "$WORKSPACE/Cargo.lock" <<'PYTHON'
import sys
import tomllib
from pathlib import Path

lock = tomllib.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
allowed = 'registry+https://github.com/rust-lang/crates.io-index'
unexpected = sorted({
    package['source']
    for package in lock.get('package', [])
    if 'source' in package and package['source'] != allowed
})
if unexpected:
    raise SystemExit(f'Cargo.lock contains non-crates.io sources: {unexpected}')
PYTHON

cd /tmp
/usr/bin/setpriv \
    --reuid=gpu-ci --regid=gpu-ci --init-groups --no-new-privs \
    --inh-caps=-all --ambient-caps=-all -- \
    /usr/bin/env -i \
        HOME="$CI_HOME" USER=gpu-ci LOGNAME=gpu-ci \
        PATH="$CI_HOME/.cargo/bin:/usr/bin:/bin" \
        CARGO_HOME="$CI_HOME/.cargo" RUSTUP_HOME="$CI_HOME/.rustup" \
        RUSTUP_TOOLCHAIN="$PINNED_TOOLCHAIN" \
        "$CI_HOME/.cargo/bin/cargo" fetch --locked --manifest-path "$WORKSPACE/Cargo.toml"

{
    /usr/bin/setpriv \
        --reuid=gpu-ci --regid=gpu-ci --init-groups --no-new-privs \
        --inh-caps=-all --ambient-caps=-all -- \
        /usr/bin/env -i \
            HOME="$CI_HOME" USER=gpu-ci LOGNAME=gpu-ci PATH=/usr/bin:/bin \
            /usr/bin/bash --noprofile --norc \
                "$WORKSPACE/scripts/hip-code-object-witness.sh" </dev/null
} 2>&1 | /usr/bin/cat
