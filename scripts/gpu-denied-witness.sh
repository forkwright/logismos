#!/usr/bin/env bash
# WHY: Synthetic host endpoints prove denial without opening or probing any real accelerator device.
set -euo pipefail
PATH=/usr/bin:/bin

SCRIPT_DIR=${BASH_SOURCE[0]%/*}
if [[ "$SCRIPT_DIR" == "${BASH_SOURCE[0]}" ]]; then
    SCRIPT_DIR=.
fi
ROOT=$(builtin cd -- "$SCRIPT_DIR/.." && builtin pwd -P)
RUNNER="$ROOT/scripts/gpu-denied-runner.sh"
FIXTURE_DIR=$(/usr/bin/mktemp -d)
LINK_FIXTURE_DIR=$(/usr/bin/mktemp -d "${ROOT%/*}/gpu-denied-hardlink.XXXXXX")
CLEANUP_DONE=

cleanup() {
    if [[ -n "$CLEANUP_DONE" ]]; then
        return
    fi
    CLEANUP_DONE=1
    /usr/bin/rm -f -- "$ROOT/target/gpu-denied-escape-link"
    /usr/bin/rm -f -- \
        "$ROOT/target/gpu-denied-hardlink-a" \
        "$ROOT/target/gpu-denied-hardlink-b"
    /usr/bin/chmod 0700 "$ROOT/target/gpu-denied-unreadable" 2>/dev/null || true
    /usr/bin/rm -rf -- "$ROOT/target/gpu-denied-unreadable"
    /usr/bin/rm -rf -- "$FIXTURE_DIR"
    /usr/bin/rm -rf -- "$LINK_FIXTURE_DIR"
}

interrupt() {
    local signal=$1
    cleanup
    trap - "$signal"
    kill -"$signal" "$$"
}

trap cleanup EXIT
trap 'interrupt INT' INT
trap 'interrupt TERM' TERM

HOST_MOUNT_NS=$(/usr/bin/readlink /proc/self/ns/mnt)
HOST_NETWORK_NS=$(/usr/bin/stat --dereference --format '%d:%i' /proc/self/ns/net)
HOST_PID_NS=$(/usr/bin/readlink /proc/self/ns/pid)

{
    /usr/bin/timeout 60 "$RUNNER" -- /usr/bin/python3 - \
        "$ROOT" "$HOST_MOUNT_NS" "$HOST_NETWORK_NS" "$HOST_PID_NS" <<'PYTHON'
import os
import fcntl
import socket
import struct
import sys
from pathlib import Path

root = Path(sys.argv[1])
host_mount_namespace, host_network_namespace, host_pid_namespace = sys.argv[2:]
sandbox_mount_namespace = os.readlink('/proc/self/ns/mnt')
sandbox_network_namespace = os.stat('/proc/self/ns/net')
sandbox_network_inode = f'{sandbox_network_namespace.st_dev}:{sandbox_network_namespace.st_ino}'
sandbox_pid_namespace = os.readlink('/proc/self/ns/pid')


def require_distinct_network_namespace(actual: str, expected: str) -> None:
    if actual == expected:
        raise AssertionError('network namespace was not isolated')


# WHY: validate the isolation predicate itself, rather than relying only
# on a host/sandbox comparison that could accidentally use unlike identifiers.
try:
    require_distinct_network_namespace(sandbox_network_inode, sandbox_network_inode)
except AssertionError:
    pass
else:
    raise AssertionError('network namespace isolation predicate accepted itself')
require_distinct_network_namespace(sandbox_network_inode, host_network_namespace)
if (
    sandbox_mount_namespace == host_mount_namespace
    or sandbox_pid_namespace == host_pid_namespace
):
    raise AssertionError('one or more required namespaces were not isolated')

# INVARIANT: this is the private network namespace created before Bubblewrap.
# Its loopback must remain down: a host listener cannot be reached through it.
with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
    loopback_flags = struct.unpack(
        '16sH14x',
        fcntl.ioctl(probe.fileno(), 0x8913, struct.pack('16sH14x', b'lo', 0)),
    )[1]
if loopback_flags & 0x1:
    raise AssertionError('private network namespace loopback is unexpectedly up')

status = {}
for line in Path('/proc/self/status').read_text(encoding='utf-8').splitlines():
    if ':' in line:
        name, value = line.split(':', 1)
        status[name] = value.strip()
if status.get('NoNewPrivs') != '1' or int(status.get('CapEff', '1'), 16) != 0:
    raise AssertionError('privilege restrictions are not active')

allowed_devices = {
    'core',
    'fd',
    'full',
    'null',
    'ptmx',
    'pts',
    'random',
    'shm',
    'stderr',
    'stdin',
    'stdout',
    'tty',
    'urandom',
    'zero',
}
unexpected_devices = set(os.listdir('/dev')) - allowed_devices
if unexpected_devices:
    raise AssertionError(f'unexpected synthetic /dev entries: {sorted(unexpected_devices)}')
if os.listdir('/sys') or os.listdir('/run'):
    raise AssertionError('masked host runtime trees are not empty')
if set(os.listdir('/etc')) - {'ld.so.cache'}:
    raise AssertionError('sandbox /etc exposes more than the dynamic-loader cache')

if os.environ['HOME'] != '/tmp/home' or os.environ['CARGO_HOME'] != '/tmp/cargo':
    raise AssertionError('synthetic home configuration is missing')
if os.environ['CARGO_NET_OFFLINE'] != 'true':
    raise AssertionError('Cargo offline mode is missing')
if os.environ['CARGO_TARGET_DIR'] != str(root / 'target'):
    raise AssertionError('Cargo target is not confined to the worktree target')

source_marker = root / 'gpu-denied-source-write'
try:
    source_marker.write_text('unexpected', encoding='utf-8')
except OSError:
    pass
else:
    source_marker.unlink(missing_ok=True)
    raise AssertionError('worktree source is writable')

target_marker = root / 'target/gpu-denied-target-write'
target_marker.write_text('expected', encoding='utf-8')
target_marker.unlink()
print('gpu-denied namespace, privilege, mount, network, and environment witness: PASS')
PYTHON
} 2>&1 | /usr/bin/cat

TRAP_MARKER="$FIXTURE_DIR/ambient-tool-ran"
TRAP_EXECUTABLE="$FIXTURE_DIR/trap-executable"
/usr/bin/printf '#!/usr/bin/bash\n/usr/bin/touch -- %q\nexit 97\n' "$TRAP_MARKER" >"$TRAP_EXECUTABLE"
/usr/bin/chmod 0700 "$TRAP_EXECUTABLE"
/usr/bin/printf '/usr/bin/touch -- %q\n' "$TRAP_MARKER" >"$FIXTURE_DIR/bash-env"
PYTHON_VERSION=$(/usr/bin/python3 -I -c \
    'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")')
PYTHON_SITE="$FIXTURE_DIR/python/lib/$PYTHON_VERSION/site-packages"
/usr/bin/mkdir -p "$PYTHON_SITE"
/usr/bin/printf '%s\n' \
    'import os' \
    'from pathlib import Path' \
    'Path(os.environ["GPU_DENIED_TRAP_MARKER"]).touch()' >"$PYTHON_SITE/sitecustomize.py"
/usr/bin/mkdir -p "$FIXTURE_DIR/bin" "$FIXTURE_DIR/home/.cargo/bin" "$FIXTURE_DIR/home/.rustup"
for name in bash bwrap dirname python3 setpriv; do
    /usr/bin/ln -s "$TRAP_EXECUTABLE" "$FIXTURE_DIR/bin/$name"
done

{
    # WHY: these variables intentionally expand in the child `/bin/sh`, after
    # the runner has cleared the hostile parent environment.
    # shellcheck disable=SC2016
    PATH="$FIXTURE_DIR/bin" \
    HOME="$FIXTURE_DIR/home" \
    CARGO_HOME="$FIXTURE_DIR/home/.cargo" \
    RUSTUP_HOME="$FIXTURE_DIR/home/.rustup" \
    BASH_ENV="$FIXTURE_DIR/bash-env" \
    PYTHONUSERBASE="$FIXTURE_DIR/python" \
    GPU_DENIED_TRAP_MARKER="$TRAP_MARKER" \
    /usr/bin/timeout 60 "$RUNNER" -- /bin/sh -ceu '
        case "$PATH:$HOME:$CARGO_HOME:${RUSTUP_HOME:-}" in
            *"$1"*) echo "ambient tool or home path entered the sandbox" >&2; exit 1 ;;
        esac
    ' /bin/sh "$FIXTURE_DIR" </dev/null
} 2>&1 | /usr/bin/cat
if [[ -e "$TRAP_MARKER" ]]; then
    builtin printf '%s\n' 'gpu-denied witness: an ambient executable ran before the boundary' >&2
    exit 1
fi

/usr/bin/python3 - "$RUNNER" "$ROOT" "$FIXTURE_DIR" "$LINK_FIXTURE_DIR" <<'PYTHON'
import os
import pty
import resource
import socket
import stat
import subprocess
import sys
from pathlib import Path

runner = Path(sys.argv[1])
root = Path(sys.argv[2])
fixture = Path(sys.argv[3])
link_fixture = Path(sys.argv[4])


def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(command, check=False, timeout=60, **kwargs)


read_descriptor, write_descriptor = os.pipe()
soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
if soft_limit == resource.RLIM_INFINITY or soft_limit > 513:
    high_descriptor = 512
else:
    high_descriptor = int(soft_limit) - 1
if high_descriptor < 32:
    raise AssertionError('descriptor limit is too low for the inherited-descriptor witness')
os.dup2(read_descriptor, high_descriptor, inheritable=True)
os.close(read_descriptor)
os.close(write_descriptor)
reduced_limit = max(16, high_descriptor // 2)


def lower_descriptor_limit() -> None:
    resource.setrlimit(resource.RLIMIT_NOFILE, (reduced_limit, hard_limit))


high_result = run(
    [
        str(runner),
        '--',
        '/usr/bin/python3',
        '-c',
        'import os,sys; assert not os.path.exists(f"/proc/self/fd/{sys.argv[1]}")',
        str(high_descriptor),
    ],
    stdin=subprocess.DEVNULL,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    pass_fds=(high_descriptor,),
    preexec_fn=lower_descriptor_limit,
)
os.close(high_descriptor)
if high_result.returncode != 0:
    raise AssertionError(high_result.stderr.decode(errors='replace'))

master, slave = pty.openpty()
try:
    standard_result = run(
        [str(runner), '--', '/usr/bin/true'],
        stdin=slave,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
finally:
    os.close(master)
    os.close(slave)
if standard_result.returncode != 69 or b'unsupported host endpoint' not in standard_result.stderr:
    raise AssertionError('inherited pseudo-terminal device was not rejected')

left, right = socket.socketpair()
try:
    socket_standard_result = run(
        [str(runner), '--', '/usr/bin/true'],
        stdin=left,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
finally:
    left.close()
    right.close()
if socket_standard_result.returncode != 69 or b'unsupported host endpoint' not in socket_standard_result.stderr:
    raise AssertionError('inherited standard-I/O socket was not rejected')

unlinked_path = fixture / 'unlinked-standard-input'
unlinked_descriptor = os.open(unlinked_path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
unlinked_path.unlink()
try:
    unlinked_result = run(
        [str(runner), '--', '/usr/bin/true'],
        stdin=unlinked_descriptor,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
finally:
    os.close(unlinked_descriptor)
if unlinked_result.returncode != 69 or b'unsupported host endpoint' not in unlinked_result.stderr:
    raise AssertionError('unlinked regular standard input was not rejected')

hardlink_a = root / 'target/gpu-denied-hardlink-a'
hardlink_b = root / 'target/gpu-denied-hardlink-b'
hardlink_a.unlink(missing_ok=True)
hardlink_b.unlink(missing_ok=True)
hardlink_a.write_text('synthetic target data', encoding='utf-8')
hardlink_b.hardlink_to(hardlink_a)
try:
    internal_hardlink_result = run(
        [str(runner), '--', '/usr/bin/true'],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
finally:
    hardlink_b.unlink(missing_ok=True)
    hardlink_a.unlink(missing_ok=True)
if internal_hardlink_result.returncode != 0:
    raise AssertionError(internal_hardlink_result.stderr.decode(errors='replace'))

hardlink_source = link_fixture / 'gpu-denied-hardlink-source'
hardlink_target = root / 'target/gpu-denied-hardlink-a'
hardlink_source.write_text('synthetic host data', encoding='utf-8')
hardlink_target.hardlink_to(hardlink_source)
try:
    escaping_hardlink_result = run(
        [str(runner), '--', '/usr/bin/true'],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
finally:
    hardlink_target.unlink(missing_ok=True)
    hardlink_source.unlink(missing_ok=True)
if (
    escaping_hardlink_result.returncode != 69
    or b'hard link escapes' not in escaping_hardlink_result.stderr
):
    raise AssertionError('target hard link to a host file was not rejected')

target_socket_path = root / 'target/gpu-denied-host-endpoint.sock'
target_socket_path.unlink(missing_ok=True)
target_alias = fixture / 'target-alias'
target_alias.symlink_to(root / 'target', target_is_directory=True)
target_listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
try:
    target_listener.bind(str(target_alias / target_socket_path.name))
    target_listener.listen()
    target_socket_result = run(
        [str(runner), '--', '/usr/bin/true'],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
finally:
    target_listener.close()
    target_socket_path.unlink(missing_ok=True)
if target_socket_result.returncode != 69 or b'host endpoint' not in target_socket_result.stderr:
    raise AssertionError('pre-existing target socket was not rejected')

unreadable_dir = root / 'target/gpu-denied-unreadable'
unreadable_socket_path = unreadable_dir / 'host-endpoint.sock'
unreadable_alias = fixture / 'unreadable-alias'
unreadable_dir.mkdir(mode=0o700)
unreadable_alias.symlink_to(unreadable_dir, target_is_directory=True)
unreadable_listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
try:
    unreadable_listener.bind(str(unreadable_alias / unreadable_socket_path.name))
    unreadable_listener.listen()
    unreadable_dir.chmod(0)
    unreadable_result = run(
        [str(runner), '--', '/usr/bin/true'],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
finally:
    unreadable_dir.chmod(0o700)
    unreadable_listener.close()
    unreadable_socket_path.unlink(missing_ok=True)
    unreadable_dir.rmdir()
if unreadable_result.returncode != 69 or b'cannot inspect' not in unreadable_result.stderr:
    raise AssertionError('unreadable target directory was not rejected')

host_file = fixture / 'host-only'
host_file.write_text('synthetic host data', encoding='utf-8')
escape_link = root / 'target/gpu-denied-escape-link'
escape_link.unlink(missing_ok=True)
escape_link.symlink_to(host_file)
try:
    link_result = run(
        [
            str(runner),
            '--',
            '/usr/bin/python3',
            '-c',
            'from pathlib import Path; import sys; p=Path(sys.argv[1]); assert p.is_symlink() and not p.exists()',
            str(escape_link),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
finally:
    escape_link.unlink(missing_ok=True)
if link_result.returncode != 0:
    raise AssertionError(link_result.stderr.decode(errors='replace'))

unix_path = fixture / 'host-listener.sock'
unix_listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
tcp_listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    unix_listener.bind(str(unix_path))
    unix_listener.listen()
    tcp_listener.bind(('127.0.0.1', 0))
    tcp_listener.listen()
    port = tcp_listener.getsockname()[1]
    network_result = run(
        [
            str(runner),
            '--',
            '/usr/bin/python3',
            '-c',
            (
                'import socket,sys; '
                'u=socket.socket(socket.AF_UNIX); assert u.connect_ex(sys.argv[1]) != 0; u.close(); '
                't=socket.socket(); assert t.connect_ex(("127.0.0.1",int(sys.argv[2]))) != 0; t.close()'
            ),
            str(unix_path),
            str(port),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
finally:
    unix_listener.close()
    tcp_listener.close()
    unix_path.unlink(missing_ok=True)
if network_result.returncode != 0:
    raise AssertionError(network_result.stderr.decode(errors='replace'))

print('gpu-denied synthetic descriptor, path, socket, and network witnesses: PASS')
PYTHON

builtin printf '%s\n' 'gpu-denied boundary witness: PASS'
