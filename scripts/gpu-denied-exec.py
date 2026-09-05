#!/usr/bin/python3
"""Enter the GPU-denied Bubblewrap boundary after validating host inputs."""

from __future__ import annotations

import errno
import os
import re
import stat
import sys
from pathlib import Path

BWRAP = Path('/usr/bin/bwrap')
SETPRIV = Path('/usr/bin/setpriv')
UNSHARE = Path('/usr/bin/unshare')
SANDBOX_CARGO_HOME = Path('/tmp/cargo')
SANDBOX_HOME = Path('/tmp/home')
SANDBOX_RUST_ROOT = Path('/opt/gpu-denied')
SYSTEM_PATH = '/opt/rocm/bin:/opt/gpu-denied/cargo/bin:/usr/local/bin:/usr/bin:/bin'
BLOCKED_MOUNT_ROOTS = (
    Path('/dev'),
    Path('/etc'),
    Path('/proc'),
    Path('/run'),
    Path('/sys'),
)
ROCM_LIBCLANG_CANDIDATES = (
    Path('/usr/lib64/rocm/llvm/lib'),
    Path('/opt/rocm/llvm/lib'),
)
SPECIAL_FILE_TYPES = (stat.S_IFBLK, stat.S_IFCHR, stat.S_IFIFO, stat.S_IFSOCK)
MOUNT_ESCAPE = re.compile(r'\\([0-7]{3})')
VERSION_COMPONENTS = re.compile(r'[0-9]+(?:\.[0-9]+)*')


class BoundaryError(Exception):
    """A condition that requires the runner to fail closed."""


def _decode_mount_path(value: str) -> Path:
    return Path(MOUNT_ESCAPE.sub(lambda match: chr(int(match.group(1), 8)), value))


def _reject_nested_mounts(source: Path) -> None:
    try:
        records = Path('/proc/self/mountinfo').read_text(encoding='utf-8').splitlines()
    except OSError as error:
        raise BoundaryError(f'cannot inspect host mount topology: {error}') from error
    for record in records:
        fields = record.split()
        if len(fields) < 5:
            raise BoundaryError('host mount topology contains a malformed record')
        mount_point = _decode_mount_path(fields[4])
        if mount_point != source and mount_point.is_relative_to(source):
            raise BoundaryError('worktree contains a nested host mount')


def _reject_special_files(
    source: Path,
    *,
    skip_top_level: frozenset[str] = frozenset(),
    allow_contained_hardlinks: bool = False,
) -> None:
    def reject_walk_error(error: OSError) -> None:
        raise BoundaryError(f'cannot inspect worktree mount source: {error}') from error

    hardlinks: dict[tuple[int, int], tuple[int, int, Path]] = {}
    for current, directories, files in os.walk(
        source, topdown=True, onerror=reject_walk_error, followlinks=False
    ):
        current_path = Path(current)
        if current_path == source and skip_top_level:
            directories[:] = [name for name in directories if name not in skip_top_level]
        for name in (*directories, *files):
            path = current_path / name
            try:
                metadata = path.lstat()
            except OSError as error:
                raise BoundaryError(f'cannot inspect worktree mount source: {error}') from error
            file_type = stat.S_IFMT(metadata.st_mode)
            if file_type in SPECIAL_FILE_TYPES:
                raise BoundaryError(f'worktree contains a host endpoint: {path}')
            if file_type == stat.S_IFREG and metadata.st_nlink != 1:
                if not allow_contained_hardlinks:
                    raise BoundaryError(f'worktree contains a multiply-linked regular file: {path}')
                key = (metadata.st_dev, metadata.st_ino)
                observed, expected, first_path = hardlinks.get(
                    key, (0, metadata.st_nlink, path)
                )
                if expected != metadata.st_nlink:
                    raise BoundaryError(f'worktree hard-link count changed while scanning: {path}')
                hardlinks[key] = (observed + 1, expected, first_path)

    for observed, expected, path in hardlinks.values():
        if observed != expected:
            raise BoundaryError(
                f'worktree target hard link escapes the writable target: {path}'
            )


def _local_account_home() -> Path | None:
    passwd = Path('/etc/passwd')
    try:
        if passwd.stat().st_size > 1_048_576:
            raise BoundaryError('local account database exceeds the 1 MiB safety limit')
        records = passwd.read_text(encoding='utf-8').splitlines()
    except OSError as error:
        raise BoundaryError(f'cannot read the local account database: {error}') from error

    uid = os.getuid()
    for record in records:
        fields = record.split(':')
        if len(fields) != 7:
            continue
        try:
            record_uid = int(fields[2])
        except ValueError:
            continue
        if record_uid == uid:
            home = Path(fields[5])
            if not home.is_absolute():
                raise BoundaryError('local account home is not absolute')
            return home
    return None


def _optional_mount_source(path: Path) -> Path | None:
    if not path.exists():
        if path.is_symlink():
            raise BoundaryError(f'toolchain mount source is a broken symlink: {path}')
        return None
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise BoundaryError(f'cannot resolve toolchain mount source {path}: {error}') from error
    if not resolved.is_dir():
        raise BoundaryError(f'toolchain mount source is not a directory: {path}')
    if resolved == Path('/') or any(
        resolved == blocked or resolved.is_relative_to(blocked) for blocked in BLOCKED_MOUNT_ROOTS
    ):
        raise BoundaryError(f'refusing sensitive toolchain mount source: {path}')
    return resolved


def _toolchain_mount_args() -> tuple[list[str], list[tuple[str, str]]]:
    home = _local_account_home()
    if home is None:
        return [], []
    cargo_home = home / '.cargo'
    cargo_bin = _optional_mount_source(cargo_home / 'bin')
    cargo_registry = _optional_mount_source(cargo_home / 'registry')
    cargo_git = _optional_mount_source(cargo_home / 'git')
    rustup_home = _optional_mount_source(home / '.rustup')

    args = [
        '--dir',
        str(SANDBOX_RUST_ROOT),
        '--dir',
        str(SANDBOX_RUST_ROOT / 'cargo'),
    ]
    environment: list[tuple[str, str]] = []
    if cargo_bin is not None:
        args.extend(('--ro-bind', str(cargo_bin), str(SANDBOX_RUST_ROOT / 'cargo/bin')))
    if rustup_home is not None:
        args.extend(('--ro-bind', str(rustup_home), str(SANDBOX_RUST_ROOT / 'rustup')))
        environment.append(('RUSTUP_HOME', str(SANDBOX_RUST_ROOT / 'rustup')))
    if cargo_registry is not None:
        args.extend(('--ro-bind', str(cargo_registry), str(SANDBOX_CARGO_HOME / 'registry')))
    if cargo_git is not None:
        args.extend(('--ro-bind', str(cargo_git), str(SANDBOX_CARGO_HOME / 'git')))
    return args, environment


def _clang_environment() -> list[tuple[str, str]]:
    for library_path in ROCM_LIBCLANG_CANDIDATES:
        if not library_path.is_dir() or not any(library_path.glob('libclang.so*')):
            continue
        resources: list[tuple[tuple[int, ...], Path]] = []
        resource_root = library_path / 'clang'
        if resource_root.is_dir():
            for candidate in resource_root.iterdir():
                if VERSION_COMPONENTS.fullmatch(candidate.name) is None:
                    continue
                if (candidate / 'include').is_dir():
                    version = tuple(int(component) for component in candidate.name.split('.'))
                    resources.append((version, candidate))
        environment = [('LIBCLANG_PATH', str(library_path))]
        if resources:
            resource_dir = max(resources, key=lambda item: item[0])[1]
            environment.append(('BINDGEN_EXTRA_CLANG_ARGS', f'-resource-dir={resource_dir}'))
        return environment
    return []


def _prepare_worktree(root_argument: str) -> tuple[Path, Path]:
    root_input = Path(root_argument)
    try:
        root = root_input.resolve(strict=True)
    except OSError as error:
        raise BoundaryError(f'cannot resolve worktree root: {error}') from error
    if not root_input.is_absolute() or root != root_input or not root.is_dir():
        raise BoundaryError('worktree root must be a canonical absolute directory')

    target = root / 'target'
    try:
        target.mkdir(mode=0o755)
    except FileExistsError:
        pass
    except OSError as error:
        raise BoundaryError(f'cannot create worktree target directory: {error}') from error
    try:
        target_metadata = target.lstat()
    except OSError as error:
        raise BoundaryError(f'cannot inspect worktree target directory: {error}') from error
    if not stat.S_ISDIR(target_metadata.st_mode) or target.resolve(strict=True) != target:
        raise BoundaryError('worktree target must be a real directory, not a link')
    if target.is_mount():
        raise BoundaryError('worktree target must not be a host mount point')

    _reject_nested_mounts(root)
    _reject_special_files(root, skip_top_level=frozenset({'target'}))
    _reject_special_files(target, allow_contained_hardlinks=True)
    return root, target


def _validate_standard_descriptors(root: Path) -> None:
    try:
        null_device = Path('/dev/null').stat().st_rdev
    except OSError as error:
        raise BoundaryError(f'cannot identify the null device: {error}') from error
    for descriptor in range(3):
        try:
            metadata = os.fstat(descriptor)
        except OSError as error:
            raise BoundaryError(
                f'standard descriptor {descriptor} is unavailable: {error}'
            ) from error
        file_type = stat.S_IFMT(metadata.st_mode)
        if file_type == stat.S_IFCHR and metadata.st_rdev == null_device:
            continue
        if file_type == stat.S_IFIFO:
            continue
        if file_type == stat.S_IFREG:
            if metadata.st_nlink == 0:
                raise BoundaryError(
                    f'standard descriptor {descriptor} is an unsupported host endpoint'
                )
            try:
                linked_path = Path(os.readlink(f'/proc/self/fd/{descriptor}')).resolve(strict=True)
            except OSError as error:
                raise BoundaryError(
                    f'cannot establish the path for standard descriptor {descriptor}: {error}'
                ) from error
            if linked_path == root or linked_path.is_relative_to(root):
                continue
        raise BoundaryError(f'standard descriptor {descriptor} is an unsupported host endpoint')


def _close_inherited_descriptors() -> None:
    try:
        descriptors = [int(name) for name in os.listdir('/proc/self/fd') if name.isdecimal()]
    except OSError as error:
        raise BoundaryError(f'cannot enumerate inherited descriptors: {error}') from error
    for descriptor in descriptors:
        if descriptor < 3:
            continue
        try:
            os.close(descriptor)
        except OSError as error:
            if error.errno != errno.EBADF:
                raise BoundaryError(
                    f'cannot close inherited descriptor {descriptor}: {error}'
                ) from error


def _environment_args(environment: list[tuple[str, str]]) -> list[str]:
    args = ['--clearenv']
    for name, value in environment:
        args.extend(('--setenv', name, value))
    return args


def _sandbox_args(root: Path, target: Path, command: list[str]) -> list[str]:
    toolchain_args, toolchain_environment = _toolchain_mount_args()
    environment = [
        ('HOME', str(SANDBOX_HOME)),
        ('USER', 'gpu-denied'),
        ('LOGNAME', 'gpu-denied'),
        ('PATH', SYSTEM_PATH),
        ('TMPDIR', '/tmp'),
        ('CARGO_HOME', str(SANDBOX_CARGO_HOME)),
        ('CARGO_NET_OFFLINE', 'true'),
        ('CARGO_TARGET_DIR', str(target)),
        ('RUSTUP_NO_UPDATE_CHECK', '1'),
        *toolchain_environment,
        *_clang_environment(),
    ]
    mounts = ['--tmpfs', '/', '--ro-bind', '/usr', '/usr']
    rocm_root = Path('/opt/rocm')
    if rocm_root.exists():
        try:
            resolved_rocm_root = rocm_root.resolve(strict=True)
        except OSError as error:
            raise BoundaryError(f'cannot resolve the system ROCm root: {error}') from error
        if not resolved_rocm_root.is_dir():
            raise BoundaryError('the system ROCm root is not a directory')
        mounts.extend(('--ro-bind', str(resolved_rocm_root), str(rocm_root)))
    mounts.extend(
        (
            '--dir',
            '/etc',
            '--symlink',
            'usr/bin',
            '/bin',
            '--symlink',
            'usr/sbin',
            '/sbin',
            '--symlink',
            'usr/lib',
            '/lib',
            '--symlink',
            'usr/lib64',
            '/lib64',
            '--dev',
            '/dev',
            '--tmpfs',
            '/sys',
            '--tmpfs',
            '/run',
            '--tmpfs',
            '/tmp',
            '--proc',
            '/proc',
            '--perms',
            '0700',
            '--dir',
            str(SANDBOX_HOME),
            '--perms',
            '0700',
            '--dir',
            str(SANDBOX_CARGO_HOME),
            *toolchain_args,
            '--ro-bind',
            str(root),
            str(root),
            '--bind',
            str(target),
            str(target),
        )
    )
    loader_cache = Path('/etc/ld.so.cache')
    if loader_cache.is_file():
        mounts.extend(('--ro-bind', str(loader_cache), str(loader_cache)))
    return [
        str(BWRAP),
        '--unshare-all',
        # WHY: the private network namespace is established by the fixed
        # unshare launcher below. Retaining it prevents Bubblewrap from
        # configuring loopback in a host policy that forbids RTM_NEWADDR.
        '--share-net',
        '--unshare-user',
        '--die-with-parent',
        '--new-session',
        '--disable-userns',
        '--assert-userns-disabled',
        '--cap-drop',
        'ALL',
        *_environment_args(environment),
        *mounts,
        '--chdir',
        str(root),
        '--',
        str(SETPRIV),
        '--no-new-privs',
        '--',
        *command,
    ]


def _namespace_launcher_args(sandbox_args: list[str]) -> list[str]:
    # WHY: util-linux unshare creates the private network namespace without
    # assigning an address. Bubblewrap would otherwise bring loopback up as
    # part of its network setup, which some ordinary container policies deny.
    return [
        str(UNSHARE),
        '--user',
        '--map-root-user',
        '--net',
        '--',
        *sandbox_args,
    ]


def main() -> int:
    if len(sys.argv) < 4 or sys.argv[2] != '--':
        print('usage: gpu-denied-exec.py ROOT -- COMMAND [ARG...]', file=sys.stderr)
        return 64
    if not BWRAP.is_file() or not os.access(BWRAP, os.X_OK):
        print('gpu-denied runner requires /usr/bin/bwrap; refusing to execute', file=sys.stderr)
        return 69
    if not UNSHARE.is_file() or not os.access(UNSHARE, os.X_OK):
        print('gpu-denied runner requires /usr/bin/unshare; refusing to execute', file=sys.stderr)
        return 69
    if not SETPRIV.is_file() or not os.access(SETPRIV, os.X_OK):
        print('gpu-denied runner requires /usr/bin/setpriv; refusing to execute', file=sys.stderr)
        return 69
    if os.geteuid() == 0:
        print(
            'gpu-denied runner refuses to establish an unprivileged-task boundary as root',
            file=sys.stderr,
        )
        return 69
    try:
        root, target = _prepare_worktree(sys.argv[1])
        _validate_standard_descriptors(root)
        _close_inherited_descriptors()
        os.execv(UNSHARE, _namespace_launcher_args(_sandbox_args(root, target, sys.argv[3:])))
    except BoundaryError as error:
        print(f'gpu-denied runner: {error}', file=sys.stderr)
        return 69
    except OSError as error:
        print(f'gpu-denied runner: cannot enter boundary: {error}', file=sys.stderr)
        return 69
    return 70


if __name__ == '__main__':
    sys.exit(main())
