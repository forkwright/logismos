#!/usr/bin/python3
"""Enter the GPU-denied Bubblewrap boundary after validating host inputs."""

from __future__ import annotations

import errno
import os
import re
import stat
import sys
from pathlib import Path
from typing import NamedTuple

BWRAP = Path('/usr/bin/bwrap')
SETPRIV = Path('/usr/bin/setpriv')
UNSHARE = Path('/usr/bin/unshare')
SANDBOX_CARGO_HOME = Path('/tmp/cargo')
SANDBOX_HOME = Path('/tmp/home')
SANDBOX_RUST_ROOT = Path('/opt/gpu-denied')
SANDBOX_READ_ONLY_INPUT_DIRECTORY = Path('/mnt/gpu-denied-input')
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


class ReadOnlyInputTarget(NamedTuple):
    """One fixed synthetic destination and its sole environment variable."""

    sandbox_path: Path
    environment: str


class ReadOnlyInputRequest(NamedTuple):
    """One unvalidated caller path assigned to a fixed input role."""

    source_argument: str
    target: ReadOnlyInputTarget


class ValidatedReadOnlyInput(NamedTuple):
    """One canonical host file that passed the shared input policy."""

    source: Path
    identity: tuple[int, int]


class AdmittedReadOnlyInput(NamedTuple):
    """One immutable admission value consumed by environment and mount setup."""

    source: Path
    identity: tuple[int, int]
    target: ReadOnlyInputTarget


LEGACY_INPUT_TARGET = ReadOnlyInputTarget(
    SANDBOX_READ_ONLY_INPUT_DIRECTORY / 'artifact', 'LOGISMOS_GPU_DENIED_INPUT'
)
MODEL_INPUT_TARGET = ReadOnlyInputTarget(
    SANDBOX_READ_ONLY_INPUT_DIRECTORY / 'model', 'LOGISMOS_GPU_DENIED_MODEL'
)
TOKENIZER_INPUT_TARGET = ReadOnlyInputTarget(
    SANDBOX_READ_ONLY_INPUT_DIRECTORY / 'tokenizer', 'LOGISMOS_GPU_DENIED_TOKENIZER'
)


def _decode_mount_path(value: str) -> Path:
    return Path(MOUNT_ESCAPE.sub(lambda match: chr(int(match.group(1), 8)), value))


def _host_mount_points() -> list[Path]:
    try:
        records = Path('/proc/self/mountinfo').read_text(encoding='utf-8').splitlines()
    except OSError as error:
        raise BoundaryError(f'cannot inspect host mount topology: {error}') from error
    mount_points: list[Path] = []
    for record in records:
        fields = record.split()
        if len(fields) < 5:
            raise BoundaryError('host mount topology contains a malformed record')
        mount_points.append(_decode_mount_path(fields[4]))
    return mount_points


def _reject_nested_mounts(source: Path) -> None:
    for mount_point in _host_mount_points():
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


def _prepare_read_only_input(
    input_argument: str, root: Path, target: Path
) -> ValidatedReadOnlyInput:
    input_path = Path(input_argument)
    if not input_path.is_absolute():
        raise BoundaryError('read-only input path must be canonical and absolute')
    try:
        resolved = input_path.resolve(strict=True)
    except OSError as error:
        raise BoundaryError('cannot resolve read-only input path') from error
    # `Path` deliberately normalizes `.` and repeated separators.  Compare the
    # original argv spelling too: otherwise `/file//name` would pass despite
    # not being the one canonical pathname the caller supplied for review.
    if input_argument != str(resolved):
        raise BoundaryError('read-only input path must be canonical and contain no symlinks')
    if resolved == root or resolved.is_relative_to(root):
        raise BoundaryError('read-only input must be outside the worktree and writable target')
    if any(
        resolved == blocked or resolved.is_relative_to(blocked)
        for blocked in BLOCKED_MOUNT_ROOTS
    ):
        raise BoundaryError('read-only input is under a sensitive host root')
    try:
        metadata = resolved.lstat()
    except OSError as error:
        raise BoundaryError('cannot inspect read-only input') from error
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise BoundaryError('read-only input must be a single-link regular file')
    _reject_read_only_input_tree_alias(metadata, root)
    if resolved in _host_mount_points():
        raise BoundaryError('read-only input must not be a host mount point')
    if not os.access(resolved, os.R_OK):
        raise BoundaryError('read-only input is not readable by the runner account')
    return ValidatedReadOnlyInput(resolved, (metadata.st_dev, metadata.st_ino))


def _admit_read_only_inputs(
    requests: tuple[ReadOnlyInputRequest, ...], root: Path, target: Path
) -> tuple[AdmittedReadOnlyInput, ...]:
    admitted: list[AdmittedReadOnlyInput] = []
    for request in requests:
        validated = _prepare_read_only_input(request.source_argument, root, target)
        admitted.append(
            AdmittedReadOnlyInput(
                source=validated.source,
                identity=validated.identity,
                target=request.target,
            )
        )
    if len({input_file.identity for input_file in admitted}) != len(admitted):
        raise BoundaryError('read-only model and tokenizer inputs must name distinct files')
    return tuple(admitted)


def _reject_read_only_input_tree_alias(input_metadata: os.stat_result, root: Path) -> None:
    """Reject a bind-directory spelling of an inode in the protected worktree.

    A file bind mount does not increment `st_nlink`; an external pathname can
    therefore name a writable `target/` inode despite the input's single-link
    requirement.  Compare inode identity while walking the already protected
    tree without following any symlinks.
    """

    def reject_walk_error(error: OSError) -> None:
        raise BoundaryError('cannot inspect worktree for a read-only input alias') from error

    identity = (input_metadata.st_dev, input_metadata.st_ino)
    for current, directories, files in os.walk(
        root, topdown=True, onerror=reject_walk_error, followlinks=False
    ):
        current_path = Path(current)
        for name in (*directories, *files):
            try:
                candidate = (current_path / name).lstat()
            except OSError as error:
                raise BoundaryError('cannot inspect worktree for a read-only input alias') from error
            if stat.S_ISREG(candidate.st_mode) and (candidate.st_dev, candidate.st_ino) == identity:
                raise BoundaryError('read-only input aliases the protected worktree')


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


def _compiler_alias_mount_args() -> list[str]:
    compiler = Path('/usr/bin/cc')
    try:
        target = os.readlink(compiler)
    except OSError as error:
        if error.errno in (errno.EINVAL, errno.ENOENT):
            return []
        raise BoundaryError(f'cannot inspect the system C compiler link: {error}') from error
    alias = Path(target)
    expected_alias = Path('/etc/alternatives/cc')
    if alias != expected_alias:
        return []
    try:
        resolved_compiler = expected_alias.resolve(strict=True)
    except OSError as error:
        raise BoundaryError(f'cannot resolve the system C compiler alias: {error}') from error
    if (
        not resolved_compiler.is_file()
        or not os.access(resolved_compiler, os.X_OK)
        or not resolved_compiler.is_relative_to(Path('/usr'))
    ):
        raise BoundaryError('the system C compiler alias does not resolve under /usr')
    # WHY: Ubuntu's /usr/bin/cc may use precisely this root-owned alternative.
    # Recreate the link to its validated /usr target instead of bind-mounting
    # it: binding dereferences the link and makes GCC believe its executable
    # lives under /etc, breaking its self-relative plugin lookup.  This is
    # optional support for compiler commands, not a requirement for pure ones.
    return [
        '--dir',
        str(expected_alias.parent),
        '--symlink',
        str(resolved_compiler),
        str(expected_alias),
    ]


def _sandbox_args(
    root: Path,
    target: Path,
    read_only_inputs: tuple[AdmittedReadOnlyInput, ...],
    command: list[str],
) -> list[str]:
    toolchain_args, toolchain_environment = _toolchain_mount_args()
    compiler_alias_args = _compiler_alias_mount_args()
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
    input_mount_args: list[str] = []
    if read_only_inputs:
        input_mount_args.extend(
            (
                '--dir',
                str(SANDBOX_READ_ONLY_INPUT_DIRECTORY.parent),
                '--dir',
                str(SANDBOX_READ_ONLY_INPUT_DIRECTORY),
            )
        )
        for input_file in read_only_inputs:
            environment.append(
                (input_file.target.environment, str(input_file.target.sandbox_path))
            )
            input_mount_args.extend(
                ('--ro-bind', str(input_file.source), str(input_file.target.sandbox_path))
            )
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
            *compiler_alias_args,
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
            *input_mount_args,
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


def _parse_input_requests(
    arguments: list[str],
) -> tuple[tuple[ReadOnlyInputRequest, ...], list[str]] | None:
    requests_by_flag: dict[str, ReadOnlyInputRequest] = {}
    input_targets = {
        '--ro-input-file': LEGACY_INPUT_TARGET,
        '--ro-model-file': MODEL_INPUT_TARGET,
        '--ro-tokenizer-file': TOKENIZER_INPUT_TARGET,
    }
    next_argument = 0
    while next_argument < len(arguments) and arguments[next_argument] != '--':
        flag = arguments[next_argument]
        target = input_targets.get(flag)
        if target is None or next_argument + 1 >= len(arguments):
            return None
        source_argument = arguments[next_argument + 1]
        if not source_argument or flag in requests_by_flag:
            return None
        requests_by_flag[flag] = ReadOnlyInputRequest(source_argument, target)
        next_argument += 2
    if next_argument >= len(arguments) or arguments[next_argument] != '--':
        return None
    command = arguments[next_argument + 1 :]
    if not command:
        return None

    has_legacy = '--ro-input-file' in requests_by_flag
    has_pair_member = (
        '--ro-model-file' in requests_by_flag or '--ro-tokenizer-file' in requests_by_flag
    )
    if has_legacy and has_pair_member:
        return None
    if has_pair_member and {
        '--ro-model-file',
        '--ro-tokenizer-file',
    } != requests_by_flag.keys():
        return None

    return tuple(requests_by_flag.values()), command


def main() -> int:
    parsed = _parse_input_requests(sys.argv[2:])
    if parsed is None:
        print(
            'usage: gpu-denied-exec.py ROOT [--ro-input-file FILE | '
            '--ro-model-file FILE --ro-tokenizer-file FILE] -- COMMAND [ARG...]',
            file=sys.stderr,
        )
        return 64
    root_argument = sys.argv[1]
    requests, command = parsed
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
        root, target = _prepare_worktree(root_argument)
        read_only_inputs = _admit_read_only_inputs(requests, root, target)
        _validate_standard_descriptors(root)
        _close_inherited_descriptors()
        os.execv(
            UNSHARE,
            _namespace_launcher_args(
                _sandbox_args(root, target, read_only_inputs, command)
            ),
        )
    except BoundaryError as error:
        print(f'gpu-denied runner: {error}', file=sys.stderr)
        return 69
    except OSError as error:
        print(f'gpu-denied runner: cannot enter boundary: {error}', file=sys.stderr)
        return 69
    return 70


if __name__ == '__main__':
    sys.exit(main())
