# GPU-denied test runner

Run non-hardware tests through the required boundary:

```bash
scripts/gpu-denied-runner.sh -- /bin/sh -ceu \
  'LOGISMOS_HIP_BUILD=cpu-only cargo test -p kernels --lib'
scripts/gpu-denied-witness.sh
scripts/check-hip-build-modes.sh
```

The runner is intentionally non-interactive. Its standard descriptors must be
pipes, `/dev/null`, or safe regular files as described below; redirect through
a pipe when launching it from a terminal.

## Enforced boundary

The runner requires the fixed system paths `/usr/bin/bash`, `/usr/bin/bwrap`,
`/usr/bin/setpriv`, and `/usr/bin/python3`. Bash privileged mode and Python
isolated mode prevent caller-supplied startup hooks from running before the
supervisor. The runner refuses to run when a prerequisite is unavailable,
refuses a root invocation, and never falls back to running the command
directly. The command runs with private user, mount, PID, IPC, UTS, cgroup, and
network namespaces; all capabilities dropped; nested user namespaces
disabled; and `no_new_privs` set.

The host or CI executor must run the wrapper as a non-root account and permit
unprivileged user namespaces. A container seccomp profile or host policy that
blocks namespace creation is a hard prerequisite failure, not a reason to run
the requested command outside the boundary.

The mount namespace contains:

- a synthetic `/dev`, empty `/sys` and `/run`, private `/proc` and `/tmp`, and
  an `/etc` containing at most the host's read-only dynamic-loader cache;
- the host `/usr` and an optional system `/opt/rocm`, read-only;
- the worktree read-only, with only its real, non-mounted `target/` directory
  rebound writable; and
- optional account Rust toolchain, registry, and Git caches read-only at fixed
  sandbox paths.

The toolchain sources come from the local account record, not `HOME`,
`CARGO_HOME`, `RUSTUP_HOME`, or `PATH`. Sources are canonicalized and rejected
if they resolve into `/dev`, `/etc`, `/proc`, `/run`, or `/sys`. The mounted
account toolchain and caches are part of the trusted host input. `HOME` and
`CARGO_HOME` are private tmpfs directories. Cargo is forced offline and its
target is fixed to the worktree's `target/`.

Before Bubblewrap starts, the supervisor enumerates `/proc/self/fd` and closes
every inherited descriptor above standard I/O. This does not depend on the
soft descriptor limit. Standard input, output, and error accept only pipes,
`/dev/null`, or regular files inside the worktree. Sockets, unlinked or
multiply-linked regular files, block devices, pseudo-terminals, and all other
character devices fail closed. Standard output and error remain deliberate
byte-egress channels to the invoking process.

The witness uses synthetic inherited descriptors, startup hooks, symlink and
hard-link targets, Unix and TCP listeners, and a pseudo-terminal. It never
opens, probes, or issues an `ioctl` to a real accelerator device. It also
proves that an existing Unix socket in the writable target makes the runner
fail closed.

## Build modes and Clang discovery

`contracts/gpu-target.txt` is the canonical HIP target (`gfx1100`). The kernel
build script reads it for every `--offload-arch` flag. Build mode is selected
with `LOGISMOS_HIP_BUILD`:

- `cpu-only` emits `logismos_no_gpu_kernels` and omits the HIP archive.
- `required` invokes `hipcc`; a missing compiler or compiler error fails the
  build.

An unset mode selects `required`. `LOGISMOS_SKIP_HIP_BUILD` is retired and
fails with its explicit replacement. The build-mode witness proves CPU mode
and required-mode missing-compiler failure inside the boundary. It does not
invoke a HIP compiler or device runtime.

The runner discovers libclang without executing an ambient program. It checks
the fixed system layouts `/usr/lib64/rocm/llvm/lib` and
`/opt/rocm/llvm/lib`, sets `LIBCLANG_PATH` when libclang exists, and selects the
highest numeric `clang/<version>` resource directory for
`BINDGEN_EXTRA_CLANG_ARGS`. A command can override that choice explicitly when
the required resource directory is already within a mounted system tree:

```bash
scripts/gpu-denied-runner.sh -- /bin/sh -ceu \
  'BINDGEN_EXTRA_CLANG_ARGS="-resource-dir=/usr/lib64/rocm/llvm/lib/clang/20" \
   LOGISMOS_HIP_BUILD=cpu-only cargo check --workspace'
```

`LOGISMOS_HIP_BUILD=required` with a real `hipcc` is compile-only evidence. It
does not execute a HIP kernel or prove hardware behavior. Real GPU execution
is a separate hardware lane and is never enabled by this runner or its default
CI job.

## Threat-model limits

The boundary contains the requested command and its descendants as an
ordinary unprivileged workload. It trusts the host kernel, Bubblewrap,
`setpriv`, the runner and supervisor, the mounted system/toolchain files, and
the process that invokes the runner. It is not a boundary against a host
administrator, a compromised kernel or sandbox binary, or a same-UID host
process racing trusted inputs before Bubblewrap enters the namespaces.

It does not provide CPU, memory, process-count, wall-time, or target-disk
quotas. The writable `target/` remains on the host and must be treated as
untrusted build output after a run. Source confidentiality is not a goal: the
command can read the worktree and mounted toolchain. Tests that require host
networking, arbitrary host files, service sockets, interactive socket-based
standard I/O, or real hardware belong in a separately authorized lane.
