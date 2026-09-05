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
`/usr/bin/unshare`, `/usr/bin/setpriv`, and `/usr/bin/python3`. Bash privileged mode and Python
isolated mode prevent their caller-supplied language startup hooks from
running before the supervisor. The runner refuses to run when a prerequisite
is unavailable, refuses a root invocation, and never falls back to running the
command directly. The command runs with private user, mount, PID, IPC, UTS,
cgroup, and network namespaces; all capabilities dropped; nested user
namespaces disabled; and `no_new_privs` set.

The host or CI executor must run the wrapper as a non-root account and permit
unprivileged user namespaces. The fixed `unshare` launcher first creates a
private user and network namespace without assigning a loopback address;
Bubblewrap retains that private network namespace while creating the remaining
namespaces. This avoids requiring Bubblewrap to configure loopback, while the
network remains isolated and loopback-down. A container seccomp profile or host
policy that blocks this namespace creation is a hard prerequisite failure, not
a reason to run the requested command outside the boundary.

GitHub-hosted Ubuntu runners may require the CI-only
`scripts/ci-hosted-userns.sh` admission helper before either boundary job. It
is guarded to those ephemeral hosted runners, reads the exact AppArmor
user-namespace sysctl, accepts only `0` or `1`, and verifies `0` before the
boundary starts. Its environment checks prevent accidental invocation on a
development host; they are not a security boundary against a deliberate host
operator. The workflow-scoped change does not alter the runner or grant the
sandbox a capability. A failed admission remains a hard failure.

The pinned ROCm container job provisions a dedicated non-root account before
entering the boundary. Its networked `cargo fetch` is a trusted provisioning
step: it runs from `/tmp`, accepts only crates.io sources from the lockfile,
uses an empty environment, and does not execute build scripts. Compilation and
object inspection begin only after the GPU-denied runner succeeds. The job
uses the exact release in `rust-toolchain.toml`, not an ambient latest-stable
toolchain.

The HIP job starts its explicit pinned container without `--privileged`,
additional device mounts, or added capabilities. Its root provisioning process
therefore retains Docker's default capability set; only the later `gpu-ci`
invocation is the sandboxed workload, with inherited and ambient capabilities
dropped before it enters the denied runner.

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
`/dev/null`, or regular files inside the worktree. Sockets, unlinked regular
files, block devices, pseudo-terminals, and all other character devices fail
closed. Source-tree hard links also fail closed. Target hard links are allowed
only when an inode census proves that every link is contained in the writable
target; this preserves ordinary Cargo artifact reuse without permitting a
writable alias to a file elsewhere on the host. Standard output and error
remain deliberate byte-egress channels to the invoking process.

The witness proves the network namespace inode differs from the host, loopback
is down, and a host TCP listener is unreachable. It also uses synthetic
inherited descriptors, startup hooks, symlink and hard-link targets, Unix and
TCP listeners, and a pseudo-terminal. It never
opens, probes, or issues an `ioctl` to a real accelerator device. It also
proves that an existing Unix socket in the writable target makes the runner
fail closed.

## Build modes and Clang discovery

`contracts/gpu-target.txt` is the canonical HIP target (`gfx1100`). The kernel
build script reads it for every `--offload-arch` flag. Build mode is selected
with `LOGISMOS_HIP_BUILD`:

- `cpu-only` emits `logismos_no_gpu_kernels` and omits the HIP archive.
- `required` invokes `hipcc`; a missing compiler, compiler error, or empty
  HIP/CPP source discovery fails the build. Only `cpu-only` may omit the
  kernel archive.

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

Run `scripts/hip-code-object-witness.sh` for the required-HIP lane. This public
entry point always enters the denied runner, then compiles the in-tree HIP
sources and inspects the archive's offload bundles and exact target metadata.
After that positive pass, it copies each HIP source to a temporary fixture,
injects a syntax error, and verifies the compiled `build.rs` path rejects the
source-specific parser diagnostic. It rejects missing objects and malformed or
wrong-architecture metadata.
`target/hip-code-object-witness/receipt.txt` records the observed compiler,
resource directory, archive membership, code-object count, and syntax-rejection
count. The receipt describes that invocation; it is neither a hardware result
nor evidence that another toolchain or revision passed.

## Threat-model limits

The boundary contains the requested command and its descendants as an
ordinary unprivileged workload. It trusts the host kernel, Bubblewrap,
`setpriv`, the runner and supervisor, the mounted system/toolchain files, and
the process that invokes the runner. That trusted invoker must also supply a
safe dynamic-loader environment: the loader processes variables such as
`LD_PRELOAD` before a shell script can clear them. It is not a boundary against
a host administrator, a compromised kernel or sandbox binary, or a same-UID
host process racing trusted inputs before Bubblewrap enters the namespaces.

It does not provide CPU, memory, process-count, wall-time, or target-disk
quotas. The writable `target/` remains on the host and must be treated as
untrusted build output after a run. Source confidentiality is not a goal: the
command can read the worktree and mounted toolchain. Tests that require host
networking, arbitrary host files, service sockets, interactive socket-based
standard I/O, or real hardware belong in a separately authorized lane.
