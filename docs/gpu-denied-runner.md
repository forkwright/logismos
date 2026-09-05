# GPU-denied test runner

Run headless tests through the required boundary:

```bash
scripts/gpu-denied-runner.sh -- /bin/sh -ceu 'LOGISMOS_HIP_BUILD=cpu-only cargo test -p kernels --lib'
scripts/gpu-denied-witness.sh
scripts/check-hip-build-modes.sh
```

The runner requires `bwrap`, `setpriv`, and Python 3. It refuses to run when
one is unavailable; it never executes the requested command outside the
boundary. It enters private user, mount, PID, IPC, UTS, cgroup, and network
namespaces, closes inherited descriptors other than standard I/O, drops
capabilities, and sets `no_new_privs`. The command receives only a writable
worktree, read-only system/toolchain directories, a synthetic `/dev`, an empty
`/sys` and `/run`, a private `/proc`, and a cleared environment. It sets
`CARGO_TARGET_DIR` to that worktree's `target/`, so a host-wide target setting
cannot escape the boundary.

The negative witness attempts AMD and raw-memory device paths plus host control
sockets. Its proof is that no handle can be opened or connected, not that every
possible `ioctl` is filtered. The CPU command in that witness proves the
boundary still executes ordinary CPU processes.

`contracts/gpu-target.txt` is the canonical HIP target (`gfx1100`). The
kernel build script reads it for every `--offload-arch` flag. Build mode is
selected with `LOGISMOS_HIP_BUILD`:

- `cpu-only` emits `logismos_no_gpu_kernels` and deliberately omits the HIP
  archive.
- `required` invokes `hipcc`; a missing compiler or a compiler error fails the
  build.

An unset mode selects `required`. `LOGISMOS_SKIP_HIP_BUILD` is retired and fails
with its explicit replacement. The build-mode witness proves the CPU mode and
the required-mode missing-compiler failure under the boundary; it does not
claim HIP compilation succeeded.

`LOGISMOS_HIP_BUILD=required` with a real `hipcc` is compile-only evidence. It
does not execute a HIP kernel or prove hardware behavior. Real GPU execution is
an explicit, separate hardware lane and is never enabled by this runner or its
default CI job.

Run compile-only HIP proof through the same device-denied boundary with:

```bash
scripts/gpu-denied-runner.sh -- /bin/sh -ceu 'LOGISMOS_HIP_BUILD=required cargo build -p kernels'
```
