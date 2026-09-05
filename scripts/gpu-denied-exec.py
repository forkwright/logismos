#!/usr/bin/env python3
"""Close inherited descriptors, then enter the GPU-denied Bubblewrap boundary."""

import os
import resource
import sys


def main() -> None:
    bwrap, setpriv, root, sandbox_home, *remaining = sys.argv[1:]
    separator = remaining.index("--")
    mount_args = remaining[:separator]
    command = remaining[separator + 1 :]

    limit = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
    if limit == resource.RLIM_INFINITY:
        limit = 1_048_576
    os.closerange(3, int(limit))

    args = [
        bwrap,
        "--unshare-all",
        "--unshare-user",
        "--die-with-parent",
        "--new-session",
        "--disable-userns",
        "--assert-userns-disabled",
        "--cap-drop",
        "ALL",
        "--clearenv",
        "--setenv",
        "HOME",
        sandbox_home,
        "--setenv",
        "USER",
        "gpu-denied",
        "--setenv",
        "LOGNAME",
        "gpu-denied",
        "--setenv",
        "PATH",
        f"{sandbox_home}/.cargo/bin:/usr/local/bin:/usr/bin:/bin",
        "--setenv",
        "TMPDIR",
        "/tmp",
        "--setenv",
        "CARGO_TARGET_DIR",
        f"{root}/target",
        "--setenv",
        "LIBCLANG_PATH",
        "/usr/lib64/rocm/llvm/lib",
        *mount_args,
        "--chdir",
        root,
        "--",
        setpriv,
        "--no-new-privs",
        "--",
        *command,
    ]
    os.execv(bwrap, args)


if __name__ == "__main__":
    main()
