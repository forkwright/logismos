#!/usr/bin/env bash
# Prove the GPU-denied runner blocks device and host-control interfaces.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
RUNNER="$ROOT/scripts/gpu-denied-runner.sh"

if PATH=/definitely-missing /usr/bin/bash "$RUNNER" -- /usr/bin/true >/dev/null 2>&1; then
    echo "gpu-denied witness: missing bwrap did not fail closed" >&2
    exit 1
fi

"$RUNNER" -- /bin/sh -ceu '
    test ! -e /dev/kfd
    test ! -e /dev/dri
    test ! -e /dev/vfio
    test ! -e /dev/mem
    test ! -e /dev/kmem
    test ! -e /dev/port
    test ! -e /sys/class/drm
    test ! -e /run/systemd/private
    printf "%s\\n" "gpu-denied CPU command: PASS"
'

exec 3<"$ROOT/Cargo.toml"
"$RUNNER" -- /bin/sh -ceu 'test ! -e /proc/self/fd/3'
exec 3>&-

"$RUNNER" -- /usr/bin/python3 - <<'PYTHON'
import errno
import fcntl
import os
import socket

device_paths = (
    "/dev/kfd",
    "/dev/dri/renderD128",
    "/dev/vfio/vfio",
    "/dev/mem",
    "/dev/kmem",
    "/dev/port",
)
control_sockets = (
    "/run/systemd/private",
    "/run/dbus/system_bus_socket",
    "/run/user/1000/bus",
    "/var/run/docker.sock",
    "/run/podman/podman.sock",
)

for path in device_paths:
    try:
        descriptor = os.open(path, os.O_RDWR | os.O_CLOEXEC)
    except OSError as error:
        if error.errno not in (errno.ENOENT, errno.ENOTDIR):
            raise AssertionError(f"device path {path} failed with {error}") from error
    else:
        try:
            fcntl.ioctl(descriptor, 0, 0)
        except OSError:
            pass
        else:
            raise AssertionError(f"device ioctl {path} unexpectedly succeeded")
        finally:
            os.close(descriptor)
        raise AssertionError(f"device path {path} unexpectedly opened")

for path in control_sockets:
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        connection.connect(path)
    except OSError as error:
        if error.errno not in (errno.ENOENT, errno.ENOTDIR):
            raise AssertionError(f"host control socket {path} failed with {error}") from error
    else:
        raise AssertionError(f"host control socket {path} unexpectedly connected")
    finally:
        connection.close()

if os.listdir("/sys"):
    raise AssertionError("sandbox /sys is not empty")

print("gpu-denied negative boundary witness: PASS")
PYTHON
