# Official ROCm 7.1.1 complete image, pinned by its linux/amd64 manifest digest.
FROM rocm/dev-ubuntu-24.04@sha256:c6648f6a60470959f5f9c653ce8397d72fc0adda455942b265a5f973c9ee5891

# WHY: This fixed unprivileged identity owns only the hosted job's target/.
# The build context is empty, so no workspace bytes are available during build.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends bubblewrap python3 util-linux \
    && groupadd --gid 10001 gpu-ci \
    && useradd --create-home --uid 10001 --gid 10001 --shell /bin/bash gpu-ci \
    && rm -rf /var/lib/apt/lists/*

USER gpu-ci
WORKDIR /workspace
