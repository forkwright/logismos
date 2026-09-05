# Official ROCm 6.4.4 development image, pinned by its linux/amd64 manifest digest.
FROM rocm/dev-ubuntu-24.04@sha256:31418ac10a3769a71eaef330c07280d1d999d7074621339b8f93c484c35f6078

# WHY: This fixed unprivileged identity owns only the hosted job's target/.
# The build context is empty, so no workspace bytes are available during build.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends bubblewrap python3 util-linux \
    && groupadd --gid 10001 gpu-ci \
    && useradd --create-home --uid 10001 --gid 10001 --shell /bin/bash gpu-ci \
    && rm -rf /var/lib/apt/lists/*

USER gpu-ci
WORKDIR /workspace
