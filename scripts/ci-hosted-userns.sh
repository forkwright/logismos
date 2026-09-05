#!/usr/bin/bash -p
# WHY: Ubuntu hosted runners can mediate unprivileged user namespaces through
# AppArmor. This is an ephemeral executor admission step, never a runner bypass.
set -euo pipefail
PATH=/usr/sbin:/usr/bin:/bin
unset CDPATH LD_LIBRARY_PATH LD_PRELOAD

if [[ "${GITHUB_ACTIONS:-}" != true || "${RUNNER_ENVIRONMENT:-}" != github-hosted ]]; then
    builtin printf '%s\n' \
        'hosted user-namespace admission is restricted to GitHub-hosted CI runners' >&2
    exit 69
fi

readonly APPARMOR_USERNS_KEY=kernel.apparmor_restrict_unprivileged_userns
if ! initial_value=$(/usr/sbin/sysctl -n "$APPARMOR_USERNS_KEY"); then
    builtin printf '%s\n' \
        'hosted user-namespace admission cannot read the AppArmor user-namespace setting' >&2
    exit 69
fi
case "$initial_value" in
    0|1)
        builtin printf 'hosted user-namespace admission initial %s=%s\n' \
            "$APPARMOR_USERNS_KEY" "$initial_value"
        ;;
    *)
        builtin printf 'hosted user-namespace admission rejected unexpected %s value: %s\n' \
            "$APPARMOR_USERNS_KEY" "$initial_value" >&2
        exit 69
        ;;
esac

if [[ "$initial_value" == 1 ]]; then
    /usr/bin/sudo /usr/sbin/sysctl -w "$APPARMOR_USERNS_KEY=0" >/dev/null
fi

final_value=$(/usr/sbin/sysctl -n "$APPARMOR_USERNS_KEY")
if [[ "$final_value" != 0 ]]; then
    builtin printf '%s\n' \
        'hosted user-namespace admission could not enable the required executor prerequisite' >&2
    exit 69
fi
builtin printf 'hosted user-namespace admission verified %s=%s\n' \
    "$APPARMOR_USERNS_KEY" "$final_value"
