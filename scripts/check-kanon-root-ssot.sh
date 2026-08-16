#!/usr/bin/env bash
# WHY: CLAUDE.md carries the single authoritative statement of how to resolve
# the kanon checkout root on this box (#65, #80-followup). Every other
# pointer (AGENTS.md, README.md, llms.txt, crate doc comments) must point AT
# that statement, never restate the mechanism — a restated copy is exactly
# how the stale-claim class #65 was filed to close regenerates: one copy
# gets corrected, siblings keep asserting the old (or a newly-broken)
# mechanism, and a reader trusts whichever copy they hit first.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

mechanism='kanon locate kanon-repo'
authoritative='CLAUDE.md'

violations=$(git grep -l -F "$mechanism" -- . ":!${authoritative}" ":!scripts/check-kanon-root-ssot.sh" || true)

if [[ -n "$violations" ]]; then
    echo "SSOT violation: '${mechanism}' restated outside ${authoritative} in:" >&2
    echo "$violations" >&2
    echo "Point at ${authoritative}'s checkout-root paragraph instead of repeating the mechanism." >&2
    exit 1
fi

if ! git grep -q -F "$mechanism" -- "$authoritative"; then
    echo "SSOT violation: ${authoritative} no longer states the kanon-root resolution mechanism at all." >&2
    exit 1
fi

echo "ok  kanon-root SSOT: '${mechanism}' appears only in ${authoritative}"
