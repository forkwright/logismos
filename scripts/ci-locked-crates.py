#!/usr/bin/python3
"""Reject non-crates.io lockfile sources before trusted Cargo fetching."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

CRATES_IO = 'registry+https://github.com/rust-lang/crates.io-index'


def main() -> int:
    if len(sys.argv) != 2:
        print('usage: ci-locked-crates.py CARGO_LOCK', file=sys.stderr)
        return 64
    try:
        lock = tomllib.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    except (OSError, tomllib.TOMLDecodeError) as error:
        print(f'cannot read Cargo.lock: {error}', file=sys.stderr)
        return 69
    packages = lock.get('package')
    if not isinstance(packages, list) or any(not isinstance(package, dict) for package in packages):
        print('Cargo.lock contains an invalid package array', file=sys.stderr)
        return 69
    unexpected = sorted({
        source if isinstance(source, str) else repr(source)
        for package in packages
        if 'source' in package
        and (source := package['source']) != CRATES_IO
    })
    if unexpected:
        print(f'Cargo.lock contains non-crates.io sources: {unexpected}', file=sys.stderr)
        return 69
    return 0


if __name__ == '__main__':
    sys.exit(main())
