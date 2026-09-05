#!/usr/bin/python3
"""Validate the repository Rust pin and publish it to GitHub Actions."""

from __future__ import annotations

import os
import re
import sys
import tomllib
from pathlib import Path


def main() -> int:
    output_path = os.environ.get('GITHUB_OUTPUT')
    if output_path is None:
        print('ci-rust-toolchain requires GITHUB_OUTPUT', file=sys.stderr)
        return 69
    config = tomllib.loads(Path('rust-toolchain.toml').read_text(encoding='utf-8'))
    channel = config['toolchain']['channel']
    components = config['toolchain'].get('components', [])
    if re.fullmatch(r'[0-9]+\.[0-9]+\.[0-9]+', channel) is None:
        print('rust-toolchain.toml must pin an exact stable release', file=sys.stderr)
        return 69
    if any(re.fullmatch(r'[a-z0-9_-]+', item) is None for item in components):
        print('rust-toolchain.toml contains an invalid component', file=sys.stderr)
        return 69
    with open(output_path, 'a', encoding='utf-8') as output:
        print(f'channel={channel}', file=output)
        print(f'components={",".join(components)}', file=output)
    return 0


if __name__ == '__main__':
    sys.exit(main())
