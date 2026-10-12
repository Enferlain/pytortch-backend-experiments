#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p .stage0-sources
if [[ ! -d .stage0-sources/cubecl/.git ]]; then
    git clone --filter=blob:none --no-checkout https://github.com/tracel-ai/cubecl.git .stage0-sources/cubecl
    git -C .stage0-sources/cubecl checkout --detach 6bba5fdf04206d9c3710dfbc46aa9850450ab405
fi
if [[ "$(git -C .stage0-sources/cubecl rev-parse HEAD)" != 6bba5fdf04206d9c3710dfbc46aa9850450ab405 ]]; then
    echo "CubeCL source is not at the experiment pin" >&2
    exit 1
fi
if [[ -n "$(git -C .stage0-sources/cubecl status --porcelain)" ]]; then
    echo "CubeCL source has local changes; use a clean pinned checkout" >&2
    exit 1
fi
cargo build --manifest-path stage0/Cargo.toml --locked "${@}"
cc -std=c11 -Wall -Wextra -Werror -fPIC -shared stage0/launcher.c -ldl -o stage0/libstage0_launcher.so
