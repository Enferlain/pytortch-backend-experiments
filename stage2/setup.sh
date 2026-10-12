#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -d .stage0-sources/cubecl/.git ]]; then
    bash stage0/setup.sh
fi
python fp_hardening/patches.py
cargo build --manifest-path stage0/Cargo.toml --locked --features llvm,hip-probe
cc -std=c11 -Wall -Wextra -Werror -fPIC -shared stage0/launcher.c -ldl -o stage0/libstage0_launcher.so
