#!/usr/bin/env bash
# Build the live native pieces: the cubecl-stage0 descriptor compiler (release),
# the CUDA driver launcher, and the runtime bridge (CubeCL's CUDA client and
# CubeK in the torch process), from sources/cubecl and sources/cubek in this
# repository. Records the sources' state the binaries are built from
# (target/source-state.json, cubecl_torch/compiler/source.py), which provenance
# carries and the runtime holds the sources to. Extra arguments are passed to
# cargo.
set -euo pipefail
NATIVE="$(cd "$(dirname "$0")" && pwd)"
REPO="$NATIVE/../.."
source_py() { PYTHONPATH="$REPO" python3 -P -m cubecl_torch.compiler.source "$@"; }

source_py record "$NATIVE/target/source-state.json"
# The upstream revision the compiler reports in each kernel manifest.
CUBECL_TORCH_CUBECL_UPSTREAM="$(source_py upstream cubecl)"
export CUBECL_TORCH_CUBECL_UPSTREAM

cargo build --manifest-path "$NATIVE/Cargo.toml" --locked --release "$@"
cargo build --manifest-path "$NATIVE/bridge/Cargo.toml" --locked --release
cc -std=c11 -Wall -Wextra -Werror -fPIC -shared "$NATIVE/launcher.c" -ldl -o "$NATIVE/libcubecl_torch_launcher.so"
