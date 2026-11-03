#!/usr/bin/env bash
# Build the live native pieces: the cubecl-stage0 descriptor compiler (release),
# the CUDA driver launcher, and the runtime bridge (CubeCL's CUDA client and
# CubeK in the torch process). Refuses unless the CubeCL checkout is exactly the
# audited pin with the full patch stack (patches/series) applied and the CubeK
# checkout is at its pin. Extra arguments are passed to cargo.
set -euo pipefail
NATIVE="$(cd "$(dirname "$0")" && pwd)"
SOURCE="${CUBECL_SOURCE:-$NATIVE/../../.stage0-sources/cubecl}"

CUBEK="${CUBEK_SOURCE:-$(dirname "$SOURCE")/cubek}"

# The checks shared with the Python side (cubecl_torch/compiler/source.py): the
# CubeCL pin with patches/series applied in order must equal the working tree,
# and CubeK must be at its pin with no local changes.
PYTHONPATH="$NATIVE/../.." python3 -P -m cubecl_torch.compiler.source "$SOURCE" "$CUBEK"

cargo build --manifest-path "$NATIVE/Cargo.toml" --locked --release "$@"
cargo build --manifest-path "$NATIVE/bridge/Cargo.toml" --locked --release
cc -std=c11 -Wall -Wextra -Werror -fPIC -shared "$NATIVE/launcher.c" -ldl -o "$NATIVE/libcubecl_torch_launcher.so"
