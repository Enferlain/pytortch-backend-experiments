#!/usr/bin/env bash
# Build the live native pieces: the cubecl-stage0 descriptor compiler (release)
# and the CUDA driver launcher. Refuses unless the CubeCL checkout is exactly the
# audited pin with the full FP patch stack (cubecl-fp.patch) applied. Extra
# arguments are passed to cargo.
set -euo pipefail
NATIVE="$(cd "$(dirname "$0")" && pwd)"
SOURCE="${CUBECL_SOURCE:-$NATIVE/../../.stage0-sources/cubecl}"
PIN=907676dca8c4cd407237892cc2915ff6aadc4f1e

if [[ "$(git -C "$SOURCE" rev-parse HEAD)" != "$PIN" ]]; then
    echo "CubeCL checkout $SOURCE is not at the audited pin $PIN" >&2
    exit 1
fi
if [[ -n "$(git -C "$SOURCE" diff --cached)" || -n "$(git -C "$SOURCE" ls-files --others --exclude-standard)" ]]; then
    echo "CubeCL checkout has staged or untracked files; inspect them before building" >&2
    exit 1
fi
if ! cmp -s <(git -C "$SOURCE" diff) "$NATIVE/cubecl-fp.patch"; then
    echo "CubeCL working tree differs from the audited FP patch stack ($NATIVE/cubecl-fp.patch)" >&2
    echo "apply it to a clean checkout at the pin with: git -C \"$SOURCE\" apply \"$NATIVE/cubecl-fp.patch\"" >&2
    exit 1
fi

cargo build --manifest-path "$NATIVE/Cargo.toml" --locked --release "$@"
cc -std=c11 -Wall -Wextra -Werror -fPIC -shared "$NATIVE/launcher.c" -ldl -o "$NATIVE/libcubecl_torch_launcher.so"
