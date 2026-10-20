#!/usr/bin/env bash
# Build the live native pieces: the cubecl-stage0 descriptor compiler (release)
# and the CUDA driver launcher. Refuses unless the CubeCL checkout is exactly the
# audited pin with the full FP patch stack applied (see fp_hardening/patches.py
# for applying it). Extra arguments are passed to cargo.
set -euo pipefail
NATIVE="$(cd "$(dirname "$0")" && pwd)"
SOURCE="${CUBECL_SOURCE:-$NATIVE/../../.stage0-sources/cubecl}"
PIN=6bba5fdf04206d9c3710dfbc46aa9850450ab405

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
    echo "apply it with: python fp_hardening/patches.py" >&2
    exit 1
fi

cargo build --manifest-path "$NATIVE/Cargo.toml" --locked --release "$@"
cc -std=c11 -Wall -Wextra -Werror -fPIC -shared "$NATIVE/launcher.c" -ldl -o "$NATIVE/libcubecl_torch_launcher.so"
