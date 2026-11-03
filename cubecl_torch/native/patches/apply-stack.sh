#!/usr/bin/env bash
# Put the CubeCL and CubeK checkouts in their audited state: each at its pin in
# runtime/native.py with its stack applied in series order (patches/series for
# CubeCL, patches/cubek/series for CubeK). Discards both checkouts' local
# changes and untracked files (not ignored ones), which the stacks hold; then
# runs the audit.
set -euo pipefail
PATCHES="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$PATCHES/../../.." && pwd)"
SOURCE="${CUBECL_SOURCE:-$REPO/.stage0-sources/cubecl}"
CUBEK="${CUBEK_SOURCE:-$(dirname "$SOURCE")/cubek}"
pin() { PYTHONPATH="$REPO" python3 -P -c "from cubecl_torch.runtime import native; print(native.$1)"; }

# apply CHECKOUT PIN STACK_DIR
apply() {
    git -C "$1" reset -q --hard
    git -C "$1" clean -fdq
    git -C "$1" checkout -q "$2"
    while read -r patch || [[ -n "$patch" ]]; do
        [[ -n "${patch// /}" ]] && git -C "$1" apply "$3/${patch// /}"
    done < "$3/series"
}
apply "$SOURCE" "$(pin CUBECL_PIN)" "$PATCHES"
apply "$CUBEK" "$(pin CUBEK_PIN)" "$PATCHES/cubek"
PYTHONPATH="$REPO" python3 -P -m cubecl_torch.compiler.source "$SOURCE" "$CUBEK"
