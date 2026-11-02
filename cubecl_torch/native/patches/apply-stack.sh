#!/usr/bin/env bash
# Put the CubeCL checkout in its audited state: the pin in runtime/native.py with
# patches/series applied in order. Discards the checkout's local changes and
# untracked files (not ignored ones), which the stack holds; then runs the audit.
set -euo pipefail
PATCHES="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$PATCHES/../../.." && pwd)"
SOURCE="${CUBECL_SOURCE:-$REPO/.stage0-sources/cubecl}"
CUBEK="${CUBEK_SOURCE:-$(dirname "$SOURCE")/cubek}"
PIN="$(PYTHONPATH="$REPO" python3 -P -c 'from cubecl_torch.runtime import native; print(native.CUBECL_PIN)')"

git -C "$SOURCE" reset -q --hard
git -C "$SOURCE" clean -fdq
git -C "$SOURCE" checkout -q "$PIN"
while read -r patch || [[ -n "$patch" ]]; do
    [[ -n "${patch// /}" ]] && git -C "$SOURCE" apply "$PATCHES/${patch// /}"
done < "$PATCHES/series"
PYTHONPATH="$REPO" python3 -P -m cubecl_torch.compiler.source "$SOURCE" "$CUBEK"
