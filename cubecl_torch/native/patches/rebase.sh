#!/usr/bin/env bash
# Move the audited CubeCL patch stack to a new pin: rebase.sh NEW_PIN [WORK_DIR]
# With STACK=cubek, the CubeK stack (patches/cubek, CUBEK_PIN) instead.
#
# Works in a shared clone of the checkout: commits each patch in series order at
# the current pin (runtime/native.py CUBECL_PIN or CUBEK_PIN), rebases those commits onto
# NEW_PIN, and rewrites each patch file from its own commit. On a conflict it
# stops in the clone (printed below); resolve there with `git rebase --continue`
# and run it again with the same WORK_DIR. The real checkout is not touched;
# apply-stack.sh moves it once the rewritten stack is reviewed.
set -euo pipefail
PATCHES="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$PATCHES/../../.." && pwd)"
case "${STACK:-cubecl}" in
    cubecl) SOURCE="${CUBECL_SOURCE:-$REPO/.stage0-sources/cubecl}"; PIN_NAME=CUBECL_PIN ;;
    cubek)
        SOURCE="${CUBEK_SOURCE:-$REPO/.stage0-sources/cubek}"; PIN_NAME=CUBEK_PIN
        PATCHES="$PATCHES/cubek" ;;
    *) echo "STACK is cubecl or cubek" >&2; exit 1 ;;
esac
NEW="$1"
WORK="${2:-$(mktemp -d)/${STACK:-cubecl}}"
OLD="$(PYTHONPATH="$REPO" python3 -P -c "from cubecl_torch.runtime import native; print(native.$PIN_NAME)")"
# Fixed identity, no signing or hooks, and plain a/ b/ diffs whatever the
# user's git config says.
GIT=(git -C "$WORK" -c user.name=rebase -c user.email=rebase@invalid -c commit.gpgsign=false
    -c core.hooksPath=/dev/null)
# One name per line of series, blank lines ignored, as runtime/native.py reads it.
mapfile -t SERIES < <(grep -v '^[[:space:]]*$' "$PATCHES/series" | tr -d '[:blank:]')

if [[ ! -d "$WORK/.git" ]]; then
    git clone -q --shared --no-checkout "$SOURCE" "$WORK"
    "${GIT[@]}" checkout -q "$OLD"
    for patch in "${SERIES[@]}"; do
        "${GIT[@]}" apply --index "$PATCHES/$patch"
        "${GIT[@]}" commit -q --no-verify -m "$patch"
    done
    if ! "${GIT[@]}" rebase -q --onto "$NEW" "$OLD"; then
        echo "conflict: resolve in $WORK, git -C $WORK rebase --continue, then: $0 $NEW $WORK" >&2
        exit 1
    fi
fi
if [[ -d "$WORK/.git/rebase-merge" || -d "$WORK/.git/rebase-apply" ]]; then
    echo "rebase in $WORK is unfinished" >&2
    exit 1
fi
mapfile -t COMMITS < <("${GIT[@]}" rev-list --reverse "$NEW..HEAD")
if [[ ${#COMMITS[@]} -ne ${#SERIES[@]} ]]; then
    echo "expected ${#SERIES[@]} commits on $NEW in $WORK, found ${#COMMITS[@]} (a patch emptied?)" >&2
    exit 1
fi
for i in "${!SERIES[@]}"; do
    "${GIT[@]}" diff --no-ext-diff --no-renames --src-prefix=a/ --dst-prefix=b/ \
        "${COMMITS[$i]}~1" "${COMMITS[$i]}" > "$PATCHES/${SERIES[$i]}"
done
echo "rewrote ${SERIES[*]} onto $NEW from $WORK; next: set $PIN_NAME, then apply-stack.sh"
