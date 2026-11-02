# CubeCL patch stack

Commands below run from the repository root.

Our changes to CubeCL, kept as patches over the pin in `cubecl_torch/runtime/native.py`
(`CUBECL_PIN`). `series` lists them in application order:

| Patch | What it changes |
| --- | --- |
| `fp.patch` | Floating-point fixes in CubeCL's C++ and LLVM dialects and optimizer (see `FP_HARDENING_REPORT.md`) |

The checkout `.stage0-sources/cubecl` carries the stack as uncommitted changes. The audit
(`cubecl_torch/compiler/source.py`, run by `build.sh` and before every compilation) applies the
stack to the pin in a temporary index and refuses unless the result equals the working tree.
Provenance records the stack's digest (`patch_stack_sha256`), so changing any patch gives every
artifact a new key.

## Changing a patch

Work in a clone with one commit per patch, never in the checkout:

```bash
PIN=$(PYTHONPATH=. python3 -P -c 'from cubecl_torch.runtime import native; print(native.CUBECL_PIN)')
WORK=$(mktemp -d)/cubecl
cubecl_torch/native/patches/rebase.sh "$PIN" "$WORK"   # the current pin: reproduces the stack
# edit $WORK and amend the patch's commit (or add a commit and list a new patch in series)
cubecl_torch/native/patches/rebase.sh "$PIN" "$WORK"   # rewrites the patch files from the commits
cubecl_torch/native/patches/apply-stack.sh
```

## Moving the pin

```bash
git -C .stage0-sources/cubecl fetch origin
WORK=$(mktemp -d)/cubecl
cubecl_torch/native/patches/rebase.sh NEW_PIN "$WORK"   # rewrites each patch onto NEW_PIN
# resolve any conflict in $WORK (git rebase --continue), rerun with the same $WORK
# set CUBECL_PIN in runtime/native.py (and cubecl_commit in native/src/main.rs)
cubecl_torch/native/patches/apply-stack.sh               # resets the checkout to the new stack
cubecl_torch/native/build.sh
```

`apply-stack.sh` discards the checkout's local changes and untracked files, so run it only once the
stack holds them. Then run the upgrade checks in `cubecl_torch/README.md` (`native_matrix` against a
build of the previous pin, the manifest).
