# cubecl-tile: our copy

Copied from CubeCL's `crates/cubecl-tile/src` so that we can change what our kernels need (bead
`f05`, OpenSpec change `cubecl-tile-stage-1`). `Cargo.toml` is ours; upstream's `tests/` are left
out because they need the `cubecl` crate's test runtime. CubeCL's FP patch does not touch this
crate, so the copy equals the pinned commit.

Upstream revision: fed6c0c0ca5f06250be8cdb15da0b12458ac083f

## Local changes

`test_tile_copy.py` fails on any file under `src/` that differs from the revision above and is
not listed here. Add a line per changed file: `` - `src/path.rs`: what changed and why ``.

- `src/space/base.rs`: `Space::with_sizes` is public, so a kernel can build a runtime space from its
  own scalars (no `Launcher`).
- `src/space/partition/base.rs`: `Partitioning::with_levels`, the same for a partitioning.
- `src/tile/kind/procedural/kind.rs`: `Procedural::within`, a recipe bounded by runtime extents;
  upstream leaves dynamic axes unmasked. Checked and masked reads evaluate at a redirected
  coordinate, never outside the box (upstream evaluates, then selects, and a recipe that loads reads
  past its tensor).
  `Procedural::in_lines` and `lined` serve a recipe in lines, a whole line from the recipe or a lane
  at a time (upstream asserts a recipe read is scalar).
- `src/tile/kind/procedural/base.rs`: recipes are documented as pure rather than memory-free, since
  a read outside the box still evaluates them. `LineRecipeExpand`.
- `src/tile/kind/procedural/mod.rs`: the same wording.
- `src/tile/kind/procedural/erased.rs`: `lines`, for recipes that compute whole lines.
- `src/tile/base.rs`: a procedural tile's vector size is its line width.
- `src/lib.rs`: exports `LineRecipeExpand`.
- `src/tile/accumulator/base.rs`: `accumulated`, an accumulator's one line, for a kernel that
  combines its units itself.
- `src/tile/kind/plane/registers.rs`: `RegisterData::first_line`, for the same.

## Merging from upstream

Merge on purpose, between changes, as with a CubeCL upgrade (`.stage0-sources/README.md`). Diff
upstream's old and new revisions, apply that here, resolve against the changes listed above, then
update the revision and list together.

Per file, from `git diff --name-status OLD NEW -- crates/cubecl-tile/src` in the checkout: copy an
added file, delete a removed one, and three-way merge a modified one into the copy with
`git merge-file <copy file> <OLD version> <NEW version>` (`git show REV:path` gives each version).
Match `Cargo.toml` to upstream's dependency versions. Last merge: 907676dc to fed6c0c0 (2026-11-02,
two conflicts, both in files listed above).
