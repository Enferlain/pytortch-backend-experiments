# cubecl-tile: our copy

Copied from CubeCL's `crates/cubecl-tile/src` so that we can change what our kernels need (bead
`f05`, OpenSpec change `cubecl-tile-stage-1`). `Cargo.toml` is ours; upstream's `tests/` are left
out because they need the `cubecl` crate's test runtime. CubeCL's FP patch does not touch this
crate, so the copy equals the pinned commit.

Upstream revision: 907676dca8c4cd407237892cc2915ff6aadc4f1e

## Local changes

`test_tile_copy.py` fails on any file under `src/` that differs from the revision above and is
not listed here. Add a line per changed file: `` - `src/path.rs`: what changed and why ``.

- `src/space/base.rs`: `Space::with_sizes` is public, so a kernel can build a runtime space from its
  own scalars (no `Launcher`).
- `src/space/partition/base.rs`: `Partitioning::with_levels`, the same for a partitioning.
- `src/tile/accumulator/smem_slots.rs`: `SmemSlotsAccumulation::sum`, the one-cell sum as a value,
  for a kernel with no output tile to land it in.

## Merging from upstream

Merge on purpose, between changes, as with a CubeCL upgrade (`.stage0-sources/README.md`). Diff
upstream's old and new revisions, apply that here, resolve against the changes listed above, then
update the revision and list together.
