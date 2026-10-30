# sk8: cubecl-tile prototype

Research for bead `sk8`: can CubeCL's tile DSL (`.stage0-sources/cubecl/crates/cubecl-tile`)
supply the kernel-to-thread mapping that `cubecl_torch` builds plan by plan? The bead's notes
hold the findings; this folder holds what produced them.

The case is the float32 sum consumer `x * x.sum(-1, keepdim=True)` at 2048×8192, against
Inductor+Triton. Triton's autotuner picks `R0_BLOCK` 8192 with 16 warps: each of 512 threads
holds 16 values of the row.

## Results (2026-10-30, ncu at base clocks)

Speed:

| Kernel (mode) | 3080 | 3080 DRAM read | 3090 | 3090 DRAM read |
|---|---|---|---|---|
| Triton | 203.9 µs | 67 MB | 167.0 µs | 67 MB |
| Tile, one row per cube, 16 planes × 32 units × vec4 (`nohint`) | 286 µs | 129 MB | 236.9 µs | 129 MB |
| Same, pass 1 `evict_last`, pass 2 `evict_first` (`hinted`, default) | 204.0 µs | 70 MB | 169.5 µs | 71 MB |
| Same with a sum per vector lane (`lanes`, hinted) | 204.7 µs | 67 MB | | |
| One row per plane, 16 rows per cube (`row_plane`) | 295 µs | 134 MB | | |

The tile DSL has no load cache hints. The kernels set them during expansion: a helper walks the
IR emitted so far and sets the CubeCL patch's `LoadOp` retained/streaming flags on the 16-byte
global loads (`hints` in `tile_proto/src/main.rs`). With them the tile kernel is level with
Triton. Without them the second pass re-reads the row from DRAM.

Accuracy (`order_model.py` models each order exactly in float32; every model below that has a GPU
kernel matches it bit for bit). "Closer/farther" counts rows whose sum is nearer to / further
from the float64 sum than Triton's:

| Order | normal: mean \|err\|, closer/farther | offset (+1000): same |
|---|---|---|
| Triton | 6.96e-6 | 0.187 |
| Tile `hinted` (per unit: each step's 4 lanes added, then into one sum) | 8.37e-6, 721/919 | 0.370, 274/887 |
| Tile `lanes` (a sum per lane) | 8.25e-6, 719/944 | 0.261, 290/659 |
| `lanes` + planes combined as a tree (model) | 7.09e-6, 789/825 | 0.221, 308/501 |
| `lanes` + planes and lanes as trees, units xor 16→1 (model) | 7.04e-6, 793/811 | 0.191, 314/353 |

Triton's `tl.sum` is a balanced tree: inside the thread over its 16 registers, across lanes xor
16 down to 1, then across warps. The tile DSL's order is fixed by its lowering: units combine xor
1 up to 16, planes in sequence through the cyclic rounds (plane 0, then 15 down to 1), the lanes
in sequence. The sequential plane combine costs the most. Matching Triton's bits would also need
its thread ownership (each thread's 4 repeats 2048 apart, not contiguous plane slices).

## Check: Inductor's loop body inside tile loops (RMSNorm)

The question: can Inductor's fused per-element code run inside tile loops, or does the tile DSL
only cover bare reductions? The case is RMSNorm with a weight, `x * rsqrt(mean(x²) + eps) * w`,
float32 at 2048×8192: work fused before the sum (`x²`) and after it (`* rsqrt`, `* w`).

`fused_probe/` takes the kernel's `semantic.json` as our pipeline produced it and runs our own
emitter (`cubecl_torch/native/src/{ir,semantic}.rs`, compiled in unchanged through `#[path]`)
inside two tile *recipes*: the value the reduction sums, and the value each output element
stores. A recipe is a hand-written expand type that holds the buffer pointers, binds the row and
column the way our reduction emitter does, and calls `semantic::emit`. The tile layout is the
Triton-shaped one above (one row per cube, 16 planes × 32 units); the summing pass's row load is
marked retained and the storing pass's streaming, through the descriptor ops' own flags.

| Kernel (3080, ncu) | Time | DRAM read |
|---|---|---|
| Triton | 228.6–231.0 µs | 85 MB |
| Ours today (`shared_tree_compensated` 512, first-use selection) | ~223 µs | 85.7 MB |
| Tile loops + our emitter (`fused_probe`, leaf width 4) | 229.8–231.3 µs | 85.6–86.7 MB |

Correct against float64 (max relative error 2.6e-7). Compile cost per kernel, same NVRTC:

| | Expansion | C++ | NVRTC | Total |
|---|---|---|---|---|
| Tile loops + our emitter | 7.6 ms (our `emit`: 0.1 ms) | 30 ms | 29 ms (1,188 lines) | ~67 ms |
| Ours today | 10–12 ms for the whole compiler process | | 20 ms (363 lines) | ~32 ms |

What the check found:

- It works with our emitter unchanged. A recipe's expansion runs in a `Scope` like ours, and a
  tensor argument's buffer (`as_slice` → `__extract_list`) is the pointer our loads index.
- Recipe values are scalar: the loads stay 4 bytes wide whatever the leaf width. It did not cost
  time here (the kernel is memory-bound), but vector loads need a recipe that returns a vector.
- Copying a recipe into an output needs a scalar destination (`fill_procedural`), so the output
  binds at width 1.
- Recipes are documented as memory-free; ours read memory. Overhangs and shared-memory staging of
  a recipe were not exercised.
- The row sum reaches the storing recipe through a second, writable binding of the sums buffer.
  A read-only second binding compiles to `const __restrict__` non-coherent loads that miss
  in-kernel writes (the `lanes` mode hit this).
- The probe covers plain ops and one sum: no mask regions, Welford, multi-output or splits.
- Compiling it inside `native/` behind an off-by-default feature still changed the default
  compiler binary's digest (the manifest change alters Cargo's crate hash), which would invalidate
  every artifact key; hence the separate crate.

## Files

- `tile_proto/`: the tile kernels (Rust, against the pinned CubeCL checkout; CubeCL's CUDA
  runtime with the C++ backend, `cuda-cpp`, as production uses).
- `triton_ref.py`: generates the inputs and writes Triton's outputs, the chosen launch
  configs and the generated code, plus float64 references.
- `compare.py`: bitwise and accuracy comparison of a tile output against Triton and float64. Its
  closer/farther counts recover Triton's sum from its output, so they are slightly noisy;
  `order_model.py` compares exact sums.
- `order_model.py`: float32 models of Triton's and the tile kernels' summation orders, with
  variants.
- `triton_time.py`: runs the compiled Triton kernel, for timing under ncu.
- `fused_probe/`: the RMSNorm check (our emitter inside tile recipes, CubeCL's CUDA runtime).
- `rms_time.py`: runs RMSNorm compiled by `cubecl` (this repo's backend) or `triton`, for ncu.

## Running

Everything writes into a data directory of your choosing (about 1.1 GB for both inputs);
delete it afterwards. `D` below is that directory, `CU13` the venv's CUDA 13 wheel
(`stage0/python-deps/nvidia/cu13`): no system CUDA is installed, and CubeCL's C++ backend
needs its headers and NVRTC.

```bash
REF=/home/imi/Projects/sd-scripts/.venv-wsl/bin/python
TORCHINDUCTOR_CACHE_DIR=$D/ind TRITON_CACHE_DIR=$D/tri $REF docs/research/sk8/triton_ref.py $D 2048 8192

cargo build --release --manifest-path docs/research/sk8/tile_proto/Cargo.toml
export CUDA_PATH=$CU13 LD_LIBRARY_PATH=$CU13/lib CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
BIN=docs/research/sk8/tile_proto/target/release/tile_proto
# modes: hinted (default), nohint, lanes, row_plane
$BIN $D/normal.x.bin $D/normal.lanes.bin 2048 8192 1 lanes
$REF docs/research/sk8/compare.py $D normal $D/normal.lanes.bin lanes
$REF docs/research/sk8/order_model.py $D normal $D/normal.lanes.bin
```

`CUDA_VISIBLE_DEVICES=0` with PCI bus order is the 3080 (`1` the 3090). For timing, run the
binary with 4 repetitions under `ncu --launch-skip 2 --launch-count 1`; time Triton with
`triton_time.py <x.bin> 3080|3090` (all GPUs visible). Set `CUBECL_DEBUG_LOG=<file>` to log the
generated CUDA. The build writes several GB under `tile_proto/target`; remove it when done.

The RMSNorm check needs a `semantic.json`: compile RMSNorm once through this repo's backend with
`CUBECL_TORCH_CACHE_DIR=$D/cc` (`rms_time.py cubecl`, from the repository root with
`PYTHONPATH=$PWD` and `stage0/.venv/bin/python`) and take it from `$D/cc/artifacts/*/`.

```bash
cargo build --release --manifest-path docs/research/sk8/fused_probe/Cargo.toml
docs/research/sk8/fused_probe/target/release/fused_probe $D/out $SEMANTIC 2048 8192 3 4
```

It prints expansion, C++ and first-launch times and the error against float64, and writes the
generated CUDA to `$D/out/tile_kernel.cu`. For Triton under ncu, run `rms_time.py triton` once
without ncu first: a fresh Triton cache fails to build its launcher under ncu.
