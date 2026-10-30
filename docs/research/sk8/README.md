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
