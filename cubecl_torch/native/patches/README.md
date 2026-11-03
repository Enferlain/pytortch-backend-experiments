# CubeCL and CubeK patch stacks

Commands below run from the repository root. The CubeK stack is described in
[its own section](#cubek-patch-stack); the commands are the same with `STACK=cubek`.

Our changes to CubeCL, kept as patches over the pin in `cubecl_torch/runtime/native.py`
(`CUBECL_PIN`). `series` lists them in application order:

| Patch | What it changes |
| --- | --- |
| `fp.patch` | Floating-point fixes in CubeCL's C++ and LLVM dialects and optimizer (see `FP_HARDENING_REPORT.md`) |
| `bridge.patch` | Running on a caller's streams and memory (the `torch-runtime-bridge` change); see below |
| `plane-dim.patch` | Performance fix: the C++ backends lower `PLANE_DIM` to the compiled warp size (32 on CUDA) instead of the runtime `warpSize`, which NVRTC cannot fold. With it, the plane polyfill loops (`plane_sum` as a shuffle tree) unroll, and the unit and plane positions take no runtime division. Unchanged order of operations. Found with ncu on bridged sums (bead `zah.9`) |
| `static-sizes.patch` | Compiling a kernel for its sizes, as Inductor writes static sizes as literals (`StaticSizes` in `KernelSettings` and `KernelId`, set with `with_static_sizes`, taken by macro-generated launches): the launcher records each index (`usize`) scalar and each tensor's shape and strides as it registers them, and the kernel reads them as constants (`Tensor::shape`/`stride` as a select chain on the dimension). Divisions and modulos by sizes fold, and loops over a fixed extent get fixed trip counts. One kernel per set of sizes; the launch still carries every value, so a kernel whose settings lack them is unchanged. Launch APIs are unchanged. The constant folder leaves an integer division or remainder by zero to run time, since a constant size of zero can sit in a branch that never runs (it panicked before; no test, as CubeCL has no source-only kernel-definition harness). Found with ncu on bridged sums: CubeK's generic addressing executed 3× Triton's instructions |

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

## `bridge.patch` and the other servers

What each change is, and how CubeCL's HIP, Metal and wgpu servers provide the same thing. Only CUDA
is built and tested. Checked against the sources at the pin, not assumed.

| Change | CUDA (built) | HIP | Metal | wgpu |
| --- | --- | --- | --- | --- |
| **Transport chosen per device** (`cubecl-common` `device/handle/selected.rs`): `set_transport(device, CallingThread)` before the device's first service; a call has run when it returns. Per-handle `blocks()` replaces the static check in collectives. | Used by the bridge | Unchanged: the transport sits above every server | Unchanged | Unchanged; `wasm` builds keep the reentrant transport they already use |
| **Bound streams** (`cubecl-server` `StreamPool::bind`, ids with both top bits set (`StreamId::BOUND_BASE`; per-task ids clear the second)): a caller's stream gets a slot of its own, never shared through `id % max_streams`; `Server::bind_stream[_with]` binds a handle once. Bound streams must outlive the server: device-wide work (cleanup, reports, relocation) reaches them. | `CudaStreamBackend::wrap_stream` around a `CUstream` | `MultiStream` too (`compute/server.rs`): a `HipStreamBackend` wrap of a `hipStream_t`, same shape | `MultiStream` over command queues (`compute/stream.rs` creates them with `newCommandQueue`): a wrap of the caller's `MTLCommandQueue` (PyTorch MPS exposes its `MPSStream` queue) | No `MultiStream`: one queue per device, which `init_device` already takes from a caller's `WgpuSetup` (device and queue); binding means using that setup |
| **External memory** (`Server::register_external` / `unregister_external`, `MemoryManagement` external map, `EXTERNAL_POOL_POS`): a caller's buffer resolved directly, never pooled, moved or freed; it counts as allocated, and no pool claims it. | `GpuStorage` `AllocationKind::External` over a device pointer | Same: `hipMalloc` pointers, the same storage shape (`compute/storage/gpu.rs`) | Buffers are `MTLBuffer` objects, not pointers: the storage takes a buffer handle behind the same calls (the shape cubecl#1749 proposes) | `wgpu::Buffer` objects, likewise |
| **Caller allocator** (`StreamAllocator`, `Server::bind_stream_with`): a bound stream's own device memory from the caller, in `Dedicated` mode so nothing is pooled. | `GpuStorage::with_caller_allocator`, `AllocationKind::Caller` | Same, in place of `hipMalloc` | The allocator returns buffer objects: `allocate` takes a backend handle type rather than `u64` | Likewise, `wgpu::Buffer` |
| **Floating-point compilation per kernel** (`FpCompilation` in `KernelSettings` and `KernelId`, set with `with_fp_compilation`, taken by macro-generated launches): contraction and division precision, part of the kernel's identity. | NVRTC flags `--fmad`, `--prec-div`; the source also leads with them as a comment, so PTX kept by source text is not shared across settings. The LLVM backend refuses a non-default setting | Not applied yet: the HIP compiler ignores the field (only the cache key changes). hiprtc takes clang options (`compute/artifact.rs` builds `--offload-arch`): `-ffp-contract=on/off`, and `-fhip-fp32-correctly-rounded-divide-sqrt` for division | Not applied yet. `MTLCompileOptions` (created in `compute/artifact.rs`) has `mathMode` and `mathFloatingPointFunctions` for both | Not applied yet, and not applicable: WGSL has no switch (naga emits no contraction, and division precision is fixed by the spec), so this compiler should refuse a non-default setting |
| **Compilation observer** (`CompilationObserver`, `Server::set_compilation_observer`): each compiled kernel's exact source and binary. | Reported in `finalize` once assembly succeeded (source and PTX) | Same point in its artifact compiler (source and code object) | Same (MSL and the library) | Same (WGSL and the shader module) |
| **Metadata without host memory on bound streams** (`compute/words.rs`): a launch's dynamic info (shapes, strides) written by a kernel carrying the words in its parameters, so a capture its owner makes records it and nothing reads host memory later. | Used for every bound stream | HIP kernels take parameters the same way (`hipModuleLaunchKernel`): same | Argument buffers (`setBytes`) are recorded by an indirect command buffer: same idea | wgpu has no external capture: not needed |
| **Failures on external memory** (`MemoryManagement` external entries carry `Taint`): a failed launch on a caller's buffer is reported by `check` like any other. | Used by the bridge after every launch | Same, in `cubecl-server` | Same | Same |
| **Logging leaves the compiled text alone**: at the `Full` level only the logged copy goes through `clang-format`. | `compute/artifact.rs` | Has the same replacement (`compute/artifact.rs`, `format_cpp`): same fix | Not affected | Not affected |

## CubeK patch stack

Our changes to CubeK, kept as patches over `CUBEK_PIN` in `cubecl_torch/runtime/native.py`, in
`cubek/series` order. The checkout `.stage0-sources/cubek` carries them as uncommitted changes and
is audited like CubeCL's (`check_cubek`); provenance records the stack's digest
(`cubek_patch_stack_sha256`). `apply-stack.sh` resets both checkouts. To change a patch:

```bash
PIN=$(PYTHONPATH=. python3 -P -c 'from cubecl_torch.runtime import native; print(native.CUBEK_PIN)')
WORK=$(mktemp -d)/cubek
STACK=cubek cubecl_torch/native/patches/rebase.sh "$PIN" "$WORK"
# edit $WORK, amend the patch's commit (or add one and list it in cubek/series)
STACK=cubek cubecl_torch/native/patches/rebase.sh "$PIN" "$WORK"
cubecl_torch/native/patches/apply-stack.sh
```

Each patch is one change meant for upstream (tracel-ai/cubek), not yet proposed there. They were
found measuring the bridged `cubek_reduce` sums against Triton's accuracy band
(`torch-runtime-bridge` task 4.2, bead `zah.7`) and time (task 4.5, bead `zah.9`); results in
`cubecl_torch/testing/results/torch-runtime-bridge.md`.

| Patch | Kind | What it changes | Upstream |
| --- | --- | --- | --- |
| `forced-planes.patch` | Bug | A forced cube blueprint with `use_planes` was validated against `cube_dim.x` (the plane size) instead of `cube_dim.y` (the plane count, as the inferred blueprint lays it out), so only a blueprint with as many planes as the plane size passed. A forced blueprint with `num_shared_accumulators == cube_dim.x` and fewer planes, accepted before, is now refused | Not proposed. Needs a test forcing a blueprint with fewer planes than the plane size |
| `lanes-first.patch` | Numerics | `ReduceInstruction::fold_lanes`: in parallel vectorization the cube routine folds each unit's vector lanes (a pairwise tree for `Sum` and `Mean`; others unchanged) before units merge, as Triton reduces a thread's own elements first. Before, the lanes were added last, in order, when they are largest: float32 sums of offset data were 1.12–1.18× Triton's error at any partial count | Not proposed. Changes the cube routine's rounding for `Sum`/`Mean`; the plane and unit routines are unchanged |
| `plane-tree.patch` | Numerics | With `use_planes`, plane 0 merges the plane accumulators with a plane reduction (one per unit, a shuffle tree) instead of adding them in order, as Triton reduces its warps' results. This applies to every cube reduction with planes, inferred or forced, and every instruction (`plane_reduce_inplace`; null accumulators fill the units past the plane count, and ArgMax keeps the lower coordinate on ties). The plane count is bounded by the plane size (validated when forced, clamped when inferred) | Not proposed. Also removes a sequential loop over shared memory |
| `unit-elements.patch` | Numerics | `CubeBlueprint::unit_elements`: in parallel vectorization each unit of the cube routine keeps that many elements as separate partials (`max(1, unit_elements / vector_size)` accumulators of consecutive vectors, folded as a tree in the unit), as a Triton thread holds `R0_BLOCK / threads` elements whatever its load width. Validated as a power of two (the in-unit tree), and with bound checks unless the row is a whole number of steps. Inferred blueprints keep 1 (unchanged). Before, a row too narrow-aligned to vectorize (2048×4099) left one partial per unit, 512 where Triton has 2048 | Not proposed. A new blueprint field; forced blueprints elsewhere (CubeK's tests) set it to 1 |
| `folded-plane-sum.patch` | Performance | `ReduceInstruction::plane_reduce_folded`: once `lanes-first` has folded a unit's lanes into lane 0 (the others zero), `Sum` reduces across the plane only lane 0 (`plane_sum` of one scalar) instead of every lane of the vector. `Mean` and `Mixed` delegate; every other instruction keeps `plane_reduce_inplace`. Same values: the other lanes are zero throughout | Not proposed. Depends on `lanes-first` |
