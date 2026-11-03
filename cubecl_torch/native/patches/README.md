# CubeCL patch stack

Commands below run from the repository root.

Our changes to CubeCL, kept as patches over the pin in `cubecl_torch/runtime/native.py`
(`CUBECL_PIN`). `series` lists them in application order:

| Patch | What it changes |
| --- | --- |
| `fp.patch` | Floating-point fixes in CubeCL's C++ and LLVM dialects and optimizer (see `FP_HARDENING_REPORT.md`) |
| `bridge.patch` | Running on a caller's streams and memory (the `torch-runtime-bridge` change); see below |

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
