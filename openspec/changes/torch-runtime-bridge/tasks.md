# Tasks

Each group closes the same way:

1. Run the gates:
   - style: ruff format, ruff check, `cargo fmt` and clippy (`-D warnings`) for `native/` and the
     bridge crate;
   - the full suite (`-n 4`, output to a file);
   - the manifest check.
2. Get a Sonnet (medium) review of the group's diff, then verify and fix its findings.
3. Commit the group.

CubeCL edits happen in the bridge worktree (`.stage0-sources/cubecl-bridge`) until group 1 moves
them into the patch stack. Never rebuild native pieces while a GPU job runs. Time on the 3080 with a
clock warm-up, and confirm important numbers on the 3090.

## 1. Patch stack and audit (decision 7)

- [x] 1.1 Move `native/cubecl-fp.patch` to `native/patches/fp.patch` and teach `build.sh` to verify the checkout against the stack applied to a temporary index at the pin. Verify: `build.sh` passes on the real checkout, refuses a scratch checkout with one extra edit and one with a patch file missing, and `native_matrix` shows byte-identical output to the build before the move.
- [x] 1.2 Pin CubeK in `build.sh` and the provenance records (revision and clean tree), and point `runtime/native.py`'s provenance at the stack digest and both pins. Verify: the spec's "Unexpected CubeK state" scenario as a test on a fake checkout; `test_cache_identity.py` shows a stack change producing new artifact keys.
- [x] 1.3 Update `.stage0-sources/README.md` and `cubecl_torch/README.md` (patch stack, CubeK as a build dependency, upgrade procedure). Verify: the documented commands run as written.

## 2. CubeCL changes (decisions 1–5), in `bridge.patch`

- [x] 2.1 Runtime transport choice per service (decision 1), including the reentrant handle's lazy `utilities()` fix. Verify: a CubeCL unit test that a blocking-transport client's launch is enqueued when `launch` returns, and that a default client still uses the channel.
- [x] 2.2 Bound streams in `cubecl-server`'s stream pool and the CUDA backend's wrapping of a caller-owned stream (decision 2). Verify: a CubeCL-level test with two caller-owned streams. Each gets its own slot, and neither is ever destroyed by CubeCL (checked by destroying them after the client drops).
- [x] 2.3 External memory registration and unregistration in `cubecl-runtime`, `cubecl-server` and the CUDA storage (decision 3). Verify: a test that a registered buffer is read and written in place, and that unregistering frees nothing (the pointer stays valid and owned by the caller).
- [x] 2.4 Caller-supplied allocator for a stream's storage, used with `Dedicated` memory management (decision 4). Verify: a test with a counting allocator. Every CubeCL device allocation for the stream goes through it, and every one is returned.
- [x] 2.5 NVRTC contraction and division options in `CudaCompilationOptions` and the kernel cache key (decision 5). Verify: a kernel with a division compiled both ways shows `div.rn` against `div.approx`/`div.full` in its PTX, and the two settings get different cache entries.
- [x] 2.6 Record, for each of 2.1–2.5, the HIP, Metal and wgpu counterpart (decision 9) in `native/patches/README.md`. Verify: every change has an entry.
- [x] 2.7 Write `bridge.patch` from the worktree into the stack and rebuild. Verify: `build.sh` passes, and `native_matrix` and the manifest check show no change to any existing plan.

## 3. Bridge library and runtime module (decision 6)

- [ ] 3.1 Create the `native/bridge` crate (CubeCL CUDA client with the blocking transport, CubeK via the audited path, C interface: device init, allocator install, launch entry points), built by `build.sh`. Verify: it builds with `--locked`, and loading it in a process that never touches CUDA creates no CUDA context.
- [ ] 3.2 Write `runtime/bridge.py`: load the library, install PyTorch's caching allocator through the callback, bind PyTorch's current stream per call, register and unregister tensors per call, record them on the stream, and map errors. Verify: `test_layers.py` passes with the module importable without the compiler or front end.
- [ ] 3.3 Ordering and lifetime tests through `runtime/bridge.py` with a CubeCL test kernel (spec: ordering, cross-stream use, freed while queued). Verify: they pass on the default and a side stream with no host synchronization, under allocator pressure.
- [ ] 3.4 Allocation and capture tests (spec: bridge allocations are PyTorch's, capture and replay, shape first seen while capturing, replay after eager use of the same shape). Verify: PyTorch's allocator statistics account for every bridge allocation, and capture works for a new shape.
- [ ] 3.5 Free-threaded concurrency test (spec: concurrent threads, one stream each). Verify: correct outputs, with lock contention measured and recorded.
- [ ] 3.6 Launch host cost against Triton's raw launch with the same arguments (spec: launch host cost), on the 3080, confirmed on the 3090. Verify: results in `testing/results/torch-runtime-bridge.md` and a test guarding the ratio with recorded noise.

## 4. First consumer: CubeK reduce (decision 8)

- [ ] 4.1 Add the `cubek_reduce` strategy (unit, plane, cube with inferred blueprints) to the catalog, Python legality with its structural rule, and a bridged kernel object behind `AutotunedKernel.compile`. Verify: legality tests for accepted (plain row sums in float32, float16, bfloat16) and refused descriptors (fused body, mask region, several reductions, non-contiguous rows, max/Welford), each with its reason.
- [ ] 4.2 Numerical band per routine and class on the stored Triton references (pinned, like stage 1's `tile_band`). Verify: results recorded per class, with admission set from them.
- [ ] 4.3 Selection records and manifest for bridged candidates: provenance (both pins, stack digest, numerics policy) and the generated CUDA source digest, through the compilation observer in `bridge.patch` (decision 8). Verify: the observed source matches the text NVRTC compiled, with and without CubeCL's compilation logging at `Full`; a manifest review by content shows only bridged candidates added, no existing artifact changed, then re-record.
- [ ] 4.4 Correctness and bounds through the Inductor route at 2048×8192, 2048×4099, 300×4096, 7×2049 and 64×31. Verify: outputs within the band's bound, and the runtime guard passes on every admitted class.
- [ ] 4.5 Pinned `ncu` timing against Triton and our best plan at those shapes on the 3080, confirmed on the 3090, plus first-call cost (in-process compile and tuning) against today's. Verify: results in the results file. Any class slower than Triton is held out and recorded on a bead with its cause, or a fix in CubeK.

## 5. Records

- [ ] 5.1 Draft the cubecl#1749 write-up: the use case, the ordering problem, the four mechanisms with their platform table, and the measurements. Verify: draft in `docs/research/zah.2/upstream-1749.md` for the user to review and post; nothing is posted by us.
- [ ] 5.2 Update beads (`zah`, `zah.2`, `17v`) and memory with what the bridge established and what the family migration needs next. Verify: `bd show` on each.

## Workflow follow-up

- Archive the change once every task is done and the user approves.
- Next change: fused bodies on bridged reductions through CubeK's `ReduceArgs` (Burn's `FusedReduceArgs` pattern), then the family migration plan.
