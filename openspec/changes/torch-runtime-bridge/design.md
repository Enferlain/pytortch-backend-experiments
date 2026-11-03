# Design

## Context

Today `cubecl_torch` compiles descriptors in a separate process (`cubecl-stage0`, CubeCL's C++
compiler and NVRTC) and launches the PTX through `launcher.c` on PyTorch's current stream, with
tensors recorded on the stream by `runtime/launch.py`. CubeCL's runtime (client, server, memory,
streams, tuner) is never used.

The prototype (`docs/research/zah.2`, `cubecl-bridge.patch`, worktree branch `bridge-proto`) found:

- CubeCL's `std` device handle is a channel to a server thread. `Client::launch` queues and returns;
  the driver launch happens later on that thread. A flush per launch restores ordering at about
  70 µs per launch on this machine. CubeCL's reentrant-mutex handle runs calls on the calling
  thread at about 10 µs, but it is a global build-time choice.
- Memory handles are ids the server resolves (`ManagedMemoryDescriptor`). Constructors are public.
  CubeCL's claim checks skip ids outside its pools, so an external id needs entries only in memory
  management and storage.
- `StreamId` is an abstract number. `cubecl-server`'s `MultiStream` maps it to a fixed pool of
  backend streams (`id % max_streams`) and adds its own events when a buffer moves between them.
  Each CUDA `Stream` owns its `CUstream`, its memory management and a cache of launch metadata
  buffers.
- CubeCL's NVRTC options are the architecture, include paths and `-lineinfo`, so NVRTC defaults
  apply: FMA contraction on and precise division. Our default policy is contraction on and fast
  division (`compiler/nvrtc.py`), matching Triton.
- CubeCL's C++ dialect is the one our FP patch already corrects, so bridged kernels inherit those
  fixes.
- CubeK main (`50cb94b2`) builds against CubeCL `fed6c0c0` with its git dependencies patched to the
  local checkout.

## Goals / Non-Goals

**Goals:**

- Bridged launches that meet the `runtime-bridge` spec on CUDA, at the measured cost.
- CubeCL changes shaped so upstream can take them: general mechanisms, not PyTorch-specific code.
- One consumer through the real Inductor route, measured against Triton and our plans.

**Non-Goals:**

- Replacing the descriptor compiler or `launcher.c` for existing plans.
- Using CubeCL's autotuner. First-use selection stays ours here; adopting the tuner is a later change.
- Fused bodies on bridged kernels.
- Building or testing HIP, Metal or wgpu paths.

## Decisions

### 1. The transport is chosen per service, at runtime

`DeviceHandle`'s inner transport becomes a runtime choice between the channel and the reentrant
mutex, fixed when a device's service is created. The default stays the channel. The bridge asks for
the blocking transport for its devices before their first client exists. A launch then runs on the
calling thread and is on its stream when `launch` returns, which the ordering requirement needs.

Alternatives considered:
- *A flush after each launch on the channel transport:* correct, but about 70 µs per launch,
  three times Triton.
- *The global build-time switch the prototype used:* it changes every CubeCL user in the process,
  and upstream would not take it.

The choice also carries the prototype's fix to the reentrant handle's `utilities()`, which panicked
before the first call built the service.

Counterparts: the transport is backend-independent, so HIP, Metal and wgpu get it unchanged.

### 2. Bound streams get their own pool slots

`cubecl-server`'s stream pool gains bound streams: a `StreamId` from a reserved range maps to a
dedicated slot whose backend stream wraps a handle the caller owns. The backend never creates or
destroys that handle. The bridge derives the `StreamId` from the PyTorch stream (device and stream
handle) and issues each call under it with `StreamId::executes`. CubeCL's cross-stream bookkeeping
then stays exact: each PyTorch stream is one CubeCL stream, never multiplexed with another.

Alternative considered: *the prototype's per-call override of the slot's handle.* It mixes several
PyTorch streams into one CubeCL stream's cursor and event history, which is wrong as soon as two
streams are used.

Counterparts: HIP streams map the same way. Metal's command queue and wgpu's queue are the
analogues of a stream; PyTorch MPS exposes its command buffer and encoder (`MPSStream`), and wgpu
already supports an externally created device and queue.

### 3. Memory is borrowed per call, through handles cached by address

`Client::register_external(ptr, size)` makes a handle that memory management resolves before its
pools, and that storage records as never freed. It counts as allocated (its own pool position), and
it carries failure claims as a pool slice does, so a launch that fails on it is reported by
`Client::check`; the bridge checks its handles after every launch and raises the failure (CubeCL
reports launch failures on the outputs, not as a return value).

The bridge keeps each handle it registered, keyed by stream, address and size, for later calls: a
handle is only an address mapping, and memory PyTorch frees and reallocates at that address is what
a later call with that address means. Nothing refers to PyTorch storage between calls, and a handle
whose call failed is released, so no failure claim outlives its call. Profiling asked for it: per
call registration and unregistration were four of six server calls per launch.

The GPU-side lifetime is PyTorch's own, with PyTorch's operator semantics by default: a bridged
launch does not record its tensors on the launch stream, as PyTorch's own operators and Inductor's
Triton kernels do not, and a caller that frees a tensor while another stream still uses it records
it there (`Tensor.record_stream`), as it must for any operator. Recording costs about 4 µs per
launch, most of the bridge's gap to Triton's launch when it was always on, and inside the compiled
graphs the bridge serves, Inductor already handles cross-stream lifetime. Recording by the bridge
is opt-in: `record_stream=True` on a call, or `CUBECL_TORCH_BRIDGE_RECORD_STREAM=1` read when
`runtime/bridge.py` loads makes it the default; it then records every tensor of the call, so the
caching allocator does not reuse the storage until the stream has passed it. `runtime/launch.py`
still records always; whether today's launcher follows the same default is a separate question,
outside this change.

Alternative considered: *handles tied to PyTorch storage lifetime by callbacks.* It needs hooks into
PyTorch storage destruction, and an address mapping needs no lifetime of its own.

Counterparts: HIP is the same. wgpu buffers and Metal `MTLBuffer`s are object handles, not
pointers, so each backend's storage takes its own handle type behind the same registration call,
the shape cubecl#1749 already proposes.

### 4. CubeCL's own memory comes from PyTorch's allocator, unpooled

The bridge's streams use a storage whose allocations go through a caller-supplied allocator
(allocate and free, on a stream). The bridge installs PyTorch's caching allocator there through a
callback (`torch.cuda.caching_allocator_alloc` and `caching_allocator_delete` on the launch's
stream). Memory management for those streams runs in the `Dedicated` mode, so CubeCL pools nothing:
PyTorch does the pooling, and its statistics and graph pools cover everything.

The metadata buffer cache stays: buffers are immutable once written and keyed by their contents.
Their allocations are rare, so the callback's cost does not matter.

During graph capture, allocations on the capturing stream come from the graph's private pool,
which is the behavior the spec asks for. CubeCL never learns that PyTorch captures a bound stream,
so nothing it does on one may need host memory or a driver allocation: the dynamic part of a
launch's metadata (shapes, strides), which CubeCL stages through pinned host memory, is written on
bound streams by a kernel carrying the words in its parameters (`cubecl-cuda`
`compute/words.rs`), which a capture records as a node with the words baked in. A recorded copy
from host memory would read memory whose lifetime only PyTorch's graph knows.

Alternative considered: *a C++ shim calling `c10::cuda::CUDACachingAllocator` directly.* It avoids
the Python callback, but ties the bridge to PyTorch's C++ ABI. It's worth revisiting only if the
allocations stop being rare.

### 5. NVRTC options come from our numerics policy, per kernel

Our policy can change between kernels (Inductor's `emulate_precision_casts` and
`division_rounding` are read at codegen), so the settings travel with each kernel, not with the
server. `FpCompilation` (contraction, precise division; `None` keeps NVRTC's default) is a field of
`KernelSettings` and of `KernelId`, hashed only when not the default so existing ids keep their
hashes. A launch takes the setting in force on its thread (`with_fp_compilation`) into its
settings, on the caller's thread, so CubeK's launch functions need no new parameter and the
channel transport sees it too. The CUDA compiler passes it as `--fmad` and `--prec-div`, and writes
it as the source's first line, so PTX kept by source text is never shared across settings. The
LLVM backend refuses a non-default setting. The bridge sets it from `compiler/nvrtc.py`'s policy
around each launch, so a bridged kernel and a descriptor-compiled kernel round the same way.

Alternative considered: *the settings in `CudaCompilationOptions`* (fixed per server). A policy
change mid-process would reuse kernels compiled under the old one.

### 6. The bridge is a C library loaded with ctypes

There is a new crate, `cubecl_torch/native/bridge`, a cdylib with a C interface built by `build.sh`.
It depends on CubeCL (`cubecl-cuda` with its `cpp` feature) and CubeK (`cubek-reduce`), with CubeK's
git dependencies patched to the audited checkout. Without `cpp`, `cubecl-cuda` compiles through its
LLVM backend (the prototype did), not the C++ dialect our FP patch corrects and our descriptor
compiler uses. NVRTC finds the CUDA 13 headers of the torch install (`compiler/nvrtc.py`
`include_dirs`) through `CUDA_PATH`, which the runtime module sets before the first compilation. `cubecl_torch/runtime/bridge.py` loads it with ctypes and does the
following:

- installs the allocator callback;
- derives each call's stream from PyTorch's current stream;
- records tensors on the stream when asked (decision 3);
- maps errors to exceptions.

Every C entry point restores the calling thread's CUDA context when it returns, as PyTorch's
`CUDAGuard` restores its device: CubeCL makes its device's primary context current on the thread
that runs its server, which with the calling-thread transport (decision 1) is the caller's, so a
launch on `cuda:1` would otherwise leave PyTorch's current device at 1. Two `cuCtxGetCurrent`
calls per launch, and a `cuCtxSetCurrent` only when the device differs.

This matches how `launcher.c` and the metadata library are loaded today. It needs no
Python-version-specific build for the free-threaded interpreter, and it keeps the runtime layer
importable without the compiler or the front end.

Alternative considered: *a pyo3 extension.* It gives richer types, but adds a Python ABI
dependency and a second binding style for no gain at this interface size.

### 7. The patch stack becomes a directory, audited as a whole

The CubeCL patch moves into `cubecl_torch/native/patches/` as ordered files:
- `fp.patch`, today's `cubecl-fp.patch`;
- `bridge.patch`, this change's hunks.

`build.sh` applies the stack to a temporary index at the pin and compares the result with the
checkout's working tree, so the check covers the stack without assuming the diff is the
concatenation of its patches. CubeK gets a stack of its own (`native/patches/cubek/`), audited the
same way, the spec's new scenario. The provenance record names both pins and both stacks' digests.

CubeK's stack holds what the bridged plans need changed in CubeK, each patch one change meant for
upstream, listed with its upstream status in `native/patches/README.md`. A patch rather than a copy
(as `cubecl-tile` is copied): the changes are small, inside CubeK's own routines, and meant to land
there. Three came from the band (decision 8):
- `forced-planes.patch` fixes a forced cube blueprint with planes, validated against the plane size
  instead of the plane count;
- `lanes-first.patch` folds each unit's vector lanes before units merge, as Triton reduces a
  thread's elements first (`reduceWithinThreads`);
- `plane-tree.patch` merges the plane accumulators with a plane reduction instead of in order, as
  Triton reduces its warps' results;
- `unit-elements.patch` lets a forced cube blueprint keep several elements per unit as separate
  partials, as a Triton thread holds `R0_BLOCK / threads` elements whatever its load width.

### 8. The first consumer: CubeK's reduce for plain row sums

A new strategy, `cubek_reduce`, is a bridged plan kind in `compiler/plan.py`. Its candidates are
Burn's grid for a reduction along contiguous rows (`burn-cubecl` `kernel/reduce/tune.rs`): each
CubeK routine with `BlueprintStrategy::Inferred` (unit; plane with independent accumulators; cube
with planes), with and without `parallel_output_vectorization`, six in all. A seventh is the cube
routine with the blueprint Inductor's own config implies for the row (`mapping="inductor"`): its
warps as planes, `R0_BLOCK / threads` elements per unit. Inductor gives a looped INNER row sum one
fixed config (no autotuning), so this is the structure of Triton's sum for the shape. The rule is
copied from Inductor's heuristic, which needs Triton to run; `testing/data/inductor_row_configs.json`
records Inductor's actual choices and a test holds the copy to them. It applies to rows over 1024
elements; shorter rows Inductor reduces persistently, with a config autotuned or chosen by row
count, so the mapping is refused there with that reason, and "rows over 1024" is a signature fact. The flag matters here:
on a contiguous last axis it makes each unit reduce as many rows as its output vector holds. Each
candidate carries `routine` and `parallel_output` fields, left out of native plans' dictionaries so
their identities stay unchanged. The candidates live in `bridge_catalog()`, not `catalog()`, which
is the set the native compiler implements (tests compile every reduction plan in it); legality
offers both to every kernel.

Python legality admits it only for a descriptor whose single reduction is a sum of one loaded
input, with no other operation in its body:
- over the last axis of a contiguous row, read from the tensor's address (offset 0, one row per
  extent), with a non-empty row;
- one output element per row, in row order;
- float32 or 16-bit storage, cast only to float32 before the sum and at most once, to the stored
  dtype, after it; accumulated in float32;
- input and output addresses aligned to 16 bytes. CubeCL vectorizes by shape and strides alone, as
  for memory it allocates itself, so a misaligned PyTorch view would fault on the device;
  `bridge.sum_rows` refuses one too.

The contiguity part is a runtime fact, so it is part of the legality signature. Admission is
numerical, from the band (task 4.2, `legality.cubek_numerical_reason`):
- the cube with Inductor's mapping, for float32 and 16-bit sums, on rows up to 32768 elements;
- Burn's six plans for 16-bit values stored as loaded, on rows up to 8192. Their float32 sums
  (unit, plane, and the cube with CubeK's inferred blueprint) were outside the band;
- anything else is rejected with the measured reason, and the row limits are signature facts.

Every other kernel records it as rejected, with the reason; an effectful kernel (random numbers,
mutation, checked indirection) takes the existing frozen-tuning reason first, and an output must be
a buffer apart from its input. It has no Rust `plan.rs` counterpart, since no descriptor is compiled
for it. CubeK checks its own problem at launch and raises what it refuses; first-use selection
excludes a candidate CubeK refuses, recording CubeK's reason, rather than failing the selection (a
bridged plan is never the fallback; strict testing keeps it fatal, as for any candidate failure).

`AutotunedKernel.compile` returns a bridged kernel object (`inductor/bridged.py`) with
`LoadedKernel`'s call signature, so first-use tuning, the runtime guard and the selection records
treat it like any plan. Each launch checks again the runtime facts legality proved for its class
(aligned addresses, contiguous non-empty rows), so a direct call outside the tuner cannot reach the
device with an address that would fault. Its launch contract is the frozen plan's `LoadedKernel` for the same
descriptor: each launch is validated by it (arguments, layouts, storage spans, aliasing), so a
bridged launch is refused exactly where the frozen plan's would be. For the benchmarker's CUPTI
events, a prepared bridged kernel names the one CUDA kernel CubeK launches for the shape (CubeK
names it by the inferred blueprint), found by profiling one launch.

The numerical band decides admission per class, as in stage 1. The band harness pins a routine and
runs the stored Triton references.

The manifest records bridged candidates in candidate, rejected and selection lists. For compiled
candidates, it records a digest of the CUDA source and PTX of each kernel the candidate launched
(no CubeCL IR of ours). `bridge.py` keeps every kernel compiled for a device, by its CUDA entry
name (CubeCL names it after a hash of the kernel id), from an observer it installs before the
device's first launch, so a kernel another candidate compiled first is still found; a bridged
kernel knows its entries from the launches it profiled. The kernels are reported by a compilation
observer `bridge.patch` adds (`CompilationObserver`, `Client::set_compilation_observer`): each
kernel's exact compiled source and PTX, as `finalize` builds them. CubeCL's compilation logging does
not serve: it writes the source only at the `Full` level, as text to file, stdout, stderr or `log`
sinks, and the level is process-wide configuration. At that level the CUDA backend also replaced
the compiled source with `clang-format` output; `bridge.patch` formats only the logged copy, so the
compiled text no longer depends on the log level or the installed formatter. CubeCL's persistent
compilation store, off by default, would serve kernels without compiling them; the bridge leaves it
off.

### 9. Platform constraints are checked in review, not code

Each decision above names its HIP, Metal and wgpu counterpart. Only CUDA code is written. The
upstream write-up for cubecl#1749 carries the same table, so the mechanisms land general enough for
the other servers.

## Risks / Trade-offs

- [The reentrant transport serializes every server call on a device across threads.] → Measured:
  four threads launching at once take about 280 µs per launch each with the GIL off, against about
  18 µs for one (`testing/results/torch-runtime-bridge.md`); correctness holds. The cause is not
  profiled; a per-stream lock, which needs CubeCL's server state split by stream, is the candidate
  fix, on a bead under `zah`.
- [Capture-time allocations come from the graph pool; CubeCL's metadata cache could later reuse
  such a buffer outside the graph.] → A buffer written inside an owner's capture holds its words
  only once the graph runs, so on a bound stream being captured the metadata cache is neither read
  nor filled. Tested with eager use of the shape after the capture and a second graph of it.
- [CubeK's summation order may sit outside Triton's band, or its kernels may be slower than our
  plans.] → That is a measured outcome, not a failure of the bridge. The candidate is held out per
  class with its reason, and the gap goes on a bead for fixing in CubeK.
- [In-process compilation puts NVRTC and CubeCL inside the torch process: a crash in them takes the
  process down, and first-call compile time is not isolated.] → First-use cost is measured against
  today's compiler process. Isolation was never a requirement (investigation, section 9).
- [A bound stream must outlive the server: device-wide work (memory cleanup, reports, relocation)
  reaches every bound stream.] → PyTorch's default and pooled streams are never destroyed. A
  `torch.cuda.ExternalStream` a user destroys is the exception. Task 3.2 checks whether PyTorch's
  stream ids tell external streams apart (c10 encodes a stream type in them); if so the bridge
  refuses them until CubeCL has an unbind, otherwise the requirement is documented on the module.
- [By default a bridged launch does not record its tensors on the stream, so a caller that frees a
  tensor while a side stream still uses it, without `record_stream`, gets its storage reused
  early.] → That is PyTorch's own operator contract, and Inductor's generated code keeps it. The
  opt-in (decision 3) restores recording per call or per process, and both cases are tested.
- [Carrying these CubeCL changes raises merge cost at each upgrade.] → They are small and general,
  and the upstream write-up aims to land them.
- [CubeK main pins a CubeCL branch commit, while we build it against CubeCL main.] → The prototype
  built and ran. The CubeK pin is audited, and an upgrade moves both together.

## Migration Plan

Additive. Existing plans, the launcher and the descriptor compiler are untouched, and bridged
candidates only add to candidate sets. Removing the `cubek_reduce` strategy from the catalog falls
back completely. The manifest re-records once, after review, for the new candidate lists.

## Open Questions

None. Whether CubeCL's compilation logging could supply the generated source (decision 8) was
checked against `fed6c0c0`: it can't, so a patch hunk exposes it.
