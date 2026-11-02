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

### 3. Memory is borrowed for the duration of one call

`Client::register_external(ptr, size)` makes a handle that memory management resolves before its
pools, and that storage records as never freed. The bridge registers each tensor at the start of a
call and unregisters it before returning: the launch is already on the stream by then (decision 1),
and CubeCL needs the handle no longer. The GPU-side lifetime is PyTorch's own: the bridge records
each tensor on the launch stream, exactly as `runtime/launch.py` does for today's kernels, so the
caching allocator does not reuse the storage until the stream has passed it.

The prototype measured per-call registration at a few microseconds over a handle registered once,
inside the launch-cost bound. A cache keyed by storage can come later if profiling asks for it.

Alternative considered: *handles cached per storage, tied to PyTorch storage lifetime by callbacks.*
It is faster in principle, but needs hooks into PyTorch storage destruction, and caching gives no
correctness benefit.

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
which is the behavior the spec asks for.

Alternative considered: *a C++ shim calling `c10::cuda::CUDACachingAllocator` directly.* It avoids
the Python callback, but ties the bridge to PyTorch's C++ ABI. It's worth revisiting only if the
allocations stop being rare.

### 5. NVRTC options come from our numerics policy

`CudaCompilationOptions` gains the contraction and division settings. `compile_to_ptx` passes them
as `--fmad` and `--prec-div`, and leaves NVRTC's defaults when they are unset. The bridge sets them
from `compiler/nvrtc.py`'s policy, so a bridged kernel and a descriptor-compiled kernel round the
same way. The settings are part of CubeCL's kernel cache key.

### 6. The bridge is a C library loaded with ctypes

There is a new crate, `cubecl_torch/native/bridge`, a cdylib with a C interface built by `build.sh`.
It depends on CubeCL (`cubecl-cuda`) and CubeK (`cubek-reduce`), with CubeK's git dependencies
patched to the audited checkout. `cubecl_torch/runtime/bridge.py` loads it with ctypes and does the
following:

- installs the allocator callback;
- derives each call's stream from PyTorch's current stream;
- records tensors on the stream;
- maps errors to exceptions.

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
concatenation of its patches. It also checks the CubeK checkout against its pin and for local
changes, the spec's new scenario. The provenance record names both pins and the stack's digest.

### 8. The first consumer: CubeK's reduce for plain row sums

A new strategy, `cubek_reduce`, is a bridged plan kind in `compiler/plan.py`. It has one candidate
per CubeK routine (unit, plane, cube), each with `BlueprintStrategy::Inferred`, the way Burn offers
them.

Python legality admits it only for a descriptor whose single reduction is a sum of one loaded
input, with no other operation in its body:
- over the last axis of a contiguous row;
- one output element per row;
- float32 or 16-bit storage, accumulated in float32.

Every other kernel records it as rejected, with the reason. It has no Rust `plan.rs` counterpart,
since no descriptor is compiled for it; CubeK validates its own problem and reports refusals, which
become rejection reasons.

`AutotunedKernel.compile` returns a bridged kernel object with `LoadedKernel`'s call signature, so
first-use tuning, the runtime guard and the selection records treat it like any plan.

The numerical band decides admission per class, as in stage 1. The band harness pins a routine and
runs the stored Triton references.

The manifest records bridged candidates in candidate, rejected and selection lists. For compiled
candidates, it records a digest of the CUDA source CubeCL generates. A `bridge.patch` hunk exposes
that source: a compile hook on the server that receives the unformatted source NVRTC compiles.
CubeCL's compilation logging does not serve. It writes the source only at the `Full` level, as text
to file, stdout, stderr or `log` sinks. At that level the CUDA backend also replaces the source with
`clang-format` output before compiling it (`cubecl-cuda/src/compute/artifact.rs`), which runs a
subprocess on every compile and makes the compiled text depend on the installed formatter. And the
level is process-wide configuration.

### 9. Platform constraints are checked in review, not code

Each decision above names its HIP, Metal and wgpu counterpart. Only CUDA code is written. The
upstream write-up for cubecl#1749 carries the same table, so the mechanisms land general enough for
the other servers.

## Risks / Trade-offs

- [The reentrant transport serializes every server call on a device across threads.] → Launches
  are about 10 µs. The free-threading test measures contention, and a per-stream lock is the escape
  if it shows.
- [Capture-time allocations come from the graph pool; CubeCL's metadata cache could later reuse
  such a buffer outside the graph.] → Cached metadata buffers are immutable and read-only. The
  "shape first seen while capturing" test replays after eager use of the same shape to catch reuse
  hazards.
- [CubeK's summation order may sit outside Triton's band, or its kernels may be slower than our
  plans.] → That is a measured outcome, not a failure of the bridge. The candidate is held out per
  class with its reason, and the gap goes on a bead for fixing in CubeK.
- [In-process compilation puts NVRTC and CubeCL inside the torch process: a crash in them takes the
  process down, and first-call compile time is not isolated.] → First-use cost is measured against
  today's compiler process. Isolation was never a requirement (investigation, section 9).
- [Carrying four CubeCL changes raises merge cost at each upgrade.] → They are small and general,
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
