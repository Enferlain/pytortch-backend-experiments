# Proposal

## Why

`cubecl_torch` connects to CubeCL at the compiler and launches through its own thin launcher, so
everything CubeCL and CubeK keep in their runtime layer has been out of reach: CubeK's launch
functions and hardware-parameterized selection, and CubeCL's autotuner. We rebuilt those above our
launcher (plan catalogs, legality, tuning), with NVIDIA assumptions baked in, while the project's
target is every platform PyTorch and CubeCL both serve. CubeCL's runtime could not take PyTorch's
memory or streams, which is why the original investigation chose the thin launcher. The prototype
in `docs/research/zah.2` (bead `zah.2`) showed that three contained CubeCL changes remove that
obstacle: CubeK's reduce ran on PyTorch tensors and streams, ordered with PyTorch's own work,
captured into a CUDA graph, at 10–17 µs per launch against Triton's 20–23 µs and our 52 µs. This
change builds that bridge properly, as the foundation the kernel families then move onto.

## What Changes

- **A CubeCL runtime on PyTorch's resources.** CubeCL's CUDA client runs inside the torch process on
  memory PyTorch allocated and on PyTorch's current stream, with PyTorch's ordering, lifetime and
  graph-capture semantics. CubeCL's own allocations (launch metadata, workspaces) come from
  PyTorch's caching allocator, so memory accounting and capture cover them.
- **CubeCL changes, carried and offered upstream:**
  - a synchronous device transport, chosen per server, so a launch is on its stream when the call
    returns;
  - device streams bound to PyTorch streams, one stream-pool slot per bound stream;
  - non-owning device memory handles that CubeCL never pools or frees (cubecl#1749);
  - storage that allocates through a caller-supplied allocator;
  - NVRTC options taken from our numerics policy.

  They join our patch stack beside the FP patch, with the build audit extended to them and to a
  pinned CubeK checkout.
- **A bridge library and runtime module.** A Rust library loaded in the torch process (CubeCL,
  CubeK, a C interface) and a `cubecl_torch.runtime` module that binds tensors, streams and the
  allocator and launches through it. Kernels compile in-process on first launch. The descriptor
  compiler and today's launcher stay as they are for every existing plan.
- **A first consumer, end to end.** CubeK's reduce, with its own routines and blueprint inference,
  offered as a candidate for plain row sums (`x.sum(-1)` with no fused body) through the existing
  Inductor route, judged by the numerical band and timed against Triton and our plans. Fused bodies
  (through CubeK's `ReduceArgs`, as Burn does) and moving other families onto the bridge are the
  next changes, not this one.

Out of scope: removing or rewriting existing plans; fused bodies on bridged kernels; CubeCL's
autotuner in our selection; platforms other than CUDA beyond keeping the design open to them (HIP,
Metal and wgpu servers are design constraints, not deliverables).

## Capabilities

### New Capabilities

- `runtime-bridge`: CubeCL's runtime running on PyTorch's memory and streams: ordering, lifetime,
  allocation, graph capture, launch cost, provenance, and which kernels may run through it.

### Modified Capabilities

- `descriptor-compiler`: "Audited CubeCL source only" extends to the bridge: the CubeCL patch stack
  now includes the bridge changes, and the pinned CubeK source is audited the same way.

## Impact

- **CubeCL checkout:** new patch hunks in `cubecl-common` (device handle), `cubecl-server` (stream
  pool, memory management), `cubecl-runtime` (server and client API), `cubecl-cuda` (external
  streams and memory, storage allocator, NVRTC options). `build.sh` and the pin records audit them.
- **CubeK:** becomes a build dependency (path, pinned), not only reference reading.
- **New code:** a bridge crate under `cubecl_torch/native/`, `cubecl_torch/runtime/bridge.py`, a
  bridged candidate kind in the plan catalog and legality layers, tests.
- **Process model:** CubeCL, CubeK and NVRTC run inside the torch process for bridged kernels. The
  separate compiler process remains for descriptor-compiled plans.
- **Unchanged:** every existing plan's generated code, the launcher, and the artifact cache. The
  manifest gains records for bridged candidates.
