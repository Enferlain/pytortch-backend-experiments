# zah.2: CubeCL's runtime on PyTorch's memory and stream (prototype)

Question: can CubeCL's CUDA runtime, and CubeK through it, run on memory and streams PyTorch
owns, with PyTorch's ordering and at a launch cost no worse than Triton's? If so, CubeCL's
runtime-level machinery (CubeK launch and selection, CubeCL's autotuner) becomes usable from
`cubecl_torch` as Burn uses it, instead of being rebuilt above our launcher.

Answer: yes, on CUDA, with three changes to CubeCL. Measured 2026-11-02 on the RTX 3080 (WSL,
PyTorch 2.14.1, Python 3.14t).

## The CubeCL changes

In the worktree `.stage0-sources/cubecl-bridge`, branch `bridge-proto`, on CubeCL `fed6c0c0`
(`cubecl-bridge.patch` here is the same diff, for applying to a fresh checkout):

1. **Synchronous device handle** (`b58579f4`). With `std`, CubeCL's device handle is a channel
   to a server thread: `Client::launch` queues the launch and returns before the driver call. On
   a borrowed stream that breaks ordering against PyTorch work issued after it. The prototype uses
   CubeCL's existing reentrant-mutex handle for every `std` build, so server calls run on the
   calling thread. It also fixes a latent bug in that handle: `utilities()` panicked before the
   first call built the service.
2. **External stream** (`f083b7ac`). `cubecl_cuda::external::on_stream(stream, f)` issues every
   server call `f` makes on this thread to `stream`. Each command sets its CubeCL stream's handle
   to the bound stream, or back to its own.
3. **External memory** (`f083b7ac`). `Client::register_external(ptr, size)` returns a handle for
   memory the caller owns: the GPU storage records it as a kind it never frees, and memory
   management resolves its id before the pools. CubeCL's claim checks already skip ids outside
   the pools.

## Results

`torch_bridge` is a C library driven from Python by ctypes (`check.py`), calling CubeK's
`reduce` (plane routine, inferred blueprint, sum) and a trivial CubeCL kernel on PyTorch tensors.

| Check | Result |
|---|---|
| Correctness, row sums at 2048×8192, 300×4096, 7×2049, 64×31 | at most 0.005 of a float32 bound (64 ulp of the summed magnitudes) |
| Ordering against PyTorch work queued before and after, no host sync, 20 iterations | correct on the default stream and on a side stream |
| CUDA graph capture and replay (`torch.cuda.graph`) | correct |

Host time per launch of a trivial kernel, 20 000 calls, three rounds:

| Path | µs per call |
|---|---|
| CubeCL client, channel transport, launch only (unordered), in Rust | 9.7–13.2 |
| CubeCL client, channel transport, launch + flush (ordered), in Rust | 69.5–70.7 |
| CubeCL client, synchronous transport, launch + flush, in Rust | 8.8–10.4 |
| Bridge from Python, handle registered once (stream fetched once) | 10.0–17.2 |
| Bridge from Python, registered every call (stream fetched once) | 14.5–20.1 |
| Bridge from Python, registered every call, `current_stream()` each call | 33.3–37.6 |
| `torch.cuda.current_stream().cuda_stream` alone | 4.6 |
| Triton raw `kernel[grid](...)` launch | 19.9–23.5 |
| Ours today, `LoadedKernel` | 50.5–54.0 |

The last bridge row also grows CubeCL's external tables by two entries per call, which the
prototype never removes.

## What the prototype does not settle

- **CubeCL's own allocations.** Per-launch metadata buffers come from CubeCL's pools, not PyTorch's
  allocator. Capture worked because the warm call had cached the buffer; a shape first seen during
  capture would allocate mid-capture. A storage backed by PyTorch's allocator would close this.
- **Lifetimes.** Registration is per call and never released. The real bridge ties a handle to the
  PyTorch storage and releases it once the stream has passed its last use, as `record_stream` does.
- **Streams.** The per-call override is a shortcut. The real change gives each bound PyTorch stream
  its own slot in `cubecl-server`'s stream pool, so CubeCL's cross-stream bookkeeping stays right.
- **Transport per service.** The prototype switches the transport for the whole build. Upstream
  would want it chosen per server.
- **Compilation in-process.** CubeCL compiles on first launch inside the torch process; first-call
  cost against our compiler process is unmeasured.
- **Other targets.** CUDA only. HIP, Metal and wgpu need the same three changes in their servers.

Upstream context: cubecl#1749 ("zero-copy interop with DLPack") asks for non-owning device handles;
nobody there has raised ordering, which is what the synchronous handle is for.
