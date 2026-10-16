# Fixed-nine final reduction

Current decisions live in [STAGE5_REPORT.md](../../../../STAGE5_REPORT.md).
This pass changes only an explicit final-kernel diagnostic. The qualified
uniform-loop partial, its work assignment, arithmetic, loop expression, lazy
loads, split9 and precision behavior remain frozen. No default eligibility,
backend final policy, Stage 6, commit or push changes are included.

## Result

One final topology is qualified: the existing independent-plane physical plan
with four 32-lane planes per block, one output per plane, and the retained
descending FP32 merge order. All **148 CPU-model cases** preserve the frozen
final bytes and existing per-element qualification results. All **152 GPU
corpus/stream/special-value cases** and three final-only dtype controls match
the original final bitwise, including output/storage canaries. These include
the original 109 inputs, heldout_10017, finite cancellation adversaries and
FP16/BF16 storage neighbors. Existing failures are preserved by byte identity;
no tolerance or qualification policy is changed.

The candidate reduces final cost but does not establish complete parity or an
automatic default. On this RTX3090 capture, final-only medians are about
**1.73–1.76 → 1.38–1.41 µs**, versus retained Triton **1.25 µs**. With the frozen
CubeCL partial immediately preceding each final, secondary final medians are
**2.304 → 1.920 µs**, versus the same retained Triton final **1.663 µs**.

The earlier 2.3-versus-1.2 comparison therefore included both a final-body cost
and a producer-dependent execution-state cost. A crossed producer/consumer
experiment independently reproduces the latter with all three final binaries.
The partial is not changed to address it.

## Inventory before implementation

`inventory.json` retains exact source paths, line ranges, hashes and revision
identities. The local CUB/CCCL source is CUDA **13.1**; it is a source comparison,
not a production CUDA13.0 compilation or a claim about a CUB kernel being run.
ATen is from the installed Torch cu130 headers. CubeK/CubeCL use the current
retained source trees. Excluded upstream/cache trees were read directly after
graph coverage checks; graph absence was not treated as source absence.

| Mechanism | Work/arithmetic/ownership | Synchronization and reuse decision |
| --- | --- | --- |
| Retained configured Triton final | Grid B, 64 threads/block; `tid & 15` reads p0…p8 or +0, replicated across four half-warps. XOR8,4,2,1; only tid0 stores. | Four shuffles; no shared memory or block barriers. Existing cubin is launched directly for comparison, without Triton runtime ownership or retuning. |
| Original CubeCL final | Grid B, 256 threads/block; lanes0…8 each accumulate one value from +0; others stay +0. Shared tree128,64,32,16,8,4,2,1; unit0 stores. | 1024 user shared bytes, publication plus eight merge barriers. Retained as the frozen comparison and ordinary backend behavior. |
| CubeK Plane body | `GlobalFullPlaneReduce` maps cube/y to independent outputs; `PlaneReader` distributes a strided lane chunk, and Sum starts at +0. Only UNIT_POS_X==0 writes/commits. Eager/Lazy merge determines where plane_sum occurs. | Kernel-body composition is reusable without its routine planner, allocator or launcher. Its stock plane_sum lowers to ascending XOR1,2,4,8,16 on this CubeCL version, so it cannot silently replace the frozen descending order. |
| CubeK Cube body | Per-unit accumulators publish to shared memory; tree merge and ownership follow its component configuration. | Synchronization remains required between shared stages. SharedSum's global atomic ownership is unsuitable for this fixed owner-only final. Stock CubeK policies were not adopted. |
| CUB WarpReduce/BlockReduce | FP32 shuffle specialization uses ascending down offsets1,2,4,8,16 with valid-lane predicates. Default block reduction publishes warp aggregates, synchronizes, and combines them; only thread0's aggregate is valid. | Mature primitives, but their stock FP32 order differs. No CUB candidate is implemented or claimed qualified. |
| ATen Reduce.cuh/block_reduce.cuh | For contiguous FP32 B×9: grid 1, block(8,8) for B=8 or(8,64) for B=64. Four private chains/thread, stride8, serial private merge, descending subwarp offsets4,2,1; x0 owns each y output. General block_x_reduce also supports descending shared stages for wider x. | Source-model and GPU controls preserve all 148 corpus outputs bitwise. Actual stock sum_out is measured and traced below; its runtime is not adopted. |

The new code is a small adapter over the **existing independent-plane mapping,
semantic emitter and CubeCL ShuffleXorOp/FAddOp**. Generic FSumOp is unsuitable
because its actual local polyfill uses a different FP32 order. A new general
load primitive, reducer API or runtime is not introduced.

The ATen nine-value dispatch was resolved explicitly after the initial source
inventory, rather than assuming its generic warp helper defines its assignment.
`aten-model/` models lane t's four initialized chains (only r=t and r=t+8 can
be active), their serial private merge, and descending4,2,1. All148 CPU and
executed GPU corpus controls preserve the frozen final bytes. FP16/BF16 here
mean an FP32 sum followed by the original output cast; a lower-precision
sum_out accumulation is not substituted.

`aten-reference/` contains 24 raw CUPTI captures. Actual FP32 sum_out medians
are 2.464µs for B=8 and 2.432µs for B=64 in repeated-final scope, and 2.464/2.848µs
after the frozen CubeCL partial. This is a separately captured mature reference,
not a pooled timing or a second custom candidate. Its generic ReduceOp kernel
reports 26 registers,16 static shared bytes and grid 1. `aten-executed-sass.txt`
and `aten-code.ncu-repz` retain the loaded library function, including generic
runtime branches and descending SHFL.DOWN. The selected path has no executed
shared loads/stores. Static instructions from unselected generic branches are
not counted as executed work. Installed headers and the actual library function
are available; the prebuilt library's original frontend IR/PTX is not retained.
The stock runtime call is consequently not an inexpensive replacement here;
the component-level owner/shuffle mechanisms remain useful.

## Exact candidate and boundary

For block c, plane y and lane t, output is `o=4*c+y`. A valid plane initializes
FP32 a=+0, lazily reads p[o,t] only for t<9, then computes a=+0+p[o,t]. Inactive
lanes retain +0. All 32 lanes execute descending XOR distances16,8,4,2,1; lane0
alone replays the existing outer epilogue/store. The root groups the nonzero
terms as `(((p0+p8)+p4)+(p2+p6))+((p1+p5)+(p3+p7))`, with explicit zero-padding
operations modeled separately. No reassociation permission is added.

This owner-only final needs no shared publication or block barrier: every
reduction result is an SSA value within its own plane. The probe explicitly
requires extent9, one reduction, scalar inputs and only outer stores. It does
not delete synchronization from multi-consumer/fused workgroup reductions.

The semantic descriptor is unchanged. Index/address arithmetic remains signed
i64; FP32 accumulation, output storage casts, lazy access, conservative CubeCL
FP policy and CUDA13.0 NVRTC `--fmad=false` are unchanged. The existing validated
Kernel/InfoBuilder/prepared launcher retains PyTorch storage, readiness,
current-stream and allocator-lifetime behavior. The physical catalog and its
eligibility rules are unchanged: only `cpp-probe-final-plane` invokes the new
descending final expression.

`code-identity/checks.json` regenerates the ordinary partial and final for all
three dtypes with the rebuilt compiler. All **18 IR/CUDA/PTX comparisons** match
the retained files byte-for-byte. `gpu-qualification.json` additionally hashes
all retained partial/final artifact files before and after execution.

Four admission controls cover extent8, eight planes, a non-owner epilogue and
the wrong physical plan. The initial eight-plane control exposed a descriptor/
hardcoded-geometry mismatch in the new diagnostic route; its generated source
is retained in `guard-controls-initial/` and was never GPU-compiled or executed.
The route now rejects that descriptor before target code. After this guard fix,
`code-identity-after-guard/` verifies all27 IR/CUDA/PTX files for the ordinary
partial, ordinary final and valid candidate across three dtypes are unchanged.
The four rejection controls pass. Numerical/timing captures remain applicable
because every valid kernel artifact is byte-identical; no samples are replaced.

## Measurements and producer control

Primary timing evidence is `validated-plane/timing.json`, its archived
`harness-gpu.py`, and 72 raw CUPTI traces. Each cell has six rotated captures of
16 calls. All samples, startup calls and outliers are retained. A gate kernel
allows the entire launch sequence to be queued; the analyzer verifies raw API
timestamps. Final-only repeated execution and same-partial pairs remain
separate scopes, not whole Inductor graph or host latency claims.

| Shape/scope | Frozen CubeCL final | Plane final | Retained Triton final |
| --- | ---: | ---: | ---: |
| 8×1048576, repeated final only | 1.728 | 1.408 | 1.247 |
| 8×1048576, after frozen CubeCL partial | 2.112 | 1.761 | 1.504 |
| 64×65539, repeated final only | 1.759 | 1.376 | 1.248 |
| 64×65539, after frozen CubeCL partial | 2.304 | 1.920 | 1.663 |

Values are µs, medians of CUPTI kernel durations, with input/output identity.
`analysis.json` retains counts, extrema, MAD and every sample.

`cross-producer.json` and 96 raw traces cross two unchanged partial binaries
with the same three final binaries. Each pair uses the same input, split9,
partial/output allocation and bitwise result. Eight randomized captures of 16
pairs are fully prequeued. Secondary medians are:

| Preceding partial | Frozen final | Plane final | Triton final |
| --- | ---: | ---: | ---: |
| Frozen CubeCL | 2.303 | 1.921 | 1.664 |
| Retained Triton | 1.728 | 1.407 | 1.248 |

This establishes a producer-history effect on the final's **execution
duration**, beyond queue gaps. It does not uniquely attribute the effect to
cache policy, memory-controller state or resource handoff. No partial change or
production priming workaround follows from this observation.

## IR → CUDA → PTX → executed SASS

All candidate boundary artifacts are in `validated-plane/candidate/<dtype>`.
`executed-sass.txt` comes from the actual driver-loaded functions in
`final-code.ncu-repz`, not an offline CUDA13.1 substitution. `ncu-raw.csv` is
code/resource evidence; its replay durations are not warm timing evidence.

- Original: nine BAR.SYNC sites, shared publication/merge loads/stores, and one
  BSSY/BSYNC region around the nine-value protected read.
- Plane candidate: zero block barriers, zero executed shared loads/stores,
  five SHFL.BFLY operations, one active-mask VOTE, and one BSSY/BSYNC region.
- Triton: four SHFL.BFLY operations, predicated scalar read, no BSSY/BSYNC or
  shared-memory reduction traffic. Its bound scalars use an i32 parameter ABI;
  our unchanged semantic/i64 metadata boundary also requires a global extent
  load and more integer address construction.

Driver attribute queries report 10/15/12 registers for original/candidate/
Triton; executed CUPTI/NCU launch allocation reports **16 for all three**.
These labels are kept separate. No local-memory spills are reported. Removing
shared work helps actual latency; the allocated-register proxy does not explain
the gain by itself.

One isolated follow-up translated the already-established uniform/grouped-load
expression to the final's one private chain. All 152 bitwise checks pass, but
NVRTC produces **identical PTX for every dtype**. Its separate timing variation
is not credited as an improvement. The retained candidate is the original plane
expression; no other final topology is tried in this pass.

The remaining ~0.13–0.29 µs matched final gap is bounded by the observed extra
shuffle/control/address work and different launch geometry, but no single
instruction is assigned the full cost. No relaxed FP, vectorization, narrowed
semantic indices or custom predicated-load mechanism is introduced.

## Validation and reproduction

The initial umbrella regression stopped because the diagnostic Rust rebuild
omitted the LLVM feature required by Stage0. `regressions/` preserves that
failure and restored history; `build-full-features.log` records the corrected
LLVM/HIP build. The fresh full run is `regressions-full/`; inspect its gate
status rather than treating the earlier incomplete run as numerical evidence.
Seventeen gates pass there; the last long-row gate initially fails on an empty
CUPTI activity census (`AssertionError: []`). Its unchanged assertions pass in
the fresh `long-row-repeat/` run. Thus all 18 distinct gates pass across the full
run and affected repeat, with both original failures preserved and no policy
assertion weakened. `validation.json` records this distinction. Every protected
historical artifact is restored after both successful and failing runs.
Native review remains disabled by the user; there is no completed independent
review for this pass.

```sh
cargo build --release --locked --features llvm,hip-probe --manifest-path stage0/Cargo.toml --bin cubecl-stage0
stage0/.venv/bin/python -m stage5.reduction_final_probe --cpu --output /tmp/new-final-model
stage0/.venv/bin/python -m stage5.reduction_final_probe --gpu --output /tmp/new-final-candidate
stage0/.venv/bin/python -m stage5.reduction_final_analyze
```

GPU work is gated by the retained CPU qualification. New GPU output paths must
not already exist. `--cross` and `--counter` operate on the already-qualified
candidate; exact commands and source/tool hashes are retained in this packet.
Earlier exploratory captures remain intact; the archived validated capture and
crossed producer control are the primary timing evidence. Full generated-graph
adoption/performance and remaining handoff/control costs stay open in Stage5.
