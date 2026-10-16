# Qualified small final in generated graphs (cwk.7)

Current decisions remain in [STAGE5_REPORT.md](../../../../../STAGE5_REPORT.md).
This packet extends the [completed component investigation](../README.md), using
its single qualified descending independent-plane mechanism. It does not reopen
cwk.5 or the partial topology. Stage 5 stays active; Stage 6 stays deferred.

## Decision and scope

The plane final is a qualified **explicit diagnostic substitution** in actual
Dynamo/AOTAutograd/Inductor-generated CubeCL graphs. The normal final compiler
route, physical catalog, automatic eligibility and backend final policy are
unchanged. The existing probe now supports constant final extents **1..32**;
the generated-graph adapter deliberately selects only extent9 with the explicit
`reference512_diagnostic` policy. No new final arithmetic topology is introduced.

The retained uniform-loop partial is byte-identical in IR, CUDA and PTX,
including its work assignment, arithmetic, split9, lazy reads, FP policy and
PyTorch storage/current-stream interface. No thread-dependent-loop candidate,
synthetic sleep, memory primer, production warmup, new planner or tuning sweep
is included. Ordinary execution for compilation/module readiness is recorded in
the harness and is outside timing.

## Why the old final costs more

The old final assigns nine FP32 inputs to lanes0..8 of a 256-thread block per
output, initializes all other lanes to +0, publishes all lanes to shared memory,
and runs the descending128,64,32,16,8,4,2,1 shared tree. Publication plus the eight
merge stages require nine block barriers. Unit0 owns the final store. This is
unnecessarily large workgroup machinery for nine values.

The reused CubeK plane mapping assigns one output per 32-lane plane, four planes
per block. Each active lane starts at +0 and lazily adds its one partial; inactive
lanes retain +0. Explicit CubeCL ShuffleXorOp/FAddOp implement descending16,8,4,2,1.
Only lane0 performs the original epilogue/store. No cross-plane consumer exists,
so no shared publication or block barrier is needed. For nine inputs the nonzero
root is `(((p0+p8)+p4)+(p2+p6))+((p1+p5)+(p3+p7))`; the CPU model also includes all
zero additions, rather than reasoning only over this abbreviated expression.

This substitution saves about0.35–0.38µs in the fresh isolated-final measurements
and about0.39–0.47µs in the generated-graph GPU sums. The earlier2.3-versus1.2µs
comparison also included the producer-history effect established in cwk.6.
Its full1µs difference cannot honestly be assigned to the shared tree alone.
The exact internal source of the environmental effect remains parked in cwk.5.

The candidate retains one more shuffle than Triton, active-mask handling,
reconvergence and i64/metadata address work. Those are verified code differences;
this experiment does not assign an exact fraction of residual latency to each.
Changing register count or removing barriers is not used as a substitute for
measured improvement.

## Mature mechanisms reused and inspected

The existing [source inventory](../inventory.json) was checked against current
source bytes: all21 entries match their recorded SHA-256 values. Excluded source
trees were read directly; graph absence was not treated as absence of code.

| Reference | Finding and decision |
| --- | --- |
| Exact retained Triton final | Tensor1×16 reduction; masked p0..p8/+0;64 threads/block, gridB; replicated half-warps; XOR8,4,2,1; tid0 store; no shared reduction. Retained IR/PTX/cubin and executed SASS are compared without retuning. |
| Current CubeK/cubek-reduce | GlobalFullPlaneReduce maps block/y to independent outputs; PlaneReader distributes reads; lane0 owns write/commit. Reuse that mapping/ownership through our IR, without its allocator, planner or launcher. |
| CubeCL plane/shared machinery | ShuffleXorOp is available through the existing CUDA emitter. Generic FSum's ascending1,2,4,8,16 polyfill has a different order and is not substituted. Shared publication remains necessary in multi-consumer/fused reductions. |
| CUB/CCCL | FP32 WarpReduce uses predicated down shuffles with ascending offsets; BlockReduce can publish warp aggregates and synchronize. Useful ownership references, but stock order is not silently adopted. Local headers are CUDA13.1 source references, not production CUDA13.0 compilation. |
| Installed ATen/PyTorch | Reduce.cuh uses an8-lane dimension for B×9, private stride8/four chains and descending4,2,1, x0 ownership. The prior source model and actual GPU reference qualify, but stock sum_out was slower here; its runtime is not adopted. |

Revision identities and exact line ranges remain in the inventory. The mechanism
is the previously qualified component, not a fresh reducer invented for cwk.7.

## Numerical and boundary qualification

CPU qualification preceded the compiler-guard extension and neighboring GPU work.
All148 existing corpus cases preserve the frozen final bytes and unchanged
reference-or-better policy. Repeating/slicing their partial buffers at final
sizes1,4,8,9,10,16,17,31,32 gives1332 further exact FP32 comparisons with the old
256-lane tree. Direct controls cover NaN positions, ±Inf/opposite infinities,
negative zero, subnormals and finite cancellation. No tolerance changes.

The GPU extension passes324 comparisons across those sizes, all three storage
dtypes, random/direct inputs, output counts around the four-plane boundary and
storage canaries. Eight rejected descriptors cover0,33,64,negative/dynamic
extents, eight-plane geometry, non-owner stores and the wrong plan. These are
bounded candidate-versus-tree GPU parity controls, not an independent absolute
GPU oracle for every neighboring size or general qualification of arbitrary reductions.

All178 existing generated-graph rows preserve their recorded inputs, results,
ATen/FP64 diagnostics and the two recorded eager-relative failures. Another33
heldout/adversarial generated-graph cases match the frozen qualification rows.
The offset-input/nondefault-stream check passes through the existing validated
Kernel and current-stream launcher. Complete historical finite graph rows are
checked at recorded-value precision; component comparisons and canaries use
byte hashes. NaN/class/zero-sign graph checks retain their original assertions.

`controls.json` verifies27 regenerated IR/CUDA/PTX files for the normal partial,
normal final and valid plane final across three dtypes. `frozen-files.json`
protects201 retained component files. `analysis.json` also compares the timed
partial directly with the retained generated graph's partial artifacts.

## Measurements

This is one RTX3090/SM86 capture; environment identities are in `timing.json`.
Clocks were not forced and per-capture clock telemetry was not collected.
The raw distributions and rotated comparisons qualify this capture, not a
clock-normalized or cross-device performance guarantee.
All values below are µs, from six rotated same-process rounds with16 calls per
cell, including every measured call. Raw CUPTI traces, extrema, MAD and per-round
medians are retained. No favorable prefix or replacement capture is selected.

| Shape/scope | Frozen final | Plane final | Retained Triton final |
| --- | ---: | ---: | ---: |
| 8×1048576, isolated final only | 1.888 | 1.536 | 1.344 |
| 64×65539, isolated final only | 1.889 | 1.505 | 1.344 |
| 8×1048576, same frozen partial immediately before final | 2.306 | 1.951 | 1.632 |
| 64×65539, same frozen partial immediately before final | 2.464 | 2.110 | 1.793 |

The latter rows hold producer binary/input/output allocation fixed across the
three finals. They measure final execution separately from partial duration.
They do not investigate new hardware-state mechanisms.

| Generated workload | Frozen partial/final GPU sum | Plane-final GPU sum | Retained Triton device sequence |
| --- | ---: | ---: | ---: |
| 8×1048576 | 43.889 | 43.423 | 45.539 |
| 64×65539 | 23.957 | 23.571 | 22.965 |

The CubeCL columns execute real generated wrappers, including their ordinary
allocation/validation/current-stream path. The reference executes the exact
retained generated Triton partial/final cubins via the existing DriverKernel ABI,
with matching input bytes/split9 and unchanged configs. It is a **device-sequence
reference**, not a fresh Triton wrapper or a host-latency comparison. This keeps
the acceptance process Triton-free/GIL-disabled while making the GPU comparison
same-process and rotated. Inter-kernel GPU gaps are retained separately and excluded
from these GPU-duration sums. No CUDA Graph replay is used.

The timed partial medians remain41.54/41.46µs at8×1M and21.43/21.40µs at64×65539
for frozen/plane graphs. Each partial binary is identical. The small differences
are not credited as a partial optimization. The plane final saves time in every
round, but the secondary complete device workload remains behind the reference.

## IR → CUDA → PTX → executed SASS

`retained/` contains the generated semantic/physical descriptors and IR/CUDA/PTX
for both timed variants. The plane final uses existing signed-i64 semantic
indices, FP32 accumulation, original output casts and conservative CUDA13.0
NVRTC options (`--fmad=false`); production storage/runtime ownership is preserved.
Retained Triton frontend/target artifacts are linked through its exact hashes.

The full executed Nsight command/provenance is in `ncu-command.json`; the command
uses the PATH-resolved2026.3.1 runner, matching section assets, SourceCounters,
three launches, kernel replay and no clock/cache controls. The replay output is
code/counter evidence only, not warm performance evidence. The loaded functions
are retained in `final-code.ncu-repz`, with raw metrics and disassembly.

Exact command (also retained as structured arguments):

```sh
/home/imi/tools/nvidia/nsight-compute-2026.3.1/ncu --target-processes all --profile-from-start off --replay-mode kernel --cache-control none --clock-control none --launch-count 3 --section-folder /home/imi/tools/nvidia/nsight-compute-2026.3.1/sections --section SourceCounters --metrics smsp__inst_executed.sum,launch__registers_per_thread,launch__registers_per_thread_allocated,smsp__sass_inst_executed_op_barrier.sum,smsp__sass_inst_executed_op_shared_ld.sum,smsp__sass_inst_executed_op_shared_st.sum --export stage5/artifacts/reduction-structure/final/graphs/final-code --csv stage0/.venv/bin/python -m stage5.reduction_final_graphs counter
```

Raw selected fields from `ncu-metrics.csv` for the64-output launch:

| Kernel | warp instructions executed | allocated registers/thread | shared loads | shared stores |
| --- | ---: | ---: | ---: | ---: |
| cubecl5_b1f1603b0e5b0a49 | 34752 | 16 | 1600 | 1280 |
| cubecl5_191aecc1359e95bb | 2816 | 16 | 0 | 0 |
| triton_per_fused_sum_1 | 3904 | 16 | 0 | 0 |

`executed-sass-counts.txt` reports4608 executed warp-level BAR operations for the
old final and zero for plane/Triton. Plane has320 warp-level SHFL executions
(64 output planes×5); Triton512 (128 participating warps×4). BSSY/BRA remain64
warp-level executions for the plane final, versus no executed Triton BRA/BSSY.
The static site counts remain9barriers/5shuffles/4shuffles respectively. Dynamic
instruction totals include the differing grid/participation geometries and are
not a claim that fewer total instructions guarantees a shorter critical path.
`driver-resources.json` records driver-query final registers10/15, separately
from the16/16 allocated launch registers above; the partial stays40 registers. No
local spills are introduced. No offline CUDA13.1 binary substitutes for this
driver-loaded SASS.

## Validation, preserved failures and limits

The first graph harness attempt imported Inductor before overlay installation
and stopped before graph execution. A second attempt used short-first/guarded
history against a large-first/inherited retained capture and correctly failed
the exact-row check. Both attempts and their source/logs remain in `failures/`.
The accepted run uses the retained large-first/inherited split9 setup. A distinct
wrapper-source comment prevents Python module cache reuse across diagnostics;
the timed trace census additionally requires the two distinct final entrypoints.

The primary numerical/timing evidence consists of108 admitted raw traces, not
the preserved failed setup attempts. All18 established gates pass in `regressions/gates.json`;
`regressions/gate-status.json` reports complete, and the historical-preservation
record confirms restoration. The required review found the change sound with one actionable cache-environment
ordering issue in the extras harness. It is fixed, and all33 extra cases plus the
offset/current-stream check pass again under an explicitly asserted cache path.
The repeat preserves the original rows and IR/CUDA/PTX bytes. The analyzer now
independently checks substitutions for all three storage types. Details are in
[review/disposition.md](review/disposition.md); the original capture is preserved.

Automatic eligibility remains **unaccepted**. This demonstrates one pure,
owner-only, scalar FP32 sum mechanism with bounded small constant extents and
three output storage dtypes. It does not qualify generic fused consumers,
arbitrary epilogues, dynamic final extents, all devices or general reductions.
Normal independent-plane FSum still has a different arithmetic order and cannot
share a production cache identity with this probe; enabling
this mechanism in ordinary selection requires a separate explicit decision.
The remaining final/secondary workload gap is visible, and cwk.5 remains parked.
