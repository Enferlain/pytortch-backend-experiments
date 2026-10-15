# Stage 5 follow-up: matched reduction references

> **Supporting experiment record.** Start with [the current Stage 5 findings](STAGE5_REPORT.md). Results and recommendations below retain this pass's historical context.

## Result and decision

**The matched investigation is complete; a new default physical policy is not
accepted.** No production scheduling, compiler, launcher, numerical eligibility
or tolerance was changed in this investigation.

The comparison supports a narrower conclusion than either "compensation is
always needed" or "copy Triton." Normal Inductor's initial shape changes the
logical split and Triton's physical tile. The efficient split reference is
faster than the current CubeCL path on representative long-row workloads.
Compensation improves accuracy, but the cheaper existing CubeCL candidates and
the reference's work assignment must be evaluated together before choosing a
default. Removing compensation alone does not close every performance gap.

The user's latest requirement in [the accumulation report](STAGE5_ACCUMULATION_REPORT.md)
governs this decision: practical behavior/performance, PyTorch operation
contracts, mature implementation references, and no materially slower default
justified solely by closer agreement with FP64. Matching every reference bit or
reproducing its numerical weaknesses is not the objective.

## Executed comparison

Both environments use PyTorch `2.14.1+cu130`, commit
`5c4886908584029761b579af026dcfb627c84070`, CUDA 13.0 and the RTX 3090 / SM86.
Both interpreters are CPython 3.14.5t. CubeCL remains GIL-disabled with Triton
unavailable and no loaded Triton native module. Triton 3.8.0 enables the GIL in
the reference environment; no `PYTHON_GIL` forcing is used. No host-time parity
claim is made.

`torch.compile(lambda x: x.sum(-1), fullgraph=True, dynamic=True)` enters normal
Dynamo/AOTAutograd/Inductor. CubeCL uses the ordinary Stage 5 `first_use` backend;
the reference uses `compile_fx` with the established matched settings:
`emulate_precision_casts=True`, `joint_graph_constant_folding=False`,
`eager_numerics.division_rounding=True`, `eager_numerics.disable_ftz=True`,
`eager_numerics.use_pytorch_libdevice=True`, CUDA graphs off, graph cache off and
one compile thread. This plain sum does not exercise transcendental/division
equivalence: those controls alone are not evidence of equivalent arbitrary math.

Fresh processes exercise two histories:

- **short first:** the original `64×31`, `64×257`, `8×257`, `8×65539` sequence;
- **long first:** initial capture on `8×65539`.

Each checks the same **109 FP32 inputs**: the original RNG history; four seeds
at 11 lengths from 31 through 1,048,576; and 64 held-out/validation seeds at
`8×65539`. Six additional storage cases cover FP32/FP16/BF16 with 8/64 rows.
All **115 input byte hashes match** across the four backend/history captures and
the explicit CubeCL baseline diagnostic. Eager and FP64 reference vectors are
identical. Three special-value fixtures check NaN, both infinities, mixed
infinities and negative-zero inputs, including zero-result sign, per dtype.

Each process captures three Dynamo graphs, one per storage dtype. The sampled
compatible FP32 sizes reuse their graph; runtime dimensions remain metadata.
This is the normal dynamic-shape path, not PTX-only dynamic length support.

## Numerical findings

The original unchanged `rtol=atol=2e-5` comparison is retained as a diagnostic,
with every failing case listed. It is not silently waived or rewritten as a
success. Core 109-case results:

| Executed path | Eager-relative failing inputs | FP64 diagnostic failing inputs |
| --- | ---: | ---: |
| CubeCL first_use, short first | 1 | 0 |
| CubeCL first_use, long first | 1 | 0 |
| Configured Triton, short first | 16 | 17 |
| Configured Triton, long first | 2 | 0 |
| CubeCL explicit baseline, long first | 1 | 1 |

The remaining CubeCL eager-relative failure is seed 30003, whose ATen result
was already shown to have the larger FP64 error. Long-first Triton also fails
that comparison and narrowly fails the original RNG-history row 7:

| Original row 7 | FP32 value |
| --- | ---: |
| Eager ATen | 2.3122177124023438 |
| Triton short first | 2.312225341796875 |
| Triton long first | 2.312286376953125 |
| Compensated CubeCL, short first | 2.3122305870056152 |

All inputs are byte matched. These are actual ordering/topology differences,
not different RNG samples. Long-first Triton still passes the same tolerance
against FP64 for every core input. PyTorch documents order-dependent floating
results and does not require bitwise equality for mathematically equivalent
computations. That contract does not make every error acceptable or establish
a universal error bound for these plans. [PyTorch 2.14 numerical accuracy](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html)

The direct physical comparison re-executes four existing CubeCL plans on those
109 identical inputs; it does not alter production eligibility:

| Physical plan | Eager-relative failures | Differences beyond 2e-5 from long-first Triton |
| --- | ---: | ---: |
| Original shared tree | 6 | 6 |
| Compensated shared tree | 1 | 0 |
| Four private accumulators + shared tree | 2 | 4 |
| Existing scalar plane/shared reduction | 6 | 5 |

This is evidence about the tested inputs, not a mandate to choose the reference's
exact output. A diagnostic attempt to run the existing vector plane plan on long
rows was rejected by the current numerical eligibility rule. That rejection is
preserved; the validation was not disabled to obtain a timing.

## Actual physical strategy

The semantic operation is the same FP32-accumulated sum, with explicit storage
conversions and direct logical indexing. Normal `Reduction.num_splits` /
`InductorChoices.reduction_split_factor` selects one stage for the initial short
FP32 graph, and **nine partials plus a final reduction** for the long-first graph.
Both backends inherit this logical split. Initial FP16/BF16 capture occurs on a
long extent, so those fixtures also split even in the short-first process.

Captured Triton configuration and IR show:

- short-first FP32: `XBLOCK=2`, `R0_BLOCK=32`, two warps. Each row has a long
  rolling accumulation chain; at N=65539 the loop runs 2,049 iterations;
- long-first FP32: nine partials per row, `XBLOCK=1`, `R0_BLOCK=2048`, 16 warps.
  The 2,048 logical values map to four private FP32 values per thread. The emitted
  LLVM uses four independent rolling sums followed by a balanced private merge,
  warp shuffle reductions, and a shared merge with one workgroup barrier;
- final reduction: nine partials, a separate generated Triton kernel.

This pattern agrees with the versioned `reduceWithinThreads` tree and
`reduceWithinWarps` lowering in [Triton 3.8 ReduceOpToLLVM](https://github.com/triton-lang/triton/blob/v3.8.0/lib/Conversion/TritonGPUToLLVM/ReduceOpToLLVM.cpp).
The saved LLVM/PTX, not a guessed translation of `tl.sum`, establishes the
executed layout and arithmetic. Ordinary FP32 additions remain uncontracted;
no compensation or FP64 arithmetic appears in this reference. Its PTX loads
these four values with scalar memory instructions; the LLVM vector arithmetic
is not proof of a contiguous vector memory transaction or an alignment
requirement to reproduce in CubeCL.

Current CubeCL short-first FP32 uses eight scalar256 workgroups, one per row,
with TwoSum private accumulation and a compensated shared binary tree.
Long-first uses the same semantic nine-part split, but partial kernels use the
scalar256 compensated tree. The final short reduction uses the existing
first-use selection (including independent planes). Neither logical split nor
dtype/storage handling needs replacement.

## Device cost

Primary measurements reuse the established Stage 5 method: seven independent
CUPTI sums of the actual generated kernels, with the same 50M-cycle activity
warmup outside every region. Kernel names/counts are checked, including native
CubeCL launch counts. CPU enqueue gaps and GPU annotations are excluded. These
are profiler-qualified, separate-process device measurements, not unprofiled
end-to-end latency or definitive small-percentage optimization claims. Every
sample and MAD is retained in [analysis.json](stage5/artifacts/reference-reduction/analysis.json).
The analyzer regenerates or verifies every numerical/timing/counter table and
quoted cost paragraph, then binds the report hash to the measured source data.
Its default verification rejects stale report numbers.

| Input | CubeCL short-first µs | Triton short-first µs | CubeCL long-first µs | Triton long-first µs |
| --- | ---: | ---: | ---: | ---: |
| FP32 8×65539 | 11.431 | 198.343 | 8.008 | 6.120 |
| FP32 64×65539 | 25.634 | 198.941 | 28.346 | 24.617 |
| FP16 64×65539 | 19.079 | 15.277 | 19.172 | 15.271 |
| BF16 64×65539 | 18.993 | 15.287 | 19.168 | 15.276 |
| FP32 8×131071 | 22.364 | 370.326 | 12.591 | 9.114 |
| FP32 8×1048576 | 147.067 | 2945.256 | 65.041 | 48.444 |

The short-first reference's bad reuse is not a sufficient target for declaring
CubeCL competitive. The efficient long-first path exposes the relevant gap.
Conversely, the long-first reference incurs unnecessary split overhead when
reused on short rows; it is not a universal scheduling oracle.

The same-topology **explicit CubeCL baseline diagnostic** performs the long-first
8×1048576 case in 48.487µs versus 65.041µs for first_use and 48.444µs for Triton.
Its eager/FP64 diagnostic differences remain listed above. At 8×65539, its
6.624µs versus Triton's 6.120µs. Its final reduction also differs
from first_use's selection, so this is not an isolated compensation-only ablation.

The prepared-native physical component confirms the direct arithmetic cost:
8×65539 is 7.145µs original, 8.708µs compensated, 9.250µs with four accumulators,
and 7.104µs for the existing scalar plane/shared plan. These component timings
are kept separate from the compiled graph's multi-kernel sums.

## Nsight confirmation

WSL-native Nsight Compute **2026.3.1.0 build 38829034** collected hardware counters
and LaunchStats/Occupancy/SpeedOfLight. Production CubeCL compilation remains
CUDA 13.0 NVRTC. No CUDA 13.1 offline tool output substitutes for these captures.

Actual collection commands (the environment supplies the existing CUDA_PATH,
TRITON_LIBCUDA_PATH and TRITON_LIBDEVICE_PATH from the reproduction section):

```bash
ncu --target-processes all --kernel-name regex:cubecl5_ --launch-skip 2 --launch-count 1 --section-folder ~/tools/nvidia/nsight-compute-2026.3.1/sections --section LaunchStats --section Occupancy --section SpeedOfLight --csv --log-file stage5/cache/reference-reduction/ncu-compensated.csv stage0/.venv/bin/python -m stage5.reference_reduction_ncu --plan compensated --output stage5/artifacts/reference-reduction/ncu-compensated.json
ncu --target-processes all --profile-from-start off --launch-count 2 --section-folder ~/tools/nvidia/nsight-compute-2026.3.1/sections --section LaunchStats --section Occupancy --section SpeedOfLight --csv --log-file stage5/cache/reference-reduction/ncu-triton-large-primed.csv /home/imi/Projects/sd-scripts/.venv-wsl/bin/python -m stage5.reference_reduction_probe --impl triton --history large_first --ncu-case float32-8 --output stage5/artifacts/reference-reduction/ncu-triton-large.json
```

Raw CSV values, curated without normalization:

| Metric | CubeCL compensated component | Triton partial | Triton final |
| --- | ---: | ---: | ---: |
| Grid Size | 8 | 72 | 8 |
| Block Size | 256 | 512 | 64 |
| Registers Per Thread | 36 | 29 | 16 |
| Dynamic Shared Memory Per Block (bytes) | 2048 | 64 | 0 |
| Waves Per SM | 0.02 | 0.29 | 0.01 |
| Duration (ns) | 14048 | 6048 | 2688 |
| Compute (SM) Throughput (%) | 2.47 | 12.63 | 0.07 |
| Memory Throughput (%) | 16.53 | 42.33 | 1.28 |

The CubeCL capture is a direct **existing physical component**; the Triton
capture is the normal compiled split graph. Nsight replay/cache/clock conditions
perturb timing, so the durations are not substituted for the primary table.
Grid eight on 82 SMs directly establishes underfill; the counter data supports
investigating work assignment. It does not isolate compensation as the only
cause. Triton's compiler reports 12 registers for the final launcher, while
Nsight reports 16 physical registers; those quantities remain distinguished.

The first reference Nsight attempt hit a Nix host-linker wrapper error while
building Triton's driver helper. An ordinary, unprofiled run primed the exact
helper cache; the repeat collected both kernels successfully. This is recorded
as an environment-qualified recovery, not a counter-permission failure or an
unavailable profiling tool.

## Changes, controls and remaining work

Only diagnostic source and documentation are added:

- `reference_reduction_probe.py`: normal compiled backend/history capture,
  input hashes, dtype/special checks, exact kernel census and selected Triton
  launcher observer, restored after collection;
- `reference_reduction_physical.py`: four already-existing physical candidates,
  byte-matched input comparison and the established native batch benchmark;
- `reference_reduction_ncu.py`: one existing pure physical kernel, avoiding
  nested CUPTI/autotuning during Nsight;
- `reference_reduction_analyze.py`: rejects incomplete/mismatched captures,
  preserves every numerical difference and curates executed target artifacts.

Initial attempts are preserved separately: an obsolete PyCodeCache observer;
unqualified sparse/power-regime timings; the existing CUDA graph-capture veto;
the long-vector eligibility veto; an empty first Nsight capture; and the Nix
linker failure. Completed packets are not overwritten by failed results.
CUDA graph support was neither enabled nor bypassed for timing.

The prior 17 regression gates remain the executed production baseline; they
are not relabelled as new runs. No production source changed, so this task's
verification is the actual matched five-process matrix, physical corpus, real
Nsight counters, analysis assertions, nine failure/data-integrity controls and
Python quality checks. The conservative
FP patch remains `9ae155efb01454be6cc61766db3564fce49bcf8495af753c60afab5df07c97d2`.

**Next bounded direction:** evaluate tiled private FP32 accumulation with a
plane/shared merge under the existing semantic/physical boundary, and evaluate
split choice for long rows without treating the first-shape hint as a permanent
performance decision. Reuse proven patterns from the inspected implementations;
do not reproduce their weaknesses or invent a new general planner. Keep
compensation available as a measured accuracy alternative rather than declaring
the existing >4096 mandate final. The selection must be tested on these exact
inputs and practical workload shapes before changing the default.

`cwk` remains open for that selection and numerical-gate qualification. No
tolerance has been widened, no new physical algorithm implemented, no next stage
started, and nothing committed by the agent.

## Reproduction

```bash
export STAGE0_CARGO_PROFILE=release
export CUDA_PATH="$PWD/stage0/cuda13-toolkit/nvidia/cu13"
export TRITON_LIBCUDA_PATH=/usr/lib/wsl/lib
export TRITON_LIBDEVICE_PATH="$CUDA_PATH/nvvm/libdevice/libdevice.10.bc"
# Fresh process for each implementation/history; no PYTHON_GIL forcing.
stage0/.venv/bin/python -m stage5.reference_reduction_probe --impl cubecl --history short_first --output stage5/artifacts/reference-reduction/cubecl-short-warm.json
stage0/.venv/bin/python -m stage5.reference_reduction_probe --impl cubecl --history large_first --output stage5/artifacts/reference-reduction/cubecl-large-warm.json
/home/imi/Projects/sd-scripts/.venv-wsl/bin/python -m stage5.reference_reduction_probe --impl triton --history short_first --output stage5/artifacts/reference-reduction/triton-short-warm.json
/home/imi/Projects/sd-scripts/.venv-wsl/bin/python -m stage5.reference_reduction_probe --impl triton --history large_first --output stage5/artifacts/reference-reduction/triton-large-warm.json
stage0/.venv/bin/python -m stage5.reference_reduction_probe --impl cubecl --history large_first --cubecl-policy baseline --output stage5/artifacts/reference-reduction/cubecl-baseline-large.json
stage0/.venv/bin/python -m stage5.reference_reduction_physical --output stage5/artifacts/reference-reduction/physical.json
# Prime Triton's host helper outside Nsight if the Nix linker wrapper fails.
/home/imi/Projects/sd-scripts/.venv-wsl/bin/python -m stage5.reference_reduction_probe --impl triton --history large_first --ncu-case float32-8 --output stage5/artifacts/reference-reduction/ncu-triton-large.json
# Run the two Nsight commands above, then validate and curate their raw records.
python -m stage5.reference_reduction_analyze --update-report
python -m stage5.reference_reduction_analyze
```
