# 5nb — actual CubeK RMS inference trial

Current decisions live in [STAGE5_REPORT.md](../../../STAGE5_REPORT.md).
The selected [8x0 boundary](../reuse-first-selection/README.md) is implemented.
**The reuse path works and qualifies numerically, but this configuration loses
on performance. The specified stop condition is reached.** It remains an
explicit diagnostic; no alternative mechanism, tuning sweep or automatic
eligibility was introduced. Stage5 remains active; Stage6 is deferred.

## Actual implementation

`stage5/rms_probe` composes the pinned, unchanged CubeK
`Reader::new` → `PlaneReader::{new,length,read}` →
`<Sum as ReduceInstruction>::{null_accumulator,reduce,plane_reduce_inplace}`.
It uses the selected scalar FP32 operand/accumulator, Lazy merge,
32×8 workgroup, eight independent rows per block and `ceil(B/8)` blocks.
`BoundChecks::Branch` protects tail reads; whole idle planes take a uniform
branch. The stock plane sum returns a value to every participating lane.
The `row < rows` guard also prevents the metadata coordinate map's modulo
from wrapping an idle plane onto row zero; it must precede every reader access.

The narrow virtual-input adapter calls **unchanged**
`stage0/src/descriptor2.rs::emit` for the original input cast chain and square.
A metadata-only virtual `[B,1]` supplies the stock reader's coordinate map.
The output adapter replays the original division, epsilon addition, rsqrt,
multiply, cast barriers and disjoint reduction-scope stores through the same
emitter. It invokes neither CubeK's lane-zero global writer nor its host
client/allocator/launcher. A separate diagnostic scheduling extension emits
the adapter into actual Inductor-generated wrappers; no artifact substitution
or compiler/selector monkeypatch is used.

Python and native code admit only the three retained RMS semantic templates.
Native validation also fixes the entire physical-plan envelope and entrypoint.
Live calls retain the common tensor/layout/storage validation and restrict
widths to the selected boundary set and positive row counts through65.
Twelve mutated descriptors/mechanism envelopes are rejected before IR emission;
21 live invalid extent/layout/dtype/device/alias controls reject before launch.

The isolated compiler expands actual components into KernelBuilder IR and the
existing `CppCompiler<Cuda>`. CUDA13.0 NVRTC (`--fmad=false`) produces PTX, loaded
by the existing native driver launcher on the current PyTorch stream. PyTorch
owns tensor storage, metadata/readiness events and allocator lifetime handling.
No production compiler, planner, catalog or runtime source changed.

CubeK remains `4ca0235e7e4fddd82f67ec27bbc3d48c085be25a`, CubeCL
`6bba5fdf04206d9c3710dfbc46aa9850450ab405` with its retained conservative FP
patch. The pre-existing isolated compatibility copy is reused; six exercised
component files match upstream byte-for-byte. Registry dependency pins match
the retained component-probe lockfile. See `component-sources.json`,
`environment.json` and the isolated `Cargo.lock`.

## Numerical gate before GPU implementation

Matched configured compiled-Triton RMS outputs were obtained first in the
separate reference environment. Then the exact CPU model qualified all66 cases
and945,534 output elements before the new Rust GPU body was written:

- Nine existing RMS corpus dtype/shape cases:64×1024,64×255,16×4099 in
  FP32/FP16/BF16.
- Twelve boundary widths31/32/33,255/256/257,1023/1024/1025,4095/4096/4099,
  with separate dynamic-range, skewed, alternating, ±zero, subnormal-input,
  square-overflow, NaN and signed-infinity rows in every storage dtype.
- Seven whole/partial output-block row counts at width33 in every dtype.

The model rounds the square to FP32, starts each lane at+0, adds the scalar
`32*k+lane` chain in ascending k with+0 tails, and performs simultaneous XOR
butterflies1,2,4,8,16. It checks all32 lane sums and replays each lane's original
epilogue. The frozen final's descending-order qualification was not reused.

The existing `reference-or-better-v1` policy and unchanged eager-relative
tolerances pass. NaN/Inf classes, zero/nonzero classes and zero/Inf signs are
checked separately. GPU execution passes the same66 cases, canaries, original
input preservation, all21 live rejection controls and nondefault-current-stream
offset views in all three dtypes. CPU/GPU final bits differ in some cases;
the CPU rsqrt model is not asserted to be a bitwise CUDA oracle. Both full
outputs are qualified against the matched reference and eager result. No
coarse compensated-reducer guard or tolerance relaxation is used.

`reference.pt` contains the frozen inputs, compiled reference, eager outputs
and FP64 diagnostics; `cpu-model.pt` contains all modeled lane sums and outputs.
`reference.json`, `cpu.json`, `cpu-zero-class.json`, `gpu.json` and
`admission.json` retain results and hashes. Reference code/cubins and the
reference/CPU runner snapshots are preserved separately. The reference process
enables the GIL through installed Triton; CubeCL acceptance remains Triton-free
and GIL-disabled. No host timing comparison is made.

## Isolated and generated-graph timing

These are **CUPTI kernel durations**, not queue-inclusive CUDA event intervals,
host time or profiler replay durations. The primary isolated comparison ran
first. Generated-graph comparisons then used the actual wrappers and ordinary
`first_use` selections, in the same process, six alternating-order rounds,
16 calls per path per round. No synthetic GPU activity, warmup workaround or
memory primer was introduced. Each generated graph has one kernel.

| Primary FP32 64×1024 scope | Ordinary µs | Actual CubeK components µs |
| --- | ---: | ---: |
| Prepared isolated kernel | 2.333 | 4.132 |
| Real generated graph kernel | 2.340 | 4.162 |

| Generated graph | Ordinary µs | Actual CubeK components µs |
| --- | ---: | ---: |
| FP32 64×255 | 2.238 | 2.464 |
| FP32 16×4099 | 3.745 | 15.069 |
| FP16 64×1024 | 2.307 | 4.419 |
| FP16 64×255 | 2.240 | 2.625 |
| FP16 16×4099 | 7.992 | 27.221 |
| BF16 64×1024 | 5.951 | 12.106 |
| BF16 64×255 | 5.853 | 7.099 |
| BF16 16×4099 | 11.674 | 43.157 |

The component loses **all six round medians in every case**, including the
primary isolated comparison. FP16 16×4099 shows substantial first-round drift
(ordinary3.779µs versus roughly8µs later; component roughly12.7µs versus27.5µs
later), with smaller drift also visible in BF16. Every round is retained;
the component still loses each matched round. No hardware-state cause is
claimed or investigated here. The primary graph round medians are
ordinary2.339/2.334/2.338/2.335/2.372/2.357µs and
component4.161/4.162/4.156/4.157/4.164/4.162µs.
Absolute cross-dtype latencies are not treated as a hardware-state diagnosis.
The data establish a paired loss for this configuration on this RTX3090,
not a general ranking of CubeK versus custom implementations.

`graphs.json` records selections, input/output identities and per-round samples.
All120 accepted raw timing traces are independently checked, including their
exact kernel census and duration arrays. Generated wrappers are retained.
Guard-only follow-up corrections preserve IR/CUDA/PTX after entrypoint renaming;
the final qualification still applies to the actual timed body.

## IR/PTX/executed SASS attribution

The same two timed FP32 kernels were captured with the PATH-resolved native
Nsight Compute2026.3.1 and matching section assets. Exact argv/version/SHA-256
are in `ncu-provenance.json`; `counter-command.txt` reproduces the actual argv.
The capture uses SourceCounters/LaunchStats, two launches,
`--profile-from-start off --replay-mode kernel --cache-control none
--clock-control none`. It does not substitute offline ptxas or cold replay
durations for warm timing. `counter-kernels.json` binds names and PTX hashes.

Relevant raw `ncu-raw-import.csv` fields (data rows3/4):

```text
Kernel Name,smsp__inst_executed.sum,launch__registers_per_thread,smsp__sass_inst_executed_op_shared_ld.sum,smsp__sass_inst_executed_op_shared_st.sum
cubecl5_b9862d901ad45222,35840.000000,46,896.000000,512.000000
cubecl5_rms_cubek_4c13774397ff2c68,47680.000000,26,0.000000,0.000000
```

The complete instruction tables are in `executed-sass.txt`. Their executed
instruction sums match the raw metric exactly. SourceCounters-derived executed
warp instruction counts show BAR896→0, SHFL0→320, IMAD2,944→12,864 and
IADD3 3,456→6,400. The raw barrier metric is unavailable; BAR counts come from
the instruction table, not an invented metric value. Both kernels have zero
local-memory bytes. Ordinary uses256 dynamic shared bytes; the component uses0.

The primary ordinary selection is a64-thread shared tree,64 blocks. The stock
component configuration packs eight rows per block, so only8 blocks participate
and each lane has twice the private input/output chain length. Its generic
reader retains a rank/coordinate walk, including a dynamic unsigned64-bit
remainder by the output extent; IR/CUDA/PTX retain this map. The stock collective
lowers to a shuffle loop with ascending masks. Fewer registers/shared operations
do not offset the larger executed instruction count and more concentrated work.
Those are source/code-supported explanations for the loss, not an additive
microsecond decomposition or a separately isolated causal proof. No second
mechanism was implemented to investigate them.

## Validation, preserved failures and limits

The offline analyzer verifies all120 accepted traces, source/code identity,
12 static rejection controls, the numerical gate records, and **12,784**
pre-existing source/evidence files. Upstream revision/diff identities and all
frozen long-row sources are unchanged. The prior8x0 packet is preserved.
Ruff F checks, Ruff formatting, isolated Cargo build/format and
`git diff --check` pass. Required review returned **no blockers**; its low-severity
findings and the affected rechecks are recorded in `review/disposition.md`.
The analyzer also verifies unchanged component-copy bodies, retained registry
pins, NCU command/provenance and environment artifacts against
`evidence-integrity.json`, including the exercised compiler/tool binaries.

The following setup/collection failures are preserved rather than hidden:

- A Python integer fixture exceeding i64 needed an explicit float conversion;
  no candidate arithmetic was involved.
- Initial unlocked resolution selected two newer registry packages; that build
  was discarded. Accepted builds use the retained dependency pins and `--locked`.
- A manifest field spelling error prevented the first host build.
- IR verification caught the adapter passing a mutable expanded handle instead
  of loading the scalar value. The failed IR never reached NVRTC/GPU execution;
  `read_value` fixes the existing boundary without changing the mechanism.
- The first graph collection hit Dynamo's default recompile cap after partial
  collection. The **whole attempt**, including every trace, is excluded from
  accepted timing. A complete rerun used the same256 cap as the reference
  harness; no favorable prefix was selected.
- Resolving the Python interpreter symlink bypassed the isolated environment
  in the first counter command. It failed to import Torch and launched no GPU
  work; the corrected command keeps the virtual-environment interpreter path.

Graph project/status/search/coverage tools return `Transport closed`; all source
verification uses direct reads and scoped searches. No graph completeness claim
is made. This is one actual reusable-component configuration, one normalization
family, one GPU and a bounded corpus. It establishes runtime-free semantic
composition and a qualified negative performance result. It does not evaluate
other CubeK algorithms, tile layouts, tuning policies or broader normalization
fusion. cwk.5 stays parked; the frozen partial, split9 and precision/storage/
stream/lifetime policies are untouched.

Reproduction commands from the repository root:

```sh
/home/imi/Projects/sd-scripts/.venv-wsl/bin/python -m stage5.rms_trial reference
stage0/.venv/bin/python -m stage5.rms_trial cpu
CARGO_TARGET_DIR="$PWD/stage5/cache/reuse-first-rms/target" cargo build --release --locked --offline --manifest-path stage5/rms_probe/Cargo.toml
stage0/.venv/bin/python -m stage5.rms_trial gpu
stage0/.venv/bin/python -m stage5.rms_trial admission
stage0/.venv/bin/python -m stage5.rms_trial graphs
# The exact NCU capture argv is in counter-command.txt; then export its report:
ncu --import stage5/artifacts/reuse-first-rms/primary-code.ncu-repz --csv --page raw > stage5/artifacts/reuse-first-rms/ncu-raw-import.csv
ncu --import stage5/artifacts/reuse-first-rms/primary-code.ncu-repz --page source --print-source sass > stage5/artifacts/reuse-first-rms/executed-sass.txt
python -m stage5.rms_trial_analyze
```

Snapshot the retained packet before rerunning these fixed-output commands.
Do not overwrite the original evidence to obtain a different performance result.
