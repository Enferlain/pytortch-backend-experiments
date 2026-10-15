# Uncompensated private-merge qualification

Current findings belong in [STAGE5_REPORT.md](../../../../STAGE5_REPORT.md).
This packet extends the existing structure investigation; it is not another root
stage report. Historical captures in the parent directory are unchanged.

The subsequent [complete work-assignment/topology follow-up](topology/README.md)
resolves the numerical question with a CPU model and one reference-derived
512-thread GPU candidate. That candidate qualifies numerically but remains
materially unattractive for FP32; the current recommendation is to stop further
long-row optimization here. The merge-only results below remain historical
evidence, including all failures.

## Result and decision

**Partial: the single candidate is not admitted.** It fixes the original
held-out failure at comparable cost but fails three elements in an existing
finite adversarial fixture against efficient configured Triton.

One candidate changes only the private four-chain merge from
`(a0+a2)+(a1+a3)` to `(a0+a1)+(a2+a3)`. It retains scalar256 work assignment,
protected load batching, FP32 chains, increasing-XOR plane reductions, two
uniform barriers, reduced-value publication, the final kernel, and all storage
precision barriers. No added compensation, fast math, vector loads, planner,
PyTorch source patch or CubeCL checkout patch.

Explicit experimental `tuning="adjacent_merge_diagnostic"` reproduces this
candidate for diagnosis; it is not a qualified working replacement. Automatic
eligibility and `first_use` remain
unchanged; compensation remains present for continuity/diagnosis, not an accepted
final policy. `cwk` stays open for practical default/performance qualification.

## Rounding mechanism

`trace.json` reproduces the old load-batch partial buffers and final output
bitwise with a CPU FP32 arithmetic model, using the exact original CUDA-generated
input hash. Its three CPU arithmetic ablations are not GPU candidates.

For heldout_10017 output4, cumulative error relative to the recorded FP64
diagnostic is:

| Boundary | Old halving private merge | Adjacent private merge |
| --- | ---: | ---: |
| Private chains, summed without further FP32 rounding | -2.71794124e-6 | -2.71794124e-6 |
| After private merge | +4.64882032e-6 | -3.87772161e-7 |
| After intra-plane merge | -3.51074777e-5 | +7.80786650e-6 |
| Partial buffers, before final rounding | -2.36633859e-5 | -7.75202352e-7 |
| Final output | -2.55707346e-5 | -2.68255098e-6 |

The final kernel contributes -1.90734863e-6 in both cases. The old result is
0.11673736572265625, adjacent is 0.11676025390625, efficient configured Triton is
0.1167755126953125, and FP64 is 0.1167629364572349. The adjacent result passes
the unchanged per-element policy. Reversing only the collective butterfly in
the CPU model instead gives error +2.78350271e-5: changing every reference-like
feature at once would obscure this mechanism.

The arithmetic is permitted reduction-order variation, not an identified
miscompile. The adjacent order appears in the retained efficient Triton LLVM/
PTX evidence described by the parent report. It uses exactly three ordinary
FP32 additions, as did the old merge. It is not universally more accurate:
`reduction_merge_checks.py` retains a counterexample where the old order gives
the exact sum and adjacent loses a unit. Qualification is corpus/contract based;
FP64 maximization and exact Triton imitation are not objectives.

That counterexample was also exercised through the actual fixed-nine-part
CubeCL kernels and an efficient configured-Triton graph, reusing the finite
adversaries in the old accumulation suite. It is a real qualification failure,
not merely a theoretical accuracy caveat:

| Input nonzeros at coordinates 0,256,512 | Old pairing | Adjacent | Efficient Triton |
| --- | ---: | ---: | ---: |
| 1e8,1,-1e8 | 1 | 0 | 1 |
| -1e8,-1,1e8 | -1 | 0 | -1 |
| 2^24,0.25,-2^24 | 0.25 | 0 | 0.25 |

Our 256-unit assignment puts the three terms into three chains of the same
unit. Triton's executed 512-thread assignment puts the first/third terms into
one thread and the small term into a different thread. Thus the same adjacent
private merge is not the same numerical topology. Partial-buffer captures locate
the loss before the unchanged final kernel. The ATen reference also returns zero
in these three rows; this is not evidence of a CubeCL semantic miscompile. It
nevertheless fails the chosen corresponding-Triton-or-no-worse-FP64 policy.
No tolerance is changed and no second candidate is implemented to bypass it.

## Qualification and cost

- 148 direct checks: 109 exact original cases, six storage neighbors, 32 new
  held-out seeds selected after the candidate decision, and one non-default
  stream/allocator-pressure repeat. The CPU model matches executed partial and
  final bytes for both plans in every case. No candidate qualification failures;
  old baseline still fails heldout_10017. Of 1352 output elements, 1351 qualify
  by reference tolerance, one by no-greater FP64 error. Special classifications
  and zero signs pass separately for all three storage dtypes.
- One additional five-row existing finite adversarial case makes **149 direct
  checks overall**, with **one failed case / three failed elements**. The other
  two rows (preserved subnormals and alternating +/-1024) pass. Both old/new CPU
  models continue to reproduce every executed partial/final byte. The complete
  failure list remains in `merge-qualification.json` and `analysis.json`; the
  148-pass count above must not be interpreted as complete qualification.
- All 178 real guarded Inductor graph cases pass corresponding efficient-regime
  references. Empty/singleton/tail/guard-boundary cases, dtype neighbors and
  generated-only kernel census are retained. These graph timings are not a
  rotated comparison against the old capture and do not imply a Triton speed win.
- All 18 RMS forward/arbitrary-upstream backward phase comparisons pass against
  the retained matched efficient-Triton arrays at unchanged operation tolerances.
- Three rotated rounds of 384 CUPTI partial-kernel events per plan/case compare
  adjacent, old load-batch and compensated plans on 12 dtype/shape combinations:
  FP32/FP16/BF16 at 8x65539,64x65539,8x131071,8x1048576. Logical split count nine
  and final packet remain fixed. No established regression versus old load-batch
  under the existing >=2/3-round plus disjoint median +/-3 MAD rule. Similar
  performance is not claimed as acceleration. `component-timings.json` preserves
  all rounds/samples and comparisons to compensation; those do not establish a
  universal win over compensation or Triton.
- Production Driver API resource queries retain 48 registers/thread and zero
  local bytes for the FP32 old/new load-batch kernels. These are not offline
  CUDA13.1 assembler estimates. The compiler remains CUDA13.0 NVRTC/driver JIT.

`analysis.json` verifies the graph/reference input hashes, trace hashes,
normalization arrays, retained kernels and unchanged final CUDA/PTX. Generated
partial CUDA differs only in two addition operands after entrypoint normalization;
PTX FP32-add counts are unchanged. `kernels/` retains IR/CUDA/PTX/ABI/resources for
the old and new partials and fixed final across three dtypes.

## Validation and reproduction

Independent Python/Rust validators reject eleven neighboring invalid plans each.
Old serialized identities omit the default `private_merge` field; new identities
record `private_merge="adjacent"` and remain automatically ineligible. All CubeCL
captures retain Triton unavailable/unimported, no native Triton mapping, no GIL
forcing and a disabled GIL on CPython3.14t. The separate reference environment
enables the GIL; no host-time comparison is claimed. Conservative FP patch hash:
`9ae155efb01454be6cc61766db3564fce49bcf8495af753c60afab5df07c97d2`.

All 18 established Stage0–5/FP gates pass in `gates.json`, including the 168-case
corpus, constructor precision, frozen order, accumulation and guarded histories.
These protect the unchanged default and do not override the candidate failure.
Historical tracked evidence was restored, and all 412 files in the parent
packet still match its original hash manifest. `precision-output-preservation.json`
records correction of the initial runner's output redirection; the fresh result
is retained here and the historical file is restored exactly. The runner now
also protects untracked prior evidence and redirects both prior output paths.
Required review returned without a blocking implementation or eligibility
finding. Low robustness/documentation items are addressed in
`review-disposition.md`, with affected checks rerun in `post-review-gates.json`.
The reviewed initial 148-case capture predates the extra finite adversary; the
additional failure is retained explicitly and no candidate admission is claimed.

The next focused question is work assignment together with arithmetic topology,
while retaining the already understood protected-load organization. Neither
compensation nor a private-merge-only default is accepted by this pass. `cwk`
remains open. No additional candidate, default promotion or later stage begins.

Run from the repository root with `STAGE0_CARGO_PROFILE=release` and
`CUDA_PATH=$PWD/stage0/cuda13-toolkit/nvidia/cu13`. The reference command uses
the existing separate Triton environment and `TRITON_LIBCUDA_PATH=/usr/lib/wsl/lib`.

```sh
cargo build --release --locked --manifest-path stage0/Cargo.toml --features llvm,hip-probe
stage0/.venv/bin/python -m stage5.reduction_rounding_probe
/home/imi/Projects/sd-scripts/.venv-wsl/bin/python -m stage5.reduction_merge_qualification --impl triton
stage0/.venv/bin/python -m stage5.reduction_merge_qualification --impl cubecl
stage0/.venv/bin/python -m stage5.reduction_merge_checks
stage0/.venv/bin/python -m stage5.reduction_merge_benchmark
stage0/.venv/bin/python -m stage5.reference_reduction_probe --impl cubecl --history large_first --cubecl-policy adjacent_merge_diagnostic --cubecl-split-policy guarded --long-row --rounds 1 --output stage5/artifacts/reduction-structure/rounding/graph-large.json
stage0/.venv/bin/python -m stage5.long_row_normalization --impl cubecl --cubecl-policy adjacent_merge_diagnostic --output-dir stage5/artifacts/reduction-structure/rounding --cache-dir stage5/cache/reduction-rounding/norm
stage0/.venv/bin/python -m stage5.reduction_merge_analyze
stage0/.venv/bin/python -m stage5.reduction_merge_gates
```

Run GPU commands sequentially. Raw arrays/profiler traces and gate logs stay in
the ignored cache. This packet preserves inspectable code and compact evidence.
