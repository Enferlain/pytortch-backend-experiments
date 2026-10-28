# Proposal

## Why

Every compiled kernel writes exactly one external output, so a fused region with several
outputs becomes one kernel per output. Each of those kernels repeats the loads and any
reduction. Per kernel we are at Triton's speed, but end to end this is now the largest
measured gap to the Inductor+Triton floor:

- **Adam step:** +26% (248 vs 197 µs, RTX 3080). Three kernels where Triton has one.
- **frexp:** +26%. Two kernels where Triton has one.
- **Norms:** var_mean, LayerNorm and GroupNorm forward take 1.7 to 3 times Triton's time. Up to
  five kernels repeat the same Welford reduction.

Kernels are also capped at 6 buffers (5 for reductions), counting inputs and outputs. Fused
Adam needs 8, the GroupNorm second stage 6, and Triton's GroupNorm backward up to 9. So
lifting the output rule cannot meet the floor without raising the limit (bead `4z8.23`).

## What Changes

- **Several external outputs per kernel.**
  - Fusion admits nodes whose union writes several external outputs, in pointwise and
    reduction kernels.
  - A reduction kernel may store per-row values (mean, rstd) and per-element values (the
    normalized output) in the same kernel.
  - The descriptor declares a general buffer list: every buffer is read-only or write-only.
    Validation (Python and Rust) checks every store against it, instead of "the last buffer
    is the only write".
  - Kernels with random numbers or atomic stores stay single-output.
- **Each reduction computed once per kernel.**
  - Nodes fused into one kernel that reduce the same value share one reduction, as Triton's
    `reduction_cache` gives.
  - This includes split first stages, whose Welford inputs are masked loads. Masked regions
    today get fresh value names in each node and are never matched.
- **Buffer limit raised to 16 in total** (inputs plus outputs). The limit moves through
  fusion, pre-fusion admission, descriptor validation, the native validators, launch
  metadata and the stage5 launch ABI. This closes `4z8.23`.
- **Per-output launch contract.** Each output gets its own extent guard: per-row outputs
  have the outer extent, per-element outputs the full extent. The runtime checks every
  output, not only the last tensor.
- **Per-output plan selection.**
  - First-use tuning gives every output its own scratch buffer.
  - Each output is compared by its own rule, chosen from the reductions it depends on:
    - exact for the Welford weight and for max/min;
    - the Welford corruption guard;
    - the sum guard;
    - element-wise comparison for pointwise outputs.
  - Kernels mixing a sum with a Welford reduction are qualified against Triton's accuracy.
- **One pass for per-element stores.** A reduction kernel with several per-element stores
  emits them in one loop over the row, as Triton does, not one loop per store.
- **Performance floor gated.** Adam step, frexp, var_mean, LayerNorm and GroupNorm forward
  are timed against Inductor+Triton on both GPUs. The Welford floor deferred by
  `welford-reductions` and the whole Adam step become gates.

Out of scope:

- **User-visible mutation** (`4z8.9`). The in-place optimizer step is rejected before
  fusion. Triton turns it into one kernel whose outputs share storage with its inputs. That
  follows this change, and adds an aliasing contract for outputs over input storage.
- **In-kernel storage reuse** (Triton's `in_out_ptr`). The backend sets
  `inplace_buffers=False`, so Inductor never gives our kernels such a buffer.
  - **Measured cost: none.** Toggling only that setting under Inductor+Triton on the RTX 3090:
    - Inductor made 4 to 199 in-place decisions per workload across eight model workloads
      (SDXL, Anima, Wan 2.2, Krea2, MiniMax-H3, Qwen-Image 2.1, MOSS-Image, KohakuMoss ops).
    - Peak memory was identical to the byte in every one, and for the optimizer helpers.
    - CosyVoice 3 fails on the Triton path itself, so it gives no data.
  - **Tracked as bead `4z8.31`.** It is re-checked under gradient checkpointing and larger
    batches in the workload matrix (`4z8.11`), and implemented if any peak differs.
- **Fusing random-number kernels** (dropout's output and mask). Fusion of effectful
  kernels stays behind its own effect-safety proof.
- **int32 storage** (`4x9.11`), which bare `torch.frexp` needs for its exponent output.

## Capabilities

### New Capabilities

- `multi-output-kernels`: Kernels with several external outputs. Covers:
  - fusion admission and the buffer limit;
  - the descriptor's buffer contract and its malformed-descriptor errors;
  - each reduction computed once per kernel;
  - per-row and per-element stores in one reduction kernel;
  - per-output launch guards and plan-selection comparison;
  - unchanged single-output kernels;
  - the performance floor for multi-output workloads.

### Modified Capabilities

- `reduction-kernels`: The performance-floor requirement turns the deferred Welford check
  into a gate on the multi-output norm kernels.
- `float-arithmetic`: The basic-float performance requirement gates the whole Adam step,
  not only the three updates separately.
- `mean-variance-reductions`: The accuracy requirement covers multi-output norm kernels and
  kernels that mix a sum with a Welford reduction.
- `kernel-runtime`: Contract-driven launch validation checks the extent of every output.

## Impact

- **Python front end**:
  - `inductor/admission.py`: fusion rule, input limits and per-node checks.
  - `inductor/codegen.py`: output count, de-duplication of masked regions, descriptor
    buffers.
  - `inductor/tuning.py`: scratch per output and per-output comparison.
  - `inductor/legality.py`: per-output comparison classes, the vector adapter's one-store
    rule, the mixed-reduction plan policy.
  - `compiler/descriptor.py`: buffer contract and `MAX_BUFFERS`.
  - `compiler/binding.py`: per-output guards and mixed store scopes.
- **Runtime**: `runtime/launch.py` checks the extent of each output; `runtime/artifact.py`
  documents the guard.
- **Rust compiler** (`cubecl_torch/native/src`):
  - `semantic.rs` and `reduction/analysis.rs`: buffer contract and limits.
  - `reduction/emit.rs`: one loop for per-element stores.
  - `lib.rs`: `cubecl_torch_metadata` buffer bound.
- **Native launcher**: `native/launcher.c` raises the `stage5_launch` pointer bound to 17.
  The historical `stage4_launch` is unchanged.
- **Tests and tools**:
  - New `test_multi_output.py`.
  - `test_float_arithmetic.py`: `adam_kernels` goes from 3 to 1.
  - `test_admission_paths.py`: the pointwise rejection becomes acceptance.
  - Contract tests for descriptors and runtime.
  - The timing tools for Welford, float arithmetic and rounding/remainder are re-run.
- **Manifest**: corpus regions whose graphs have several outputs change kernel count. Each
  changed case is reviewed and explained before re-recording. Single-output kernels must
  stay byte-identical.
- **Beads**: closes `4z8.21` and `4z8.23`; `4z8.9` follows. The coverage inventory notes
  multi-output kernels.
