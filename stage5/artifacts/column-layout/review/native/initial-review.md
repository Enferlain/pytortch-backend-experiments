# Native independent review — original l1m snapshot

Reviewer: specialized native `adversarial_reviewer`, task
`/root/column_layout_review`. Authorized by the user after review-mcp failed
with HTTP429/code1308. Read-only review against HEAD `c5457e92` plus untracked
diagnostic files. The initial result is **changes requested**, one Medium
finding. A separate focused follow-up reviews remediation.

## Medium: caller mutation desynchronizes validation and compiled PTX

The original `generic()` and `compile_kernel()` retain the caller-owned semantic
dictionary in Dispatch and Kernel. The base kernel stores that reference, and
the storage validator retains nested buffer dictionaries. The cache fingerprint
is computed before retention. Unlike the prior s1a Compiled boundary, this path
does not deep-copy the descriptor.

Trigger: compile a mutable admitted descriptor, mutate it or a nested buffer
after compilation, then invoke the retained dispatch or compile an untouched
copy of the original descriptor and receive the already-cached object.

The reviewer ran a host-only cache probe using a real admitted semantic:

```text
same_cache_object True
retained_caller_alias True
future_original_gets_mutated_semantic True
```

A caller could shrink the linear input size/stride descriptor and pass a small
tensor satisfying those mutated host layout/span checks, while loaded PTX still
indexes the original `B*N` input. This permits out-of-bounds reads. The existing
cache entry also remains under the original fingerprint, allowing another caller
to retrieve the poisoned object without renewed admission. This is a Medium
admission/ownership defect within the public diagnostic runtime.

Generated wrappers use private literal dictionaries, and recorded runs did not
mutate them. The reviewer found no basis to invalidate captured GPU results.
Recommended fix: deep-copy at both boundaries before admission, fingerprinting
or retained use; retain only the owned copy. Test post-compilation mutation and
an untouched subsequent cache lookup.

Original locations: packet `sources/stage5/column_layout_runtime.py` generic
lines89–97, compile_kernel100–178, Dispatch78; base `stage2/runtime.py`40–42
and nested descriptor retention in `stage5/artifact.py`167–172.

## Remaining result

No other actionable finding survived verification. The three admitted column
templates support the bounded identity with positive `B,N`, `i<B*N`, U64
addressing and the unchanged scalar256 launch. Protected loads remain inside
the original branch; actual FastDivmod::div and PlainLayout produce the required
row address. Non-load arithmetic, output addresses and protected fallbacks are
unchanged. Declared scalar tuples, span/disjointness, current-stream readiness
and lifetime mechanisms remain. FP32/FP16/BF16 protected reads and special-value
class/signed-zero checks were reviewed; outputs match frozen generic bytes.

The reviewer independently corroborated three templates/candidate artifacts,
12 graphs × six permutations × three paths ×16 calls =3,456 CUPTI samples,
68/72 generic wins and43/72 ordinary wins. Coarse clock limits are stated
correctly, with no stable intrinsic speed claim. Nine NCU targets bind the
IR/CUDA/PTX/SASS evidence, support21→16 registers and about34.5% fewer executed
warp instructions. All374 initial packet hashes matched and covered every
packet file except the ledger itself. Original source snapshots matched before
concurrent remediation. The failed review-mcp is accurately preserved as having
no verdict.

Independent checks: Ruff F, Python bytecode compilation, cargo fmt --check,
git diff --check, packet SHA/inventory verification, excluded upstream
FastDivmod/PlainLayout source reads, graph-assisted ownership tracing with
direct source fallback. Upstream source was graph-excluded; live runtime became
metadata-changed during the parent's concurrent fix. GPU compilation/performance
and offline analyzer were not rerun by this reviewer.

The initial review explicitly did not certify the concurrent fix or post-review
outputs. Requested parent gates: full78-case/90-live/12-stream repeat, normalized
PTX identity, refreshed source binding and integrity. No production/default or
reduction promotion is supported.
