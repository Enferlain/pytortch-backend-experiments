# Required review — 2b9

2026-10-18, `review-mcp.review_with_context`, diff target HEAD.
Nine focus files: the new candidate, runtime/frozen runtime/backend, observer
generator, phase driver, offline analyzer, packet README and STAGE5_REPORT.md.
Context: analysis.json, packing.json and candidate-source.diff. Review completed
after20 model calls. No fallback reviewer was required.

The returned verdict was:

> **Approve — no blockers.** The implementation matches the stated intent exactly (one storage-access adaptation, byte-identical PlaneReader/Sum body and epilogue, unchanged casts), numerical equivalence is established at the bit level on all 36 cases including instrumented operand/sum parity, alignment/unit-stride/full-vector and exact-template guards are present on both the Python and native sides, the frozen control is provably immutable, and every quantitative claim in the README/STAGE5_REPORT is supported by — and honestly bounded by — the retained artifacts, including the disclosed debug/release inventory discrepancy and the unqualified isolated FP32 cost. The negative result (ordinary wins all 54 paired graph rounds) is reported without promotion claims. The minor notes above are robustness/cosmetic and do not warrant changes to this closed diagnostic packet.

## Review findings retained

The reviewer read all nine focus files plus directly referenced validation,
descriptor5, frozen wqc source/model, timing table and artifact tree. It found:

- The complete Rust diff is confined to imports, operand storage access and
  distinct entrypoint/mechanism/manifest identity. The input adapter faithfully
  mirrors descriptor5, with its narrower assumptions covered by exact templates.
- The1,356-byte component body and original casts/epilogue remain unchanged;
  packed elements and scalar replay use the same original coordinates.
- Runtime layout/span/alignment/disjointness and native exact-template/plan
  admission cover the full-vector proof. The Branch protects the entire load.
- Observer derivation, x/trace/y argument ordering and unmodified descriptor
  x/y array mapping are consistent. Operand/padding/sum parity and final equality
  are correctly scoped to36 cases and qualification-only instrumentation.
- SASS classification excludes metadata, checks all executed reader loads have
  the intended width, and cross-checks512 input warp loads and NCU totals.
- The54 ordinary wins,216 traces, full permutation schedule,3–4% FP32 gains,
  small/mixed low-precision results, complete unstable isolated record and
  debug/release inventory disclosure are supported by the recorded artifacts.

Five nonblocking observations were explicitly retained:

1. `arrays[0]` and one pre-reduction load are implicit local assumptions. Exact
   include_str templates and parity protect this scope; explicit local asserts
   would matter if future templates broadened the adapter.
2. Frozen-control Kernel bypasses the wqc constructor's explicit metadata tuple
   recheck. Copied manifest byte identity and inherited ABI validation cover it.
3. The alignment error still uses the phrase "vector4 input"; cosmetic.
4. The original packing.json producer was outside the reviewed focus set. The
   offline analyzer independently reconstructs its target-code evidence, so the
   reviewer requested no action.
5. Template files are re-read in admit(); harmless at diagnostic call frequency.

The reviewer did not request GPU changes, retiming, promotion or a second
mechanism. Its statements about offline verification refer to inspected checker
code/artifacts; the parent separately executed the checks.
