# Warm-dispatch review disposition

Two required review-mcp reviews returned successfully. Raw provider output/tool traces remain in ignored `stage5/cache/launch-review-1.md` and `launch-review-2.md`.

## First review

- Missing pointwise output-extent parity: restored `expected=numel` for non-mutation pointwise; kept Stage4's mutation exemption. Added a negative runtime component compared with Stage2.
- Schema3 RNG bound: returned-extent guard now spans both schema branches; no new RNG/reduction operation is admitted.
- Native pointer state: explicit validated-invocation guard; exception-safe restoration retained.
- Shared-cache PTX assertion: frozen files supply actual old adapter provenance; old/new directories must differ before exact PTX comparison. Legacy legality is also frozen.
- Phase ablation: instrument both signature globals; document serial-only reference lock/global use.
- Gate integration: warm controls added to standard final gates. Zero-grid launch records filtered from trace census.
- Counters: selection hits checked by real rank-zero capture; ordinary nonempty launch counting avoids duplicate geometry evaluation.

## Follow-up review

The reviewer verified validation parity, pointer/lock safety, descriptor/cache bounds, counters, gate wiring and independent artifact pairing. No dispatch correctness regression was found.

- Reference routing now requires exactly one expected import; injected import changes fail loudly.
- Factory cleanup uses `try/finally`; an injected benchmark failure restores the original factory.
- Unwarmed reference selection produces an explicit diagnostic.
- Zero-grid Python/native counter equivalence has an explicit assertion.
- Old final-gate resume compatibility is documented.
- The older unbounded CUDA metadata/event retention is explicitly distinguished from the new 64-key descriptor LRU and tracked as `pytortch-backend-experiments-dm4`. No eviction/readiness redesign here.

Affected Stage5 controls and the complete168-case corpus passed after the production repairs. Final paired capture passed with equal plans and independently produced exact PTX. Small subsequent harness/control guard changes pass targeted hook/control checks, Ruff and diff checks; they do not alter measured production hot paths. No third full review was required for these small corrections. Stage0–4 production sources were never modified.
