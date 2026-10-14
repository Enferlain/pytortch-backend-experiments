# Coordinated suite protocol
This byte-identical reference is bundled in each skill so each directory also works independently.

| Question | Lead |
| --- | --- |
| Where did operation/layout/cast/memory behavior become X? | compiler-lowering-trace |
| Is X → Y permitted by the governing contract? | compiler-semantic-audit |
| Why does generated code cost time or resources? | gpu-codegen-profiler |
| Which change introduced a reproducible failure? | compiler-regression-investigation |

Keep one lead. A versioned slowdown stays a regression investigation with profiling support; semantic review can consult history without becoming a bisect.

## Mandatory rules
- Establish the path exercised from revision-specific source and artifacts. Check graph breaks, decompositions, external/library calls, precompiled source, caches, training/backward and target choices. Treat example pipelines as hypotheses.
- Separate Python, PyTorch/framework, frontend, compiler IR, optimization assumptions, backend IR, driver/compiler, machine-code and hardware semantics. Never derive upper-layer contracts from one lowering, similar names, common compiler practice or Triton behavior.
- Label important claims: **contract**, **source implementation**, **artifact**, **measurement**, **history intent**, or **inference**. Cite repository/revision/path/symbol, test or document/version/section. Tests establish only their covered conditions. Report conflicts and missing evidence.
- Verify concrete commands, flags, APIs, env variables and paths in authoritative source/docs for the relevant version AND check local availability. Record executable/library identity and actual options. Label proposed interfaces/instrumentation **hypothetical** until implemented. Never invent a pass-disable or dump switch.
- Preserve raw artifacts before normalization; compare equivalent configurations and mark unavailable stages. PTX, ptxas, SASS and Nsight are NVIDIA-specific. Verify target-specific alternatives.
- Respect authorized scope. Reading source does not authorize changes to investigated repositories. In read-only work, propose instrumentation; in development, preserve reversible instrumentation diffs.

## Evidence packet
Use one investigation directory. Record:
1. Question, scope, lead and success/failure predicate.
2. Exact input/reproducer/data; dtype/shape/strides/layout/masks/aliasing/mutation and forward/backward mode.
3. Revisions and dirty changes, dependency/lockfile versions, compiler/framework versions and Python GIL/threading mode when relevant.
4. Device/target/driver/runtime/compiler identities, compiler/fast-math/env/launch/stream options and cache/autotune state.
5. Boundary table: stage → producer/consumer symbols → artifact → observation → confidence. Preserve relevant IR/source/target IR, diagnostics, binary/disassembly, profiler reports, numeric outputs and benchmark samples. Record missing artifacts; hash when useful.
6. Claims, counterevidence, conclusion and next discriminating experiment.

Hand off the packet path, specific unresolved question, observed boundary, accepted assumptions/contract and falsification criterion. Reuse evidence rather than restarting. Supporting skills return findings to the lead. Discover installed complementary skills by purpose; do not invent their names or assume installation. If unavailable, use verified official tooling and state the limitation.
