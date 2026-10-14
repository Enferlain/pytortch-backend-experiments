---
name: compiler-lowering-trace
description: Trace an operation, datatype, layout, cast, scheduling/fusion decision or memory construct through current compiler boundaries. Use for PyTorch torch.compile/Dynamo/FX/AOTAutograd/Inductor to Triton or a proposed CubeCL backend, CubeCL/Pliron and target-code provenance, and differing generated GPU code. Owns where behavior originates; coordinate semantic legality, performance diagnosis and version attribution with the suite.
---
# Compiler lowering trace
Read [suite protocol](references/suite-protocol.md) first. Use [stack navigation](references/stack-navigation.md) for source entry points and [artifact capture](references/artifact-capture.md) for verified dump interfaces. Refresh these dated maps against the investigated checkout.

## Workflow
1. **Define the exact trace target.** Specify operation overload, dtype, shape/stride/layout, mode and observable signature: cast/rounding modifier, index/layout map, synchronization, launch choice or instruction family. Bound the question.
2. **Establish the executed path.** Follow caller, backend selection/registration, dispatch and invocation into producer/consumer interfaces. Distinguish Dynamo backend registration from Inductor device scheduling/codegen registration. Check decompositions, graph breaks, external kernels, forward/backward partitioning, precompiled source and cache reuse. Mark a proposed Inductor → CubeCL adapter boundary hypothetical.
3. **Build the boundary ledger.** Record each input/output representation, owning project, symbols, flags and capture route. Read IR definitions, not only directory names. Pliron may supply infrastructure for CubeCL dialects; do not presume CubeCL IR → Pliron is a mandatory conversion between unrelated IRs.
4. **Capture adjacent stages.** Align operations across fusion/decomposition/splitting using origin metadata, operands, types and index maps. Save wrapper/launch code too. A final instruction's absence does not locate its removal.
5. **Identify the earliest established change.** Record the last boundary without and first with the behavior. If a stage is unavailable, report an interval of uncertainty. Do not blame the next visible pass by default.
6. **Navigate source when dumps are absent.** Find constructors, registrations, traits/interfaces, macro definitions, rewrite match conditions, pass assembly, target guards and emitters; read caller and callee. Distinguish constant evaluation, canonicalization, optimization, dialect conversion, source emission and target compilation. Locate tests/history. Source establishes a conditional implementation, not proof the path ran.
7. **Confirm the mechanism.** Check pass order, repeated passes/fixpoints, target prologue/epilogue, analyses and operation-level modes. When authorized, use minimal instrumentation or controlled pass replay while keeping valid IR/ABI. Disabling a pass demonstrates sensitivity, not sole causation.
8. **Determine intent separately.** Read introduction/change diffs, tests, comments and discussion. Pattern existence is implementation evidence; intentional requirement needs intent/contract evidence. Avoid deriving intent from author identity or common practice.

## Inductor backend development
Inventory scheduling/features/fusion eligibility, symbolic sizes/strides, masks/reductions, dtype/layout, metadata/ABI, allocations/aliasing, wrappers/launches/streams, caching/autotuning, errors and external fallbacks. Trace Triton dependencies including startup imports and capability checks; one registration hook is not a complete replacement. Preserve PyTorch tensor/autograd semantics above the adapter. Assess Python/Rust integration and free-threading separately; GPU compilation APIs guarantee neither.

## Handoffs and output
Send legality and assumption questions to **compiler-semantic-audit**, measured-cost questions to **gpu-codegen-profiler**, good/bad-version attribution to **compiler-regression-investigation**. Use installed PyTorch skills for capture/setup and Triton skills for Triton-specific mechanisms.

Return actual path, boundary table, earliest change, responsible source/conditions, intent evidence, missing stages and next experiment. Distinguish source-confirmed from runtime-confirmed provenance.
