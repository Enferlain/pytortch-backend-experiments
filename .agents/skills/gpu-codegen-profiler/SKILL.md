---
name: gpu-codegen-profiler
description: Connect reproducible GPU performance observations to generated compiler IR/source, target IR, assembler diagnostics and machine code, then to scheduling/lowering decisions. Use for Inductor/CubeCL/Pliron/Triton fusion or launch pathologies, register pressure/spills, memory and instruction bottlenecks. Orchestrate existing PyTorch/Nsight/CUDA/HIP skills instead of replacing kernel-tuning workflows.
---
# GPU codegen profiler
Read [suite protocol](references/suite-protocol.md). Use [performance decision guide](references/performance-decisions.md) and [verified tools/artifacts](references/tools-and-artifacts.md) as needed.

## Workflow
1. **Define cost and correctness baseline.** Specify wall latency, steady-state GPU time, throughput, compilation time or memory. Save passing contract-appropriate results and exact input/configuration. A 25% slowdown needs samples and a stable baseline.
2. **Validate timing.** Separate compilation/cache/autotune warmup from steady state; measure cold behavior separately if relevant. Account for asynchronous work and streams/graphs without inserting synchronization into the evaluated path. Record scope, repetitions/dispersion, clocks/thermal/power/background load and allocation/transfers. Alternate comparable cases where useful. Keep independent unprofiled timing because profilers perturb execution.
3. **Attribute high-level work.** Use installed PyTorch profiling skills for capture/operator attribution and suitable activities. Match fused, external-library and backward kernels, recompiles and launches via wrapper/origin information and invocation identity. Operators and kernels need not correspond one-to-one.
4. **Choose system or kernel depth.** Use existing Nsight Systems skills on NVIDIA for launch gaps, CPU overhead, synchronization/transfers/concurrency; verified ROCm tooling on AMD. If a particular kernel dominates, delegate focused counter collection to existing Nsight Compute or target skills. Avoid full-model exhaustive metrics.
5. **Form competing hypotheses.** Low occupancy is a symptom: distinguish theoretical resource limits, achieved occupancy, small grids/underfill and insufficient eligible work. Inspect scheduler/block choices, generated live ranges, register allocation/spills, shared allocation and synchronization. Occupancy alone is not the optimization objective.
6. **Descend to answer a concrete question.** Inspect scheduling/fusion and source for launch count/layout/vectorization/masks. Inspect target IR for memory widths, instructions/math/barriers and live values. Use assembler diagnostics for resources, machine code for unresolved spills/instruction mix/dependencies or IR-to-machine discrepancies. PTX virtual registers do not give allocated SASS register count; PTX does not map one-to-one to SASS.
7. **Trace cause upward.** Send the measured limitation and generated construct to **compiler-lowering-trace**. Distinguish CubeCL, Pliron infrastructure, LLVM, NVRTC/HIPRTC, offline assembler and driver-JIT ownership. Never silently substitute offline ptxas output for the actual driver-JIT binary.
8. **Validate causality.** When authorized change one relevant compiler decision, save raw before/after artifacts, rerun correctness and original unprofiled timings. Check representative shape/dtype/layout/target cases and tradeoffs. Demonstrate the predicted resource/code change and end-to-end benefit. Correlation or a favorable microbenchmark is insufficient.

## Handoffs and output
Use **compiler-regression-investigation** for version attribution and **compiler-semantic-audit** for semantic tradeoffs. Engage existing global/shared-memory, launch configuration, Triton or CUDA→HIP skills only after identifying the compiler-owned mechanism; hand-edited kernels do not prove what the compiler emits.

Return timing protocol/distributions, launch attribution, chosen profiling depth/reason, competing hypotheses, measurements/artifacts, compiler cause and correctness results. Without device/counters/binary, report bounded hypotheses and a verified collection plan rather than measured conclusions.
