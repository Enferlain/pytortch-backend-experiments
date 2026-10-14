# Choosing profiling depth
Use a working correctness baseline, unprofiled timing distributions and one lead investigation. Deeper inspection is warranted only when it can distinguish competing explanations.

| Symptom | First evidence | Descend when | Trace compiler decision |
| --- | --- | --- | --- |
| Many short kernels / gaps | Operator attribution, system timeline, launch count | GPU work alone cannot explain wall time | Fusion split, host wrapper, allocation or dispatch |
| Synchronization/transfer overhead | Timeline and API/stream dependencies | Locate implicit waits, copies or lost overlap | Dependency scheduling, buffer/device ownership |
| Fusion vanished | Pre/post scheduling, wrapper/kernel count | Determine eligibility/layout/mutation change | Fusion guard, scheduler or fallback path |
| Bandwidth limitation | Time, bytes moved, attainable bandwidth, targeted counters | Transactions exceed useful bytes or cache behavior changes | Layout/indexing/vector widths, redundant loads/stores |
| Cache/coalescing | Address/transaction facts and relevant counters | IR/source cannot explain effective traffic | Index/layout decision, masking, access ordering |
| Shared-memory limitation | Allocation/bank/access/barrier evidence | Resource or conflict data explains critical cost | Buffer placement, bank mapping, synchronization |
| Low occupancy | Theoretical limits, achieved occupancy, grid coverage | Registers/shared/block shape limit useful concurrency | Block policy, live ranges, unrolling, allocation |
| Register growth / spills | Final resource diagnostics and spill traffic | Locate pressure threshold or backend allocation change | Source/IR live ranges, vectorization, target register allocator |
| Instruction throughput | Targeted pipeline utilization and instruction mix | Source choice fails to explain costly instructions | Math/intrinsic/type lowering, legalization |
| Divergence/dependency stalls | Branch/activity and eligible-work evidence | Determine why relevant lanes or operands wait | Mask/control-flow lowering, instruction scheduling |
| Code-size / compile-time growth | Per-stage compilation timing and emitted size | Identify unrolling/template/pass growth | Pass complexity, specialization or code expansion |

Distinguish “low achieved occupancy” from resource-limited theoretical occupancy and small-grid underfill. More occupancy may hurt instruction count or bandwidth. Do not equate profiler metric labels to causal explanations.

## Mapping and causation
Preserve process/device/stream, kernel identity, invocation number, grid/block/shared bytes, input shape and timing region. Compiler-generated names may change or collide; correlate via wrapper and origin mapping rather than name alone. Profiling replay changes cache/concurrency; compare equivalent replay/cache policies, then validate with unprofiled runs.

For a 20–30% regression: reproduce steady-state slowdown; inspect high-level attribution; use system timeline if gaps/copies/kernel count explain it; use focused kernel counters if one launch dominates; inspect source/IR to locate differing compiler choices; ask for target IR/resource diagnostics only to resolve a hypothesis; inspect final code only if lowering/assembler/JIT changes remain plausible. This order is adaptive, not a demand to run every tool.

## Complementary skills
Ask installed Nsight Systems skills to collect/interpret timelines and Nsight Compute skills to choose supported metrics/replay/kernel filters. Ask CUDA global/shared-memory or launch-configuration skills for focused mechanism analysis, Triton writing skills for Triton-specific experiments, numerical/test skills for validation, CUDA→HIP portability skills for target adaptation. Keep compiler attribution and before/after evidence here.

For AMD, use its verified system/kernel profiler and AMDGPU IR/code object/ISA/metadata. Distinguish VGPR/SGPR/scratch/LDS from NVIDIA registers/local/shared terms; wave/workgroup capabilities vary. For other targets, discover available equivalents and explicitly mark unsupported stages.
