# FP hardening review disposition

The required review read the tracked diff, untracked implementation/patch files,
affected upstream source and retained numerical/runtime evidence. It found no
correctness bug on the tested surface and confirmed the live conservative policy,
patch composition, compiler routes and reported counters.

| Finding | Disposition |
| --- | --- |
| Generic LLVM replacement could delete its own callee or an in-use declaration | Guard deletion with `callee != declaration`, no body and no uses; validate existing function signatures before reuse |
| Exotic LLVM frem type failed inside the shared helper | Leave unsupported widths to the original target lowering; report whether any instruction was actually redirected so no library is linked unnecessarily |
| Promoted/vector wrapper has a doubled internal prefix | Cosmetic, inherited naming composition; retained. That path is not claimed as runtime-tested |
| Historical versus active patch hash ambiguous | Active `cubecl_patch_sha256` now names the full patch; historical base hash is explicitly `cubecl_patch_sha256_historical`; stack hash is retained |
| Probe's alias check weaker than general storage interval validation | Check non-overlapping byte intervals for all three contiguous zero-offset buffers |
| Pre-O3 contract check alone insufficient | No change needed: exact device counterexamples and PTX FMA inspection already cover final behavior |
| Unsupported-operation fixtures depended on final dtype-loop tensors | Use dedicated FP32 tensors for the negative probes |
| Vector/splat and non-SM86 portability not established | Retain explicit scope limits; neither feature is added by this task |

These are small guards/diagnostic/fixture corrections rather than a new numerical
algorithm or architecture change. A second broad review is not required. The
affected runtime, real-Inductor, Stage 0 LLVM, IR and quality checks are rerun.
The full Stage 0/1/1.5/2 regression suites passed before these guards; the C++
backend, scheduler, launcher, production dtype/index/mask surface remain unchanged.
