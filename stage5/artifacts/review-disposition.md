# Independent review disposition

The initial review inspected the complete implementation and identified one blocking proof mismatch plus smaller robustness and measurement gaps. Its raw transcript is local at `review-initial.md`.

* Static reduction-axis proof: fixed. Python and Rust now require literal signed-i64 coefficients0/1 for vector inputs; runtime scalar axis multipliers are rejected on both sides. The emitter chooses vector access from the coefficient, so literal `r*0` remains a scalar uniform load. Executed `proof-controls.json` checks literal zero with a misaligned three-element backing allocation and rejects symbolic zero/one before vector compilation.
* Output isolation: existing schema3/4 validators already require one final write buffer. The alleged multi-output exposure was outside the accepted descriptor contract. An additional explicit Python assertion now rejects it before compilation, with an executed control.
* Persisted selection identity: compare canonical JSON instead of tuple/list object equality. Fresh-process reload evidence is captured separately.
* Deterministic cold cache: executed a distinct semantic kernel, proving no empirical benchmarking, unchanged RNG and frozen strategy selection.
* Packed-plane unused shared scratch: removed. Shared allocation is optional and absent for independent planes; components require zero static shared bytes, manifests and Nsight verify zero dynamic shared bytes.
* Compiler comparison: explicitly disables Inductor graph benchmarking, consistent with the adapter's unsupported capture boundary.
* Effectful reductions: schema3 has no effectful operation surface. Existing deliberate rejection remains; no speculative effectful reduction fallback was added.
* Duplicate bypass removed. Finite alignment over-keying and diagnostic EVENTS growth are documented limitations.

Measurement refinement after the initial review: WSL driver enqueue intervals and the generic `_CALLABLE` GPU annotation include queue gaps. Selection now uses exact named CUPTI kernel events after generic Inductor event warmup, verifies384 kernels/round, and excludes annotations. Three rotated rounds plus a conservative median/MAD decision remain. Whole graph comparisons use seven independently profiled GPU-kernel sums; state reset and synchronization happen outside each region. These are kept separate from twenty unprofiled event/host samples. Nsight counter access now works after the Windows-host permission correction.

The focused follow-up review returned: no blocking correctness or benchmark-validity defects. It confirmed all eight fixes. Two low-severity notes were also addressed: literal multipliers require exact signed Python int (bool/float excluded), and a selection retry reconstructs complete key_data after an exception. Component controls verify both; the source hash invalidates affected tuning selections. Bounded profiling retries and the controlled activity warmup are small measurement follow-ups to that review, with the same strict count gate preserved.

Final recapture exposed one incomplete CUPTI buffer (371/384 expected kernels). The strict count gate failed rather than timing partial data. The timer now retries a complete capture at most three times, records every observed count per candidate/round, and still fails if no384-event capture is obtained. The tuning source hash invalidates all prior selections. Physical/native/semantic code is unchanged.

Separately profiled whole-graph kernel durations varied substantially for byte-identical normalized CUDA/PTX frozen strategies. Those captures are excluded as physical-speedup evidence. A controlled GPU activity warmup (CUDA sleep outside each measured/profiler region, no RNG or mutation) reduced frozen long-row timing from82us to about10us. All five families are being recaptured under the same warmup. Physical improvement claims use same-process rotated candidate measurements, not differences between identical kernels.
