# zah.11: what upstream's default path costs on a first call

Bead `zah.11` asks how to choose a plan at a first-call cost comparable to Triton's. Before
designing anything, this runs what upstream already provides: Burn, unmodified at its own pins
(`.stage0-sources/burn` `35589bf`, CubeCL `fdba0a84`, CubeK `06e1c8ec`), whose `sum_dim` takes
`burn-cubecl` `autotune_reduce`. That is CubeCL's `LocalTuner` over Burn's six reduce candidates
(unit, plane and cube with CubeK's inferred blueprints, each with and without parallel output),
keyed by CubeK's `ReduceAutotuneKey`, with roofline bounds and priority groups.

`burn_probe/` (fusion off, so the plain reduce path): one process per case, fresh tuner, input
bytes written by PyTorch from the seeds `cubek_timing.py` uses, dtype given explicitly (without it
`from_data` takes the device's default, float32). RTX 3080, 2026-11-04.

| Case | Burn first call (ms, median of 5) | Ours, today (ms) | Triton (ms) |
| --- | --- | --- | --- |
| float32 2048×8192 | 2541 | 1354 | 192 |
| float32 2048×4099 | 2286 | 779 | 187 |
| float32 300×4096 | 2324 | 2215 | 192 |
| float32 7×2049 | 2309 | 1345 | 191 |
| float32 64×31 | 2447 | 1334 | 465 |
| bfloat16 2048×8192 | 2433 | 3921 | 259 |
| bfloat16 2048×4099 | 2628 | 1702 | 271 |
| bfloat16 300×4096 | 2658 | 2620 | 278 |
| bfloat16 7×2049 | 2337 | 1502 | 315 |
| bfloat16 64×31 | 2300 | 1360 | 651 |

"Ours, today" and Triton are `cubecl_torch/testing/results/cubek-first-call-3080.json`.

What the autotune log (`CUBECL_DEBUG_LOG=stderr`, separate runs: full logging also dumps every
kernel's source and inflates the first call) shows:

- **Tuning itself is cheap; compiling the candidates is not.** The tuner compiles every candidate in
  one batch first (`tune/schedule.rs`), then times them: the timing took about 45 ms for 7×2049 and
  about 200 ms for 2048×8192, and it short-circuits once a candidate is "close enough to peak
  throughput" (roofline bounds). The rest of the 2.3–2.6 s is compiling five or six kernels
  (through CubeCL's LLVM backend, Burn's default).
- **Its picks are not stable on this machine.** The same case picked different candidates in
  different runs (float32 2048×8192: plane-parallel, plane, plane-parallel; bfloat16 7×2049: three
  different picks in three runs). Its samples are CUDA events around single launches: 50–80 µs per
  sample for kernels ncu puts at 2–3 µs, so launch overhead decides.
- **Keys are bucketed.** `ReduceAutotuneKey` anchors sizes, so 2048×4099 reuses 2048×8192's tuning
  within a process (a compile-only first call of 31–53 ms).
- Burn checks candidate outputs only under its `autotune-checks` feature ("debugging"), off by
  default.

**So upstream's runtime tuning does not reach Triton's first-call cost either.** Triton's
advantage is compiling one kernel, chosen by rule, for a looped row sum. Upstream's mechanism
worth taking is the bucketed key (one tuning per size class), not runtime tuning by default.
