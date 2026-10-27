# Paired first-use selection (bead 4z8.28)

First-use tuning (`inductor/tuning.py`, `select`) measures every legal plan over three rotated
rounds and keeps the frozen baseline unless another plan has proven faster. Until 2026-10-27 the
proof was "the best plan's median + 3 MAD lies below the baseline's median − 3 MAD, and the best
plan is the fastest in 2 of 3 rounds", with medians and MADs pooled over all rounds. The rule is
now: **the plan with the lowest median of its three round medians wins if its median is below the
baseline's in every round**; otherwise the baseline stays.

## Why the old rule failed

`calibration-3080.json` holds every candidate's per-round median for every kernel of the four
timing harnesses (integer, float arithmetic, extrema, Welford): three fresh-cache processes on the
RTX 3080, 222 selections.

- **The SM clock drifts between and within rounds.** 83 of the 222 selections had a round whose
  median over candidates was more than 10% above the fastest round. Within one selection, plans
  move together: `var_mean` 300×4096 ran every round at 22.2 µs in one process and at 14.9 µs in
  the last round of another. Memory-bound 4M-element kernels are not affected (floor division:
  round medians within 0.1%); compute-bound kernels of 3–20 µs are. Bead `sil.1` tracks the clock
  itself.
- **The pooled MAD measured that drift, not the precision of a median.** A median of 384 samples
  is far more precise than the sample spread, but the margin of 3 sample MADs on each side
  rejected real gains: 1.2% on two-input 4M pointwise kernels (round medians 74.40/74.37/74.37 µs
  against the baseline's 75.29 three times), and 10–30% on short reduction rows and split finals.
  The old rule kept the baseline in 130 of the 222 selections.
- **Round wins split among near-equal plans.** `independent_planes` 32×4 and 32×8 often take one
  round each. The wins condition then kept the baseline although both beat it in every round.

## Why paired rounds

Plans measured in the same round share most of the clock state, so each plan is compared with
the baseline round by round. The rotation moves every plan one place earlier each round, so a
steady clock ramp cannot favour the same plan in all three.

Cross-validation on the calibration data: for each selection, the plan a rule picks is compared
with the baseline in the other two processes (median over their six paired rounds).

| Rule | Moved off the baseline | Picked plan slower than the baseline in the other runs |
|---|---|---|
| old (3-MAD disjoint, 2 of 3 wins) | 92 / 222 | 0 |
| paired, 2 of 3 wins | 157 / 222 | 0 |
| paired (adopted) | 174 / 222 | 0 |

No margin is added. A plan of truly equal speed passes by chance with probability about 1/8, and
with ten or so candidates one usually does, so near-ties often leave the baseline; that costs
nothing in speed. A slower plan would have to appear faster in all three rotated rounds, which the
cross-validation never saw (3080 data, three processes; the 3090 was checked only through the
harness results below). The pooled MAD is still written to `selection.json` (`mad_ms`) as a
diagnostic; it no longer decides anything. CubeCL's own autotuner and Inductor's
Triton autotuner take the fastest candidate with no significance margin at all
(`cubecl-runtime` `tune/tuner.rs` sorts by `score()`, Inductor `autotune_to_one_config` takes
`min(timings)`); this rule stays more conservative than both.

## Buffer placement

How fast a memory-bound plan runs depends on where its buffers sit. The Adam lerp update
(two 4M float32 inputs, RTX 3080) over six placements of the same tensors, CUPTI medians in µs:

| Plan | Range over placements |
|---|---|
| `vector_pointwise` 128×4 (selected) | 74.26–74.55 |
| `scalar_pointwise` 64 | 74.50–75.40 |
| `scalar_pointwise_baseline` 256 | 74.99–75.68 |
| `scalar_pointwise` 128 | 74.60–76.28 |

The selected plan is the fastest or tied in every placement and the least sensitive. One graph
call can land on a placement that favours another plan (a single probe measured 64 at 74.22 against
74.74 for the selected plan). Differences of a few tenths of a percent between backends at 4M
are therefore inside the placement band. The launch pattern does not matter: dense batches, a GPU
sleep between launches and a host synchronize between launches rank the plans the same way.

## Against Inductor+Triton

Same harnesses and convention as before (median of three alternating rounds; slower when every
CubeCL round exceeds every Triton round). The old column is the last recorded result
(`../integer-timing-*.json`, `../float-arithmetic-timing-*.json`, `../extrema-timing*.json`). 4M
elements, gap of the CubeCL median against Triton's:

| Case | 3080 old | 3080 new | 3090 old | 3090 new |
|---|---|---|---|---|
| int64 floor division | +0.7% | +0.1% | +0.6% | −2.8%¹ |
| int64 shift/xor mix | +1.6% | +0.2% | +1.2% | +0.2% |
| float32 pad mask | +0.7% | +0.6% | +1.2% | +1.6% |
| int8 wrap | −0.7% | −0.4% | −1.5% | −1.4% |
| Adam lerp update | +1.2% | +0.4% | +2.3% | +0.4% |
| Adam addcmul update | +1.2% | +0.3% | +2.3% | −0.5% |
| Adam addcdiv update | 0.0% | −0.2% | +0.2% | +0.1% |
| addcdiv, value 0.3 | +0.1% | −0.2% | +0.4% | −0.2% |
| addcdiv, value 1 | +0.2% | +0.2% | +0.7% | +0.4% |
| abs/relu | −0.7% | −0.6% | +0.4% | −0.6% |

¹ Triton's rounds were 122.6, 139.0 and 125.7 µs; CubeCL's 122.2–122.5 µs is level with Triton's
fastest round, not 2.8% ahead.

The pad mask on the 3090 is the widest remaining margin: CubeCL 42.8–43.0 µs against Triton's
42.3–42.5 µs here, but 42.0–42.8 µs in an earlier run of the paired rule (still with the wins
condition). That is the placement band above. Its masked load still keeps vector plans out (`legality.py`, "vector lazy/effect region
unsupported"); on the 3080 the scalar plans it selects among overlap Triton's range in both runs.

Reductions (`extrema-timing-*.json`, `welford-timing-*.json`): the rule now selects the plans it
used to leave on the table. amax/amin 300×4096 run in 5.0–5.1 µs on the 3080 (5.9 µs before, Triton
8.8 µs), bfloat16 in 3.4–3.5 µs (4.4–4.5 µs). What stays slower than Triton is not selection:

- **Split finals** (2×1048579 float32 on the 3080, bfloat16 8×65539 on the 3090, +2.6% and +15–19%):
  the final kernel now selects its fastest legal plan, but Triton finishes with a persistent
  one-block reduction no plan here provides. Bead `6cl`.
- **Sub-2 µs kernels** (37×129 rows, every 4096-element pointwise kernel): 0.05–0.3 µs slower, and
  the selection varies between processes because the plans differ by less than the clock drift
  (amax float32 37×129 on the 3080: 1.47–1.91 µs over six processes). Bead `6cl`, within the
  recorded short-kernel exception.
- **Multi-output kernels** (Adam step, var_mean, layer norm, group norm): Triton fuses what
  CubeCL emits as several kernels. Bead `4z8.21`.
