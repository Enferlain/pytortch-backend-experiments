Yeah. After reading the current branch and then checking the actual CubeK/Burn machinery, I think your framing is the better center of gravity. But I’d make one important correction to the response you quoted, because otherwise you can accidentally replace one wrong architecture with a different wrong architecture.

The sentence I would change is:

> “consume Inductor's decisions rather than re-making them.”

I’d replace it with:

> **Consume Inductor's semantic schedule and facts; let the CubeCL ecosystem choose the physical realization.**

That distinction is basically the architecture.

## What I think the project actually is

The goal is **not** really:

`PyTorch/Inductor → our CubeCL compiler → CubeCL`

and it is also not:

`PyTorch/Inductor → take Triton's block/warp decisions → reproduce them in CubeK`

It should converge toward:

`Inductor scheduled kernel → thin cubecl_torch translation/contract layer → CubeK / CubeCL machinery → GPU`

with Burn used as a **reference implementation for how the ecosystem is meant to be composed**, rather than necessarily pulling Burn itself into the runtime.

That makes `cubecl_torch` an Inductor backend/adaptor, not a second GPU compiler living beside Inductor and CubeK.

And I think the repo has accumulated enough evidence now to justify making that the explicit architectural rule.

---

## The big thing the recent discussion got right

There was an earlier assumption hiding under a lot of Stage 5 work:

**CubeK has useful primitive kernels, but it isn't expressive enough to take the fused kernel Inductor gives us, so we need our own physical machinery around it.**

That assumption now looks wrong.

Burn is the smoking gun.

In `burn-cubecl-fusion/src/optim/reduce/args.rs`, `FusedReduceArgs` implements CubeK's `ReduceArgs`. Its `read_input()` literally calls `fuse_on_read(...)`, and its `write_output()` calls `fuse_on_write(...)`.

So CubeK reduction machinery doesn't fundamentally require:

`load tensor → reduce → store tensor`

It can be:

`execute fused computation → produce reduction operand → CubeK reduction → execute fused output computation`

And Burn's `reduce_broadcasted` goes further and composes elementwise and reduction blocks.

Even more importantly, `burn-cubecl-fusion/src/optim/reduce/tune.rs` already autotunes the fused reduction across:

- Unit
- Plane
- Cube

using CubeCL's tuner, with a key that contains both the reduction properties and information about the fused body.

That changes the interpretation of a lot of your existing work.

Your `compiler/plan.py` currently has a sizeable native physical universe: shared trees, compensated trees, plane-shared variants, independent planes, coalesced columns, vector plans, 512-unit plans, tile plans, Welford-specific variants, etc. Then `legality.py` contains the increasingly substantial body of knowledge deciding which ones apply.

Meanwhile CubeK itself has exactly the abstraction you'd expect to own this:

`BlueprintStrategy::Forced(...)` versus `BlueprintStrategy::Inferred(...)`

And its own comment is revealing: the strategy expresses **what the caller cares about**, while the routine infers the rest.

That is much closer to what we want.

---

## But I wouldn't go as far as “Inductor already makes the decisions”

This is where I'd correct the quoted response.

There are really several kinds of “Inductor decisions,” and they're not equally authoritative for a CubeCL backend.

| Decision | Correct owner |
|---|---|
| Which FX/IR operations exist and their semantics | PyTorch / Inductor |
| Which operations have been fused into one scheduled kernel | Inductor scheduler + backend fusion capability |
| Loop/index expressions, reads, writes, mutation/effects | Inductor scheduled kernel |
| Whether a reduction was actually split into separate scheduled kernels | Inductor |
| PyTorch stream, storage, aliasing and lifetime contract | `cubecl_torch` integration |
| Unit/plane/cube mapping | **CubeK** |
| Threads/planes, vector widths, partial arrangement | **CubeK / CubeCL ecosystem** |
| Physical reduction tree | **CubeK routine / blueprint** |
| Kernel variant tuning | **CubeCL/CubeK tuner** |
| Triton `R0_BLOCK`, `num_warps`, persistent implementation choices | **Triton-specific reference information**, not our contract |

Current PyTorch makes this distinction a little murky because parts of “Inductor” are explicitly Triton codegen heuristics.

For example, current PyTorch has `triton_config_reduction`, `persistent_reduction`, and a whole `heuristics/triton_codegen/reduction.py` area specifically for configuring **Inductor-generated Triton kernels**.

Your own `CubeCLScheduling` derives from `SIMDScheduling`, not from Triton scheduling. That is a giant hint about the intended boundary: a SIMD backend gets the scheduled computation and implements its own backend policy.

Even `should_use_persistent_reduction()` is something I'd now treat cautiously. Your `codegen.py` currently uses `InductorChoices.should_use_persistent_reduction()` when deciding how to express Welford. That may produce the right Triton-compatible behavior, but “persistent” is principally a physical implementation strategy, not a mathematical property of variance.

So:

**Inductor is upstream of us. Triton's implementation of Inductor is beside us.**

That's a much safer mental model.

---

## This also tells us exactly what's wrong with `mapping="inductor"`

Funny enough, the repo itself arrived at this conclusion overnight.

Commit `3073248b3d` added the paragraph in the runtime-bridge OpenSpec saying that `mapping="inductor"` is temporary because otherwise you grow a second Inductor scheduler inside `cubecl_torch`.

I agree with that completely.

But I'd sharpen it one more step:

`mapping="inductor"` isn't bad because copying an Inductor heuristic is intrinsically bad. It's bad **as a production abstraction** because the thing you're copying is actually a Triton physical configuration heuristic.

It has been extremely useful experimentally.

You learned:

Triton gets a certain reduction quality/performance → forcing CubeK into a similar partial structure gets CubeK into the same band → therefore CubeK's routine is capable, but its inferred blueprint is currently missing something.

That's excellent evidence.

The correct conclusion isn't therefore:

> Keep feeding Inductor mappings into CubeK.

It's:

> **Teach CubeK how to generate/consider that class of blueprint itself.**

Which is exactly what `zah.10` is now proposing.

Once CubeK can offer the good structure naturally, `_inductor_blueprint()` should die.

Triton's mapping was the microscope, not the architecture.

---

## This changes what “catalog”, “strategies” and “rules” should mean

I don't think the conclusion is “rules are bad.” A performant GPU compiler is inevitably full of heuristics, selection and legality constraints.

The issue is **where those rules live and what they describe**.

A rule like:

> “this load isn't 16-byte aligned, so this vectorized CubeK routine cannot be used”

is perfectly legitimate integration/capability information.

A rule like:

> “this operation has mutation/RNG semantics this ecosystem route cannot currently preserve”

is legitimate.

A CubeK rule like:

> “for this reduction count and device, consider Cube rather than Unit”

is legitimate backend implementation policy.

But a `cubecl_torch` rule saying approximately:

> “for FP32 rows between these sizes, choose this 512-thread physical structure with these private accumulators because this is the one we measured against Triton”

is much more suspicious.

That's how the current `catalog()` + `legality.py` combination starts becoming a second backend compiler.

The scoreboard/corpus, on the other hand, is great. **Keep the catalog of cases; get rid of the catalog as the runtime architecture.**

That's an important distinction.

---

## And I would slightly change “the kind of kernel picks the ecosystem routine”

That's still a bit too categorical.

Consider an Inductor kernel containing:

`x → x*x → sum → rsqrt → multiply x`

Calling that “a reduction kernel” is useful, but you're already seeing that real fused kernels can have several computational regions.

Then consider LayerNorm, softmax, Welford, multiple reductions, multiple outputs, masks, etc.

So instead of:

> kernel class → routine

I'd use:

### **capability-directed lowering**

Take the semantic structure Inductor scheduled and find the most-native ecosystem mechanism capable of expressing it.

Sometimes that's one CubeK reduce routine with fused read/write hooks.

Sometimes it's Burn's `reduce_broadcasted` pattern.

Sometimes it's CubeK matmul.

Sometimes `cubecl-tile`.

Sometimes there genuinely isn't machinery yet.

That last case is where you extend the ecosystem.

This still requires routing, but routing is a tiny semantic question:

> “Which ecosystem mechanism can express this program?”

It is **not** the huge physical question:

> “Which thread topology, vector width, private accumulator count, merge ordering and launch configuration should this program use?”

That second question belongs downstream.

---

## So yes: your “machine should normally be capable of doing its task” intuition is basically right

With one qualification.

You shouldn't expect CubeK to magically be optimal for PyTorch workloads without changes. Burn and PyTorch generate different workload distributions, numerical expectations, fusion patterns, dynamic shapes, etc.

But the normal reaction when CubeK falls short should increasingly be:

**why does the general CubeK machinery fail here?**

rather than:

**what new `cubecl_torch` strategy do we add for this case?**

The bridged reduction work is a really nice demonstration of that.

You needed:

- forced-plane validation fixes;
- lanes-first accumulation;
- plane-tree merge;
- multiple elements per unit;
- CubeCL plane/static-size machinery.

Those are general GPU-kernel capabilities. They make sense as CubeK/CubeCL improvements.

And after fixing them, the same CubeK reduction machinery can hit Triton's performance and numerical band.

That is much more encouraging than having another highly optimized project-local reduction emitter hit Triton's performance.

Because the former tells you the ecosystem can be the backend.

---

## Triton should stay extremely important — just in a different role

I would keep using Triton almost obsessively during development.

But as:

**differential oracle + performance floor + architectural clue.**

Not as production policy.

If CubeK generates something physically completely different from Triton but:

- has equivalent PyTorch-visible behavior;
- is inside or better than your numerical qualification;
- is as fast or faster;
- uses no more problematic memory;
- has acceptable compilation/launch cost;

then that's a success.

There is no reason to force it toward Triton's topology.

Conversely, when CubeK loses badly, looking at Triton's PTX/SASS, mapping, load pattern, reduction tree and launch config is exactly what you should do to understand *why*.

Then convert the discovered difference into a general CubeCL/CubeK capability where possible.

That's basically compiler differential diagnosis rather than imitation.

---

## What I would settle on now

I'd make the architecture something like this:

> **`cubecl_torch` is an Inductor codegen backend that translates Inductor-scheduled computation into native CubeCL-ecosystem programs while preserving PyTorch's semantic and runtime contracts. It does not maintain an independent physical GPU scheduler. CubeK/CubeCL own physical mapping, blueprint inference and tuning; missing capabilities are fixed there in upstreamable form. Triton is the differential reference and performance/numerical floor, not the source of production scheduling policy.**

That, to me, is a much more durable north star than either the current catalog model or “copy whatever Inductor/Triton decided.”

And then I'd apply one very simple design test whenever new code is proposed:

1. **Freeze new growth of the project-native physical catalog.** Keep all of it as a reference/fallback and keep all the measurement infrastructure. Don't rip anything out yet.
2. **Build the generic fused-reduction bridge next**, modeled on Burn's `FusedReduceArgs`: replay the Inductor body in CubeK's read/write hooks instead of only supporting today's plain `sum_rows`. RMSNorm's `sum(x²)` is the ideal first proof because it's small but crosses the important boundary.
3. **Make the currently-good Triton-shaped CubeK structure a native CubeK candidate**, rather than something produced by `_inductor_blueprint()`. Once CubeK itself can offer/select it and the results survive, delete `mapping="inductor"`.
4. **Expand by semantic capability rather than by shape strategy:** fused prologue, fused epilogue, multiple reductions/softmax/LayerNorm, Welford, multiple outputs, argmax/ties, masks, then split/cooperative cases. Burn should be mined heavily here because it already demonstrates several of these compositions.
5. **Retire native catalog families one at a time.** When the ecosystem path covers the semantic cases and meets the existing correctness/performance gates, its corresponding native plans stop being production candidates. Your current implementation becomes an extremely useful differential reference during that transition.
6. **For every future project-local rule, ask what kind of rule it is.** If it's a PyTorch contract, keep it here. If it's physical kernel policy, push it into CubeK/CubeCL. If it's merely a Triton-parity experiment, mark it temporary and give it a deletion criterion.

That would prevent this same architectural ambiguity from creeping back in six weeks from now.

### The biggest immediate experiment, IMO

The most informative next thing is **not another plain-reduction mapping experiment**.

You've more or less established that CubeK can do the physical reduction when properly configured.

The question that determines whether this whole architecture works is now:

**Can we take the arbitrary fused scalar program that `CubeCLKernel` is already recording from Inductor and use it as CubeK's `ReduceArgs` read/write computation, the same way Burn does?**

If yes, a huge amount of the native reduction compiler starts looking transitional almost immediately.

If no, the exact reason it fails tells you the *actual* missing abstraction in CubeK/CubeCL. That's vastly more useful than guessing ahead of time and making another project-local planner.

So I think you're at a pretty meaningful pivot in the repo: Stage 5 wasn't wasted at all — it gave you the numerical/performance oracle, the corpus, the runtime contracts, and enough understanding of the physical mechanisms to discover what needed fixing upstream. But **I would no longer treat the Stage 5 physical architecture as the thing to consolidate into production**. I'd treat it as the reference implementation you're now trying to make unnecessary.

That distinction makes basically all of the recent work click into place.