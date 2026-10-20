# Trainer feedback

## Assumptions

A = Agent in charge of trainer design and work
U = User

Agent:
Mostly sound **as an initial benchmark profile**, not as repo defaults or requirements for the future Trainer. After checking the code, I’d tighten these points:

1. **Model compilation matches exactly.** Those are the settings in [trainer_utils.py](/home/imi/Projects/sd-scripts/library/training/trainer_utils.py:277), enabled when `torch_compile=True`. Regional compilation means the harness should preserve the model’s regional boundaries—not substitute whole-model compilation and call it equivalent. [Accelerate documentation](https://huggingface.co/docs/accelerate/usage_guides/compilation)

2. **“Optimizer compile” should say “compiled optimizer helpers.”** The Newton–Schulz and `bias_rms` wrappers have exactly those decorators in [orthograd.py](/home/imi/Projects/sd-scripts/library/optimization/optimizers/utils/orthograd.py:37). That does **not** establish compilation of the complete optimizer step. Also:
   - They do not specify `dynamic=True`; PyTorch’s default is `dynamic=None`.
   - They are separate compilation entry points. Configuring the model’s Accelerate plugin does not configure these helpers—ensure the backend under test reaches both paths. [PyTorch documentation](https://docs.pytorch.org/docs/2.14/generated/torch.compile.html)

3. **The bf16 description is consistent, but preserve mixed internal computation.** `full_bf16` explicitly casts trainable models to bf16 in [finetune_mode.py](/home/imi/Projects/sd-scripts/library/training/modes/finetune_mode.py:173). It does not mean every operation or optimizer state is bf16. For example, Newton–Schulz explicitly starts with `.float()`, and Anima’s RMSNorm computes its normalization in fp32. Don’t simplify those casts away in the benchmark or inadvertently put optimizer work inside the model’s autocast scope.

4. **“SDPA” does not identify one attention implementation.** PyTorch selects an implementation based on the inputs and enabled backends. Record the actual implementation used so a comparison doesn’t silently become “different compiler **and** different attention kernel.” [SDPA documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)

The other choices are reasonable, but make their scope explicit:

- “Full fine-tune gradients” → all denoiser parameters trainable; separately state whether text encoders/VAE are excluded.
- “Batch 1” → per-device microbatch 1.
- “1024 px” → `1024 × 1024` images, with model-specific latent shapes and conditioning lengths recorded.
- No gradient checkpointing → a valid baseline, not evidence about checkpointed execution.
- Pin both model source revisions and configurations.

Finally, `reduce-overhead` can use CUDA graphs, but the mode alone does not prove capture occurred. Record that separately. [PyTorch documentation](https://docs.pytorch.org/docs/2.14/generated/torch.compile.html)

U:
and this is not just about what we have now, but about what we will have after the post trainer rework is in place, to the extent that we can tell from now. down to the level of "did we even decide to use accelerate" and such

A:
Right. My previous answer checked fidelity to today’s code; it did not adequately check whether those choices describe the future system.

**We have not decided that the reworked training system must use Accelerate.** The direction explicitly treats Accelerate/Fabric/Lightning as prior art, not an assumed architecture or adoption target. That does not rule out retaining useful Accelerate functionality behind an integration. [Recorded direction](/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/strategy_system_direction.md:1402)

Against the future design, I’d classify the assumptions like this:

| Assumption | Status after the rework |
|---|---|
| SDXL and Anima | Representative numerical workloads—not a restriction on supported models or training arrangements. |
| Full fine-tuning | An intended supported form, alongside PEFT and compatible combinations—not the universal training configuration. |
| bf16 weights and autocast | A useful precision profile to qualify. Neither bf16-only execution nor today’s `full_bf16` configuration surface is an architectural commitment. |
| Batch 1, 1024² | Initial workload choices. They must not become assumptions embedded in the backend or training engine. |
| SDPA | A selected attention implementation for these tests—not a requirement that all future model behavior use it. |
| No gradient checkpointing | A baseline test condition. The design also accommodates supported recomputation, with state, RNG and gradient semantics preserved. |
| Accelerate’s `TorchDynamoPlugin` and its flags | Today’s integration path. **Not the decided post-rework compilation interface or policy.** |
| Compiled Newton–Schulz / `bias_rms` helpers | Concrete workloads worth supporting. Their current compilation boundaries and modes are not frozen future boundaries. |

The design deliberately allows different prepared realizations, provided they preserve the accepted computation, gradients, numerical policy and coordination requirements. It also requires rejecting unsupported combinations rather than silently changing the algorithm. [Execution requirements](/home/imi/Projects/sd-scripts/openspec/changes/rework-model-strategy-trainer/specs/accepted-training-execution/spec.md:92), [Preparation requirements](/home/imi/Projects/sd-scripts/openspec/changes/rework-model-strategy-trainer/specs/training-runtime-preparation/spec.md:103)

So I would introduce that assumptions section with:

> These are initial backend qualification workloads and current-reference integration settings. They are intended to remain useful under the reworked training system, but do not prescribe its orchestration dependency, compilation interface, region boundaries or global runtime configuration.

**The backend work should support the numerical work regardless of whether the eventual caller is Accelerate or our own prepared execution machinery.** Matching today’s plugin invocation is useful compatibility evidence, not proof of future integration.

## Questions

- **Question 1:**

**Families:** which model families will the reworked trainer support first? Are SDXL and
Anima the right first two? Will Anima be a port of kohya's implementation or a new one?

- **Answer 1:**

**SDXL and Anima are sensible first backend workloads—not a settled production rollout order.** They exercise different model structures and conditioning paths. Our intended diffusion coverage is broader: SDXL, Anima, Krea2, Moss, Qwen Image 2.1, Wan 2.2, HI3 and YuE. That is a target list, not a claim that every integration will ship immediately, or that there will be no other models.

The existing design also carries SD and SD3 as maintained migration cases; choosing this backend pair doesn’t remove them. [Recorded direction](/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/strategy_system_direction.md:799)

**For Anima, my recommendation is adapted reuse of kohya’s implementation, not a from-scratch rewrite.** Concretely:

- Reuse verified neural-module definitions, checkpoint compatibility and necessary numerical behavior.
- Adapt loading, conditioning and execution integration to our architecture.
- Don’t import its old strategy/Trainer organization as part of adopting the model.
- Change underlying computation only for an identified reason, with correctness checks.

That is a recommendation, not an implementation choice we have already finalized. For the backend work now, a pinned kohya implementation is an appropriate reference; it needn’t wait for our integration.

**The research index establishes broader architectural coverage, not a release checklist.** It includes language models, autoencoders, distillation, control networks, video/audio, RL and changing topology. Those cases inform what our system must accommodate or avoid obstructing; they don’t all become first-release implementations. [Research scope](/home/imi/Projects/sd-scripts/openspec/changes/rework-model-strategy-trainer/research/README.md:5)

So the short answer to send back is:

> Start backend qualification with SDXL and Anima. They are representative starting workloads, not the complete family scope or a fixed production rollout. Use pinned kohya Anima computation as the initial reference; our eventual integration should preferably adapt that implementation to the new training architecture rather than rewrite it unnecessarily.

- **Question 2:**

**Typical training setups per family:**

- full fine-tune or LoRA/adapters (or both);
- batch size and resolution, and how bucketing varies shapes within a run;
- gradient checkpointing on or off;
- `full_bf16` or fp32 master weights with autocast;
- fp8 base weights.

- **Answer 2:**

| Concern | Intended coverage |
|---|---|
| Full fine-tuning versus adapters | Both, where the selected model/method integration supports them. Also allow compatible mixtures of directly trained parameters and adapters. |
| Batch size | Not fixed at 1. Include microbatch 1 and larger feasible batches; distinguish microbatch size from gradient accumulation and effective batch size. |
| Resolution and bucketing | Vary spatial dimensions within a run, including square, portrait and landscape buckets. Record actual latent/token shapes and conditioning lengths—not just nominal image resolution. Video/audio later add frame-count and duration variation. |
| Gradient checkpointing | Both on and off. **Checkpointing-on belongs in early qualification**, rather than being treated as an unusual extension. Record the actual checkpoint implementation and boundaries. |
| Ordinary precision | Cover both FP32 trainable parameters with bf16 autocast and bf16 trainable parameters with the intended mixed-compute behavior. Separately identify gradients, accumulation, optimizer state and any explicit FP32 master copy. |
| Quantized base weights | FP8, INT8 and other quantizations are intended coverage, but each needs a concrete implementation/profile. Quantized storage, compute precision and trainability are separate choices. |

Those distinctions already appear in our [precision research](/home/imi/Projects/sd-scripts/openspec/changes/rework-model-strategy-trainer/research/precision-and-quantization-training.md:25) and [bucketing research](/home/imi/Projects/sd-scripts/openspec/changes/rework-model-strategy-trainer/research/training-data-pipeline.md:166).

Two particularly important qualifications:

- **Checkpointing is not merely a memory flag.** It changes execution through recomputation, so compilation must preserve the required gradients, RNG and state behavior. PyTorch also has distinct checkpoint implementations. [Checkpoint documentation](https://docs.pytorch.org/docs/2.14/checkpoint.html)
- **A frozen or quantized base is not necessarily inference-only computation.** Gradients may still need to pass through it to reach trainable adapters or earlier components. Our [execution requirements](/home/imi/Projects/sd-scripts/openspec/changes/rework-model-strategy-trainer/specs/accepted-training-execution/spec.md:121) explicitly preserve that distinction.

For **first-in-line backend work**, my recommendation is:

1. Ordinary full-training and LoRA workloads, checkpointing on/off, both ordinary precision profiles, and realistic shape variation.
2. An early quantized-base-plus-adapter integration, chosen from an actual intended implementation. Then expand FP8/INT8/other formats according to observed gaps.
3. Treat native low-precision training and quantization-aware training separately; supporting a frozen quantized base does not demonstrate those.

That does **not** require benchmarking every combination immediately. Use representative combinations, then focused tests for interactions.

- **Question 3:**

**Compilation plans:**
- Will the reworked trainer keep accelerate's dynamo plugin with regional compilation and
  `dynamic=True`?
- Will anything besides model blocks and optimizer helpers be compiled, for example the loss
  or objective, or text encoders?

- **Answer 3:**

**Will we retain Accelerate’s plugin, regional compilation and `dynamic=True`?**

Not necessarily. Accelerate, Fabric or selected functionality from either may serve the implementation, but must fit our architecture—not determine it. [Recorded direction](/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/strategy_system_direction.md:1402)

Likewise:

- Regional compilation is one useful mechanism, not a mandatory partitioning scheme.
- `dynamic=True` is not a universal future policy. Supported shape variation could use dynamic kernels, guarded specialized variants, or a combination.
- Compilation choices may differ between prepared regions. There need not be one model-wide or run-wide configuration.

**Can compilation include objectives, losses and text encoders?**

Yes. The intended scope is broader than model blocks and optimizer helpers. Eligible numerical work includes:

- model computation;
- objectives, prediction transformations and losses;
- text encoders, conditioning adapters and representation producers such as VAEs;
- optimizer calculations and, where supported, larger update regions.

These are compilation candidates—not a promise that every implementation will automatically compile from day one.

Importantly, **today’s Python function boundaries should not become permanent compilation boundaries**. A supported realization might combine prediction and loss computation. Another might keep them separate. Both must preserve required gradients, numerical behavior, observations, state access and failure outcomes. The design explicitly permits fusion and partitioning under those obligations. [Execution requirements](/home/imi/Projects/sd-scripts/openspec/changes/rework-model-strategy-trainer/specs/accepted-training-execution/spec.md:92)

There are also two connected but distinct levels:

- Our run-IR lowering produces specialized execution and coordination machinery.
- Numerical compilation optimizes supported computation within that realization.

The whole-run graph does not therefore have to become one `torch.compile` invocation.

For the backend agent, I’d summarize:

> Do not assume the future Trainer retains Accelerate’s Dynamo plugin or its current flags. Plan for selective numerical compilation beyond model blocks, including objectives/losses, encoders, conditioning and optimizer computation. Use current compilation settings as compatibility profiles, while keeping supported shapes and compilation boundaries independent of the orchestration library.

- **Question 4:**

**Free-threading:**

- What runs concurrently with training: data production, caching, encoders?
- Does any compilation happen while training runs?
- Is native Windows 3.14t a goal? triton-windows has no cp314t wheel.

- **Answer 4:**

Yes—**free-threading is intended to benefit the whole-run system, not just a DataLoader**. I’d answer the three parts as follows.

### What should run concurrently with training?

Where selected, supported and useful:

- **Data preparation:** discovery, remote fetching, decoding, augmentation, tokenization, bucket/packing preparation and prefetching.
- **Cache production:** latent and conditioning computation, cache reads/writes, shard publication and registry updates. Training should not universally require the entire dataset’s preparation to finish first.
- **Encoders:** particularly frozen text encoders producing conditioning for upcoming, possibly augmented captions. CPU, another GPU or supported same-device execution are possible placements—not one mandatory arrangement.
- **Runtime services:** logging, metadata routing and persistence I/O after the required coherent capture.
- **Backend work:** communication and parameter/activation/gradient transfers may overlap computation under a supported execution schedule.

That matches the [async data direction](/home/imi/Projects/sd-scripts/docs_design/future_ideas/async_data.md:19), [encoder-prefetch research](/home/imi/Projects/sd-scripts/docs_design/future_ideas/async_te_offload_otf.md:74) and [distributed execution research](/home/imi/Projects/sd-scripts/openspec/changes/rework-model-strategy-trainer/research/distributed-training-execution.md:210).

But **independently progressing does not mean unrestricted simultaneous access**:

- Queues and resource use must be bounded.
- Results must retain the correct sample, caption/transformation and producer-state associations.
- Readiness must not silently change the accepted sampling/exposure policy.
- A trainable encoder cannot be treated like a frozen producer: required gradients and numerical state must survive, or that execution arrangement must be rejected.
- Validation/sampling cannot casually read models while optimization or temporary parameter changes conflict with that access.

Threads, processes, separate devices and backend streams remain implementation choices. We should exploit free-threaded Python where it helps, without requiring every activity to become a thread or promising that same-GPU overlap always improves performance.

### Can compilation happen during training?

**We should not assume all compilation finishes before training starts.** New shapes, newly used capabilities or an accepted arrangement change may require preparation or compilation during a run. An encoder producer may also encounter its first compilation independently of training computation.

However, we have **not selected a concurrent-compilation implementation**. My recommendation is:

- Prewarm predictable work where worthwhile.
- Permit later compilation/preparation.
- Allow unrelated execution to continue where dependencies and resource access permit.
- Serialize or isolate compiler services where necessary; don’t assume arbitrary simultaneous `torch.compile` calls are safe.
- Publish new executable state only after verification, without replacing state still in use.

Ordinary weight updates do not, by themselves, imply recompilation. Nor should every changing caption cause it. Shape-dependent specialization is a separate concern. [PyTorch compilation behavior](https://docs.pytorch.org/docs/2.14/generated/torch.compile.html)

### Is native Windows 3.14t a goal?

**Yes—native Windows, not merely WSL.**

The currently published `triton-windows` release lists ordinary `cp314-cp314` Windows wheels, not a free-threaded `cp314t` ABI wheel. That is a dependency-path gap, not a reason to redefine the goal. [Current release files](https://pypi.org/project/triton-windows/#files)

Qualification must distinguish:

1. Running on 3.14t.
2. Keeping the GIL disabled—including after lazy imports and first compilation.
3. Correct concurrent execution of the actual native/runtime stack.

CPython explicitly warns that importing an extension without free-threading support can enable the GIL. [CPython documentation](https://docs.python.org/3.14/howto/free-threading-extensions.html)

Our current CPU candidate provides bounded worker/producer interaction evidence—not proof of the complete GPU, compiler or native-Windows stack. [Recorded limits](/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/execution_candidate/README.md:997)

- **Question 5:**

**Attention:** SDPA as the default? Any plans for varlen or packed batches (as KohakuMoss does),
or flash/sage/xformers?

- **Answer 5:**

**SDPA is a sensible baseline, not an exclusive architectural choice.** We haven’t finalized the default-selection policy. Also, “SDPA versus FlashAttention” is not always a clean distinction: PyTorch’s SDPA can itself select a flash implementation. [PyTorch documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)

**Variable-length and packed batches are intended capabilities.** Our research explicitly covers packing while preserving sample boundaries, attention masks, positions and loss accounting. That means more than accepting variable tensor dimensions: unrelated samples must not attend to each other or receive incorrectly weighted losses. It does not mean every model must use packing. [Recorded research](/home/imi/Projects/sd-scripts/openspec/changes/rework-model-strategy-trainer/research/training-data-pipeline.md:132)

For implementation targets:

| Option | Intended role |
|---|---|
| PyTorch SDPA | Baseline and compatible fallback where it satisfies the workload. |
| FlashAttention implementations | Optimized dense/varlen paths, qualified for the specific hardware and required backward behavior. |
| xFormers | An explicit integration target—not something to discard because SDPA exists. |
| Sage and similar approaches | Supported candidates where the particular implementation meets the selected training or inference requirements and numerical policy. |
| Model-specific/custom attention | Allowed where the model requires behavior or performance that existing integrations do not supply. |

The important qualification for **Sage** is not to confuse inference acceleration with training support. Training-focused work such as SageBwd exists, but it requires its own accuracy and training validation; a fast inference path is not automatically a valid training replacement. [SageAttention](https://github.com/thu-ml/SageAttention), [SageBwd research](https://arxiv.org/abs/2603.02170) Even LoRA can require gradients through frozen base attention.

KohakuMoss is a useful reference because it compares **packed attention forward and backward**, custom kernels and PyTorch alternatives across different hardware—not just one inference benchmark. We should study those mechanisms without inheriting its model-specific choices wholesale. [Local attention study](/home/imi/Projects/pytortch-backend-experiments/.stage0-sources/KohakuMossTrainer/docs/performance/attention.md:1)

The support decision should therefore depend on the actual workload: masks, sequence layout, dtype, required gradients, hardware, numerical tolerance, compilation compatibility—and our Windows/free-threading goals. Unsupported combinations need a clear rejection or compatible fallback, not silent changes to attention behavior.

**For the backend work: do not assume attention is always SDPA, always padded, or always generated by Inductor.** External attention kernels and their integration are part of the intended workload coverage.

- **Question 6:**

**Performance expectations:** how much compile time per run is acceptable? (Triton's first step for
SDXL at 256 px took about 150 s in our harness.) What step-time gap against Triton would make a
GIL-free compile path worth using?

- **Answer 6:**

- **Acceptance target:** Triton parity or better on correctness-matched, representative workloads.
- **Near-parity candidates:** keep investigating small gaps when there is a plausible, inexpensive improvement. That is permission to continue development—not to declare a slower implementation accepted.
- **Optimization effort:** take straightforward gains during initial implementation; defer expensive micro-optimization unless measurements identify a significant bottleneck.
- **Measurement uncertainty:** a repeatable 2% slowdown is still a slowdown. An unresolved difference within measurement noise is inconclusive, not automatically a win.

One refinement I recommend: apply the hard gate to **actual training workloads**, using individual-kernel timings to explain gaps. Requiring every individual kernel to beat Triton could reject a backend whose complete training step is faster.

For **compilation time**, we haven’t established a numerical ceiling. The reported 150-second first step is a comparison point, not an acceptable budget we’ve agreed to.

My proposed target would be:

- Competitive cold-start time against Triton on the same workload.
- Effective cache reuse on subsequent runs.
- No unnecessary repeated compilation or tuning.
- Separate reporting of cold startup, cache-warm startup, steady-state steps and total run time.

Any extra startup cost should be visible and justified against the intended run length—not hidden behind a steady-state speed chart.

So the direct answer to **“what step-time gap makes GIL-free compilation worthwhile?”** is:

> **Parity is sufficient; a training-speed penalty is not the intended tradeoff. Small shortfalls may justify bounded development work, but not acceptance. Free-threading and its concurrency benefits must also be demonstrated separately.**

These are backend performance expectations; the existing engine-overhead budgets are a different comparison and don’t grant this backend a slowdown allowance.

## Bug finding responses

Two are confirmed; the third needs qualification.

1. **SDXL regional compilation: confirmed with Accelerate 1.15.0.**  
   Using the actual UNet instantiated on `meta`, its inner `ModuleList` becomes `OptimizedModule`, and iteration raises the reported error. [The forward path requires iteration](/home/imi/Projects/sd-scripts/library/models/sdxl/unet.py:1170).  
   This invalidates that automatic wrapping approach—not regional compilation itself. In-place compilation of callable blocks preserves containers; I verified that approach on a small forward/backward fixture, not a full SDXL GPU run.

2. **`paper_orthograd_compile`: confirmed.**  
   The actual wrapper fails on the tensor-dependent branch. The underlying body also fails full-graph capture with `eager` and `aot_eager`, so this is not a CubeCL code-generation defect. It affects **Compass as well as ADOPT AO**. Any eventual correction must preserve the small/zero-norm early-return behavior. [Source](/home/imi/Projects/sd-scripts/library/optimization/optimizers/utils/orthograd.py:56)

3. **`orthograd_atan`: not independently confirmed on the reported path.**  
   CPU Inductor passed the tested shape changes with both `dynamic=None` and `dynamic=True`, matching eager results. Those probes required disabling header precompilation because of an unrelated local `openssl` issue. The original device, dtype, flags and traceback are needed before classifying this further. SCORN’s current eager call is verified; this report does not establish an eager SCORN bug.
