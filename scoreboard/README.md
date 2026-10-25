# Workload scoreboard

Measures the experimental CubeCL Inductor backend on real trainer workloads, against
the two alternatives a CPython 3.14t user has today:

| Path | Environment | GIL | Role |
| --- | --- | --- | --- |
| `eager` | isolated 3.14t, no Triton (`stage0/.venv`) | off | floor: what users have without compilation |
| `triton` | sd-scripts environment (`.venv-wsl`) | re-enabled by libtriton | target: get within a small % of this |
| `cubecl` | isolated 3.14t, no Triton | off | the CubeCL backend (`cubecl_torch.inductor`; results before 2026-10-20 used `stage5.backend`) |

Tracked as beads epic `4z8`. Acceptance target: Triton parity or better on complete training steps
for workloads matched for correctness. Individual kernel timings explain gaps; they are not the gate.

## Workloads

Defined in [workloads.py](workloads.py).

These are initial backend qualification workloads and current-reference integration settings. They
are meant to stay useful under the reworked sd-scripts training system, but they do not prescribe its
orchestration dependency (Accelerate is not a decided dependency), compilation interface, region
boundaries, shapes or precision profile. Matching today's compile settings is compatibility evidence,
not proof of future integration.

- **`sdxl`**: SDXL UNet from sd-scripts `library/models/sdxl`. Forward and backward with full fine-tune
  gradients, bf16 weights under bf16 autocast and SDPA attention. Regional compilation (`mode="default"`,
  `fullgraph=False`, `dynamic=True`) applies in place to every ResNet, Transformer, Downsample and
  Upsample block.
- **`anima`**: the kohya upstream Anima (a Cosmos-Predict2 DiT with the LLM adapter), from
  `sd-scripts-resources/sd-scripts/library/anima_models.py`. Same settings as `sdxl`, with
  each `Block` compiled. Defaults to 768 px: at 1024 px a full fine-tune without gradient
  checkpointing exceeds 24 GiB, even in eager.
- **`optimizer`**: compiled optimizer *helpers*, not a complete compiled optimizer step. These are
  the Newton–Schulz and `bias_rms` functions sd-scripts compiles, with `fullgraph=True` and
  `mode="reduce-overhead"`. `dynamic` is left at PyTorch's default (`None`). They run over
  representative SDXL weight shapes, as separate compile entry points outside the model's autocast.
  `mode="reduce-overhead"` alone doesn't prove CUDA graph capture, so the result records
  `cudaGraphLaunch` calls.

Scope of the model workloads:

- All denoiser parameters are trainable; text encoders and the VAE are excluded.
- Per-device microbatch 1, no gradient accumulation.
- `full_bf16`-style bf16 weights under bf16 autocast, keeping the models' explicit fp32 internal casts.
- No gradient checkpointing. That's a baseline, not evidence about checkpointed execution.
- Each result records actual latent shapes, conditioning lengths, the attention kernels used,
  and pinned source revisions.

Planned additions (beads `4z8`): LoRA, gradient checkpointing, an fp32-weights profile, bucketed
shape variation within a run, larger microbatches, compiled losses and encoders, a quantized base
with adapters, and cold-start versus cache-warm compile time.

Deviations from the trainer, deliberately recorded:

- accelerate's `compile_regions` (what sd-scripts' `use_regional_compilation=True` calls) cannot run
  the SDXL UNet: it wraps the UNet's inner `ModuleList`s, which `forward` iterates. The harness uses
  in-place `module.compile` on the repeated blocks instead.
- `paper_orthograd_compile` is excluded because it fails under `fullgraph=True` on every backend: it
  branches on a tensor value.
- Upstream kohya `library.utils` is stubbed when importing Anima, because it imports `cv2`, which
  re-enables the GIL.
- `torch._functorch.config.force_non_lazy_backward_lowering = True` on both compiled paths, so
  backward compilation happens inside the backend call and CubeCL rejections are attributed to a
  region.
- The CubeCL backend cannot honour `mode="reduce-overhead"`; each result records this.

## Running

```bash
stage0/.venv/bin/python scoreboard/run_all.py                      # all workloads/paths, CubeCL fail-fast
stage0/.venv/bin/python scoreboard/run_all.py --on-reject record   # CubeCL inventory: record every rejection
stage0/.venv/bin/python scoreboard/run_all.py --workloads sdxl --paths triton,cubecl --passthrough --resolution 512
stage0/.venv/bin/python scoreboard/report.py scoreboard/results/<run>   # re-render summary.md
```

Raw results go to `scoreboard/results/<run>/` (not tracked in git). Each run writes one JSON and
one log per (workload, path), plus a `summary.md` with these views:

- **Coverage**: compiled regions, graph breaks, CubeCL regions by status, and the most frequent
  rejection or failure.
- **Rejection reasons**: per workload, each distinct reason, the number of regions that have it
  and how often it was found in them (nodes, for reasons found before fusion).
- **Cost**: median step time, kernel time split into generated, vendor-library and ATen/other,
  ratios against eager and Triton, first-step (compile) time, peak memory, and error against an
  eager reference computed in the same process.

### CubeCL rejection modes

- `fail` (default): stop at the first rejected region. This shows exactly what blocks a workload.
- `record`: run each rejected region eagerly and continue, recording every rejection. This mode is
  for inventory only. Rejected regions never count as CubeCL coverage, and the run is marked as not
  a CubeCL performance result. Each region is compiled with `rejections="all"`: the scan continues
  past the first unsupported feature (every node and every check within a node, then the backward
  graph even when the forward one was rejected), so one run shows a region's whole gap set. A
  graph's scan stops at its pre-fusion boundary, so code-generation rejections behind pre-fusion
  ones only appear once those are fixed.

## Known limitations

- GPU clocks aren't fixed yet (`sil.1`). Each JSON records `nvidia-smi` readings before and after
  the run.
- The *ATen/other* kernel class mixes eager code outside compiled regions, extern ATen calls and
  fallbacks.
- CubeCL kernels (`cubecl5_<hash>`) are not yet mapped to kernel families. That belongs to the
  census (`4z8.4`).
