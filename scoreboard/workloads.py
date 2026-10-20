"""Scoreboard workloads, mirroring how sd-scripts compiles models and optimizer steps.

Model workloads use sd-scripts' model compile settings (Inductor mode "default",
fullgraph=False, dynamic=True, regional compilation) with full-bf16 weights under
bf16 autocast. Regional compilation compiles each repeated block in place with
``module.compile``; accelerate's ``compile_regions`` cannot run the SDXL UNet
(it wraps the UNet's inner ModuleLists, which ``forward`` iterates).

Function workloads use sd-scripts' optimizer helpers exactly as decorated there:
``torch.compile(fullgraph=True, mode="reduce-overhead")``.

Every workload is deterministic for a given seed so each path can be compared
against an eager reference computed in the same process.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import types
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

SD_SCRIPTS = Path(
    os.environ.get("SCOREBOARD_SD_SCRIPTS", "/home/imi/Projects/sd-scripts")
)
KOHYA = Path(
    os.environ.get(
        "SCOREBOARD_KOHYA", "/home/imi/Projects/sd-scripts-resources/sd-scripts"
    )
)

MODEL_COMPILE = {"fullgraph": False, "dynamic": True, "mode": "default"}
OPTIMIZER_COMPILE = {"fullgraph": True, "mode": "reduce-overhead"}


@dataclass
class Instance:
    """A built workload: what to compile, and one deterministic training step."""

    step: Callable[[], dict[str, torch.Tensor]]
    regions: list[torch.nn.Module] = field(default_factory=list)
    functions: dict[str, Callable] = field(default_factory=dict)
    parameters: list[torch.nn.Parameter] = field(default_factory=list)
    settings: dict = field(default_factory=dict)


@dataclass
class Workload:
    name: str
    source: Path
    compile: dict
    build: Callable[[torch.device, dict], Instance]
    description: str


@contextlib.contextmanager
def default_dtype(dtype):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def source_revision(path: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _model_step(model, forward, target, parameters):
    def step():
        for p in parameters:
            p.grad = None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = forward()
        loss = F.mse_loss(out.float(), target.float())
        loss.backward()
        return {"output": out.detach(), "loss": loss.detach()}

    return step


def build_sdxl(device, options):
    sys.path.insert(0, str(SD_SCRIPTS))
    from library.models.sdxl import unet as U

    batch, resolution = options.get("batch", 1), options.get("resolution", 1024)
    torch.manual_seed(0)
    with torch.device(device), default_dtype(torch.bfloat16):
        model = U.SdxlUNet2DConditionModel()
    for m in model.modules():
        if isinstance(m, U.CrossAttention):
            m.set_use_sdpa(True)
    model.train()
    g = torch.Generator(device).manual_seed(1)
    latent = resolution // 8
    x = torch.randn(
        batch, 4, latent, latent, generator=g, device=device, dtype=torch.bfloat16
    )
    t = torch.randint(0, 1000, (batch,), generator=g, device=device)
    context = torch.randn(
        batch, 77, 2048, generator=g, device=device, dtype=torch.bfloat16
    )
    y = torch.randn(batch, 2816, generator=g, device=device, dtype=torch.bfloat16)
    target = torch.randn(x.shape, generator=g, device=device, dtype=torch.bfloat16)
    blocks = (U.ResnetBlock2D, U.Transformer2DModel, U.Downsample2D, U.Upsample2D)
    parameters = [p for p in model.parameters() if p.requires_grad]
    return Instance(
        step=_model_step(model, lambda: model(x, t, context, y), target, parameters),
        regions=[m for m in model.modules() if isinstance(m, blocks)],
        parameters=parameters,
        settings={
            "microbatch": batch,
            "gradient_accumulation": 1,
            "image": f"{resolution}x{resolution}",
            "latent_shape": list(x.shape),
            "conditioning": {"context": list(context.shape), "y": list(y.shape)},
            "weights": "bf16 (full_bf16 profile) under bf16 autocast",
            "attention": "SDPA (implementation chosen by PyTorch; see attention kernels)",
            "gradient_checkpointing": False,
            "regions": [b.__name__ for b in blocks],
            "trainable": "all UNet (denoiser) parameters; text encoders and VAE excluded",
        },
    )


def _import_kohya_anima():
    """Import upstream kohya Anima without its heavy ``library.utils`` (cv2/diffusers).

    cv2 re-enables the GIL on 3.14t; only ``setup_logging`` is used from that module.
    """
    package = types.ModuleType("library")
    package.__path__ = [str(KOHYA / "library")]
    sys.modules["library"] = package
    utils = types.ModuleType("library.utils")
    utils.setup_logging = lambda *a, **k: None
    sys.modules["library.utils"] = utils
    from library import anima_models

    return anima_models


ANIMA_CONFIG = {  # kohya library/anima_utils.py dit_config; num_blocks from the released checkpoint.
    "max_img_h": 512,
    "max_img_w": 512,
    "max_frames": 128,
    "in_channels": 16,
    "out_channels": 16,
    "patch_spatial": 2,
    "patch_temporal": 1,
    "model_channels": 2048,
    "concat_padding_mask": True,
    "crossattn_emb_channels": 1024,
    "pos_emb_cls": "rope3d",
    "pos_emb_learnable": True,
    "pos_emb_interpolation": "crop",
    "min_fps": 1,
    "max_fps": 30,
    "use_adaln_lora": True,
    "adaln_lora_dim": 256,
    "num_blocks": 28,
    "num_heads": 16,
    "extra_per_block_abs_pos_emb": False,
    "rope_h_extrapolation_ratio": 4.0,
    "rope_w_extrapolation_ratio": 4.0,
    "rope_t_extrapolation_ratio": 1.0,
    "extra_h_extrapolation_ratio": 1.0,
    "extra_w_extrapolation_ratio": 1.0,
    "extra_t_extrapolation_ratio": 1.0,
    "rope_enable_fps_modulation": False,
    "use_llm_adapter": True,
    "attn_mode": "torch",
    "split_attn": False,
}


def build_anima(device, options):
    A = _import_kohya_anima()
    # 1024 px full fine-tune without gradient checkpointing exceeds 24 GiB even in eager; 768 px peaks at ~17 GiB.
    batch, resolution = options.get("batch", 1), options.get("resolution", 768)
    blocks = options.get("blocks", ANIMA_CONFIG["num_blocks"])
    torch.manual_seed(0)
    with torch.device(device), default_dtype(torch.bfloat16):
        model = A.Anima(**{**ANIMA_CONFIG, "num_blocks": blocks})
    model.train()
    g = torch.Generator(device).manual_seed(1)
    latent, tokens = resolution // 8, 512
    x = torch.randn(
        batch, 16, 1, latent, latent, generator=g, device=device, dtype=torch.bfloat16
    )
    # kohya builds timesteps with dit_weight_dtype; fp32 timesteps break the bf16 block path.
    t = torch.rand(batch, generator=g, device=device).to(torch.bfloat16)
    prompt = torch.randn(
        batch, tokens, 1024, generator=g, device=device, dtype=torch.bfloat16
    )
    source_mask = torch.ones(batch, tokens, device=device, dtype=torch.long)
    t5_ids = torch.randint(0, 32000, (batch, tokens), generator=g, device=device)
    t5_mask = torch.ones(batch, tokens, device=device, dtype=torch.long)
    padding = torch.zeros(batch, 1, latent, latent, device=device, dtype=torch.bfloat16)
    target = torch.randn(x.shape, generator=g, device=device, dtype=torch.bfloat16)

    def forward():
        return model(
            x,
            t,
            prompt,
            padding_mask=padding,
            source_attention_mask=source_mask,
            t5_input_ids=t5_ids,
            t5_attn_mask=t5_mask,
        )

    parameters = [p for p in model.parameters() if p.requires_grad]
    return Instance(
        step=_model_step(model, forward, target, parameters),
        regions=list(model.blocks),
        parameters=parameters,
        settings={
            "microbatch": batch,
            "gradient_accumulation": 1,
            "image": f"{resolution}x{resolution}",
            "latent_shape": list(x.shape),
            "image_tokens": (latent // ANIMA_CONFIG["patch_spatial"]) ** 2,
            "conditioning": {
                "qwen3_prompt": list(prompt.shape),
                "t5_ids": list(t5_ids.shape),
            },
            "blocks": blocks,
            "weights": "bf16 (full_bf16 profile) under bf16 autocast; RMSNorm computes in fp32",
            "attention": "attn_mode=torch, i.e. SDPA (implementation chosen by PyTorch; see attention kernels)",
            "gradient_checkpointing": False,
            "llm_adapter": True,
            "regions": ["Block"],
            "trainable": "all DiT and LLM-adapter parameters; Qwen3 text encoder and VAE excluded",
        },
    )


# Representative SDXL weight shapes (attention/FF projections and convolution-free linears).
OPTIMIZER_SHAPES = [
    (1280, 1280),
    (10240, 1280),
    (1280, 5120),
    (640, 640),
    (5120, 640),
    (640, 2560),
    (1280, 2048),
]


def build_optimizer(device, options):
    sys.path.insert(0, str(SD_SCRIPTS))
    from library.optimization.optimizers.utils import orthograd as O

    g = torch.Generator(device).manual_seed(2)
    grads = [
        torch.randn(s, generator=g, device=device) * 1e-3 for s in OPTIMIZER_SHAPES
    ]
    biases = [
        torch.randn(s[0], generator=g, device=device, dtype=torch.bfloat16) * 1e-3
        for s in OPTIMIZER_SHAPES
    ]
    instance = Instance(
        step=None,
        functions={
            # The sd-scripts bodies behind its *_compile wrappers used by dehaze/ademamix.
            # paper_orthograd_compile (adopt_schedulefree_ao) is excluded: its data-dependent
            # `if weight_norm_sq <= eps` cannot compile under fullgraph=True on any backend.
            "newton_schulz_6": O.zero_power_via_newton_schulz_6,
            "bias_rms": O.bias_rms,
        },
        settings={
            "shapes": OPTIMIZER_SHAPES,
            "dtype": "bf16",
            "excluded": "paper_orthograd_compile: fails under fullgraph=True with Triton too (data-dependent branch)",
        },
    )

    def step():
        fn = instance.functions
        out = {}
        for i, (gr, b) in enumerate(zip(grads, biases)):
            # Clone at once: reduce-overhead (CUDA graph) outputs are overwritten by the next call.
            out[f"ns6.{i}"] = fn["newton_schulz_6"](gr.bfloat16()).clone()
            out[f"bias.{i}"] = fn["bias_rms"](b).clone()
        return out

    instance.step = step
    return instance


WORKLOADS = {
    w.name: w
    for w in [
        Workload(
            "sdxl",
            SD_SCRIPTS,
            MODEL_COMPILE,
            build_sdxl,
            "SDXL UNet forward+backward, regional compile of ResNet/Transformer/Down/Upsample blocks",
        ),
        Workload(
            "anima",
            KOHYA,
            MODEL_COMPILE,
            build_anima,
            "Anima (Cosmos-Predict2 DiT + LLM adapter) forward+backward, regional compile of Blocks",
        ),
        Workload(
            "optimizer",
            SD_SCRIPTS,
            OPTIMIZER_COMPILE,
            build_optimizer,
            "sd-scripts compiled optimizer helpers (Newton-Schulz, bias_rms) over SDXL weight shapes",
        ),
    ]
}
