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

RESOURCES = Path(
    os.environ.get("SCOREBOARD_RESOURCES", "/home/imi/Projects/sd-scripts-resources")
)
MUSUBI = RESOURCES / "musubi-tuner"
AI_TOOLKIT = RESOURCES / "ai-toolkit"
COSYVOICE = RESOURCES / "CosyVoice"
# Pure-Python packages a census source needs that the isolated venv lacks
# (CosyVoice's pinned x-transformers 2.11.24 with loguru, einx, frozendict).
CENSUS_DEPS = RESOURCES / "_scoreboard_deps"
KOHAKU_MOSS = (
    Path(__file__).resolve().parent.parent / ".stage0-sources/KohakuMossTrainer"
)

MODEL_COMPILE = {"fullgraph": False, "dynamic": True, "mode": "default"}
# musubi-tuner's compile_transformer and ai-toolkit's block compile defaults: each
# block in place, dynamic=None (auto).
BLOCK_COMPILE = {"fullgraph": False, "dynamic": None, "mode": "default"}
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


def _import_musubi():
    """Put musubi-tuner on the path without the media libraries the census never uses.

    cv2 re-enables the GIL on 3.14t, so musubi's own no-OpenCV stub stands in for it
    (as when opencv-python is not installed); PyAV is only used by its media helpers.
    """
    path = str(MUSUBI / "src")
    if path not in sys.path:
        sys.path.insert(0, path)
    import importlib

    sys.modules.setdefault(
        "cv2", importlib.import_module("musubi_tuner.utils._cv2_stub")
    )
    sys.modules.setdefault("av", types.ModuleType("av"))


def _census_settings(blocks, total, **settings):
    """Settings shared by the census workloads: real block configs, reduced depth."""
    return {
        **settings,
        "blocks": f"{blocks} of {total} (reduced depth: same block config, so the same regions)",
        "weights": "random init, bf16, under bf16 autocast",
        "gradient_checkpointing": False,
        "trainable": "all DiT parameters (a superset of the LoRA trainers' gradients)",
    }


WAN_A14B = {  # musubi-tuner wan/configs/wan_t2v_A14B.py (each A14B expert)
    "dim": 5120,
    "ffn_dim": 13824,
    "freq_dim": 256,
    "in_dim": 16,
    "out_dim": 16,
    "num_heads": 40,
    "num_layers": 40,
    "eps": 1e-6,
    "text_len": 512,
}


def build_wan22(device, options):
    _import_musubi()
    from musubi_tuner.wan.modules.model import WanModel

    blocks = options.get("blocks", 2)
    # 480x832, 17 frames: VAE stride (4, 8, 8) gives a 5x60x104 latent.
    frames, height, width = 5, 60, 104
    torch.manual_seed(0)
    with torch.device(device):
        model = WanModel(
            model_type="t2v",
            model_version="2.2",
            **{**WAN_A14B, "num_layers": blocks},
            attn_mode="torch",
        )
    model.to(torch.bfloat16).train()
    g = torch.Generator(device).manual_seed(1)
    batch = options.get("batch", 1)
    x = [
        torch.randn(16, frames, height, width, generator=g, device=device).bfloat16()
        for _ in range(batch)
    ]
    context = [
        torch.randn(512, 4096, generator=g, device=device).bfloat16()
        for _ in range(batch)
    ]
    t = torch.randint(0, 1000, (batch,), generator=g, device=device).float()
    seq_len = frames * (height // 2) * (width // 2)
    target = torch.randn(
        batch, 16, frames, height, width, generator=g, device=device
    ).bfloat16()

    def forward():
        return torch.stack(model(x, t=t, context=context, seq_len=seq_len), dim=0)

    parameters = [p for p in model.parameters() if p.requires_grad]
    return Instance(
        step=_model_step(model, forward, target, parameters),
        regions=list(model.blocks),
        parameters=parameters,
        settings=_census_settings(
            blocks,
            WAN_A14B["num_layers"],
            microbatch=batch,
            config="Wan 2.2 T2V A14B expert (dim 5120, 40 heads)",
            video="17 frames at 480x832",
            latent_shape=[batch, 16, frames, height, width],
            tokens=seq_len,
            text_tokens=512,
            attention="attn_mode=torch, i.e. SDPA",
            regions=["WanAttentionBlock"],
        ),
    )


def build_krea2(device, options):
    _import_musubi()
    import dataclasses

    from musubi_tuner.krea2.krea2_mmdit import SingleStreamDiT
    from musubi_tuner.krea2.krea2_utils import single_mmdit_large_wide as config

    blocks = options.get("blocks", 2)
    batch, resolution = options.get("batch", 1), options.get("resolution", 1024)
    h = w = resolution // 16  # VAE stride 8, patch 2
    text, valid = 256, 200  # padded text tokens; the tail is masked out
    torch.manual_seed(0)
    with torch.device(device):
        model = SingleStreamDiT(dataclasses.replace(config, layers=blocks))
    model.to(torch.bfloat16).train()
    g = torch.Generator(device).manual_seed(1)
    img = torch.randn(batch, h * w, 16 * 4, generator=g, device=device).bfloat16()
    context = torch.randn(
        batch, text, config.txtlayers, config.txtdim, generator=g, device=device
    ).bfloat16()
    # As krea2_train_network.call_dit: image ids (0, row, col), zero text ids.
    ids = torch.zeros(h, w, 3, device=device)
    ids[..., 1] = torch.arange(h, device=device)[:, None]
    ids[..., 2] = torch.arange(w, device=device)[None, :]
    pos = torch.cat(
        (
            ids.reshape(1, h * w, 3).expand(batch, -1, -1),
            torch.zeros(batch, text, 3, device=device),
        ),
        dim=1,
    )
    txtmask = torch.zeros(batch, text, dtype=torch.bool, device=device)
    txtmask[:, :valid] = True
    mask = torch.cat(
        (torch.ones(batch, h * w, dtype=torch.bool, device=device), txtmask), dim=1
    )
    t = torch.rand(batch, generator=g, device=device)
    target = torch.randn(img.shape, generator=g, device=device).bfloat16()

    def forward():
        return model(img=img, context=context, t=t, pos=pos, mask=mask)

    parameters = [p for p in model.parameters() if p.requires_grad]
    return Instance(
        step=_model_step(model, forward, target, parameters),
        regions=list(model.blocks),
        parameters=parameters,
        settings=_census_settings(
            blocks,
            config.layers,
            microbatch=batch,
            config="Krea2 single_mmdit_large_wide (6144 features, 48 heads, 12 KV heads)",
            image=f"{resolution}x{resolution}",
            image_tokens=h * w,
            text_tokens=f"{text} ({valid} valid), 12 Qwen3-VL layers of 2560",
            attention="attn_mode=torch, i.e. SDPA with a key-padding mask",
            regions=["SingleStreamBlock"],
        ),
    )


def build_minimax_h3(device, options):
    _import_musubi()
    import dataclasses

    from musubi_tuner.minimax_h3.media import audio_latent_frames
    from musubi_tuner.minimax_h3.model import MiniMaxH3Config, MiniMaxH3Model
    from musubi_tuner.minimax_h3.packing import build_h3_layout

    config = MiniMaxH3Config()  # the published R1 transformer config
    blocks = options.get("blocks", 2)
    # Text-to-video+audio, batch 1 (R1 requires it): 7 latent frames (5n+2, i.e. 22
    # pixel frames) of 30x52 latents (480x832) and their 40 Hz audio latents.
    frames, height, width, text = 7, 30, 52, 256
    audio = audio_latent_frames(17 * ((frames - 2) // 5) + 5)
    layout = build_h3_layout(
        task="t2va",
        text_length=text,
        target_video=(frames, height, width),
        target_audio_frames=audio,
    )
    torch.manual_seed(0)
    model = MiniMaxH3Model(
        dataclasses.replace(config, num_layers=blocks), attn_mode="torch", device=device
    )
    model.train()
    g = torch.Generator(device).manual_seed(1)
    video = torch.randn(
        1, 24, frames, height, width, generator=g, device=device
    ).bfloat16()
    sound = torch.randn(1, 32, 2, audio, generator=g, device=device).bfloat16()
    hidden = torch.randn(
        1, text, config.text_dim, generator=g, device=device
    ).bfloat16()
    tags = torch.ones(1, text, dtype=torch.int64)  # all text rows (no vision rows)
    target = torch.randn(video.numel() + sound.numel(), generator=g, device=device)

    def forward():
        out = model(
            video_latents=video,
            audio_latents=sound,
            text_hidden_states=hidden,
            text_token_tags=tags,
            layout=layout,
            model_t_video=0.7,
            model_t_audio=0.7,
        )
        return torch.cat((out.video.flatten(), out.audio.flatten()))

    parameters = [p for p in model.parameters() if p.requires_grad]
    return Instance(
        step=_model_step(model, forward, target, parameters),
        regions=list(model.blocks),
        parameters=parameters,
        settings=_census_settings(
            blocks,
            config.num_layers,
            microbatch=1,
            config="MiniMax-H3 R1 (hidden 5376, 56 heads; packed video+audio+text rows)",
            task="t2va",
            video="22 frames at 480x832 (latent 7x30x52)",
            audio_latent_frames=audio,
            packed_rows=layout.row_count,
            text_tokens=text,
            attention="attn_mode=torch, i.e. SDPA",
            regions=["DiTBlock"],
        ),
    )


def _import_ai_toolkit(model):
    """ai-toolkit's ``toolkit`` package and one model extension's ``src`` package."""
    import importlib

    for path in (
        AI_TOOLKIT,
        AI_TOOLKIT / "extensions_built_in/diffusion_models" / model,
    ):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return importlib.import_module("src.transformer")


def build_qwen_image21(device, options):
    T = _import_ai_toolkit("qwen_image_2")

    blocks = options.get("blocks", 2)
    batch, resolution = options.get("batch", 1), options.get("resolution", 1024)
    side = resolution // 16  # 2.1 consumes 64-channel latents unpatched
    tokens, text, valid = side * side, 256, 200
    torch.manual_seed(0)
    with torch.device(device):
        # Constructor defaults (no config.json locally): 32 layers, 32x128 heads.
        model = T.QwenImage21Transformer2DModel(num_layers=blocks)
    model.to(torch.bfloat16).train()
    g = torch.Generator(device).manual_seed(1)
    hidden = torch.randn(batch, tokens, 64, generator=g, device=device).bfloat16()
    prompt = torch.randn(batch, text, 4096, generator=g, device=device).bfloat16()
    prompt_mask = torch.zeros(batch, text, dtype=torch.bool, device=device)
    prompt_mask[:, :valid] = True
    # As ai-toolkit's run_transformer for text-to-image: no reference slots, then
    # one slot per 2x2 group of target tokens.
    img_mask = torch.cat(
        (
            torch.zeros(batch, text, dtype=torch.bool, device=device),
            torch.ones(batch, tokens // 4, dtype=torch.bool, device=device),
        ),
        dim=1,
    )
    t = torch.rand(batch, generator=g, device=device)
    target = torch.randn(hidden.shape, generator=g, device=device).bfloat16()

    def forward():
        return model(
            hidden_states=hidden,
            encoder_hidden_states=prompt,
            encoder_hidden_states_mask=prompt_mask,
            timestep=t,
            img_shapes=[[(1, side, side)]] * batch,
            img_mask=img_mask,
            return_dict=False,
        )[0][:, -tokens:]  # the joint sequence; the target image is its tail

    parameters = [p for p in model.parameters() if p.requires_grad]
    return Instance(
        step=_model_step(model, forward, target, parameters),
        regions=list(model.transformer_blocks),
        parameters=parameters,
        settings=_census_settings(
            blocks,
            32,
            microbatch=batch,
            config="Qwen-Image 2.1 transformer constructor defaults (4096 dim, 32 heads, block-causal attention)",
            image=f"{resolution}x{resolution}",
            image_tokens=tokens,
            text_tokens=f"{text} ({valid} valid)",
            attention="diffusers dispatch_attention_fn (default backend)",
            regions=["QwenImage21TransformerBlock"],
        ),
    )


def build_moss_image(device, options):
    import dataclasses

    T = _import_ai_toolkit("moss_image")

    blocks = options.get("blocks", 2)
    batch, resolution = options.get("batch", 1), options.get("resolution", 1024)
    side, text, valid = resolution // 16, 256, 200
    params = T.MossImageParams()
    torch.manual_seed(0)
    with torch.device(device):
        model = T.MossImageTransformer(dataclasses.replace(params, num_layers=blocks))
    model.to(torch.bfloat16).train()
    g = torch.Generator(device).manual_seed(1)
    latents = torch.randn(batch, 128, side, side, generator=g, device=device).bfloat16()
    prompt = torch.randn(batch, text, 4096, generator=g, device=device).bfloat16()
    mask = torch.zeros(batch, text, dtype=torch.bool, device=device)
    mask[:, :valid] = True
    t = torch.randint(0, 1000, (batch,), generator=g, device=device).float()
    target = torch.randn(latents.shape, generator=g, device=device).bfloat16()

    def forward():
        return model(latents, t, prompt, mask)

    parameters = [p for p in model.parameters() if p.requires_grad]
    return Instance(
        step=_model_step(model, forward, target, parameters),
        regions=list(model.layers),
        parameters=parameters,
        settings=_census_settings(
            blocks,
            params.num_layers,
            microbatch=batch,
            config="MOSS-Image (3072 dim, 24 heads, 128-channel latents)",
            image=f"{resolution}x{resolution}",
            image_tokens=side * side,
            text_tokens=f"{text} ({valid} valid)",
            regions=["_Block"],
        ),
    )


# CosyVoice's trainer does not compile; each component's tensor modules are one region.
COSYVOICE_COMPILE = {"fullgraph": False, "dynamic": None, "mode": "default"}


def _slaney_mel(*, sr, n_fft, n_mels=128, fmin=0.0, fmax=None, **_):
    """librosa.filters.mel's default (Slaney scale and norm) filterbank, [n_mels, n_fft/2+1]."""
    import numpy as np

    def hz_to_mel(f):
        f = np.asanyarray(f, dtype=np.float64)
        mel = f / (200.0 / 3)
        log = f >= 1000.0
        return np.where(
            log,
            15.0 + np.log(np.maximum(f, 1e-10) / 1000.0) / (np.log(6.4) / 27.0),
            mel,
        )

    def mel_to_hz(m):
        m = np.asanyarray(m, dtype=np.float64)
        f = m * (200.0 / 3)
        log = m >= 15.0
        return np.where(log, 1000.0 * np.exp((np.log(6.4) / 27.0) * (m - 15.0)), f)

    fmax = sr / 2.0 if fmax is None else fmax
    freqs = np.fft.rfftfreq(n=n_fft, d=1.0 / sr)
    mel_f = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))
    fdiff, ramps = np.diff(mel_f), np.subtract.outer(mel_f, freqs)
    lower, upper = -ramps[:-2] / fdiff[:-1, None], ramps[2:] / fdiff[1:, None]
    weights = np.maximum(0, np.minimum(lower, upper))
    weights *= (2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels]))[:, None]
    return weights.astype(np.float32)


@contextlib.contextmanager
def _stand_ins(**modules):
    """Temporarily register stand-in modules for one import, then remove them.

    Left registered, a stand-in would make other packages' availability probes
    (transformers, diffusers) believe the real package is installed.
    """
    import importlib.machinery

    added = {}
    for name, attributes in modules.items():
        name = name.replace("__", ".")
        if name in sys.modules:
            continue
        m = types.ModuleType(name)
        m.__spec__ = importlib.machinery.ModuleSpec(name, None)
        m.__dict__.update(attributes)
        added[name] = m
    for name, m in added.items():
        parent, _, child = name.rpartition(".")
        if parent in added:
            setattr(added[parent], child, m)
    sys.modules.update(added)
    try:
        yield
    finally:
        for name in added:
            sys.modules.pop(name, None)


class _NoTorchaudio(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("census torchaudio stand-in: import only")


def _import_cosyvoice():
    """CosyVoice and Matcha-TTS on the path, with stand-ins for what the venv lacks.

    ``cosyvoice.utils.onnx``: precomputed speech tokens (its ``online_feature=False``
    path), so no onnxruntime. torchaudio, only while importing the modules that
    name it: file loading, and a ``Spectrogram`` / ``MelSpectrogram`` the recipe's
    discriminator and DiT never build. librosa, only while importing
    Matcha's audio module: ``filters.mel`` (the mel loss's constant filterbank),
    reimplemented (Slaney, librosa's default).
    """
    import importlib

    for path in (CENSUS_DEPS, COSYVOICE, COSYVOICE / "third_party/Matcha-TTS"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    onnx = types.ModuleType("cosyvoice.utils.onnx")
    onnx.SpeechTokenExtractor, onnx.online_feature, onnx.onnx_path = None, False, None
    onnx.embedding_extractor = None
    sys.modules.setdefault("cosyvoice.utils.onnx", onnx)
    # transformers and diffusers load before any stand-in exists.
    importlib.import_module("cosyvoice.flow.flow")

    with _stand_ins(
        torchaudio={},
        torchaudio__transforms={
            "Spectrogram": _NoTorchaudio,
            "MelSpectrogram": _NoTorchaudio,
        },
    ):
        for name in (
            "cosyvoice.utils.file_utils",  # torchaudio.load: data loading only
            "cosyvoice.hifigan.discriminator",
            "cosyvoice.flow.DiT.dit",
        ):
            importlib.import_module(name)
    # matcha.utils as a bare package: its __init__ imports lightning and hydra, which
    # the audio module does not use.
    utils = types.ModuleType("matcha.utils")
    utils.__path__ = [str(COSYVOICE / "third_party/Matcha-TTS/matcha/utils")]
    sys.modules.setdefault("matcha.utils", utils)
    with _stand_ins(librosa={}, librosa__filters={"mel": _slaney_mel}):
        importlib.import_module("matcha.utils.audio")
    # Matcha's logger wraps methods in lightning's rank_zero_only: one process here.
    with _stand_ins(
        lightning={},
        lightning__pytorch={},
        lightning__pytorch__utilities={"rank_zero_only": lambda fn: fn},
    ):
        importlib.import_module("cosyvoice.flow.flow_matching")


QWEN25_05B = {  # CosyVoice3's Qwen2Encoder base (Qwen2.5-0.5B, hidden 896)
    "hidden_size": 896,
    "intermediate_size": 4864,
    "num_attention_heads": 14,
    "num_key_value_heads": 2,
    "num_hidden_layers": 24,
    "vocab_size": 151936,
    "rope_theta": 1000000.0,
    "rms_norm_eps": 1e-6,
    "max_position_embeddings": 32768,
    "tie_word_embeddings": True,
}


def build_cosyvoice3(device, options):
    """CosyVoice 3 (LibriTTS recipe): one training step of each component it trains
    separately, LLM, flow and HiFi-GAN (generator and discriminator turns), with the
    recipe's --use_amp (fp32 weights, bf16 autocast)."""
    import functools
    import random

    _import_cosyvoice()
    from cosyvoice.flow.DiT.dit import DiT
    from cosyvoice.flow.flow import CausalMaskedDiffWithDiT
    from cosyvoice.flow.flow_matching import CausalConditionalCFM
    from cosyvoice.hifigan.discriminator import (
        MultipleDiscriminator,
        MultiResSpecDiscriminator,
    )
    from cosyvoice.hifigan.f0_predictor import CausalConvRNNF0Predictor
    from cosyvoice.hifigan.generator import CausalHiFTGenerator
    from cosyvoice.hifigan.hifigan import HiFiGan
    from cosyvoice.llm.llm import CosyVoice3LM, Qwen2Encoder
    from cosyvoice.transformer.upsample_encoder import PreLookaheadLayer
    from matcha.hifigan.models import MultiPeriodDiscriminator
    from matcha.utils.audio import mel_spectrogram
    from omegaconf import DictConfig
    from transformers import Qwen2Config, Qwen2ForCausalLM

    blocks = options.get("blocks", 2)
    torch.manual_seed(0)
    with torch.device(device):
        qwen = Qwen2Encoder.__new__(Qwen2Encoder)  # from config, not from_pretrained
        torch.nn.Module.__init__(qwen)
        qwen.model = Qwen2ForCausalLM(
            Qwen2Config(**{**QWEN25_05B, "num_hidden_layers": blocks})
        )
        llm = CosyVoice3LM(896, 896, 6561, qwen, sampling=None, mix_ratio=[5, 15])
        cfm = DictConfig(
            {
                "sigma_min": 1e-06,
                "solver": "euler",
                "t_scheduler": "cosine",
                "training_cfg_rate": 0.2,
                "inference_cfg_rate": 0.7,
                "reg_loss_type": "l1",
            }
        )
        flow = CausalMaskedDiffWithDiT(
            input_size=80,
            output_size=80,
            spk_embed_dim=192,
            output_type="mel",
            vocab_size=6561,
            input_frame_rate=25,
            only_mask_loss=True,
            token_mel_ratio=2,
            pre_lookahead_len=3,
            pre_lookahead_layer=PreLookaheadLayer(80, 1024, 3),
            decoder=CausalConditionalCFM(
                240,
                cfm,
                n_spks=1,
                spk_emb_dim=80,
                estimator=DiT(
                    dim=1024,
                    depth=blocks,
                    heads=16,
                    dim_head=64,
                    ff_mult=2,
                    mel_dim=80,
                    mu_dim=80,
                    spk_dim=80,
                    out_channels=80,
                    static_chunk_size=50,
                    num_decoding_left_chunks=-1,
                ),
            ),
        )
        generator = CausalHiFTGenerator(
            in_channels=80,
            base_channels=512,
            nb_harmonics=8,
            sampling_rate=24000,
            nsf_alpha=0.1,
            nsf_sigma=0.003,
            nsf_voiced_threshold=10,
            upsample_rates=[8, 5, 3],
            upsample_kernel_sizes=[16, 11, 7],
            istft_params={"n_fft": 16, "hop_len": 4},
            resblock_kernel_sizes=[3, 7, 11],
            resblock_dilation_sizes=[[1, 3, 5]] * 3,
            source_resblock_kernel_sizes=[7, 7, 11],
            source_resblock_dilation_sizes=[[1, 3, 5]] * 3,
            lrelu_slope=0.1,
            audio_limit=0.99,
            conv_pre_look_right=4,
            f0_predictor=CausalConvRNNF0Predictor(1, 80, 512),
        )
        hifigan = HiFiGan(
            generator,
            MultipleDiscriminator(
                mpd=MultiPeriodDiscriminator(), mrd=MultiResSpecDiscriminator()
            ),
            [
                functools.partial(
                    mel_spectrogram,
                    n_fft=1920,
                    num_mels=80,
                    sampling_rate=24000,
                    hop_size=480,
                    win_size=1920,
                    fmin=0,
                    fmax=None,
                    center=False,
                )
            ],
        )
    for m in (llm, flow, hifigan):
        m.train()
    # Dynamo (any backend, eager included) turns the sine generators' numpy
    # upsample_scale (np.prod of the rates) into a 0-d array that F.interpolate
    # rejects after a graph break; the same value as a Python int.
    for m in hifigan.modules():
        if hasattr(m, "upsample_scale"):
            m.upsample_scale = int(m.upsample_scale)
    g = torch.Generator(device).manual_seed(1)

    def ints(high, *shape):
        return torch.randint(0, high, shape, generator=g, device=device)

    # Two ~6 s LibriTTS utterances: 25 Hz speech tokens, 50 Hz mel frames.
    speech_len = torch.tensor([150, 120], device=device)
    text_len, instruct_len = (
        torch.tensor([40, 30], device=device),
        torch.tensor([8, 6], device=device),
    )
    llm_batch = {
        "text_token": ints(151936, 2, 40),
        "text_token_len": text_len,
        "speech_token": ints(6561, 2, 150),
        "speech_token_len": speech_len,
        "instruct_token": ints(151936, 2, 8),
        "instruct_token_len": instruct_len,
    }
    flow_batch = {
        "speech_token": llm_batch["speech_token"],
        "speech_token_len": speech_len,
        "speech_feat": torch.randn(2, 300, 80, generator=g, device=device),
        "speech_feat_len": speech_len * 2,
        "embedding": torch.randn(2, 192, generator=g, device=device),
    }
    # The recipe's 24960-sample GAN crops (52 mel frames at hop 480).
    gan_batch = {
        "speech": (torch.rand(2, 24960, generator=g, device=device) - 0.5) * 1.8,
        "speech_feat": torch.randn(2, 52, 80, generator=g, device=device),
        "pitch_feat": torch.rand(2, 52, generator=g, device=device) * 300,
    }
    parameters = [
        p for m in (llm, flow, hifigan) for p in m.parameters() if p.requires_grad
    ]

    def step():
        random.seed(0)  # streaming / bistream / prompt choices in the forwards
        torch.manual_seed(0)
        for p in parameters:
            p.grad = None
        out = {}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            losses = {
                "llm": llm(llm_batch, device)["loss"],
                "flow": flow(flow_batch, device)["loss"],
                "generator": hifigan({**gan_batch, "turn": "generator"}, device)[
                    "loss"
                ],
                "discriminator": hifigan(
                    {**gan_batch, "turn": "discriminator"}, device
                )["loss"],
            }
        for name, loss in losses.items():
            loss.backward()
            out[name] = loss.detach().reshape(())
        return out

    return Instance(
        step=step,
        regions=[
            llm.llm,
            llm.criterion_ce,
            flow.pre_lookahead_layer,
            flow.decoder.estimator,
            hifigan.generator,
            hifigan.discriminator,
        ],
        parameters=parameters,
        settings={
            "recipe": "examples/libritts/cosyvoice3 (conf/cosyvoice3.yaml), --use_amp",
            "weights": "random init, fp32, under bf16 autocast (the recipe's --use_amp)",
            "blocks": f"Qwen2 {blocks} of 24 layers, DiT {blocks} of 22 (reduced depth)",
            "llm": "CosyVoice3LM over Qwen2.5-0.5B config; speech-token CE (6761 classes)",
            "flow": "CausalMaskedDiffWithDiT + CausalConditionalCFM, DiT 1024 dim",
            "hifigan": "CausalHiFTGenerator + MPD/MRSD discriminators, generator and discriminator turns",
            "batch": "2 utterances: text 40/30, instruct 8/6, speech tokens 150/120 (mel 300/240); GAN crops 24960 samples",
            "regions": [
                "Qwen2Encoder",
                "LabelSmoothingLoss",
                "PreLookaheadLayer",
                "DiT",
                "CausalHiFTGenerator",
                "MultipleDiscriminator",
            ],
            "stand_ins": "cosyvoice.utils.onnx (precomputed tokens), torchaudio (import only), librosa.filters.mel (Slaney)",
            "deviation": "sine generators' numpy upsample_scale as int (Dynamo fails on it with every backend)",
            "gradient_checkpointing": False,
        },
    )


# KohakuMoss compiles each block op's eager reference with dynamic shapes
# (kohakumoss_kernels.backends.compile.ops.CompiledOp).
KOHAKU_COMPILE = {"fullgraph": False, "dynamic": True, "mode": "default"}


def build_kohaku_moss_ops(device, options):
    """KohakuMoss's MOSS-Image block ops, as its `compile` backend compiles them.

    Each op of its eager reference table is a compiled function, run on a packed batch
    (flat token axis, per-sample modulation by segment id). DiT ops run forward and
    backward; text-encoder ops forward only (the encoder is frozen).
    """
    path = str(KOHAKU_MOSS / "src")
    if path not in sys.path:
        sys.path.insert(0, path)
    from kohakumoss.types.seqinfo import build_seqinfo
    from kohakumoss_kernels.backends.compile.ops import EAGER

    dtype = torch.bfloat16
    seq = build_seqinfo(torch.tensor([[64, 64]] * 2), torch.tensor([300, 120]))
    n, d, heads, samples = seq.n_joint, 3072, 24, 2
    g = torch.Generator(device).manual_seed(2)

    def rand(*shape, scale=1.0, dt=dtype, grad=True):
        x = (torch.randn(shape, generator=g, device=device) * scale).to(dt)
        return x.requires_grad_(grad)

    seg = seq.seg_joint.to(device)
    pos = seq.pos_joint.to(device)
    bands = (1 / (10000.0 ** (torch.arange(16) / 16))).to(device)
    cu = torch.tensor([0, 300, 420], dtype=torch.int32, device=device)
    seg_img = seq.seg_img.to(device)
    calls = {
        "rms_norm": (rand(n, d), rand(d, dt=torch.float32), 1e-6),
        "rms_norm_mod": (
            rand(n, d),
            rand(d, dt=torch.float32),
            rand(samples, d, scale=0.1, dt=torch.float32),
            rand(samples, d, scale=0.1, dt=torch.float32),
            seg,
            1e-6,
        ),
        "gated_residual": (
            rand(n, d),
            rand(n, d),
            rand(samples, d, dt=torch.float32),
            seg,
            1.0,
        ),
        "qk_norm_rope": (
            rand(n, heads, 128),
            rand(n, heads, 128),
            rand(128, dt=torch.float32),
            rand(128, dt=torch.float32),
            pos,
            bands,
            1e-6,
        ),
        "situ_glu": (rand(n, 16384, scale=4.0),),
        "token_mse": (
            rand(seq.n_img, 128),
            rand(seq.n_img, 128, grad=False),
            seg_img,
            torch.ones(samples, device=device),
            None,
        ),
        "linear_situ": (rand(n, d), rand(16384, d, scale=1 / 55), None),
        "linear_gated_residual": (
            rand(n, d),
            rand(d, d, scale=1 / 55),
            None,
            rand(n, d),
            rand(samples, d, dt=torch.float32),
            seg,
            1.0,
        ),
        "linear_qk_rope": (
            rand(n, d),
            rand(d, d, scale=1 / 55),
            rand(d, d, scale=1 / 55),
            rand(128, dt=torch.float32),
            rand(128, dt=torch.float32),
            pos,
            bands,
            1e-6,
            heads,
        ),
        # Qwen3.5 gated-delta text encoder: q/k/v conv channels, 32 value heads of 128.
        "causal_conv_varlen": (
            rand(420, 8192, grad=False),
            rand(8192, 4, grad=False),
            cu,
        ),
        "rms_norm_gated": (
            rand(420 * 32, 128, grad=False),
            rand(420 * 32, 128, grad=False),
            rand(128, dt=torch.float32, grad=False),
            1e-6,
        ),
    }
    instance = Instance(
        step=None,
        functions={name: EAGER[name] for name in calls},
        settings={
            "ops": list(calls),
            "excluded": "group_hadamard (rotated-quantization frozen bases only)",
            "dtype": "bf16 activations, fp32 per-sample modulation and norm weights",
            "batch": "packed: 2 samples of 64x64 latents + 300/120 text tokens "
            f"({n} joint tokens), MOSS-Image dim {d}, {heads} heads",
            "backward": "DiT ops forward+backward; text-encoder ops forward only",
            "source": "KohakuMossTrainer kohakumoss_kernels.backends.eager (EAGER table)",
        },
    )

    def step():
        out = {}
        for name, args in calls.items():
            for a in args:
                if isinstance(a, torch.Tensor) and a.requires_grad:
                    a.grad = None
            result = instance.functions[name](*args)
            results = result if isinstance(result, tuple) else (result,)
            if any(r.requires_grad for r in results):
                torch.autograd.backward(results, [torch.ones_like(r) for r in results])
            for i, r in enumerate(results):
                out[f"{name}.{i}"] = r.detach()
            for i, a in enumerate(args):
                if isinstance(a, torch.Tensor) and a.grad is not None:
                    out[f"{name}.grad{i}"] = a.grad.detach().clone()
        return out

    instance.step = step
    return instance


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
        for i, (gr, b) in enumerate(zip(grads, biases, strict=True)):
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
        Workload(
            "wan22",
            MUSUBI,
            BLOCK_COMPILE,
            build_wan22,
            "Wan 2.2 T2V A14B DiT forward+backward (musubi-tuner), per-block compile",
        ),
        Workload(
            "krea2",
            MUSUBI,
            BLOCK_COMPILE,
            build_krea2,
            "Krea2 single-stream MMDiT forward+backward (musubi-tuner), per-block compile",
        ),
        Workload(
            "minimax_h3",
            MUSUBI,
            BLOCK_COMPILE,
            build_minimax_h3,
            "MiniMax-H3 packed video+audio DiT forward+backward (musubi-tuner), per-block compile",
        ),
        Workload(
            "qwen_image21",
            AI_TOOLKIT,
            BLOCK_COMPILE,
            build_qwen_image21,
            "Qwen-Image 2.1 DiT forward+backward (ai-toolkit), per-block compile",
        ),
        Workload(
            "moss_image",
            AI_TOOLKIT,
            BLOCK_COMPILE,
            build_moss_image,
            "MOSS-Image DiT forward+backward (ai-toolkit), per-block compile",
        ),
        Workload(
            "kohaku_moss_ops",
            KOHAKU_MOSS,
            KOHAKU_COMPILE,
            build_kohaku_moss_ops,
            "KohakuMoss MOSS-Image block ops (eager references), compiled per op as its compile backend does",
        ),
        Workload(
            "cosyvoice3",
            COSYVOICE,
            COSYVOICE_COMPILE,
            build_cosyvoice3,
            "CosyVoice 3 TTS training step (LLM, flow, HiFi-GAN) from its LibriTTS recipe",
        ),
    ]
}
