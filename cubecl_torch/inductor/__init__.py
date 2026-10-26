"""torch.compile backend: Inductor scheduling and codegen through the descriptor compiler.

Call ``install()`` before anything imports ``torch._inductor`` (it copies and
edits Inductor into the cache root, pinned to one PyTorch wheel), then pass
``compile_backend`` to ``torch.compile``. Importing this package imports
neither Inductor nor the overlay.
"""

from cubecl_torch.compiler.errors import UnsupportedCubeCL

POLICIES = (
    "first_use",
    "baseline",
    "combined_diagnostic",
    "compensated_diagnostic",
    "load_batch_diagnostic",
    "adjacent_merge_diagnostic",
    "reference512_diagnostic",
)

__all__ = ["POLICIES", "UnsupportedCubeCL", "compile_backend", "install"]


def install(*, precision_casts=True):
    """Install the pinned Inductor overlay; idempotent."""
    from cubecl_torch.inductor import overlay

    overlay.install(precision_casts=precision_casts)


def compile_backend(
    gm,
    inputs,
    *,
    tuning="first_use",
    rng_policy="generated",
    split_policy="guarded",
    rejections="first",
):
    """``torch.compile`` backend.

    ``tuning`` is one of ``POLICIES``. ``first_use`` times the legal candidate
    plans on first launch per shape bucket and caches the choice; effectful
    kernels and kernels with one legal plan use the baseline plan.
    ``baseline`` always uses the baseline plan, without timing. The
    ``*_diagnostic`` policies force one alternative plan on reductions longer
    than ``legality.LONG_REDUCTION_THRESHOLD`` (an error where it is not legal)
    and the baseline plan elsewhere. ``split_policy="guarded"``
    (``choices.CubeCLChoices``) records shape guards on Inductor's
    split/no-split reduction regime and plans the split from a representative
    size per regime; ``"inherited"`` takes Inductor's decision unguarded.

    ``rejections="all"`` (inventory runs) scans past the first unsupported
    feature and rejects once; the ``UnsupportedCubeCL`` carries ``reasons``,
    every distinct reason with its count (see ``inductor.rejections``).
    """
    install()
    from cubecl_torch.inductor.backend import compile_backend as compile_cubecl

    return compile_cubecl(
        gm,
        inputs,
        tuning=tuning,
        rng_policy=rng_policy,
        split_policy=split_policy,
        rejections=rejections,
    )
