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
    gm, inputs, *, tuning="first_use", rng_policy="generated", split_policy="guarded"
):
    """``torch.compile`` backend; same policies and options as Stage 5's."""
    install()
    from cubecl_torch.inductor.physical import compile_backend as compile_physical

    return compile_physical(
        gm, inputs, tuning=tuning, rng_policy=rng_policy, split_policy=split_policy
    )
