"""``compile_backend``: register the CubeCL scheduling for CUDA and run Inductor."""

import contextlib
import contextvars
import threading
from functools import partial

from torch._inductor import config
from torch._inductor.codegen.common import device_codegens, register_backend_for_device
from torch._inductor.virtualized import V

from cubecl_torch.compiler.errors import UnsupportedCubeCL
from cubecl_torch.inductor import POLICIES, admission, diagnostics
from cubecl_torch.inductor import rejections as _rejections
from cubecl_torch.inductor.codegen import CubeCLScheduling

TRACE = admission.TRACE
_registered = False
_register_lock = threading.Lock()
# Inductor's cuda_backend name -> scheduling factory, one per tuning policy.
_schedulers = {}
# Set while Inductor's two-step variance rule runs (_install_two_step_variance).
_two_step_rule = contextvars.ContextVar("cubecl_two_step_rule", default=False)


def compile_backend(
    gm,
    inputs,
    *,
    tuning="first_use",
    rng_policy="generated",
    split_policy="guarded",
    rejections="first",
):
    """Compile ``gm``; ``rejections="all"`` rejects once with every reason found."""
    if tuning not in POLICIES:
        raise ValueError(
            f"unsupported tuning policy {tuning!r}; expected one of {', '.join(POLICIES)}"
        )
    if split_policy not in ("guarded", "inherited"):
        raise ValueError(
            f"unsupported split policy {split_policy!r}; expected guarded or inherited"
        )
    options = {"tuning": tuning, "rng_policy": rng_policy, "split_policy": split_policy}
    if rejections == "first":
        return _compile(gm, inputs, **options)
    if rejections != "all":
        raise ValueError("rejections: first or all")
    compiled = None
    # Stop: a boundary the scan cannot pass; ``reasons`` say why.
    with _rejections.collecting() as reasons, contextlib.suppress(_rejections.Stop):
        compiled = _compile(gm, inputs, **options)
    if reasons:
        raise _rejections.rejected(reasons)
    return compiled


def _record_graph(inner):
    """Record mode: a rejected graph yields a stub, so later graphs are scanned too."""
    from torch._inductor.output_code import OutputCode

    class Unlowered(OutputCode):
        def __call__(self, inputs):
            raise RuntimeError("record mode: this graph was rejected, not compiled")

        def post_compile(self, example_inputs, constants, graph_kwargs):
            pass

        def prepare_for_serialization(self):
            pass

        def set_triton_bundle(self, triton_bundle):
            pass

    def compile_graph(*args, **kwargs):
        try:
            return inner(*args, **kwargs)
        except Exception as e:
            cause = getattr(e, "inner_exception", e)
            if not isinstance(cause, UnsupportedCubeCL):
                raise
            if not isinstance(cause, _rejections.Stop):
                # Rejected during scheduling or code generation.
                _rejections.reject(str(cause))
            return Unlowered()

    return compile_graph


def _install_two_step_variance():
    """Lower variances as Inductor does for Triton (wide-reduction-workgroups 8).

    ``lowering.use_two_step_variance`` turns an unsplit 16-bit variance of
    1024 to 32768 elements into two sums only under ``is_triton()``, which our
    ``cuda_backend`` name makes False. While the rule runs, ``is_triton``
    answers True for CUDA nodes under a CubeCL backend; every other caller and
    the global config are unchanged, and the rule's split check runs under the
    real configuration that later splits (or not) the two sums. A call nested
    in the rule sees True too: today only ``_make_reduction_inner``'s argreduce
    check, whose answer the rule does not use.
    """
    from torch._inductor import lowering
    from torch._inductor.ir import get_device_type

    rule, is_triton = lowering.use_two_step_variance, lowering.is_triton

    def use_two_step_variance(*args, **kwargs):
        token = _two_step_rule.set(True)
        try:
            return rule(*args, **kwargs)
        finally:
            _two_step_rule.reset(token)

    def is_triton_or_cubecl(x):
        if (
            _two_step_rule.get()
            and config.cuda_backend in _schedulers
            and get_device_type(x) == "cuda"
        ):
            return True
        return is_triton(x)

    lowering.use_two_step_variance = use_two_step_variance
    lowering.is_triton = is_triton_or_cubecl


def _compile(gm, inputs, *, tuning, rng_policy, split_policy):
    if rng_policy not in ("generated", "aten_extern"):
        raise UnsupportedCubeCL("unsupported explicit RNG execution policy")
    from cubecl_torch.inductor.choices import CubeCLChoices
    from cubecl_torch.inductor.index_masks import widen_mask_indices

    global _registered
    backend_name = "cubecl_torch_" + tuning
    _schedulers[backend_name] = partial(CubeCLScheduling, tuning=tuning)
    choices_class = CubeCLChoices if split_policy == "guarded" else None
    admission.check_graph(gm, inputs)
    from torch._inductor.codegen.common import init_backend_registration
    from torch._inductor.compile_fx import compile_fx, compile_fx_inner

    init_backend_registration()
    with _register_lock:
        if not _registered:
            previous = device_codegens["cuda"]
            register_backend_for_device(
                "cuda",
                lambda scheduler: (
                    _schedulers[config.cuda_backend](scheduler)
                    if config.cuda_backend in _schedulers
                    else previous.scheduling(scheduler)
                ),
                previous.wrapper_codegen,
                previous.cpp_wrapper_codegen,
                previous.fx_wrapper_codegen,
            )
            _install_two_step_variance()
            _registered = True
    if diagnostics.tracing():
        TRACE.append({"event": "graph", "graph": str(gm.graph)})
    previous_pass = config._pre_fusion_custom_pass

    def pre_fusion(nodes):
        nodes = previous_pass(nodes) if previous_pass else nodes
        return admission.pre_fusion(widen_mask_indices(nodes))

    extra_patches = (
        {} if choices_class is None else {"inductor_choices_class": choices_class}
    )
    compile_options = {}
    if choices_class is not None:
        # Initialize the caller's lazy handler before compile_fx patches the
        # factory; a scoped bind must not leave that temporary factory cached.
        _ = V.choices

        def inner_with_choices(*args, **kwargs):
            with V.set_choices_handler(choices_class()):
                return compile_fx_inner(*args, **kwargs)

        compile_options["inner_compile"] = inner_with_choices
    if _rejections.recording():
        compile_options["inner_compile"] = _record_graph(
            compile_options.get("inner_compile", compile_fx_inner)
        )
    return compile_fx(
        gm,
        inputs,
        **compile_options,
        config_patches={
            **extra_patches,
            "cuda_backend": backend_name,
            "compile_threads": 1,
            "fx_graph_cache": False,
            "fallback_random": rng_policy == "aten_extern",
            "_pre_fusion_custom_pass": pre_fusion,
            "triton.cudagraphs": False,
            "inplace_buffers": False,
            "benchmark_kernel": False,
            "combo_kernels": False,
            "triton.coalesce_tiling_analysis": False,
            "joint_graph_constant_folding": False,
        },
    )
