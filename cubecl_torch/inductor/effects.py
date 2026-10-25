"""Stage4 default seeded uniform RNG; state remains in normal Inductor externs."""

import contextlib
import threading

import sympy
import torch
from torch._inductor import config
from torch._inductor.codegen.common import device_codegens, register_backend_for_device
from torch._inductor.codegen.simd import SIMDScheduling
from torch._inductor.scheduler import SchedulerNode
from torch._inductor.virtualized import V

from cubecl_torch.inductor import diagnostics
from cubecl_torch.inductor import rejections as _rejections
from cubecl_torch.inductor.capabilities import UnsupportedCubeCL
from cubecl_torch.inductor.diagnostics import TraceLog
from cubecl_torch.inductor.reduction_checks import (
    fusion_allowed,
)
from cubecl_torch.inductor.reduction_checks import (
    node_reasons as previous_reasons,
)
from cubecl_torch.inductor.reductions import (
    CubeCLKernel as PreviousKernel,
)
from cubecl_torch.inductor.reductions import (
    CubeCLOps as PreviousOps,
)
from cubecl_torch.inductor.reductions import (
    CubeCLScheduling as PreviousScheduling,
)
from cubecl_torch.inductor.semantic_checks import (
    FLOATS,
    check_inputs,
)
from cubecl_torch.inductor.semantic_checks import (
    node_reasons as pointwise_reasons,
)
from cubecl_torch.inductor.semantics import DTYPES

TRACE = TraceLog()


class CubeCLOps(PreviousOps):
    @staticmethod
    def rand(seed, counter):
        if seed.dtype != torch.int64 or counter.dtype != torch.int32:
            raise UnsupportedCubeCL("unsupported RNG dtype/algorithm contract")
        return V.kernel.emit(
            "rand",
            seed,
            counter,
            dtype=torch.float32,
            semantic=torch.float32,
            algorithm="philox4x32_10_top24",
        )

    @staticmethod
    def indirect_indexing(value, size, check=True):
        return V.kernel.indirect_indexing(value, size, check)

    @staticmethod
    def load_seed(name, offset):
        return V.kernel.load(name, offset)


class CubeCLKernel(PreviousKernel):
    overrides = CubeCLOps
    trace = TRACE
    supported_input_counts = (0, 1, 2, 3, 4, 5)
    dtype_names = DTYPES

    def __init__(self, *args, **kwargs):
        self.indirect_values = {}
        super().__init__(*args, **kwargs)

    def indirect_indexing(self, value, size, check=True):
        if value.dtype != torch.int64:
            raise UnsupportedCubeCL("unsupported indirect dtype/unchecked bounds")
        value = self.emit(
            "indirect",
            value,
            dtype=torch.int64,
            semantic=torch.int64,
            size=self.check_index(size),
            bounds="require_in_range",
        )
        symbol = sympy.Symbol(value.ssa, integer=True)
        self.indirect_values[symbol] = value.ssa
        return symbol

    def indirect_load(self, name, index):
        return self.load(name, index)

    def check_bounds(self, expr, size, lower, upper):
        # The explicit indirect op checks both bounds before the address is used.
        return None

    def check_index(self, index):
        if not self.indirect_values:
            return super().check_index(index)
        from cubecl_torch.compiler.indexing import encode

        expanded = self.simplify_indexing(index)
        substitutions = {s: e.expr for s, e in self.range_tree_nodes.items()}
        for _ in range(len(substitutions) + 1):
            new = expanded.xreplace(substitutions)
            if new == expanded:
                break
            expanded = new
        names = list(self.indirect_values)
        tree = encode(
            sympy.simplify(expanded),
            lambda s: -1 - names.index(s) if s in names else self.symbol_slot(s),
            self.range_trees[0].index_sym(),
        )

        def rewrite(v):
            if v.get("op") == "scalar" and v["slot"] < 0:
                return {
                    "op": "value",
                    "id": self.indirect_values[names[-1 - v["slot"]]],
                }
            return {
                k: [rewrite(x) for x in a] if k == "args" else a for k, a in v.items()
            }

        return rewrite(tree)

    def operation(self, name, *args):
        typed = next((x for x in args if hasattr(x, "ssa")), None)
        if typed is None:
            raise UnsupportedCubeCL("operation requires a typed operand")
        args = tuple(
            x if hasattr(x, "ssa") else self.overrides.constant(x, typed.semantic_dtype)
            for x in args
        )
        return super().operation(name, *args)

    def store(self, name, index, value, mode=None):
        if mode is None:
            return super().store(name, index, value, mode)
        if (
            mode != "atomic_add"
            or self.reduction_kernel
            or self.check_buffer(name) != torch.float32
        ):
            raise UnsupportedCubeCL(
                "unsupported atomic operation/dtype/physical topology"
            )
        self.args.output(name)
        value = self.convert(value, torch.float32, semantic=torch.float32)
        self.operations.append(
            {
                "op": "atomic_add",
                "args": [value.ssa],
                "dtype": "float32",
                "semantic_dtype": "float32",
                "buffer_name": name,
                "index": self.check_index(index),
                "effect": "atomic_rmw",
            }
        )

    def check_buffer(self, name):
        dtype = V.graph.get_dtype(name)
        if dtype == torch.int64:
            return dtype
        return super().check_buffer(name)

    def codegen_kernel(self):
        # KernelArgs canonicalizes mutation versions to physical storage names.
        # Match its existing mapping in serialization; CSE retains logical names.
        def canonicalize(ops):
            for op in ops:
                if "buffer_name" in op:
                    name = op["buffer_name"]
                    op["buffer_name"] = V.graph.scheduler.mutation_real_name.get(
                        name, name
                    )
                if "body" in op:
                    canonicalize(op["body"])

        canonicalize(self.operations)
        if diagnostics.tracing():
            TRACE.append(
                {
                    "event": "argument_bookkeeping",
                    "inputs": {
                        k: v if isinstance(v, str) else type(v).__name__
                        for k, v in self.args.input_buffers.items()
                    },
                    "outputs": {
                        k: v if isinstance(v, str) else type(v).__name__
                        for k, v in self.args.output_buffers.items()
                    },
                    "removed": list(self.removed_buffers),
                    "operations": self.operations.copy(),
                }
            )
        return super().codegen_kernel()

    def finalize_descriptor(self):
        super().finalize_descriptor()
        if not self.reduction_kernel:
            self.descriptor["schema"] = 4
            self.descriptor["effects"] = {
                "rng": "explicit_seed_counter"
                if any(op["op"] == "rand" for op in self.operations)
                else None,
                "mutation": any(op["op"] == "atomic_add" for op in self.operations),
            }


def rng_type(op, types):
    if op.target == "load_seed":
        if V.graph.get_dtype(op.args[1]) != torch.int64:
            raise UnsupportedCubeCL("unsupported RNG seed storage")
        return torch.int64
    if op.target == "rand":
        if types.get(op.args[1]) != torch.int64 or types.get(op.args[2]) != torch.int32:
            raise UnsupportedCubeCL("unsupported RNG operands")
        return torch.float32


def node_reason(node):
    return next(node_reasons(node), None)


def node_reasons(node):
    if not isinstance(node, SchedulerNode):
        return
    ops = {x.target for x in node._body.root_block.graph.nodes if x.op == "call_method"}
    if ops & {"load_seed", "rand", "randn", "rand_eager", "randint64"}:
        if node.is_reduction() or node._body.subblocks:
            yield "RNG in reduction/lazy subblock unsupported"
            return
        # This lowering explicitly asks for i32 counters; it does not prove a
        # global fits-in-i32 property for us. Add a normal ShapeEnv guard here.
        extent = sympy.prod(node.get_ranges()[0])
        if not V.graph.sizevars.guard_or_false(sympy.Le(extent, (1 << 31) - 1)):
            yield "unsupported RNG physical extent: signed-i32 logical counter domain"
        try:
            yield from pointwise_reasons(
                node,
                extra_ops=("rand", "load_seed"),
                allow_empty_input=True,
                float_index=True,
                storage_dtypes=(*FLOATS, torch.bool, torch.int64),
                operation_type=rng_type,
            )
        except UnsupportedCubeCL as e:
            yield str(e)
        return
    if node._body.indirect_vars:
        for expr in node._body.indexing_exprs.values():
            if any(
                set(node._body.indirect_vars) & term.free_symbols
                for term in expr.atoms(sympy.Function)
            ):
                yield "unsupported indirect quotient/modulo/function addressing"
                break
        if node.is_reduction() or node._body.subblocks:
            # The pointwise checks below would only restate this.
            yield "indirect reduction/lazy subblock scheduling unsupported"
            return
        for d in node.read_writes.writes:
            if hasattr(d, "name") and V.graph.get_dtype(d.name) not in FLOATS:
                yield "indirect writes require floating storage"
                break
        atomic = any(
            op.target == "store" and op.args[-1] == "atomic_add"
            for op in node._body.root_block.graph.nodes
            if op.op == "call_method"
        )
        if atomic:
            writes = list(node.read_writes.writes)
            if len(writes) != 1 or not hasattr(writes[0], "name"):
                yield "unsupported atomic destination dependency"
            elif (
                torch.are_deterministic_algorithms_enabled()
                or V.graph.get_dtype(writes[0].name) != torch.float32
            ):
                yield "unsupported atomic dtype or deterministic-mode incompatibility"
        yield from pointwise_reasons(
            node,
            operation_type=rng_type,
            allow_indirect=True,
            allow_atomic=True,
            storage_dtypes=(*FLOATS, torch.bool, torch.int64),
        )
        return
    if node.is_reduction():
        yield from previous_reasons(node)
        return
    yield from pointwise_reasons(
        node,
        allow_reductions=True,
        float_index=True,
        extra_ops=("reduction", "store_reduction", "rsqrt"),
        allow_empty_input=True,
        storage_dtypes=(*FLOATS, torch.bool, torch.int64),
        max_inputs=5,
    )


def pre_fusion(nodes):
    if diagnostics.tracing():
        TRACE.append(
            {
                "event": "capability_graph",
                "is_backward": bool(getattr(V.graph, "is_backward", False)),
                "nodes": [n.get_name() for n in nodes],
            }
        )
    rejected = False
    for n in nodes:
        reasons = _rejections.scan(node_reasons(n))
        reason = reasons[0] if reasons else None
        if diagnostics.tracing():
            TRACE.append(
                {
                    "event": "capability",
                    "boundary": "pre_fusion",
                    "node": n.get_name(),
                    "scheduler_type": type(n).__name__,
                    "ir_type": type(n.node).__name__,
                    "is_reduction": isinstance(n, SchedulerNode) and n.is_reduction(),
                    "semantic_ops": [
                        str(op.target)
                        for op in n._body.root_block.graph.nodes
                        if op.op == "call_method"
                    ]
                    if isinstance(n, SchedulerNode)
                    else [],
                    "decision": "reject" if reason else "accept",
                    "reason": reason,
                    "operation_graph": str(n._body.root_block.graph)
                    if isinstance(n, SchedulerNode)
                    else None,
                    "dependencies": str(n.read_writes),
                    "execution_path": "cubecl_generated"
                    if isinstance(n, SchedulerNode)
                    else "extern",
                    "kernel": getattr(n.node, "python_kernel_name", None),
                    "target": str(getattr(n.node, "op_overload", None)),
                }
            )
        if reasons:
            rejected = True
            _rejections.reject(
                *("CubeCL Stage4 unsupported before fusion: " + r for r in reasons)
            )
    if rejected:
        # Record mode: every node was scanned; rejected nodes are not scheduled.
        raise _rejections.Stop("graph rejected before fusion")
    return nodes


class CubeCLScheduling(PreviousScheduling):
    kernel_type = CubeCLKernel

    def can_fuse(self, a, b):
        # Keep RNG generation separate until effect-aware duplicate replay is proved.
        random = any(
            x.target in ("rand", "load_seed")
            for n in [*a.get_nodes(), *b.get_nodes()]
            if isinstance(n, SchedulerNode)
            for x in n._body.root_block.graph.nodes
            if x.op == "call_method"
        )
        if random or any(
            isinstance(n, SchedulerNode) and n._body.indirect_vars
            for n in [*a.get_nodes(), *b.get_nodes()]
        ):
            TRACE.append(
                {
                    "event": "capability",
                    "boundary": "fusion",
                    "decision": "reject",
                    "nodes": [a.get_name(), b.get_name()],
                    "reason": "RNG/indirect fusion requires separate effect-safe proof",
                }
            )
            return False
        return fusion_allowed(a, b, TRACE) and SIMDScheduling.can_fuse(self, a, b)

    can_fuse_vertical = can_fuse
    can_fuse_horizontal = can_fuse


_registered = False
_register_lock = threading.Lock()
_schedulers = {}


def compile_backend(gm, inputs, *, rejections="first", **options):
    """Compile ``gm``; ``rejections="all"`` rejects once with every reason found."""
    if rejections == "first":
        return _compile_backend(gm, inputs, **options)
    if rejections != "all":
        raise ValueError("rejections: first or all")
    compiled = None
    # Stop: a boundary the scan cannot pass; ``reasons`` say why.
    with _rejections.collecting() as reasons, contextlib.suppress(_rejections.Stop):
        compiled = _compile_backend(gm, inputs, **options)
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


def _compile_backend(
    gm,
    inputs,
    *,
    rng_policy="generated",
    scheduling=None,
    backend_name="cubecl_torch",
    choices_class=None,
):
    if rng_policy not in ("generated", "aten_extern"):
        raise UnsupportedCubeCL("unsupported explicit RNG execution policy")
    global _registered
    if scheduling is not None:
        _schedulers[backend_name] = scheduling
    elif backend_name not in _schedulers:
        raise ValueError("unregistered experimental scheduling implementation")
    if diagnostics.tracing():
        TRACE.append(
            {
                "event": "input_contract",
                "inputs": [
                    {
                        "type": type(x).__name__,
                        "device": str(getattr(x, "device", None)),
                        "dtype": str(getattr(x, "dtype", None)),
                        "shape": str(getattr(x, "shape", None)),
                    }
                    for x in inputs
                ],
                "graph": str(gm.graph),
            }
        )
    check_inputs(inputs, TRACE, storage_dtypes=(*FLOATS, torch.bool, torch.int64))
    for n in gm.graph.nodes:
        schema = getattr(n.target, "_schema", None)
        if (
            (n.op == "call_method" and str(n.target).endswith("_"))
            or (schema and schema.is_mutable)
            or n.kwargs.get("out") is not None
            or n.kwargs.get("inplace") is True
        ):
            TRACE.append(
                {
                    "event": "capability",
                    "boundary": "backend_selection",
                    "decision": "reject",
                    "reason": "unsupported user-visible mutation/alias pattern",
                }
            )
            _rejections.reject(
                "unsupported user-visible mutation/alias pattern; functional atomic destinations only"
            )
            break
    if config.align_random_eager or config.numerics == "strict":
        _rejections.reject("unsupported RNG alignment/exact ordering contract")
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
            _registered = True
    if diagnostics.tracing():
        TRACE.append({"event": "graph", "graph": str(gm.graph)})
    previous_pass = config._pre_fusion_custom_pass
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
            "_pre_fusion_custom_pass": lambda nodes: pre_fusion(
                previous_pass(nodes) if previous_pass else nodes
            ),
            "triton.cudagraphs": False,
            "inplace_buffers": False,
            "benchmark_kernel": False,
            "combo_kernels": False,
            "triton.coalesce_tiling_analysis": False,
            "joint_graph_constant_folding": False,
        },
    )
