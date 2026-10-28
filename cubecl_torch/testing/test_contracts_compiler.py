"""Ported live compiler contracts from the frozen gates (see contract_inventory.md).

Sources: stage5/descriptor_controls.py, stage5/proof_controls.py,
stage4/components.py, stage3/components.py, stage2/verify.py (width tests).
"""

import copy
import json
import subprocess

import pytest
import sympy
import torch
from torch.utils._sympy.functions import FloorDiv, ModularIndexing

from cubecl_torch.compiler.build import compiler_path
from cubecl_torch.compiler.errors import UnsupportedCubeCL
from cubecl_torch.compiler.indexing import encode, evaluate
from cubecl_torch.compiler.plan import PhysicalCandidate
from cubecl_torch.testing.fixtures import captured, physical
from cubecl_torch.testing.test_contracts_runtime import rng_descriptor


def rust_rejects(tmp_path, name, packet):
    out = tmp_path / name
    out.mkdir()
    (out / "descriptor.json").write_text(json.dumps(packet))
    proc = subprocess.run(
        [str(compiler_path()), str(out), "cpp", str(out / "descriptor.json")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0 and not (out / "cubecl.ir").exists(), (
        proc.stdout + proc.stderr
    )
    return proc.stderr


def packet(semantic, plan):
    return {
        "schema": 5,
        "semantic": semantic,
        "physical_plan": plan,
        "entrypoint": "cubecl5_validation",
    }


VECTOR = PhysicalCandidate("vector_pointwise", (128, 1, 1), 4).to_dict()
PHYSICAL_DEFECTS = {
    "oversized_dimensions": {"workgroup": [4294967295, 4294967295, 2]},
    "uncatalogued_scalar256": {
        "strategy": "scalar_pointwise",
        "workgroup": [256, 1, 1],
        "vector_width": 1,
    },
    "invalid_vector_transaction": {"vector_width": 8},
    "invalid_plane_policy": {"plane_width": 64},
    "invalid_mapping": {"elements_per_unit": 7},
}


@pytest.mark.parametrize("name", sorted(PHYSICAL_DEFECTS))
def test_malformed_physical_packet_fails_before_ir(tmp_path, name):
    plan = {**VECTOR, **PHYSICAL_DEFECTS[name]}
    rust_rejects(tmp_path, name, packet(captured("add-float32"), plan))


def test_invalid_envelope_and_unknown_op_fail_before_ir(tmp_path):
    rust_rejects(tmp_path, "envelope", packet(None, VECTOR))
    semantic = captured("add-float32")
    semantic["ops"][0]["op"] = "unknown"
    rust_rejects(tmp_path, "unknown", packet(semantic, VECTOR))


BASELINE = PhysicalCandidate("scalar_pointwise_baseline").to_dict()


def rng_defect(name):
    from cubecl_torch.testing.test_contracts_runtime import masked_indirect_descriptor

    d = rng_descriptor()
    if name == "counter_width":
        d["ops"][2] = {
            "id": "counter",
            "op": "constant",
            "args": [],
            "dtype": "int32",
            "semantic_dtype": "int32",
            "value": 1 << 31,
        }
    elif name == "rng_algorithm":
        d["ops"][3]["algorithm"] = "host_random"
    elif name == "missing_effect":
        d["effects"]["rng"] = None
    elif name == "index_storage":
        d["buffers"][0]["dtype"] = "int32"
        d["ops"][0]["dtype"] = d["ops"][0]["semantic_dtype"] = "int32"
    elif name == "oversized_rng_extent":
        d["numel"] = {"op": "constant", "value": 1 << 31}
    elif name == "untyped_counter":
        d["ops"][2]["args"] = ["missing"]
    else:
        d = masked_indirect_descriptor(False)
        if name == "unchecked_address":
            d["ops"][3]["body"][2]["index"]["id"] = "index"
        else:
            d["ops"][-1]["index"] = {"op": "value", "id": "checked"}
    return d


@pytest.mark.parametrize(
    "name",
    [
        "counter_width",
        "rng_algorithm",
        "missing_effect",
        "index_storage",
        "untyped_counter",
        "unchecked_address",
        "escaped_checked_address",
        "oversized_rng_extent",
    ],
)
def test_invalid_effect_descriptor_fails_before_ir(tmp_path, name):
    rust_rejects(tmp_path, name, packet(rng_defect(name), BASELINE))


REDUCTION = PhysicalCandidate("shared_tree_reduction_baseline").to_dict()


def reduction_defect(name):
    d = captured("row_sum-float32")
    reduce = next(i for i, op in enumerate(d["ops"]) if op["op"] == "reduction")
    if name == "missing_conversion":
        d["ops"][reduce - 1]["dtype"] = "bool"
    elif name == "unsupported_accumulator":
        d["ops"][reduce]["accumulator_dtype"] = "float16"
    elif name == "outer_varying_store":
        d["ops"][-1]["index"] = {"op": "reduction"}
    elif name == "wrong_semantic_dtype":
        d["ops"][reduce]["semantic_dtype"] = "int64"
    else:
        sum_op = d["ops"][reduce]
        d["ops"][reduce] = {
            "id": sum_op["id"],
            "op": "masked",
            "args": sum_op["args"],
            "dtype": "float32",
            "semantic_dtype": "float32",
            "body": [sum_op],
            "result": sum_op["id"],
            "other": 0.0,
        }
    return d


@pytest.mark.parametrize(
    "name",
    [
        "missing_conversion",
        "unsupported_accumulator",
        "outer_varying_store",
        "masked_reduction",
        "wrong_semantic_dtype",
    ],
)
def test_invalid_reduction_descriptor_fails_before_ir(tmp_path, name):
    rust_rejects(tmp_path, name, packet(reduction_defect(name), REDUCTION))


def test_signed_i64_floor_mod_encoding():
    i = sympy.Symbol("i", integer=True)
    n = sympy.Symbol("n", positive=True, integer=True)
    for expr in (
        i * 3 + 2,
        FloorDiv(i - 9, 7),
        sympy.Mod(i - 9, 7),
        ModularIndexing(i, 2, n),
    ):
        tree = encode(expr, lambda s: 0, i)
        for v in (-(2**32) - 1, -9, -1, 0, 2**31 + 1, 2**32 + 257):
            assert evaluate(tree, [257], v) == int(expr.subs({i: v, n: 257}))
    unknown = sympy.Symbol("modulus", integer=True)
    for expr in (ModularIndexing(i, 2, unknown), FloorDiv(i, -n)):
        with pytest.raises(UnsupportedCubeCL, match="positive"):
            encode(expr, lambda s: 0, i)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_literal_zero_axis_coefficient_broadcasts_from_misaligned_scalar():
    d = captured("row_sum-float32")
    d["buffers"][0]["stride"] = [
        {"op": "constant", "value": 1},
        {"op": "constant", "value": 0},
    ]
    d["ops"][0]["index"] = {
        "op": "add",
        "args": [
            {"op": "linear"},
            {
                "op": "mul",
                "args": [{"op": "reduction"}, {"op": "constant", "value": 0}],
            },
        ],
    }
    kernel = physical(d, "plane_shared", (128, 1, 1), 4, plane_width=32)
    x = (
        torch.tensor([123.0, 2.0, 3.0, 4.0], device="cuda")[1:]
        .view(3, 1)
        .expand(3, 257)
    )
    y = torch.empty(3, device="cuda")
    kernel((257, 3), x, y)
    torch.testing.assert_close(y, x.sum(-1), rtol=0, atol=0)
    assert x.data_ptr() % 16 != 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("stride", [0, 1])
def test_runtime_axis_multiplier_excluded_by_python_and_rust(stride):
    from cubecl_torch.inductor.legality import candidates, reduction_coefficient

    s = captured("row_sum-float32")
    s["scalars"].append({"id": 2, "dtype": "int64", "positive": False})
    s["buffers"][0]["stride"] = [
        {
            "op": "mul",
            "args": [{"op": "scalar", "slot": 0}, {"op": "scalar", "slot": 2}],
        },
        {"op": "scalar", "slot": 2},
    ]
    s["ops"][0]["index"] = {
        "op": "mul",
        "args": [s["ops"][0]["index"], {"op": "scalar", "slot": 2}],
    }
    x = (
        torch.ones(3, 257, device="cuda")
        if stride
        else torch.ones(1, device="cuda").expand(3, 257)
    )
    y = torch.empty(3, device="cuda")
    baseline = physical(s, "shared_tree_reduction_baseline")
    legal, _ = candidates(s, (257, 3, stride), (x, y), baseline.kernel.resources)
    assert all(p.vector_width == 1 for p in legal)
    assert reduction_coefficient(s["ops"][0]["index"]) is None
    with pytest.raises(RuntimeError, match="not proven unit stride"):
        physical(s, "plane_shared", (128, 1, 1), 4, plane_width=32)
    baseline((257, 3, stride), x, y)
    torch.testing.assert_close(y, x.sum(-1), rtol=0, atol=0)


@pytest.mark.parametrize("value", [0.0, True, 1 << 63])
def test_axis_literal_is_strict_signed_i64(value):
    from cubecl_torch.inductor.legality import reduction_coefficient

    tree = {
        "op": "mul",
        "args": [{"op": "reduction"}, {"op": "constant", "value": value}],
    }
    assert reduction_coefficient(tree) is None


def buffer_defect(base, name):
    """A captured descriptor breaking the buffer contract: inputs, then outputs,
    each output stored exactly once, at most 16 buffers."""
    d = captured(base)
    store = next(op for op in d["ops"] if op["op"] == "store")
    output = len(d["buffers"]) - 1
    if name == "store_into_input":
        store["buffer"] = 0
        return d, "store into input buffer 0"
    if name == "unwritten_output":
        d["buffers"].append(copy.deepcopy(d["buffers"][output]))
        return d, f"output buffer {output + 1} is never written"
    if name == "outputs_first":
        d["buffers"].insert(0, d["buffers"].pop())
        store["buffer"] = 0
        for op in d["ops"]:
            if op["op"] == "load":
                op["buffer"] += 1
        return d, "buffer 1 has access read, expected write: inputs precede outputs"
    if name == "two_stores":
        d["ops"].append(copy.deepcopy(store))
        return d, f"output buffer {output} is written by 2 stores"
    # 17 buffers: extra unread inputs before the output.
    extra = 17 - len(d["buffers"])
    d["buffers"][output:output] = [copy.deepcopy(d["buffers"][0])] * extra
    store["buffer"] = output + extra
    return d, "17 buffers declared, outside 1..=16"


CONTRACT_DEFECTS = [
    (base, name)
    for base in ("add-float32", "row_sum-float32")
    for name in ("store_into_input", "unwritten_output", "outputs_first", "too_many")
] + [
    # Several stores to one output are a pointwise program order (fill, then a
    # predicated store); a reduction stores each output once.
    ("row_sum-float32", "two_stores"),
]


@pytest.mark.parametrize(("base", "name"), CONTRACT_DEFECTS)
def test_buffer_contract_violation_named_before_ir(tmp_path, base, name):
    from cubecl_torch.compiler.descriptor import validate
    from cubecl_torch.compiler.errors import MalformedDescriptor

    d, message = buffer_defect(base, name)
    with pytest.raises(MalformedDescriptor) as error:
        validate(d)
    assert str(error.value) == message
    plan = REDUCTION if base.startswith("row_sum") else BASELINE
    assert message in rust_rejects(tmp_path, name, packet(d, plan))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_broken_buffer_contract_rejected_before_compile():
    from cubecl_torch.inductor import kernels, tuning

    multi = copy.deepcopy(captured("add-float32"))
    multi["buffers"][0]["access"] = "write"
    before = kernels.COMPILER_INVOCATIONS
    with pytest.raises(ValueError, match="load from output buffer 0"):
        tuning.compile_kernel(multi, 0)
    assert before == kernels.COMPILER_INVOCATIONS
