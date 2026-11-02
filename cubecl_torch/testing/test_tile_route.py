"""Tile kernels through the production route (cubecl-tile-stage-1 group 2).

The plan is pinned: no kernel is offered tile plans until they qualify.
"""

import json
import os
import subprocess
import sys
import tomllib

import numpy as np
import pytest
import torch

from cubecl_torch.compiler.plan import PhysicalCandidate, catalog, tile_catalog
from cubecl_torch.testing.fixtures import REPO
from cubecl_torch.testing.test_wide_workgroups import geometry_error

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

RMSNORM = """
import json
import sys
from cubecl_torch.runtime import guard
guard.install()
import cubecl_torch.inductor as ci
ci.install()
import torch
import torch._inductor.config
from cubecl_torch.compiler.plan import tile_catalog
from cubecl_torch.inductor import legality, tuning

rows, cols, width, validate = json.loads(sys.argv[1])
# Inductor splits rows past 65536 by default; these tests want one unsplit row.
torch._inductor.config.split_reductions = False
plan = next(p for p in tile_catalog() if p.vector_width == width)
candidates = tuning.candidates
def forced(semantic, *args, **kwargs):
    accepted, rejected = candidates(semantic, *args, **kwargs)
    if semantic["schema"] != 3:
        return accepted, rejected
    if validate:
        semantic["validate_reads"] = validate
    return [plan], rejected
tuning.candidates = forced
numerical = legality.numerical_reason
legality.numerical_reason = lambda p, *a: None if p == plan else numerical(p, *a)

def rms(x, w):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w

g = torch.Generator().manual_seed(0)
x = torch.randn(rows, cols, generator=g).cuda()
w = torch.randn(cols, generator=g).cuda()
y = torch.compile(rms, backend=ci.compile_backend, dynamic=False)(x, w)
torch.cuda.synchronize()
reference = rms(x.double(), w.double())
print("error", (y.double() - reference).abs().div(reference.abs().clamp_min(1e-30)).max().item())
"""


def pinned_rmsnorm(cache, rows, cols, width=4, validate=None):
    """The worst relative error against float64, the tile artifact's directory, and
    the reads the validation reported."""
    run = subprocess.run(
        [sys.executable, "-c", RMSNORM, json.dumps([rows, cols, width, validate])],
        cwd=REPO,
        env={
            **os.environ,
            "PYTHONPATH": str(REPO),
            "CUBECL_TORCH_CACHE_DIR": str(cache),
        },
        capture_output=True,
        text=True,
        check=True,
    )
    plans = (cache / "artifacts").glob("*/physical-plan.json")
    tile = [p.parent for p in plans if "tile_row_sum" in p.read_text()]
    assert len(tile) == 1
    lines = run.stdout.splitlines()
    error = next(line for line in lines if line.startswith("error "))
    return (
        float(error.split()[1]),
        tile[0],
        sum("[VALIDATION" in line for line in lines),
    )


@pytest.mark.parametrize(
    ("shape", "shared"),
    [((32, 16, 1), 64), ((32, 16, 1), 6144)],
)
def test_launcher_accepts_the_tile_block(shape, shared):
    assert geometry_error(shape, shared) == "null kernel handle"


@pytest.mark.parametrize("shape", [(16, 32, 1), (32, 32, 1), (32, 8, 2)])
def test_launcher_refuses_other_wide_blocks(shape):
    assert geometry_error(shape, 64) == "unsupported launch geometry"


def test_tile_plans_are_in_the_catalog():
    assert set(tile_catalog()) <= set(catalog())


def test_tile_plans_round_trip_but_other_blocks_do_not():
    for plan in tile_catalog():
        assert PhysicalCandidate.from_dict(plan.to_dict()) == plan
    other = {**tile_catalog()[0].to_dict(), "workgroup": [512, 1, 1]}
    with pytest.raises(ValueError, match="unsupported physical plan"):
        PhysicalCandidate.from_dict(other)


def test_compiler_links_no_device_runtime():
    with (REPO / "cubecl_torch" / "native" / "Cargo.lock").open("rb") as lock:
        packages = {p["name"]: p for p in tomllib.load(lock)["package"]}
    seen, stack = set(), ["cubecl-stage0"]
    while stack:
        name = stack.pop()
        if name not in seen and name in packages:
            seen.add(name)
            stack += [d.split()[0] for d in packages[name].get("dependencies", ())]
    assert "cubecl-tile" in seen
    assert "cubecl-cuda" not in seen


@pytest.mark.parametrize(("rows", "cols"), [(2048, 8192), (300, 4096)])
def test_rmsnorm_is_correct_with_the_existing_abi(tmp_path, rows, cols):
    worst, artifact, _ = pinned_rmsnorm(tmp_path, rows, cols)
    # Every plane read its own steps: one plane's share misses by a factor of 4.
    assert worst < 1e-6

    source = (artifact / "kernel.cu").read_text()
    assert "__grid_constant__" not in source
    manifest = json.loads((artifact / "manifest.json").read_text())
    assert manifest["grid_constants"] is False
    assert manifest["block"] == [32, 16, 1]
    assert manifest["arguments"][-1] == "info device pointer"
    assert manifest["metadata"]["scalar_count"] == 1
    # 16 planes x 4 lanes x 4 bytes.
    assert manifest["dynamic_shared_bytes"] == 256


@pytest.mark.parametrize(
    ("rows", "cols"), [(2048, 4099), (64, 31), (256, 100), (16, 65539)]
)
def test_no_read_past_a_tensor_at_uneven_rows(tmp_path, rows, cols):
    worst, _, reports = pinned_rmsnorm(tmp_path, rows, cols, 1, True)
    assert reports == 0
    assert worst < 1e-6


def test_validation_finds_an_unguarded_read(tmp_path):
    """The redirect and the mask are what keep the reports at zero: without them
    the last row's last step reads past ``x`` (bead sk8: columns 4099 to 4127)."""
    _, _, reports = pinned_rmsnorm(tmp_path, 2048, 4099, 1, "unguarded")
    assert reports == 29


@pytest.mark.parametrize(("rows", "cols"), [(2048, 8192), (300, 4096)])
def test_no_read_past_a_tensor_at_whole_steps(tmp_path, rows, cols):
    worst, _, reports = pinned_rmsnorm(tmp_path, rows, cols, 4, True)
    assert reports == 0
    assert worst < 1e-6


EMPTY_ROW = """
from cubecl_torch.runtime import guard
guard.install()
import cubecl_torch.inductor as ci
ci.install()
import torch
from cubecl_torch.compiler.plan import tile_catalog
from cubecl_torch.inductor import legality, tuning
from cubecl_torch.runtime import launch

plan = next(p for p in tile_catalog() if p.vector_width == 1)
candidates = tuning.candidates
def forced(semantic, *args, **kwargs):
    accepted, rejected = candidates(semantic, *args, **kwargs)
    if semantic["schema"] != 3:
        return accepted, rejected
    semantic["validate_reads"] = True
    for scalar in semantic["scalars"]:
        scalar["positive"] = False
    return [plan], rejected
tuning.candidates = forced
numerical = legality.numerical_reason
legality.numerical_reason = lambda p, *a: None if p == plan else numerical(p, *a)

# Inductor fills an empty reduction without a kernel and declares its shape symbols
# positive, so the launch contract refuses a 0. Undeclare them, and launch the
# tile kernel compiled for 40 columns at a row length of 0 itself.
calls = []
call = launch.LoadedKernel.__call__
def captured(self, scalars, *tensors):
    if tuple(self.artifact.launch.block) == (32, 16, 1):
        calls.append((self, scalars))
    return call(self, scalars, *tensors)
launch.LoadedKernel.__call__ = captured

f = torch.compile(lambda x: x.sum(-1), backend=ci.compile_backend, dynamic=True)
print("full", f(torch.ones(4, 40).cuda()).tolist())
kernel, scalars = calls[0]
assert 40 in scalars
empty = tuple(0 if s == 40 else s for s in scalars)
output = torch.full((4,), 7.0, device="cuda")
kernel(empty, torch.empty(4, 0, device="cuda").as_strided((4, 0), (0, 1)), output)
torch.cuda.synchronize()
print("empty", output.tolist(), len(calls))
"""


def test_zero_length_row_reads_nothing(tmp_path):
    run = subprocess.run(
        [sys.executable, "-c", EMPTY_ROW],
        cwd=REPO,
        env={
            **os.environ,
            "PYTHONPATH": str(REPO),
            "CUBECL_TORCH_CACHE_DIR": str(tmp_path),
        },
        capture_output=True,
        text=True,
        check=True,
    )
    assert "[VALIDATION" not in run.stdout
    assert "full [40.0, 40.0, 40.0, 40.0]" in run.stdout
    # Stored over the 7.0, so the kernel ran, on the row of no elements.
    assert "empty [0.0, 0.0, 0.0, 0.0] 2" in run.stdout


def test_vector_plan_refuses_a_partial_row(tmp_path):
    with pytest.raises(subprocess.CalledProcessError) as failure:
        pinned_rmsnorm(tmp_path, 8, 4099, 4)
    assert "tile vector plan requires a row of whole steps" in failure.value.stderr


def test_vector_recipe_loads_lines(tmp_path):
    worst, artifact, reports = pinned_rmsnorm(tmp_path, 300, 4096, 4, True)
    assert reports == 0
    assert worst < 1e-6
    source = (artifact / "kernel.cu").read_text()
    assert "ld.global.L1::evict_last.v4.b32" in source
    assert "ld.global.nc.L1::evict_first.v4.b32" in source


GUARDED = """
from cubecl_torch.runtime import guard
guard.install()
import cubecl_torch.inductor as ci
ci.install()
import torch
from cubecl_torch.compiler.plan import tile_catalog
from cubecl_torch.inductor import legality, tuning

plan = next(p for p in tile_catalog() if p.vector_width == 1)
candidates = tuning.candidates
def forced(semantic, *args, **kwargs):
    accepted, rejected = candidates(semantic, *args, **kwargs)
    if semantic["schema"] != 3:
        return accepted, rejected
    semantic["validate_reads"] = True
    return [plan], rejected
tuning.candidates = forced
numerical = legality.numerical_reason
legality.numerical_reason = lambda p, *a: None if p == plan else numerical(p, *a)

rows, cols = 2048, 4099
g = torch.Generator().manual_seed(0)
numel = rows * cols
storage = torch.full((numel + 8192,), 7.0, device="cuda")
storage[:numel] = torch.randn(numel, generator=g).cuda()
x = storage[:numel].view(rows, cols)
f = torch.compile(lambda x: x * x.sum(-1, keepdim=True), backend=ci.compile_backend, dynamic=False)
y = f(x)
torch.cuda.synchronize()
reference = x.double() * x.double().sum(-1, keepdim=True)
print("canaries", bool((storage[numel:] == 7.0).all()))
print("error", ((y.double() - reference).abs().max() / x.abs().max()).item())
"""


def test_sum_consumer_stays_inside_pytorch_storage(tmp_path):
    run = subprocess.run(
        [sys.executable, "-c", GUARDED],
        cwd=REPO,
        env={
            **os.environ,
            "PYTHONPATH": str(REPO),
            "CUBECL_TORCH_CACHE_DIR": str(tmp_path),
        },
        capture_output=True,
        text=True,
        check=True,
    )
    # Validation covers the reads, canaries the writes. The worst row sum's error
    # (the output over the largest input): 29 stray elements would add about 30.
    assert "[VALIDATION" not in run.stdout
    assert "canaries True" in run.stdout
    error = next(line for line in run.stdout.splitlines() if line.startswith("error "))
    assert float(error.split()[1]) < 0.1


SUM_ORDER = """
import json
import sys
from cubecl_torch.runtime import guard
guard.install()
import cubecl_torch.inductor as ci
ci.install()
import numpy as np
import torch
from cubecl_torch.testing.tile_band import pin

rows, cols, width = json.loads(sys.argv[1])
pin(width)
x = torch.randn(rows, cols, generator=torch.Generator().manual_seed(0))
f = torch.compile(lambda x: x.sum(-1), backend=ci.compile_backend, dynamic=False)
y = f(x.cuda())
torch.cuda.synchronize()
np.save(sys.argv[2], y.cpu().numpy())
np.save(sys.argv[3], x.numpy())
"""


def butterfly(values, axis):
    """Reduce ``axis`` by adding pairs xor 2**n down to 1, in float32."""
    bits = int(np.log2(values.shape[axis]))
    for bit in range(bits - 1, -1, -1):
        pairs = np.arange(values.shape[axis]) ^ (1 << bit)
        values = (values + np.take(values, pairs, axis=axis)).astype(np.float32)
    return np.take(values, 0, axis=axis)


@pytest.mark.parametrize(
    ("cols", "width"),
    [(8192, 4), (4096, 4), (31, 1), (100, 1), (129, 1), (1024, 1)],
)
def test_row_sum_is_in_triton_order_bit_for_bit(tmp_path, cols, width):
    """A sum per lane along a plane's steps, units xor 16 to 1, then planes and lanes
    as trees: column = plane * run * step + step index * step + width * unit + lane.
    Past the row a lane adds nothing, and a plane with no steps publishes zero: a slot
    it left unwritten would read stale shared memory."""
    rows = 32
    sums, inputs = tmp_path / "sums.npy", tmp_path / "x.npy"
    subprocess.run(
        [
            sys.executable,
            "-c",
            SUM_ORDER,
            json.dumps([rows, cols, width]),
            str(sums),
            str(inputs),
        ],
        cwd=REPO,
        env={
            **os.environ,
            "PYTHONPATH": str(REPO),
            "CUBECL_TORCH_CACHE_DIR": str(tmp_path),
        },
        capture_output=True,
        text=True,
        check=True,
    )
    step = 32 * width
    steps = -(cols // -step)
    run = -(steps // -16)
    padded = np.zeros((rows, 16 * run * step), np.float32)
    padded[:, :cols] = np.load(inputs)
    columns = padded.reshape(rows, 16, run, 32, width)
    along = np.zeros((rows, 16, 32, width), np.float32)
    for walked in range(run):
        along = (along + columns[:, :, walked]).astype(np.float32)
    across_units = butterfly(along, 2)
    across_planes = butterfly(across_units, 1)
    assert np.array_equal(butterfly(across_planes, 1), np.load(sums))


# The worst |output - float64| over a bound for a correct kernel, at most 1 when in
# it: the output's own rounding (two units, the float32 result being rounded once
# more on the way out, and one subnormal step at zero) plus float32 rounding
# accumulated over the sum of magnitudes (the kernel of |x|, |w|, |a|). One dropped
# element of a row exceeds it at any dtype.
BOUND = """
def excess(kernel, y, x, w, a):
    reference = kernel(x.double(), w.double(), a.double())
    magnitude = kernel(x.double().abs(), w.double().abs(), a.double().abs()).abs()
    limits = torch.finfo(y.dtype)
    subnormal = limits.smallest_normal * limits.eps
    rounding = 2 * limits.eps * reference.abs() + subnormal
    accumulated = 64 * torch.finfo(torch.float32).eps * magnitude
    return ((y.double() - reference).abs() / (rounding + accumulated)).max().item()
"""

FUSED = (
    """
import json
import sys
from cubecl_torch.runtime import guard
guard.install()
import cubecl_torch.inductor as ci
ci.install()
import torch
from cubecl_torch.inductor import tuning
from cubecl_torch.testing.tile_band import pin
"""
    + BOUND
    + """
width, shapes, dtypes = json.loads(sys.argv[1])
import torch._inductor.config
# Inductor splits rows past 65536 by default; these tests want one unsplit row.
torch._inductor.config.split_reductions = False
pin(width, validate=True)
KERNELS = {
    "sum": lambda x, w, a: x.sum(-1),
    "mean": lambda x, w, a: x.mean(-1),
    "consumer": lambda x, w, a: x * x.sum(-1, keepdim=True),
    "weighted": lambda x, w, a: (x * w).sum(-1) * a,
    "rmsnorm": lambda x, w, a: x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w,
}
for name, kernel in KERNELS.items():
    for rows, cols in shapes:
        for dtype in dtypes:
            g = torch.Generator().manual_seed(rows + cols)
            x = torch.randn(rows, cols, generator=g).cuda().to(getattr(torch, dtype))
            w = torch.randn(cols, generator=g).cuda().to(getattr(torch, dtype))
            a = torch.tensor(1.5, device="cuda", dtype=getattr(torch, dtype))
            torch._dynamo.reset()
            tuning.CACHE.clear()
            y = torch.compile(kernel, backend=ci.compile_backend, dynamic=False)(x, w, a)
            torch.cuda.synchronize()
            kernels = list(tuning.CACHE.values())
            reductions = [k for k in kernels if k.semantic["schema"] == 3]
            tile = [
                k for k in reductions
                if {p.strategy for p in k.selections.values()} == {"tile_row_sum"}
            ]
            per_element = sum(
                op["op"] == "store" and op.get("iteration_scope") != "outer"
                for k in reductions
                for op in k.semantic["ops"]
            )
            print(
                "case", name, rows, cols, dtype, len(kernels), len(tile), per_element,
                excess(kernel, y, x, w, a), flush=True,
            )
"""
)


@pytest.mark.parametrize(
    ("width", "shapes", "dtypes"),
    [
        (4, [(2048, 8192), (300, 4096)], ["float32"]),
        (1, [(2048, 4099), (7, 2049), (64, 31), (8, 100), (4, 65539)], ["float32"]),
        (8, [(300, 4096)], ["float16", "bfloat16"]),
        (1, [(300, 4099)], ["float16", "bfloat16"]),
    ],
)
def test_fused_row_sums_through_the_route(tmp_path, width, shapes, dtypes):
    """Each kernel is one fused tile kernel, correct and in bounds at every shape."""
    run = subprocess.run(
        [sys.executable, "-c", FUSED, json.dumps([width, shapes, dtypes])],
        cwd=REPO,
        env={
            **os.environ,
            "PYTHONPATH": str(REPO),
            "CUBECL_TORCH_CACHE_DIR": str(tmp_path),
        },
        capture_output=True,
        text=True,
        check=True,
    )
    assert "[VALIDATION" not in run.stdout
    cases = [
        line.split() for line in run.stdout.splitlines() if line.startswith("case")
    ]
    assert len(cases) == 5 * len(shapes) * len(dtypes)
    for case in cases:
        name, kernels, tile, per_element, excess = case[1], *case[5:]
        assert (kernels, tile) == ("1", "1"), case
        # The consumer and RMSNorm store per element in the reduction's own kernel.
        assert (int(per_element) > 0) == (name in ("consumer", "rmsnorm")), case
        assert float(excess) <= 1, case


ACROSS_SHAPES = (
    """
from cubecl_torch.runtime import guard
guard.install()
import cubecl_torch.inductor as ci
ci.install()
import torch
from cubecl_torch.compiler import build
from cubecl_torch.testing.tile_band import pin
"""
    + BOUND
    + """
pin(4, validate=True)
kernel = lambda x, w, a: x * x.sum(-1, keepdim=True)
f = torch.compile(lambda x: kernel(x, x, x), backend=ci.compile_backend, dynamic=True)
for rows, cols in ((2048, 8192), (1024, 8192), (2048, 4096)):
    x = torch.randn(rows, cols, generator=torch.Generator().manual_seed(0)).cuda()
    y = f(x)
    torch.cuda.synchronize()
    print("shape", build.COMPILER_INVOCATIONS, excess(kernel, y, x, x, x))
"""
)


def test_one_kernel_across_shapes_with_identical_bytes(tmp_path):
    """Runtime extents: a shape in the same class compiles nothing, and three fresh
    compiler processes write the same IR, CUDA and PTX. The reduction pass reads with
    the hint that keeps its row and the storing pass with the one that releases it."""
    records = []
    for attempt in range(3):
        cache = tmp_path / str(attempt)
        run = subprocess.run(
            [sys.executable, "-c", ACROSS_SHAPES],
            cwd=REPO,
            env={
                **os.environ,
                "PYTHONPATH": str(REPO),
                "CUBECL_TORCH_CACHE_DIR": str(cache),
            },
            capture_output=True,
            text=True,
            check=True,
        )
        assert "[VALIDATION" not in run.stdout
        shapes = [
            line.split() for line in run.stdout.splitlines() if line.startswith("shape")
        ]
        assert len(shapes) == 3
        assert len({invocations for invocations, _ in (s[1:] for s in shapes)}) == 1
        assert all(float(excess) <= 1 for _, _, excess in shapes)
        artifacts = list((cache / "artifacts").glob("*/physical-plan.json"))
        assert [p.parent for p in artifacts if "tile_row_sum" in p.read_text()]
        tile = [p.parent for p in artifacts if "tile_row_sum" in p.read_text()]
        assert len(tile) == 1
        source = (tile[0] / "kernel.cu").read_text()
        kept, released = source.index("evict_last"), source.index("evict_first")
        assert kept < source.index("__syncthreads") < released
        assert "evict_first" not in source[:released]
        assert "evict_last" not in source[released:]
        records.append(
            {
                name: (tile[0] / name).read_bytes()
                for name in ("cubecl.ir", "kernel.cu", "kernel.ptx")
            }
        )
    assert records[0] == records[1] == records[2]
