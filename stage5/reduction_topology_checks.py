"""Independent arithmetic, validator, diagnostic eligibility and native bounds."""

import ctypes as C
import json
import os
import struct
import subprocess

from stage5.reduction_topology_model import ROOT, OUTPUT, model
from stage5.reduction_topology_probe import candidate


def scalar_model(values):
    """Owner-only scalar recurrence, independent of tensor reshaping/shuffles."""

    def add(a, b):
        return struct.unpack("f", struct.pack("f", a + b))[0]

    def tree(xs):
        while len(xs) > 1:
            half = len(xs) // 2
            xs = [add(xs[i], xs[i + half]) for i in range(half)]
        return xs[0]

    extent = (len(values) + 8) // 9
    parts = []
    for part in range(9):
        private = []
        for thread in range(512):
            chains = []
            for j in range(4):
                acc = 0.0
                for r in range(thread + 512 * j, extent, 2048):
                    coordinate = part * extent + r
                    acc = add(
                        acc, values[coordinate] if coordinate < len(values) else 0.0
                    )
                chains.append(acc)
            private.append(add(add(chains[0], chains[1]), add(chains[2], chains[3])))
        parts.append(tree([tree(private[i : i + 32]) for i in range(0, 512, 32)]))
    return parts, tree(parts + [0.0] * 7)


def main():
    import torch

    from stage5.legality import numerical_reason
    from stage5.plan import PhysicalCandidate
    from stage5.reduction_policy import qualified_for_runtime

    torch.set_num_threads(1)
    torch.set_flush_denormal(False)
    torch.manual_seed(901)
    for n in (0, 1, 31, 512, 513, 65539):
        x = torch.randn((1, n))
        parts, out = scalar_model(x[0].tolist())
        result = model(x)
        assert result["partials"].tolist() == [parts]
        assert result["output"].tolist() == [out]
    plan = candidate()
    assert PhysicalCandidate.from_dict(plan.to_dict()) == plan
    for n in (0, 9, 7283, 1048576):
        sem = {"schema": 3, "reduction_extent": {"op": "constant", "value": n}}
        assert numerical_reason(plan, sem, ())
        assert not qualified_for_runtime(sem, plan)
    packet = json.loads(
        (OUTPUT / "kernels/float32/candidate/descriptor.json").read_text()
    )
    cache = ROOT / "cache/reduction-topology/negative"
    cache.mkdir(parents=True, exist_ok=True)
    changes = (
        {"private_merge": "halving"},
        {"private_merge": None},
        {"private_accumulators": 1},
        {"private_accumulators": 2},
        {"private_accumulators": True},
        {"workgroup": [256, 1, 1]},
        {"workgroup": [32, 16, 1]},
        {"vector_width": 2},
        {"plane_width": 64},
        {"outputs_per_workgroup": 2},
        {"elements_per_unit": 2},
        {"strategy": "plane_shared_load_batch"},
    )
    for i, change in enumerate(changes):
        invalid = {**plan.to_dict(), **change}
        try:
            PhysicalCandidate.from_dict(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(change)
        path = cache / f"{i}.json"
        path.write_text(json.dumps({**packet, "physical_plan": invalid}))
        run = subprocess.run(
            [
                str(ROOT.parent / "stage0/target/release/cubecl-stage0"),
                str(cache / str(i)),
                "cpp",
                str(path),
            ],
            capture_output=True,
            text=True,
            env={**os.environ, "STAGE0_CARGO_PROFILE": "release"},
        )
        assert run.returncode and "Error" in run.stderr, (change, run.stderr)
    native = C.CDLL(str(ROOT.parent / "stage0/libstage0_launcher.so"))
    launch = native.stage5_launch
    launch.argtypes = [
        C.c_void_p,
        C.POINTER(C.c_uint64),
        C.c_size_t,
        C.c_uint64,
        C.POINTER(C.c_uint),
        C.c_uint,
        C.c_uint,
        C.c_void_p,
        C.c_void_p,
        C.c_size_t,
    ]
    launch.restype = C.c_int
    bad_geometry = (
        ((512, 1, 1), 0),
        ((512, 1, 1), 32),
        ((512, 1, 1), 128),
        ((512, 1, 1), 2048),
        ((256, 2, 1), 64),
        ((1024, 1, 1), 64),
    )
    for dims, shared in bad_geometry:
        error = C.create_string_buffer(256)
        rc = launch(
            None,
            (C.c_uint64 * 2)(0, 0),
            2,
            0,
            (C.c_uint * 3)(*dims),
            shared,
            1,
            None,
            error,
            len(error),
        )
        assert rc and b"unsupported Stage5 launch geometry" in error.value
    result = {
        "complete": True,
        "independent_scalar_cases": 6,
        "python_rust_negative_neighbors": len(changes),
        "native_geometry_negatives": len(bad_geometry),
        "automatic_eligibility_unchanged": True,
    }
    (OUTPUT / "checks.json").write_text(json.dumps(result, indent=2))
    print(result)


if __name__ == "__main__":
    main()
