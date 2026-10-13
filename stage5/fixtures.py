"""Narrow physical-plan fixtures, independent of scheduler specialization."""

import copy
import json
from pathlib import Path


def captured(family, dtype):
    artifacts = json.loads(
        (Path(__file__).parent / "artifacts/baseline-cubecl_baseline.json").read_text()
    )["artifacts"]
    for item in artifacts:
        d = item["semantic"]
        if d["buffers"][0]["dtype"] != dtype:
            continue
        if (
            family == "add"
            and d["schema"] == 4
            and len(d["buffers"]) == 2
            and len(d["scalars"]) == 1
            and any(op["op"] == "add" for op in d["ops"])
            and not any(op["op"] in ("sin", "cos", "mul", "masked") for op in d["ops"])
        ):
            d = copy.deepcopy(d)
            d["scalars"][0]["positive"] = False
            return d
        if (
            family == "row_sum"
            and d["schema"] == 3
            and len(d["buffers"]) == 2
            and len(d["scalars"]) == 2
            and sum(op["op"] == "reduction" for op in d["ops"]) == 1
            and all(
                op["op"] in ("load", "cast", "reduction", "store") for op in d["ops"]
            )
        ):
            d = copy.deepcopy(d)
            d["scalars"][0]["positive"] = False
            return d
    raise ValueError("captured semantic not found: " + family + "-" + dtype)
