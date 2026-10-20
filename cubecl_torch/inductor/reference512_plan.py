"""Bounded admission of the frozen FP32 split9 producer/final pair."""

import copy
import json
from functools import cache
from pathlib import Path

STRATEGY = "plane_shared_reference512"
VERSION = "qualified-reference512-pair-v1"
SHAPES = frozenset(((5, 65539), (8, 65539), (64, 65539), (8, 131071), (8, 1048576)))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


@cache
def _templates():
    data = json.loads(
        Path(__file__).with_name("reference512_templates.json").read_text()
    )
    return tuple((name, canonical(value)) for name, value in data.items())


def templates():
    return {name: json.loads(value) for name, value in _templates()}


def role(semantic):
    for name, template in _templates():
        if canonical(semantic) == template:
            return name
    return None


def context(producer, final, selected_role):
    return {
        "version": VERSION,
        "producer": copy.deepcopy(producer),
        "final": copy.deepcopy(final),
        "role": selected_role,
    }


def validate(semantic, value):
    if value is None:
        return None
    expected = context(templates()["producer"], templates()["final"], role(semantic))
    if expected["role"] is None or canonical(value) != canonical(expected):
        raise ValueError("unsupported reference512 pair admission context")
    return copy.deepcopy(value)


def admits(value, scalars):
    return (
        value is not None
        and value["role"] == "producer"
        and len(scalars) == 2
        and (int(scalars[1]), int(scalars[0])) in SHAPES
    )


def scheduler_context(record, records):
    """Use existing scheduler ownership, then serialize only static contracts."""
    if role(record["semantic"]) is None:
        return None
    for producer in records:
        if role(producer["semantic"]) != "producer" or not producer.get(
            "shape_admitted"
        ):
            continue
        outputs = producer["outputs"]
        if len(outputs) != 1 or len(outputs[0].users) != 1:
            continue
        user = outputs[0].users[0]
        if user.is_weak:
            continue
        for final in records:
            if (
                role(final["semantic"]) != "final"
                or final["nodes"] != [user.node.get_name()]
                or producer["output_names"] != final["input_names"]
            ):
                continue
            outs = final["outputs"]
            if (
                len(outs) != 1
                or len(outs[0].users) != 1
                or outs[0].users[0].is_weak
                or type(outs[0].users[0].node).__name__ != "OutputNode"
            ):
                continue
            if record is producer or record is final:
                return context(
                    producer["semantic"], final["semantic"], role(record["semantic"])
                )
    return None
