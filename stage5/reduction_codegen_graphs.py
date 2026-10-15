"""Fresh graph capture with existing PyTorch lookup-table configuration pins.

Only the two retained efficient-reference configs are admitted. No sweep or
arithmetic rewrite; the ordinary reference harness still captures all artifacts.
"""

import json


def main():
    from stage1 import overlay
    from stage5 import reference_reduction_probe

    original_install = overlay.install
    state = {}
    pins = {
        "triton_red_fused_sum_0": {
            "XBLOCK": 1,
            "R0_BLOCK": 2048,
            "num_warps": 16,
            "num_stages": 1,
        },
        "triton_per_fused_sum_1": {"XBLOCK": 1, "num_warps": 2, "num_stages": 1},
    }
    observed = []

    def install_and_pin(*args, **kwargs):
        original_install(*args, **kwargs)
        from torch._inductor import config
        from torch._inductor.runtime import triton_heuristics as heuristics

        original = heuristics.lookup_autotune_config
        state.update(heuristics=heuristics, original=original)

        def lookup(size_hints, fn):
            if fn.__name__ in pins:
                key = heuristics.generate_lookup_hash_from_source_code(
                    str(size_hints), fn.src
                )
                config.autotune_lookup_table[key] = pins[fn.__name__]
                observed.append(
                    {
                        "kernel": fn.__name__,
                        "lookup_hash": key,
                        "size_hints": str(size_hints),
                        "config": pins[fn.__name__],
                    }
                )
            return original(size_hints, fn)

        heuristics.lookup_autotune_config = lookup

    overlay.install = install_and_pin
    try:
        reference_reduction_probe.main()
    finally:
        overlay.install = original_install
        if state:
            state["heuristics"].lookup_autotune_config = state["original"]
        from pathlib import Path
        import sys

        path = Path(sys.argv[sys.argv.index("--output") + 1])
        path.with_suffix(".pins.json").write_text(json.dumps(observed, indent=2))


if __name__ == "__main__":
    main()
