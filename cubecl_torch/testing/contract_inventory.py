"""Classification of every frozen gate assertion: ported live contract or historical.

``RULES`` maps each gate script to line ranges. A live range names the package
test that now holds the contract; a historical range names a reason code from
``REASONS``. ``python -m cubecl_torch.testing.contract_inventory`` regenerates
``contract_inventory.md``; ``test_contract_inventory.py`` checks completeness.
"""

import sys
from pathlib import Path

from cubecl_torch.testing.gate_assertions import SCRIPTS, sites

HERE = Path(__file__).resolve().parent
MARKDOWN = HERE / "contract_inventory.md"

REASONS = {
    "H-BACKEND": "Drives a superseded Stage 1-3 backend entry point with its own capability "
    "set, wire schema and runtime. The live front end is the Stage 4/5 chain; its "
    "behaviour on these semantics is covered by the 168-case Stage 5 corpus "
    "(equivalent to Stage 5 at the consolidation, guarded since by the manifest, "
    "testing/manifest_check.py) and the linked live contracts.",
    "H-ABI": "Checks a superseded per-stage native entry or Python wrapper "
    "(stage0_launch..stage4_launch, Stage 0 Kernel._enqueue, Stage 2 prepare(), "
    "Stage 3 ReductionPlan). The live runtime has one contract-driven launch path.",
    "H-COUNTERS": "Cross-checks per-stage native counters (Stage 3/4 launch and "
    "shared-byte counters) against per-stage Python counters: bookkeeping of the "
    "superseded runtimes. Launch-count parity for the live runtime is covered by "
    "test_contracts_runtime.py::test_trace_scopes_and_counters.",
    "H-LLVM": "LLVM route: an offline comparison compiler route, not the production "
    "C++/NVRTC route.",
    "H-CAPTURE": "CUDA graph capture through the Stage 0 ABI. Every later stage and "
    "the live runtime refuse capture (test_contracts_runtime.py::test_capture_refused).",
    "H-HARNESS": "Script-level aggregation or precondition (exit status after "
    "failures, dynamo error suppression off); not a behaviour contract.",
    "H-EVIDENCE": "Profiler naming evidence for a one-off audit kernel; generated "
    "kernel census for the live path is covered by "
    "test_contracts_effects.py::test_profiler_census_generated_kernels_only.",
    "H-STAGE1-PLAN": "Stage 1 schema-1 wire plan (bridge_description); schema 1 is "
    "not produced or accepted by the live compiler.",
}

R = "test_contracts_runtime.py::"
C = "test_contracts_compiler.py::"
D = "test_contracts_descriptors.py::"
T = "test_contracts_tuning.py::"
E = "test_contracts_effects.py::"
F = "test_contracts_fp.py::"
O = "test_contracts_overlay.py::"
P = "test_contracts_frontend.py::"

RULES = {
    "stage5/components.py": [
        (1, 135, T + "test_every_catalog_plan_tails_and_rejections")
    ],
    "stage5/controls.py": [
        (30, 38, R + "test_vector_misalignment_refused_before_launch"),
        (39, 63, R + "test_vector_special_values_and_canaries"),
        (64, 97, T + "test_backend_dynamic_buckets_streams_dropout_and_rank0"),
        (98, 139, R + "test_rng_candidates_exact_and_not_repeatable"),
        (140, 165, T + "test_backend_dynamic_buckets_streams_dropout_and_rank0"),
        (166, 247, R + "test_atomic_candidates_exact_and_not_repeatable"),
    ],
    "stage5/proof_controls.py": [
        (
            17,
            50,
            C + "test_literal_zero_axis_coefficient_broadcasts_from_misaligned_scalar",
        ),
        (51, 90, C + "test_runtime_axis_multiplier_excluded_by_python_and_rust"),
        (91, 102, C + "test_multiple_outputs_rejected_before_compile"),
        (103, 112, C + "test_axis_literal_is_strict_signed_i64"),
        (113, 170, T + "test_deterministic_cold_cache_and_failed_selection_retry"),
        (171, 194, R + "test_prepared_scope_releases_tensors"),
    ],
    "stage5/descriptor_controls.py": [
        (1, 61, C + "test_malformed_physical_packet_fails_before_ir")
    ],
    "stage5/launch_controls.py": [
        (44, 48, R + "test_zero_grid_is_uncounted_noop"),
        (51, 60, R + "test_warm_rejections_before_launch"),
        (61, 69, R + "test_layout_cache_is_bounded"),
        (70, 92, R + "test_warm_rejections_before_launch"),
        (93, 113, R + "test_inconsistent_descriptor_extent_refused"),
        (114, 130, T + "test_backend_dynamic_buckets_streams_dropout_and_rank0"),
        (131, 150, R + "test_warm_key_rechecks_storage_capacity"),
        (151, 158, R + "test_compiled_descriptor_is_a_snapshot"),
        (159, 188, R + "test_trace_scopes_and_counters"),
        (189, 225, R + "test_shared_kernel_independent_streams_and_trace_scopes"),
        (226, 247, R + "test_producer_readiness_and_no_retention"),
        (248, 260, R + "test_prepared_scope_releases_tensors"),
    ],
    "stage5/cache_probe.py": [
        (1, 32, T + "test_fresh_process_reload_reuses_artifacts_and_selection")
    ],
    "stage5/compiler_compare.py": [
        (
            1,
            120,
            "test_compiler_binding.py::test_hand_authored_descriptor_without_inductor",
            "C++ route correctness; the LLVM iteration of the same site is H-LLVM",
        ),
    ],
    "stage4/verify.py": [
        (28, 32, "test_compiler_source.py::test_mismatch_is_refused"),
        (33, 33, "H-HARNESS"),
        (120, 160, E + "test_profiler_census_generated_kernels_only"),
        (161, 235, E + "test_uniform_rng_exact_state_and_reuse"),
        (236, 320, E + "test_dropout_forward_backward"),
        (321, 420, E + "test_indirect_atomic_embedding"),
        (421, 520, E + "test_training_steps_match_reference"),
        (521, 590, E + "test_unsupported_capabilities_rejected_without_compiling"),
        (600, 605, R + "test_trace_scopes_and_counters"),
        (606, 611, "H-COUNTERS"),
        (612, 631, "H-HARNESS"),
    ],
    "stage4/runtime_checks.py": [
        (40, 245, E + "test_stream_lifetime_rng_streams_special_values_determinism"),
        (246, 258, R + "test_trace_scopes_and_counters"),
    ],
    "stage4/verify_bounds.py": [(1, 24, E + "test_device_bounds_trap")],
    "stage4/counter_guard_probe.py": [
        (1, 54, E + "test_oversized_rng_counter_domain_rejected")
    ],
    "stage4/components.py": [
        (100, 120, R + "test_rng_descriptor_exact_with_canaries"),
        (121, 150, R + "test_rng_counter_bits"),
        (151, 300, R + "test_masked_indirect_never_touches_invalid_indices"),
        (
            301,
            350,
            C + "test_invalid_effect_descriptor_fails_before_ir",
            "the bad_plan case targets the Stage 1 plan and is H-STAGE1-PLAN",
        ),
        (351, 375, O + "test_missing_native_component_fails_availability"),
        (376, 386, R + "test_rng_runtime_counter_domain_guard"),
        (387, 387, "H-COUNTERS"),
    ],
    "stage0/verify.py": [
        (20, 35, "test_runtime_trace_gil.py::test_free_threaded_launch_without_triton"),
        (36, 63, "test_runtime_launch.py::test_tails_and_output_canaries"),
        (64, 67, "test_runtime_launch.py::test_special_values"),
        (68, 70, "test_runtime_launch.py::test_non_default_stream_ordering"),
        (71, 95, R + "test_shared_kernel_independent_streams_and_trace_scopes"),
        (96, 125, R + "test_record_stream_prevents_early_reuse"),
        (126, 150, "H-CAPTURE"),
        (151, 162, "test_runtime_validation.py::test_dtype"),
        (163, 168, "test_runtime_validation.py::test_alias"),
        (169, 195, R + "test_shared_kernel_independent_streams_and_trace_scopes"),
        (
            196,
            217,
            "test_runtime_trace_gil.py::test_free_threaded_launch_without_triton",
        ),
    ],
    "stage1/verify.py": [(1, 252, "H-BACKEND")],
    "stage1/verify_boundaries.py": [
        (20, 33, O + "test_missing_native_component_fails_availability"),
        (34, 56, O + "test_default_inductor_still_requires_triton"),
        (57, 67, "H-STAGE1-PLAN"),
        (68, 87, O + "test_two_output_fusion_vetoed_before_codegen"),
        (88, 92, "test_inductor_backend.py::test_pointwise_matches_eager"),
    ],
    "stage2/verify.py": [
        (1, 175, "H-BACKEND"),
        (176, 269, P + "test_pointwise_special_values_signed_zero_and_storage"),
        (270, 311, "H-BACKEND"),
        (312, 328, P + "test_pointwise_special_values_signed_zero_and_storage"),
        (329, 605, "H-BACKEND"),
        (606, 690, P + "test_unsupported_inputs_rejected_without_compiling"),
        # Float minimum was a rejection case; it is supported since
        # generic-reduction-combiners and held by the acceptance test.
        (691, 695, "test_float_extrema.py::test_pointwise_extrema_scenarios"),
        (696, 716, P + "test_unsupported_inputs_rejected_without_compiling"),
        (717, 750, C + "test_signed_i64_floor_mod_encoding"),
        (751, 786, "H-COUNTERS"),
    ],
    "stage2/components.py": [
        (60, 92, D + "test_signed_i64_floor_mod_on_device"),
        (93, 125, D + "test_floor_mod_near_i64_boundary"),
        (126, 136, D + "test_signed_i64_floor_mod_on_device"),
        (137, 186, D + "test_nested_lazy_load_and_predicated_store"),
        (187, 205, D + "test_rust_rejects_unknown_ssa_operand"),
        (206, 240, D + "test_rust_rejects_unproven_positive_divisor"),
        (241, 268, D + "test_positive_runtime_divisor_enforced_at_launch"),
        (269, 286, "H-ABI"),
    ],
    "stage0/tests/fp_rewrites.rs": [(1, 250, F + "test_fp_rewrites_rust_suite")],
    "fp_hardening/verify.py": [
        (160, 189, "H-ABI"),
        (
            190,
            300,
            F + "test_fp_audit_cases_cpp_route",
            "C++ route; LLVM iterations of the same sites are H-LLVM",
        ),
        (301, 310, "H-EVIDENCE"),
        (311, 327, "H-COUNTERS"),
    ],
    "fp_hardening/verify_inductor.py": [
        (25, 90, F + "test_contraction_policy_through_backend"),
        (91, 103, "H-COUNTERS"),
    ],
    "stage3/verify.py": [
        (1, 66, "H-BACKEND"),
        (67, 73, P + "test_row_reduction_reuse_specials_and_accumulation"),
        (74, 147, "H-BACKEND"),
        (148, 191, P + "test_row_reduction_reuse_specials_and_accumulation"),
        (192, 294, "H-BACKEND"),
        (295, 324, P + "test_row_reduction_reuse_specials_and_accumulation"),
        (325, 357, "H-BACKEND"),
        (358, 375, P + "test_row_reduction_reuse_specials_and_accumulation"),
        (376, 440, "H-BACKEND"),
        (441, 480, E + "test_unsupported_capabilities_rejected_without_compiling"),
        (481, 486, "H-ABI"),
        (487, 500, E + "test_unsupported_capabilities_rejected_without_compiling"),
        (501, 510, "test_inductor_backend.py::test_pointwise_matches_eager"),
        (511, 530, "H-COUNTERS"),
        (531, 540, "test_compiler_source.py::test_mismatch_is_refused"),
        (541, 549, "H-HARNESS"),
    ],
    "stage3/components.py": [
        (100, 155, D + "test_reduction_tails_canaries_and_extent_guard"),
        (156, 222, D + "test_uniform_masked_values_feed_reduction"),
        (223, 270, C + "test_invalid_reduction_descriptor_fails_before_ir"),
        (271, 290, "H-COUNTERS"),
    ],
}


def classify(script, line):
    """``(class, target, note)`` for one site, or None if no rule covers it."""
    for rule in RULES.get(script, ()):
        low, high, target, *note = rule
        if low <= line <= high:
            kind = "historical" if target in REASONS else "live"
            return kind, target, note[0] if note else ""
    return None


def render():
    lines = [
        "# Historical gate contract inventory",
        "",
        "Generated by `python -m cubecl_torch.testing.contract_inventory`; do not edit by hand.",
        "",
        "Every assertion site in the frozen Stage 0-5 gate scripts (as enumerated by",
        "`gate_assertions.py`) is either a **live** contract, ported to the linked package",
        "test under `cubecl_torch/testing/`, or **historical**, with the reason below.",
        "",
        "## Historical reasons",
        "",
    ]
    lines += [f"- **{code}**: {text}" for code, text in REASONS.items()]
    found = sites()
    live = sum(
        1
        for script, rows in found.items()
        for s in rows
        if (c := classify(script, s["line"])) and c[0] == "live"
    )
    total = sum(len(rows) for rows in found.values())
    lines += [
        "",
        f"## Sites ({total} total: {live} live, {total - live} historical)",
        "",
    ]
    for script in SCRIPTS:
        lines += [
            f"### `{script}`",
            "",
            "| Line | Function | Assertion | Class | Ported to / reason |",
            "|---|---|---|---|---|",
        ]
        for s in found[script]:
            kind, target, note = classify(script, s["line"]) or ("UNCLASSIFIED", "", "")
            text = s["text"].replace("|", "\\|").replace("`", "'")
            where = f"`{target}`" + (f" ({note})" if note else "")
            lines.append(
                f"| {s['line']} | `{s['function']}` | `{text}` | {kind} | {where} |"
            )
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    MARKDOWN.write_text(render())
    print("wrote", MARKDOWN, file=sys.stderr)
