"""Pinned generic Inductor precision corrections for the experimental overlay.

A graph cast to FP16/BF16 the user wrote (``.to``, ``.half``, ``.type``, autocast:
lowered from ``aten._to_copy``) always rounds; a cast a decomposition inserts
(``std``'s result dtype, dropout's random values) follows Triton and stays in
compute precision inside fused code (multi-output-kernels group 5). The test is
by registration, the FX node's ``original_aten``: a cast inside a composite op
(``softmax(dtype=...)`` calls ``.to``) still counts as written. Under precision emulation, tensor
producers also retain eager's storage-rounding barriers, while Python scalar
literals follow promote_constants' operation/device policy instead. These edits
do not select a backend or define its compute dtype.
"""


def source_edits():
    return {
        "index_propagation.py": [
            (
                (
                    "        use_compute_types: bool = False,\n"
                    "    ) -> TypedExpr:\n"
                    "        return TypedExpr(value.expr, dtype)"
                ),
                (
                    "        use_compute_types: bool = True,\n"
                    "    ) -> TypedExpr:\n"
                    "        if not use_compute_types and dtype in (torch.float16, torch.bfloat16):\n"
                    "            # A dtype label cannot represent the requested storage rounding.\n"
                    "            return NotImplemented\n"
                    "        return TypedExpr(value.expr, dtype)"
                ),
            ),
        ],
        "lowering.py": [
            (
                "                    if emulate_precision_casts and inp_dtype in low_pr_fp:",
                (
                    "                    # Scalar literals already follow promote_constants' policy.\n"
                    "                    # It represents them as Constant or BaseView(Constant).\n"
                    "                    # Tensor producers still need their storage precision barrier.\n"
                    "                    if (\n"
                    "                        emulate_precision_casts\n"
                    "                        and inp_dtype in low_pr_fp\n"
                    "                        and not isinstance(\n"
                    "                            inputs[inp_index].unwrap_view()\n"
                    "                            if isinstance(inputs[inp_index], ir.BaseView)\n"
                    "                            else inputs[inp_index],\n"
                    "                            ir.BaseConstant,\n"
                    "                        )\n"
                    "                    ):"
                ),
            ),
            (
                (
                    "    use_compute_types = not (\n"
                    "        config.emulate_precision_casts\n"
                    "        and (src_dtype in low_pr_fp or dtype in low_pr_fp)\n"
                    "    )"
                ),
                (
                    "    # A graph cast to FP16/BF16 the user wrote is requested rounding: it\n"
                    "    # materializes even without precision emulation, which only adds the\n"
                    "    # per-operation barriers eager's storage implies. A cast a\n"
                    "    # decomposition inserted (its original aten op is not _to_copy) follows\n"
                    "    # Triton. Without metadata the cast counts as written. Compared by\n"
                    "    # name: the overlay's op objects are not the tracer's.\n"
                    "    origin = getattr(V.graph.current_node, 'meta', {}).get('original_aten')\n"
                    "    written = origin is None or str(origin) in (\n"
                    "        'aten._to_copy.default',\n"
                    "        'prims.convert_element_type.default',\n"
                    "    )\n"
                    "    use_compute_types = not (\n"
                    "        (dtype in low_pr_fp and (written or config.emulate_precision_casts))\n"
                    "        or (config.emulate_precision_casts and src_dtype in low_pr_fp)\n"
                    "    )"
                ),
            ),
        ],
    }
