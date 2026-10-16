# cwk.7 review disposition

The required review-mcp review returned successfully. Its verdict is that the
change is sound and the diagnostic claims are supported, with one actionable
medium cache-provenance finding. [Full review and trace](review.md).

- **Cache environment ordering:** fixed by setting TORCHINDUCTOR_CACHE_DIR before
  Torch/overlay/Inductor imports in extra_graphs. The fresh `extras-reviewed/`
  run asserts the actual resolved cache directory and repeats all33 extra cases
  plus the offset/current-stream check. Recorded rows and IR/CUDA/PTX match the
  original run; the original evidence is preserved.
- **Neighbor oracle scope:** the report explicitly labels GPU checks as parity
  against the old tree at each size. The descriptor replacement now requires
  exactly four extent-related constants before codegen, catching future unrelated
  uses of9. No reducer or GPU arithmetic changes follow from this assertion.
- **Independent substitution admission:** the analyzer now requires actual plane
  substitutions for all three storage dtypes in the178-row graph qualification.
- **Partial cache identity:** in-process frozen/plane equality can reuse one
  immutable cache entry. The substantive independent comparison remains against
  the retained r0 partial's IR/CUDA/PTX, and201 component files retain their hashes.
- **Gap label:** renamed to inter_kernel_gap and described as GPU activity
  timestamps, rather than a host measurement. Finite-row dictionary comparisons
  remain separate from exceptional-value byte/class/zero-sign controls.

The extra repeat refreshed a cached resources.json module_load_ms field. The
initial revalidation correctly rejected a changed retained-file comparison at
`retained/float32-8-1048576/frozen/0/resources.json`. The validator now preserves
the original timing-run resource snapshot while independently requiring identical
attributes, static resources, PTX hash and compiler identity. Only the volatile
module-load timing differs; kernel/descriptor/IR/CUDA/PTX checks remain exact.
`resources-refresh.json` retains that difference. This is metadata handling,
not a numerical or kernel-identity waiver.

Post-review checks pass: the affected33 graph cases/current-stream control,
108-trace evidence admission, resource queries, Python lint/compile and Rust
format checks. All18 established regression gates had already passed; these
small harness/provenance corrections do not change production compiler code,
the retained partial/final kernels or any timing capture. No second review was
needed for these bounded follow-up corrections.
