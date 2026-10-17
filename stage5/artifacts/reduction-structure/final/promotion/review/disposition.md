# Required review disposition — cwk.9

[review.md](review.md) is the unabridged review-mcp response. It found the change
sound with no blocking defects. The following minor corrections are implemented
and covered by the post-review controls and affected regression repeats:

1. The ordinary static9 graph's observed winner is asserted in both the executable
   harness and offline analyzer. This verifies this capture; it does not force
   production selection or promise the same winner elsewhere.
2. `semantic_reason` rejects malformed root/buffer/op containers cleanly. Five
   further descriptor mutations are rejected by Python and native compilation;
   None/list/string roots also return reasons. Valid admission is unchanged.
3. FP16/BF16 alias controls now reinterpret the actual FP32 input's shared storage
   as the output type. All three types must raise the disjoint-output error with
   no native launch. The original mislabeled evidence is retained as historical.
4. SASS parsing explicitly checks column order and rejects instruction rows that
   precede their header. The real tables still match the qualified instruction
   sequence after relocation and retain4608/0 BAR and0/320 SHFL executions.
5. The analyzer identifies partial/final by semantic role. Graph-storage records
   actual choices after each dynamic call; warm-creation records from the earlier
   full-feature pass remain preserved and are superseded by the corrected records.
6. Components evaluates the admission predicate once. Static enqueue revalidation
   is deliberately retained, along with all live ABI checks; no host-speed claim.

These are defensive rejection/evidence corrections, with no expansion of valid
admission or change to generated arithmetic. The normal and frozen-producer
integration controls pass again, as do148 corpus,30 direct and39 negative controls,
cache init/warm/fresh reload, guard transitions and ordinary split9 generated
wrappers with the existing normal compensated producer. All15 affected/resumed
regression groups pass after the changes; the accepted18-group manifest combines
these with3 unchanged groups. No second review is needed for these small follow-ups.

The accepted same-symbol IR/CUDA/PTX bytes, frozen partial sources and201 retained
files remain identical. Stage5 stays active; Stage6 is deferred. No commit/push.
