# Attribution review disposition

Required review-mcp completed against HEAD with the seven diagnostic/report focus files. It cross-checked histories, timing/census data, SQL accounting, precision probe anchors and production hashes. It found no correctness defect invalidating the conclusions. Raw review/trace is retained in ignored `stage5/cache/attribution/review.md`.

- Failed-attempt disclosure: the report now enumerates two naive cast-veto failures, one bare-constant/view failure, the reference Python-counter compatibility failure and missing-report NVTX attempt. Numerical cancellation is separately reproduced and tracked as `cwk`; no tolerance changed.
- Packet integrity: preliminary snapshots were moved to ignored `cache/attribution/preliminary` during the review. The review's directory inventory included the older paths. Current report links, all packet files and manifest hashes were verified against disk; no broken packet symlink was found. Final evidence and the naive-veto counterexample are retained in the curated directory.
- Reference correctness: every CubeCL process now also checks its warm frozen-reference output, gradients, updated parameters and RNG advancement. Targeted tiny/larger checks passed after this setup-only change. Their 10-sample rounds are validation observations, not replacements for the primary 100-sample measurements.
- Triton comparison qualification: the report states that the probe also uses the Stage1 generic scheduling overlay, while its baseline uses the wheel.
- Native API population: analyzer now requires correlation to a `cubecl5_` kernel, preventing vendor driver launches from entering generated-only statistics. A synthetic vendor-driver test passes; all 19 retained captures were reanalyzed and evidence checks pass.
- Transcription and duration scopes: two tiny host medians were corrected; analysis qualification distinguishes full correlated kernel durations from device-busy intervals clipped to the measured step.

Follow-ups are small diagnostic guards, postprocessing and disclosure changes; production source and measured execution body remain unchanged. Ruff, formatting, diff checks, analysis controls, packet consistency and targeted reference checks pass. A second review was not required for these corrections.
