Bounded follow-up to review-initial.json; do not scan the whole artifact tree.

Initial findings 1–3: removed unreachable mode; analyze() now requires equal
metric lists, interval, sampler binary/source, clock method, reset, harness and
primer-iteration metadata within a comparison; verifies archived literal SOURCE,
single store substitution, wrapper self-hash and the wb/wt PTX operator.
11 validation controls exercise positive admission, config/harness mismatches,
a policy tamper with consistent JSON, unsupported selector/reset, and four
invalid FFI states. Full corrected dataset reanalysis still validates all96
captures; summary-reviewed.log retains the unchanged numeric results.

Finding4: future store controls archive original_argv and rewritten_argv;
legacy captures remain immutable, with full command in qualified-commands.json.
Finding5: future decode errors save the partial image and decode status before
being rejected; original failures remain as originally captured.
Finding6: null/uninitialized clock entry points fail without crossing the C ABI;
PM reinitialization is explicitly rejected. No sampler success algorithm change.
Finding7: analyzer explicitly admits only dense read_write/write_read selectors.
Finding8: README now describes wb as an intrinsic plumbing control rather than
semantic equivalence to the original volatile store.
Finding9: future no-PM counter-harness captures also archive the sampler binary
hash/source; removed unused groups field. stats() is intentionally imported by
the retained summarize.py, so it is a used public summary helper.

New GPU work was one reviewed-smoke pair with frozen target/input/output bits,
using the hardened sampler. It is validation, not a new performance conclusion.
All earlier captures keep their original harness/source/binary identities.

Review only these fixes and any material newly introduced issue in the four
listed source files. Read validator-controls.json and reviewed-smoke-analysis.json
if needed. No GPU execution, edits or native agents. The repository runtime and
all reduction kernels remain byte-frozen. Initial review confirmed the core
C10 calibration and PM capture design; the current pass already documents the
device-wide counter limitation and unproved dirty-cache cause.
