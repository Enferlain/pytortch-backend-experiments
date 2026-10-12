# Stage 2 review disposition

The required review read the tracked diff and the untracked Stage 2 implementation,
report and retained acceptance evidence. It found no wrong-result bug on the
executed surface. The focused follow-up review verified all four fixes and found no indexing/domain
regression. Its remaining low-severity modulus check was fixed with a neighboring
rejection test; this small correction does not require another review.

| Finding | Disposition | Validation |
| --- | --- | --- |
| Masked subblocks lacked the operation/dtype whitelist | Fixed: the same check runs recursively during the pre-fusion type walk | Real `cat([tan(x), x[:1]])+1` rejects before fusion; captured live lowering confirms `tan` is in a masked subblock; zero compilations/launches |
| Inherited prepare/capture APIs unpacked a wrong metadata key | Fixed: Stage 2 rejects both unsupported APIs explicitly | Component calls receive `NotImplementedError` before metadata preparation |
| Rust divisor validation lacked positivity | Fixed: schema-2 scalar declarations retain positive domains; Rust proves constant/scalar/Add/Mul divisor provenance; runtime enforces declared-positive scalar values | Rust rejects zero, negative and undeclared-positive divisors before IR; runtime divisors 0/-7 reject before launch; divisor 7 produces expected floor results |
| Example metadata registration order differed | Fixed: example uses runtime registration order | C++ compiler probe and all regression suites pass |
| ModularIndexing modulus lacked an early positivity check | Fixed: encoder rejects an unproven-positive modulus before descriptor/IR emission | Index component checks unknown modulus and negative FloorDiv divisor rejection; accepted broadcast/mask matrix rerun |
| Pinned CUDA header location | Retained experimental environment requirement | Documented in README/report; all three floating storage types compile |
| Metadata cache has no eviction | Retained baseline lifetime policy | Report explicitly records unbounded growth for unbounded shape variation |
| Symbolic stride validation creates guards | Retained intended nonnegative-stride contract | Added source comment; dynamic stride tests and guard evidence pass |

Final regression: 34 Stage 2 category checks, 94 compiler invocations,
165 Python/native kernel launches, no failures. Components pass signed-i64
boundary, nested lazy load/store, empty extent and new divisor/capture tests.
Stage 0 C++ and LLVM, Stage 1 acceptance and Stage 1.5 boundary suites all pass
in fresh processes. The two Inductor baseline suites were rerun sequentially after
an initially concurrent overlay-install setup race; concurrent overlay setup is
outside the experiment's contract and did not reach compilation or execution.
No Triton Python/native module loaded; CPython 3.14t remained GIL-disabled.
Ruff, Python syntax, rustfmt, Clippy and C warnings-as-errors pass.
