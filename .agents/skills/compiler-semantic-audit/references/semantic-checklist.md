# Semantic proof checklist
Use only obligations supported by the governing contract; mark irrelevant, permitted variation or unspecified behavior explicitly. This is a test-design checklist, not a universal specification.

## Value semantics
| Topic | Establish before accepting a transformation | Discriminating cases |
| --- | --- | --- |
| NaNs | Propagation/classification; quiet/signaling support; quieting, payload/sign preservation, exceptions if observable | qNaN/sNaN only when representable/relevant, each operand position, constants/runtime, scalar/vector |
| Infinities | Domain restrictions and results/exception behavior | ±Inf with zero, ±Inf and finite extrema |
| Signed zero | Whether sign is observable and which rounding rules apply | ±0 in both operands, sign-sensitive consumers |
| Subnormals | Input/output denorm modes and where FTZ is introduced | Smallest/largest subnormal, underflow and normal boundary |
| Reassociation | Exact allowed reordering, precision and domain | Cancellation, huge/tiny mixed terms, reduction trees |
| Contraction/FMA | Separate versus single rounding, explicit FMA versus permitted contraction | Product near rounding boundary, cancellation/overflow |
| Reciprocal/division | Approximation error/domain/rounding, division by zero | Zero divisor, negative operands, extrema and subnormals |
| Remainder/modulo | Truncating, floor or other quotient convention, signed zero, zero divisor | Negative lhs/rhs, nonintegral lhs, x % 1, extremes |
| Min/max/clamp | NaN propagation versus number preference, zero ties, argument-order symmetry | One/both NaNs, ±0, reversed bounds if in domain |
| Comparisons | Ordered/unordered predicates, equality and mask effects | NaNs, signed zero, infinities |
| Casts/conversions | Rounding/saturation, width/sign, intermediate precision, out-of-range behavior | Boundary values, double rounding, NaN/Inf, bf16/f16/fp8 where supported |
| Integers | Wrapping, trapping, saturation, poison/UB; signed/unsigned widths; shifts/division/remainder | INT_MIN/-1, zero divisor, overflow, oversized/negative shifts |
| Reduced math modes | Actual per-op assumptions and downstream flags | Same supported input under strict/relaxed variants; excluded inputs are not conformance failures |

For NaN payloads or signaling exceptions, first establish whether that layer/device exposes a requirement at all. An unavailable floating-point exception model is not permission to erase observable value classification. Avoid assuming bitwise equality for a contract allowing numeric tolerance; conversely tolerance must not hide signed-zero/class differences.

## Memory and execution
- **Atomics:** verify operation, dtype/width, alignment, return value, memory order/scope, visibility, forward progress and atomic reduction order. Do not transfer C++ or PTX ordering guarantees to a frontend automatically. Distinguish race-free atomic updates from deterministic sums.
- **Memory ordering:** distinguish fences/barriers/atomics, participating lanes and convergence, cross-workgroup visibility, execution dependencies and async completion. A fence is not necessarily an execution barrier.
- **Aliasing:** prove or test input/output overlap, view aliasing, in-place updates, restrict/noalias assumptions and buffer lifetime. Check whether fusion moves loads across writes.
- **Address spaces:** track global/shared/local/private and pointer width/provenance/alignment/bounds; do not equate CUDA local memory with portable private storage without mapping.
- **Vectorization/layout:** preserve element-to-lane/index correspondence, alignment, memory effects, tail handling, padding, transposes and inter-lane communication. Check scalar/vector constants and result shape; zero tensor and scalar zero are different materializations.
- **Dynamic shapes/strides:** audit guards, specialization/caching keys, zero/one extents, overflow in index arithmetic, noncontiguous layouts and repeated runs with changing extents. Include address-width constraints.
- **Masking/speculation:** distinguish guarded access from computing both operands then selecting. Audit invalid masked-off loads, division, conversions and poison; inactive lanes must not acquire effects beyond the contract.
- **Reductions:** establish accumulation dtype/tree, identity/empty input, tail masks, overflow, order/determinism/tolerances and backward. Fusion may change reassociation, memory traffic and mutation ordering.
- **Training/framework:** preserve dtype promotion, broadcasting/alpha, aliases/mutation, gradients and saved tensor semantics. Burn and PyTorch wrappers can choose different defaults without changing a common low-level instruction.

## Proof record
For a candidate contract requiring round-to-nearest addition and observable signed zero, test runtime `x = -0.0` with constant `+0.0`: addition gives `+0.0`, while replacement with x gives `-0.0`. This is a conditional counterexample, not proof that CubeCL has that contract. Check the governing rounding rule and permitted observables first. Ignoring zero signs alone does not settle NaN quieting or denormal obligations.

Use one row per obligation:
`contract evidence → rewrite/lowering precondition → actual enforcement → counterexample or proof → test → conclusion`.
A source comment claiming “pure” or “speculatable” is an assertion to audit, not proof that all exceptional or side-effect behavior disappears.

## Comparing accuracy with a reference implementation
When the obligation is "no less accurate than reference R", measure each output unit (for example each reduced row) against a higher-precision oracle for both implementations on identical inputs and settings. Then decide with paired tests, not aggregate ratios:
- A sign test over units whose errors differ catches consistent small losses.
- A paired mean-difference test catches rare large ones, which a sign or rank test can outvote.

Avoid tolerance margins on "not worse" claims: a margin reads as permission to be slightly worse and tends to widen.

Reduction edge cases worth testing explicitly:
- **Approximate division in combines.** `w / w` may not be exactly 1, so identities must be guarded rather than trusted.
- **Overflow inside a combine term.** A product can overflow against an empty partial and give inf·0 = NaN (for example `delta*delta*w*ratio`).
- **Association with non-power-of-two weight ratios.** Row tails make association observable.
- **Constant and huge-constant rows shorter than the lane count.** These exercise empty partials.
- **Kernels that fuse two reduction families.** A plan qualified for one family must not apply silently to the other's arithmetic.
