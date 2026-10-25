//! Reduction combiners (sum, max/min, Welford): identity, combine, plane
//! collectives, and the compensated sum.
use crate::ir::constant;
use cubecl_core::ir::{
    Scope,
    dialect::{
        cmp::{FEqualOp, FLessThanOrEqualOp, FMaxNanOp, FMinNanOp},
        general::SelectOp,
        math::{FAbsOp, FAddOp, FDivOp, FMulOp, FSubOp},
        memory::StoreOp,
    },
    pliron::value::Value as IrValue,
};
use serde_json::{Value, json};
type Error = Box<dyn std::error::Error>;

// Knuth TwoSum / Ogita-Rump-Oishi Sum2: retain the low-order addition
// residual separately. No reassociation or contraction permission is required.
// When ordinary addition becomes nonfinite, discard compensation so inf/NaN
// propagation follows ordinary addition rather than an inf-inf residual.
pub(crate) fn compensated_add(
    s: &Scope,
    a: IrValue,
    c: IrValue,
    b: IrValue,
) -> Result<(IrValue, IrValue), Error> {
    let t = s.register_with_result(&FAddOp::new(s.ctx_mut(), a, b));
    let z = s.register_with_result(&FSubOp::new(s.ctx_mut(), t, a));
    let tz = s.register_with_result(&FSubOp::new(s.ctx_mut(), t, z));
    let low_a = s.register_with_result(&FSubOp::new(s.ctx_mut(), a, tz));
    let low_b = s.register_with_result(&FSubOp::new(s.ctx_mut(), b, z));
    let residual = s.register_with_result(&FAddOp::new(s.ctx_mut(), low_a, low_b));
    let corrected = s.register_with_result(&FAddOp::new(s.ctx_mut(), c, residual));
    let magnitude = s.register_with_result(&FAbsOp::new(s.ctx_mut(), t));
    let finite = s.register_with_result(&FLessThanOrEqualOp::new(
        s.ctx_mut(),
        magnitude,
        constant(s, &json!(f32::MAX as f64), "float32")?,
    ));
    let correction = s.register_with_result(&SelectOp::new(
        s.ctx_mut(),
        finite,
        corrected,
        constant(s, &json!(0.0), "float32")?,
    ));
    Ok((t, correction))
}

/// A reduction type as its identity, ordered two-operand combine and
/// cross-lane step, after CubeK's `ReduceInstruction`. Values are slices of
/// `arity()` components; every site that once emitted a sum asks the
/// combiner, in the same order, so the sum's IR is unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Combiner {
    Sum,
    /// NaN-propagating, -0 < +0 (PTX max.NaN): exact under any order.
    Max,
    Min,
    /// Welford (mean, m2, weight) of one value per element.
    WelfordReduce,
    /// Welford of (mean, m2, weight) partials: split second stages, and
    /// masked first stages over (x, 0, 1).
    WelfordCombine,
}

impl Combiner {
    pub(crate) fn from_descriptor(op: &Value) -> Result<Self, Error> {
        match (
            op["reduction_type"].as_str(),
            op["accumulator_dtype"].as_str(),
        ) {
            (Some("sum"), Some("float32")) => Ok(Combiner::Sum),
            (Some("max"), Some("float32")) => Ok(Combiner::Max),
            (Some("min"), Some("float32")) => Ok(Combiner::Min),
            (Some("welford_reduce"), Some("float32")) => Ok(Combiner::WelfordReduce),
            (Some("welford_combine"), Some("float32")) => Ok(Combiner::WelfordCombine),
            (kind, accumulator) => Err(format!(
                "unsupported reduction operator/accumulator: {}/{}",
                kind.unwrap_or("missing"),
                accumulator.unwrap_or("missing")
            )
            .into()),
        }
    }

    pub(crate) fn arity(self) -> usize {
        match self {
            Combiner::Sum | Combiner::Max | Combiner::Min => 1,
            Combiner::WelfordReduce | Combiner::WelfordCombine => 3,
        }
    }

    /// Operands of the reduction op: values lifted together per element.
    pub(crate) fn operands(self) -> usize {
        match self {
            Combiner::WelfordCombine => 3,
            _ => 1,
        }
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            Combiner::Sum => "sum",
            Combiner::Max => "max",
            Combiner::Min => "min",
            Combiner::WelfordReduce => "welford_reduce",
            Combiner::WelfordCombine => "welford_combine",
        }
    }

    /// Plans whose arithmetic is specific to one combiner. `final_plane` is
    /// the frozen constant final9 template reached via `independent_planes`.
    pub(crate) fn supports(self, strategy: &str, final_plane: bool) -> bool {
        match self {
            Combiner::Sum => true,
            // Compensation, the reference512 partial and the final9 template
            // are qualified sum arithmetic.
            _ => {
                !final_plane
                    && !matches!(
                        strategy,
                        "shared_tree_compensated" | "plane_shared_reference512"
                    )
            }
        }
    }

    pub(crate) fn identity(self, s: &Scope) -> Result<Vec<IrValue>, Error> {
        match self {
            Combiner::Sum => Ok(vec![constant(s, &json!(0.0), "float32")?]),
            Combiner::Max => Ok(vec![constant(s, &json!("-inf"), "float32")?]),
            Combiner::Min => Ok(vec![constant(s, &json!("inf"), "float32")?]),
            Combiner::WelfordReduce | Combiner::WelfordCombine => {
                let zero = constant(s, &json!(0.0), "float32")?;
                Ok(vec![zero; 3])
            }
        }
    }

    /// Fresh locals, each initialised to its identity component; also returns
    /// the identity values for later neutral stores.
    pub(crate) fn identity_locals(
        self,
        s: &Scope,
        ty: cubecl_core::ir::pliron::r#type::TypeHandle,
    ) -> Result<(Vec<IrValue>, Vec<IrValue>), Error> {
        // Identity constants are created once, after the first local, which
        // is where the sum emitted its single constant.
        let mut locals = Vec::new();
        let mut identity = Vec::new();
        for i in 0..self.arity() {
            let local = s.create_local_mut(ty, None);
            if identity.is_empty() {
                identity = self.identity(s)?;
            }
            s.register(&StoreOp::new(s.ctx_mut(), local, identity[i]));
            locals.push(local);
        }
        Ok((locals, identity))
    }

    /// One reduced element, given as its `operands()` values, as
    /// accumulator components.
    pub(crate) fn lift(self, s: &Scope, values: &[IrValue]) -> Result<Vec<IrValue>, Error> {
        Ok(match self {
            Combiner::Sum | Combiner::Max | Combiner::Min => vec![values[0]],
            Combiner::WelfordReduce => vec![
                values[0],
                constant(s, &json!(0.0), "float32")?,
                constant(s, &json!(1.0), "float32")?,
            ],
            Combiner::WelfordCombine => values.to_vec(),
        })
    }

    /// The logical reduction result of a final accumulator.
    /// Multi-value results are read through their components; the
    /// reduction's own id names the first one.
    pub(crate) fn output(self, value: &[IrValue]) -> IrValue {
        value[0]
    }

    /// Add one element (its operand values) to a private accumulator.
    /// Welford uses Triton's incremental update (`triton_helpers.
    /// welford_reduce`): one division and fewer roundings than combining the
    /// lifted (x, 0, 1). Every other type combines the lifted element.
    pub(crate) fn accumulate(
        self,
        s: &Scope,
        old: &[IrValue],
        operands: &[IrValue],
    ) -> Result<Vec<IrValue>, Error> {
        if self != Combiner::WelfordReduce {
            let lifted = self.lift(s, operands)?;
            return Ok(self.combine(s, old, &lifted));
        }
        let (mean, m2, weight, x) = (old[0], old[1], old[2], operands[0]);
        let one = constant(s, &json!(1.0), "float32")?;
        let delta = s.register_with_result(&FSubOp::new(s.ctx_mut(), x, mean));
        let weight = s.register_with_result(&FAddOp::new(s.ctx_mut(), weight, one));
        let step = s.register_with_result(&FDivOp::new(s.ctx_mut(), delta, weight));
        let mean = s.register_with_result(&FAddOp::new(s.ctx_mut(), mean, step));
        let spread = s.register_with_result(&FSubOp::new(s.ctx_mut(), x, mean));
        let spread = s.register_with_result(&FMulOp::new(s.ctx_mut(), delta, spread));
        let m2 = s.register_with_result(&FAddOp::new(s.ctx_mut(), m2, spread));
        Ok(vec![mean, m2, weight])
    }

    pub(crate) fn combine(self, s: &Scope, lhs: &[IrValue], rhs: &[IrValue]) -> Vec<IrValue> {
        match self {
            Combiner::Sum => {
                vec![s.register_with_result(&FAddOp::new(s.ctx_mut(), lhs[0], rhs[0]))]
            }
            Combiner::Max => {
                vec![s.register_with_result(&FMaxNanOp::new(s.ctx_mut(), lhs[0], rhs[0]))]
            }
            Combiner::Min => {
                vec![s.register_with_result(&FMinNanOp::new(s.ctx_mut(), lhs[0], rhs[0]))]
            }
            Combiner::WelfordReduce | Combiner::WelfordCombine => welford_combine(s, lhs, rhs),
        }
    }

    /// The plane collective for this combiner. CubeCL's plane max/min use
    /// the NaN-ignoring max, so max/min take the explicit butterfly.
    /// Welford's combine is not bitwise commutative, so butterfly lanes end
    /// with different bits; lane 0's result, the one every writer stores,
    /// is broadcast so all units read the same value.
    pub(crate) fn plane(self, s: &Scope, value: &[IrValue]) -> Result<Vec<IrValue>, Error> {
        match self {
            Combiner::Sum => Ok(vec![s.register_with_result(
                &cubecl_core::ir::dialect::plane::FSumOp::new(s.ctx_mut(), value[0]),
            )]),
            Combiner::Max | Combiner::Min => self.butterfly(s, value.to_vec(), &[16, 8, 4, 2, 1]),
            Combiner::WelfordReduce | Combiner::WelfordCombine => {
                let value = self.butterfly(s, value.to_vec(), &[16, 8, 4, 2, 1])?;
                Ok(broadcast(s, &value))
            }
        }
    }

    /// Explicit shuffle-XOR butterfly over `distances`, combining with this
    /// combiner's own combine.
    pub(crate) fn butterfly(
        self,
        s: &Scope,
        mut value: Vec<IrValue>,
        distances: &[u32],
    ) -> Result<Vec<IrValue>, Error> {
        for distance in distances {
            let mask = constant(s, &json!(distance), "int32")?;
            let rhs: Vec<_> = value
                .iter()
                .map(|v| {
                    s.register_with_result(&cubecl_core::ir::dialect::plane::ShuffleXorOp::new(
                        s.ctx_mut(),
                        *v,
                        mask,
                    ))
                })
                .collect();
            value = self.combine(s, &value, &rhs);
        }
        Ok(value)
    }
}

/// Lane 0's value on every lane of the plane.
pub(crate) fn broadcast(s: &Scope, value: &[IrValue]) -> Vec<IrValue> {
    value
        .iter()
        .map(|v| {
            s.register_with_result(&cubecl_core::ir::dialect::plane::BroadcastOp::new(
                s.ctx_mut(),
                *v,
                cubecl_core::ir::attributes::IndexAttr::new(0),
            ))
        })
        .collect()
}

/// Chan's parallel Welford update in Triton's guarded form
/// (`triton_helpers.welford_combine`): equal means give delta 0 (equal
/// infinities, no inf - inf) and an empty pair gives ratio 0, so a finite
/// accumulator combined with the (0, 0, 0) identity is unchanged. An empty
/// lhs gives ratio exactly 1 (not Triton's w_b / w), so the identity on the
/// left is neutral too under approximate division. The m2 term keeps
/// Triton's left-to-right product, so non-power-of-two ratios round as
/// Triton's do, but its delta is 0 when either side is empty: that contributes
/// exactly 0 even when delta * delta would overflow (Triton's inf * 0 = NaN).
fn welford_combine(s: &Scope, a: &[IrValue], b: &[IrValue]) -> Vec<IrValue> {
    let f = |v: f64| constant(s, &json!(v), "float32").unwrap();
    let add = |x, y| s.register_with_result(&FAddOp::new(s.ctx_mut(), x, y));
    let mul = |x, y| s.register_with_result(&FMulOp::new(s.ctx_mut(), x, y));
    let select = |c, x, y| s.register_with_result(&SelectOp::new(s.ctx_mut(), c, x, y));
    let (mean_a, m2_a, w_a) = (a[0], a[1], a[2]);
    let (mean_b, m2_b, w_b) = (b[0], b[1], b[2]);
    let same = s.register_with_result(&FEqualOp::new(s.ctx_mut(), mean_a, mean_b));
    let difference = s.register_with_result(&FSubOp::new(s.ctx_mut(), mean_b, mean_a));
    let delta = select(same, f(0.0), difference);
    let weight = add(w_a, w_b);
    let empty = s.register_with_result(&FEqualOp::new(s.ctx_mut(), weight, f(0.0)));
    let ratio = s.register_with_result(&FDivOp::new(s.ctx_mut(), w_b, weight));
    let ratio = select(empty, f(0.0), ratio);
    // An empty lhs takes the rhs exactly: w_b / w_b under approximate
    // division can be 1 - ulp, which would drift the mean of constant rows.
    let lhs_empty = s.register_with_result(&FEqualOp::new(s.ctx_mut(), w_a, f(0.0)));
    let ratio = select(lhs_empty, f(1.0), ratio);
    let mean = add(mean_a, mul(delta, ratio));
    let rhs_empty = s.register_with_result(&FEqualOp::new(s.ctx_mut(), w_b, f(0.0)));
    let spread = select(lhs_empty, f(0.0), delta);
    let spread = select(rhs_empty, f(0.0), spread);
    let m2 = add(add(m2_a, m2_b), mul(mul(mul(spread, spread), w_a), ratio));
    vec![mean, m2, weight]
}

// compensated_add's finite correction bound relies on the <=1,048,576 extent
// enforced in stage5.artifact.Kernel._layout; this is not arbitrary-length Sum2.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn combiners_map_descriptor_types() {
        for (kind, expected, arity, operands) in [
            ("sum", Combiner::Sum, 1, 1),
            ("max", Combiner::Max, 1, 1),
            ("min", Combiner::Min, 1, 1),
            ("welford_reduce", Combiner::WelfordReduce, 3, 1),
            ("welford_combine", Combiner::WelfordCombine, 3, 3),
        ] {
            let op = json!({"reduction_type": kind, "accumulator_dtype": "float32"});
            let combiner = Combiner::from_descriptor(&op).unwrap();
            assert_eq!(combiner, expected);
            assert_eq!(combiner.name(), kind);
            assert_eq!(combiner.arity(), arity);
            assert_eq!(combiner.operands(), operands);
        }
        for op in [
            json!({"reduction_type": "prod", "accumulator_dtype": "float32"}),
            json!({"reduction_type": "max", "accumulator_dtype": "float16"}),
            json!({"reduction_type": "welford_reduce", "accumulator_dtype": "float64"}),
            json!({"accumulator_dtype": "float32"}),
        ] {
            let error = Combiner::from_descriptor(&op).unwrap_err().to_string();
            assert!(error.starts_with("unsupported reduction operator/accumulator: "));
        }
    }

    #[test]
    fn sum_only_plans_reject_other_combiners() {
        let strategies = [
            "shared_tree",
            "shared_tree_compensated",
            "plane_shared",
            "plane_shared_load_batch",
            "plane_shared_reference512",
            "independent_planes",
            "coalesced_columns",
        ];
        for strategy in strategies {
            assert!(Combiner::Sum.supports(strategy, false));
            let exact = !matches!(
                strategy,
                "shared_tree_compensated" | "plane_shared_reference512"
            );
            for other in [
                Combiner::Max,
                Combiner::Min,
                Combiner::WelfordReduce,
                Combiner::WelfordCombine,
            ] {
                assert_eq!(other.supports(strategy, false), exact, "{strategy}");
                assert!(!other.supports("independent_planes", true));
            }
        }
        assert!(Combiner::Sum.supports("independent_planes", true));
    }
}
