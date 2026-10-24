//! Semantic sum DAG + independently validated cooperative physical plan.
//! No handwritten target source: all loops, shared accesses and barriers are IR.
use crate::descriptor2::{check_index, constant, elem, emit, index, validate_ops};
use cubecl_core::{
    compute::KernelBuilder,
    ir::{
        AddressType, Builtin, OpInserter, Scope,
        dialect::{
            branch::{IfOp, RangeLoopOp, ReturnOp},
            cmp::{FEqualOp, FLessThanOrEqualOp, FMaxNanOp, FMinNanOp, IEqualOp, SLessThanOp},
            general::{CastOp, ReadBuiltinOp, ReadScalarOp, SelectOp},
            math::{FAbsOp, FAddOp, FDivOp, FMulOp, FSubOp, IAddOp, IMulOp, SDivOp, SRemOp},
            memory::{IndexOp, LoadOp, StoreOp},
            synchronization::{SyncOp, SyncScope, SyncScopeAttr},
        },
        pliron::value::Value as IrValue,
        settings::{Dim3, ExecutionMode, KernelSettings},
        types::{ArrayType, scalar::IndexType},
    },
};
use serde_json::{Value, json};
use std::collections::{HashMap, HashSet};
type Error = Box<dyn std::error::Error>;

// Knuth TwoSum / Ogita-Rump-Oishi Sum2: retain the low-order addition
// residual separately. No reassociation or contraction permission is required.
// When ordinary addition becomes nonfinite, discard compensation so inf/NaN
// propagation follows ordinary addition rather than an inf-inf residual.
fn compensated_add(
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
            Combiner::WelfordReduce | Combiner::WelfordCombine => Ok(self
                .butterfly(s, value.to_vec(), &[16, 8, 4, 2, 1])?
                .into_iter()
                .map(|v| {
                    s.register_with_result(&cubecl_core::ir::dialect::plane::BroadcastOp::new(
                        s.ctx_mut(),
                        v,
                        cubecl_core::ir::attributes::IndexAttr::new(0),
                    ))
                })
                .collect()),
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

/// Chan's parallel Welford update in Triton's guarded form
/// (`triton_helpers.welford_combine`): equal means give delta 0 (equal
/// infinities, no inf - inf) and an empty pair gives ratio 0, so a finite
/// accumulator combined with the (0, 0, 0) identity is unchanged. An empty
/// lhs gives ratio exactly 1 (not Triton's w_b / w), so the identity on the
/// left is neutral too under approximate division. The m2
/// term is `delta * (delta * (w_a * r))` rather than Triton's left-to-right
/// product: the same operations, but an empty lhs contributes exactly 0
/// even when delta * delta would overflow.
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
    let m2 = add(add(m2_a, m2_b), mul(delta, mul(delta, mul(w_a, ratio))));
    vec![mean, m2, weight]
}

// Physical coordinates are bound only here. A semantic reduction coordinate is
// mapped to an extra emitter value slot, never to a runtime metadata argument.
fn bind_coordinate(v: &Value, slot: usize) -> Value {
    if v["op"] == "reduction" && v.get("dtype").is_none() {
        return json!({"op":"scalar", "slot":slot});
    }
    match v {
        Value::Array(a) => Value::Array(a.iter().map(|x| bind_coordinate(x, slot)).collect()),
        Value::Object(o) => Value::Object(
            o.iter()
                .map(|(k, v)| (k.clone(), bind_coordinate(v, slot)))
                .collect(),
        ),
        _ => v.clone(),
    }
}

fn sync(s: &Scope) {
    s.register(&SyncOp::new(
        s.ctx_mut(),
        SyncScopeAttr::new(SyncScope::Cube),
    ));
}

fn pointer(s: &Scope, array: IrValue, coordinate: IrValue) -> IrValue {
    let index = s.register_with_result(&CastOp::new(
        s.ctx_mut(),
        IndexType::get(s.ctx()).to_handle(),
        coordinate,
    ));
    s.register_with_result(&IndexOp::new(s.ctx_mut(), array, index, None))
}

fn load_all(s: &Scope, locals: &[IrValue]) -> Vec<IrValue> {
    locals
        .iter()
        .map(|l| s.register_with_result(&LoadOp::new(s.ctx_mut(), *l)))
        .collect()
}

fn store_all(s: &Scope, locals: &[IrValue], values: &[IrValue]) {
    for (l, v) in locals.iter().zip(values) {
        s.register(&StoreOp::new(s.ctx_mut(), *l, *v));
    }
}

fn branch<F>(s: &Scope, cond: IrValue, body: F) -> Result<(), Error>
where
    F: FnOnce(&Scope) -> Result<(), Error>,
{
    let op = IfOp::new(s.ctx_mut(), cond);
    let child = s.branch_child(OpInserter::new_at_block_end(op.then_block(s.ctx())));
    body(&child)?;
    child.terminate_yield();
    s.branch_child(OpInserter::new_at_block_end(op.else_block(s.ctx())))
        .terminate_yield();
    s.register(&op);
    Ok(())
}

fn dependencies(id: &str, ops: &[Value], ready: &HashSet<String>, needed: &mut HashSet<String>) {
    if ready.contains(id) || !needed.insert(id.into()) {
        return;
    }
    if let Some(op) = ops.iter().find(|op| op["id"] == id) {
        for a in op["args"].as_array().unwrap() {
            dependencies(a.as_str().unwrap(), ops, ready, needed);
        }
        if let Some(result) = op["result"].as_str() {
            dependencies(result, ops, ready, needed);
        }
        // Mask bodies may refer to enclosing SSA values. Local body ids are
        // emitted by the shared lazy-region emitter, not extracted from it.
        fn region(v: &Value, ops: &[Value], ready: &HashSet<String>, needed: &mut HashSet<String>) {
            if let Some(a) = v["args"].as_array() {
                for x in a {
                    dependencies(x.as_str().unwrap(), ops, ready, needed);
                }
            }
            if let Some(result) = v["result"].as_str() {
                dependencies(result, ops, ready, needed);
            }
            if let Some(body) = v["body"].as_array() {
                for x in body {
                    region(x, ops, ready, needed);
                }
            }
        }
        if let Some(body) = op["body"].as_array() {
            for v in body {
                region(v, ops, ready, needed);
            }
        }
    }
}

/// The ops computing `ids`, in descriptor order. Reductions and their
/// components are values in `ready` once reduced, never re-emitted.
fn closure(ids: &[&str], ops: &[Value], ready: &HashMap<String, IrValue>) -> Vec<Value> {
    let mut needed = HashSet::new();
    let ready = ready.keys().cloned().collect();
    for id in ids {
        dependencies(id, ops, &ready, &mut needed);
    }
    ops.iter()
        .filter(|op| {
            !matches!(op["op"].as_str(), Some("reduction" | "reduction_component"))
                && op["id"].as_str().is_some_and(|id| needed.contains(id))
        })
        .cloned()
        .collect()
}

fn operand_ids(op: &Value) -> Vec<&str> {
    op["args"]
        .as_array()
        .unwrap()
        .iter()
        .map(|a| a.as_str().unwrap())
        .collect()
}

fn contains_coordinate(v: &Value) -> bool {
    if v["op"] == "reduction" && v.get("dtype").is_none() {
        return true;
    }
    match v {
        Value::Array(a) => a.iter().any(contains_coordinate),
        Value::Object(o) => o.values().any(contains_coordinate),
        _ => false,
    }
}

fn region_has_effect(v: &Value) -> bool {
    if matches!(v["op"].as_str(), Some("store" | "reduction")) && v.get("dtype").is_some() {
        return true;
    }
    v["body"]
        .as_array()
        .is_some_and(|a| a.iter().any(region_has_effect))
}

pub(crate) fn validate_description(d: &Value) -> Result<(), Error> {
    if d["schema"] != 3 || d["index_dtype"] != "int64" {
        return Err("unsupported reduction schema".into());
    }
    let buffers = d["buffers"].as_array().ok_or("buffers missing")?;
    if !(2..=5).contains(&buffers.len()) {
        return Err("buffer count unsupported".into());
    }
    let defs = d["scalars"]
        .as_array()
        .ok_or("scalar definitions missing")?;
    if defs.len() > 32 {
        return Err("scalar limit".into());
    }
    for (i, v) in defs.iter().enumerate() {
        if !(v["dtype"] == "int64" || v["dtype"] == "float64" && v["positive"] == false)
            || v["id"] != i
            || !v["positive"].is_boolean()
        {
            return Err("scalar schema".into());
        }
    }
    check_index(&d["outer"], defs, false)?;
    check_index(&d["reduction_extent"], defs, false)?;
    if d["numel"] != d["outer"] {
        return Err("outer logical domain mismatch".into());
    }
    for (i, b) in buffers.iter().enumerate() {
        if b["access"]
            != if i == buffers.len() - 1 {
                "write"
            } else {
                "read"
            }
            || !matches!(
                b["dtype"].as_str(),
                Some("float32" | "float16" | "bfloat16" | "bool")
            )
        {
            return Err("storage contract".into());
        }
        let size = b["size"].as_array().ok_or("size missing")?;
        let stride = b["stride"].as_array().ok_or("stride missing")?;
        if size.len() != stride.len() || size.len() > crate::descriptor2::MAX_RANK {
            return Err("layout rank".into());
        }
        for v in size
            .iter()
            .chain(stride)
            .chain(std::iter::once(&b["offset"]))
        {
            check_index(v, defs, false)?;
        }
    }
    let ops = d["ops"].as_array().ok_or("operations missing")?;
    // Runtime symbols and the private reduction coordinate are distinct even
    // before the physical adapter binds the latter into an emitter argument.
    fn runtime_slots(v: &Value, count: usize) -> Result<(), Error> {
        if v["op"] == "scalar" && v["slot"].as_u64().ok_or("scalar slot")? >= count as u64 {
            return Err(
                "invalid runtime scalar slot; reduction coordinate must be explicit".into(),
            );
        }
        match v {
            Value::Array(a) => {
                for x in a {
                    runtime_slots(x, count)?;
                }
            }
            Value::Object(o) => {
                for x in o.values() {
                    runtime_slots(x, count)?;
                }
            }
            _ => {}
        }
        Ok(())
    }
    runtime_slots(&d["ops"], defs.len())?;
    let mut bound = bind_coordinate(&Value::Array(ops.clone()), defs.len())
        .as_array()
        .unwrap()
        .clone();
    let mut extended_defs = defs.clone();
    extended_defs.push(json!({"dtype":"int64","positive":false}));
    let mut reductions = 0;
    let mut combiners = HashMap::new();
    for op in &mut bound {
        if op["op"] == "reduction" {
            reductions += 1;
            let combiner = Combiner::from_descriptor(op)?;
            if !matches!(combiner.arity(), 1 | 3) {
                return Err("multi-value reduction accumulators unsupported".into());
            }
            let operands = op["args"].as_array().ok_or("operand list")?.len();
            if operands != combiner.operands() {
                return Err(format!(
                    "reduction type {} takes {} operand(s), got {}",
                    combiner.name(),
                    combiner.operands(),
                    operands
                )
                .into());
            }
            combiners.insert(
                op["id"].as_str().ok_or("reduction id")?.to_string(),
                combiner,
            );
            if op["dtype"] != "float32"
                || !matches!(
                    op["source_dtype"].as_str(),
                    Some("float32" | "float16" | "bfloat16")
                )
            {
                return Err("unsupported reduction operator/accumulator".into());
            }
            if !matches!(
                op["semantic_dtype"].as_str(),
                Some("float32" | "float16" | "bfloat16")
            ) {
                return Err("reduction result semantic dtype".into());
            }
            // Type-check the same SSA edge as a float32 identity; no IR is built
            // until every scalar op and every reduction record is validated.
            // Every operand is checked as an explicit float32 conversion below.
            op["op"] = "cast".into();
            op["args"] = json!([op["args"][0]]);
        } else if op["op"] == "reduction_component" {
            // One component of an earlier multi-value reduction, typed float32.
            let source = op["args"]
                .as_array()
                .filter(|a| a.len() == 1)
                .and_then(|a| a[0].as_str())
                .ok_or("reduction component requires one reduction operand")?;
            let combiner = combiners
                .get(source)
                .ok_or("reduction component of an unknown or later reduction")?;
            let index = op["index"].as_u64().ok_or("reduction component index")?;
            if combiner.arity() == 1 || index >= combiner.arity() as u64 {
                return Err(format!(
                    "reduction type {} has no component {index}",
                    combiner.name()
                )
                .into());
            }
            if op["dtype"] != "float32" || op["semantic_dtype"] != "float32" {
                return Err("reduction component dtype".into());
            }
            op["op"] = "cast".into();
        } else if op["op"] == "store"
            && !matches!(op["iteration_scope"].as_str(), Some("outer" | "reduction"))
        {
            return Err("store iteration scope missing".into());
        }
        if let Some(body) = op["body"].as_array()
            && body.iter().any(region_has_effect)
        {
            return Err("masked reduction/effect unsupported".into());
        }
    }
    if !(1..=4).contains(&reductions) {
        return Err("one to four reductions per cooperative schedule required".into());
    }
    let mut types = HashMap::new();
    for (original, checked) in ops.iter().zip(&bound) {
        if original["op"] == "reduction" {
            for argument in original["args"].as_array().ok_or("operand list")? {
                let argument = argument.as_str().ok_or("reduction operand missing")?;
                if types.get(argument).map(String::as_str) != Some("float32") {
                    return Err("reduction requires explicit float32 input conversion".into());
                }
            }
        }
        validate_ops(
            std::slice::from_ref(checked),
            buffers,
            &extended_defs,
            &mut types,
            0,
        )?;
    }
    let reduced_ids: HashSet<_> = ops
        .iter()
        .filter(|op| matches!(op["op"].as_str(), Some("reduction" | "reduction_component")))
        .map(|op| op["id"].as_str().unwrap().to_string())
        .collect();
    for store in ops
        .iter()
        .filter(|op| op["op"] == "store" && op["iteration_scope"] == "outer")
    {
        let mut needed = HashSet::new();
        dependencies(
            store["args"][0].as_str().unwrap(),
            ops,
            &reduced_ids,
            &mut needed,
        );
        if contains_coordinate(&store["index"])
            || ops.iter().any(|op| {
                op["id"].as_str().is_some_and(|id| needed.contains(id)) && contains_coordinate(op)
            })
        {
            return Err("outer store depends on a varying reduction coordinate".into());
        }
    }

    Ok(())
}

pub fn build(d: &Value) -> Result<cubecl_runtime::kernel::KernelDefinition, Error> {
    if d["schema"] != 3 || d["index_dtype"] != "int64" {
        return Err("unsupported reduction schema".into());
    }
    let expected = json!({"iteration_mapping":"one_workgroup_per_outer", "workgroup":[256,1,1],
        "elements_per_unit":"strided_loop_step_workgroup", "local_vector_width":1,
        "reduction_mapping":"shared_binary_tree", "accumulator_dtype":"float32", "shared_elements":256,
        "barriers":"uniform_after_publish_and_each_tree_level", "result_owner":"unit_zero_store_all_units_read",
        "bounds_strategy":"loop_reduction_coordinate_less_than_extent", "phases":1});
    if d["physical_plan"] != expected {
        return Err(
            "unsupported physical reduction topology (including divergent barriers)".into(),
        );
    }
    let entry = d["entrypoint"].as_str().ok_or("entry missing")?;
    if !entry.starts_with("cubecl3_")
        || !entry.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
    {
        return Err("entry invalid".into());
    }
    validate_description(d)?;
    build_physical(d, [256, 1, 1], "shared_tree", 1, 1, "halving")
}

pub(crate) fn build_physical(
    d: &Value,
    workgroup: [u32; 3],
    strategy: &str,
    vector_width: usize,
    private_accumulators: usize,
    private_merge: &str,
) -> Result<cubecl_runtime::kernel::KernelDefinition, Error> {
    build_physical_codegen(
        d,
        workgroup,
        strategy,
        vector_width,
        private_accumulators,
        private_merge,
        CodegenOptions {
            uniform_loop: strategy == "plane_shared_reference512",
            ..Default::default()
        },
    )
}

#[derive(Clone, Copy, Default)]
pub(crate) struct CodegenOptions {
    pub uniform_loop: bool,
    pub owner_result: bool,
    pub narrow_loop: bool,
    pub final_plane: bool,
    pub final_load_batch: bool,
}

/// The qualified reference512 path translates CubeK's uniform chunk loop.
/// Other options are offline probes, never selected by the physical planner.
/// No chain coordinate, protected body, or accumulator-update order changes.
pub(crate) fn build_physical_codegen(
    d: &Value,
    workgroup: [u32; 3],
    strategy: &str,
    vector_width: usize,
    private_accumulators: usize,
    private_merge: &str,
    codegen: CodegenOptions,
) -> Result<cubecl_runtime::kernel::KernelDefinition, Error> {
    let CodegenOptions {
        uniform_loop,
        owner_result,
        narrow_loop,
        final_plane,
        final_load_batch,
    } = codegen;
    if (uniform_loop || owner_result || narrow_loop) && strategy != "plane_shared_reference512" {
        return Err("codegen probe requires the frozen reference512 plan".into());
    }
    if final_load_batch && !final_plane {
        return Err("final load batching requires the final-plane probe".into());
    }
    if final_plane
        && (strategy != "independent_planes"
            || workgroup != [32, 4, 1]
            || vector_width != 1
            || private_accumulators != 1
            || d["reduction_extent"]["op"] != "constant"
            || d["reduction_extent"]["value"]
                .as_u64()
                .is_none_or(|extent| !(1..=32).contains(&extent))
            || d["ops"].as_array().is_none_or(|ops| {
                ops.iter().filter(|op| op["op"] == "reduction").count() != 1
                    || ops
                        .iter()
                        .any(|op| op["op"] == "store" && op["iteration_scope"] != "outer")
            }))
    {
        return Err(
            "final-plane probe requires the small owner-only independent-plane plan (extent 1..32)"
                .into(),
        );
    }
    if private_merge == "adjacent" && private_accumulators != 4 {
        return Err("adjacent private merge requires four accumulators".into());
    }
    validate_description(d)?;

    let entry = d["entrypoint"].as_str().ok_or("entry missing")?;
    let buffers = d["buffers"].as_array().unwrap();
    let defs = d["scalars"].as_array().unwrap();
    let ops = d["ops"].as_array().unwrap();
    let mut shared_arity = 1;
    for op in ops.iter().filter(|op| op["op"] == "reduction") {
        let combiner = Combiner::from_descriptor(op)?;
        if !combiner.supports(strategy, final_plane) {
            return Err(format!(
                "strategy {} does not support reduction type {}",
                if final_plane {
                    "independent_planes_final9"
                } else {
                    strategy
                },
                combiner.name()
            )
            .into());
        }
        shared_arity = shared_arity.max(combiner.arity());
    }
    if owner_result
        && (ops.iter().filter(|op| op["op"] == "reduction").count() != 1
            || ops
                .iter()
                .any(|op| op["op"] == "store" && op["iteration_scope"] != "outer"))
    {
        return Err("owner-result probe requires one reduction and only owner stores".into());
    }
    let mut b = KernelBuilder::new(
        KernelSettings::new(
            Dim3::new_3d(workgroup[0], workgroup[1], workgroup[2]),
            ExecutionMode::Unchecked,
            AddressType::U64,
        )
        .kernel_name(entry),
    );
    let arrays: Vec<_> = buffers
        .iter()
        .map(|v| b.buffer(elem(v["dtype"].as_str().unwrap()).unwrap().to_type(b.ctx())))
        .collect();
    let ity = elem("int64")?.to_type(b.ctx());
    let fty = elem("float32")?.to_type(b.ctx());
    let scalars: Vec<_> = (0..=defs.len())
        .map(|_| {
            let id = b.scalar(elem("int64").unwrap());
            b.register_with_result(&ReadScalarOp::new(b.ctx_mut(), ity, id))
        })
        .collect();
    let builtin = |kind| {
        let x = b.register_with_result(&ReadBuiltinOp::new(
            b.ctx_mut(),
            IndexType::get(b.ctx()).to_handle(),
            kind,
        ));
        b.register_with_result(&CastOp::new(b.ctx_mut(), ity, x))
    };
    let cube = builtin(Builtin::CubePosX);
    let lane = builtin(Builtin::UnitPosX);
    let column = strategy == "coalesced_columns";
    let planes = strategy == "independent_planes";
    let unit = if column {
        builtin(Builtin::UnitPosY)
    } else {
        lane
    };
    let outputs = if column {
        workgroup[0]
    } else if planes {
        workgroup[1]
    } else {
        1
    };
    let lane_outer = if column {
        lane
    } else if planes {
        builtin(Builtin::UnitPosY)
    } else {
        constant(&b, &json!(0), "int64")?
    };
    let scale = constant(&b, &json!(outputs), "int64")?;
    let outer = b.register_with_result(&IMulOp::new(b.ctx_mut(), cube, scale));
    let outer = b.register_with_result(&IAddOp::new(b.ctx_mut(), outer, lane_outer));
    let cooperation = if column { workgroup[1] } else { workgroup[0] };
    let zero = constant(&b, &json!(0), "int64")?;
    let step = constant(&b, &json!(cooperation), "int64")?;
    let outer_extent = index(&b, &d["outer"], zero, &scalars[1..])?;
    let reduction_extent = index(&b, &d["reduction_extent"], zero, &scalars[1..])?;
    let reference512 = strategy == "plane_shared_reference512";
    let batched_loads = strategy == "plane_shared_load_batch" || reference512;
    let plane_shared = strategy == "plane_shared" || batched_loads;
    let compensated = strategy == "shared_tree_compensated";
    let shared_size = if plane_shared {
        workgroup[0] / 32
    } else {
        workgroup[0] * workgroup[1] * if compensated { 2 } else { 1 }
    };
    // One scratch array per accumulator component.
    let shared: Vec<_> = if planes {
        Vec::new()
    } else {
        (0..shared_arity)
            .map(|_| b.create_shared(ArrayType::get(b.ctx(), fty, shared_size as usize), None))
            .collect()
    };
    let shared_index = |s: &Scope, r: IrValue| {
        let stride = constant(s, &json!(outputs), "int64").unwrap();
        let scaled = s.register_with_result(&IMulOp::new(s.ctx_mut(), r, stride));
        s.register_with_result(&IAddOp::new(s.ctx_mut(), scaled, lane_outer))
    };
    let valid = b.register_with_result(&SLessThanOp::new(b.ctx_mut(), outer, outer_extent));
    // CubePosX and runtime extents are workgroup-uniform; no unit predicate
    // encloses a barrier. Reduction loops and masked bodies contain no barriers.
    let collective_guard = if column {
        constant(&b, &json!(true), "bool")?
    } else {
        valid
    };
    branch(&b, collective_guard, |s| {
        let mut ready = HashMap::new();
        let bound_ops = bind_coordinate(&Value::Array(ops.clone()), defs.len())
            .as_array()
            .unwrap()
            .clone();
        let reductions: Vec<_> = bound_ops
            .iter()
            .filter(|op| op["op"] == "reduction")
            .collect();
        for (reduction_index, op) in reductions.iter().enumerate() {
            let combiner = Combiner::from_descriptor(op)?;
            let correction = compensated.then(|| s.create_local_mut(fty, None));
            if let Some(c) = correction {
                s.register(&StoreOp::new(
                    s.ctx_mut(),
                    c,
                    constant(s, &json!(0.0), "float32")?,
                ));
            }
            let partial = if private_accumulators > 1 {
                // ATen's independent strided chains, scalar loads, followed by
                // a balanced private merge. This is a physical choice only;
                // logical replay and all semantic/storage barriers are reused.
                let accumulators = (0..private_accumulators)
                    .map(|_| combiner.identity_locals(s, fty).map(|(locals, _)| locals))
                    .collect::<Result<Vec<_>, _>>()?;
                let count = constant(s, &json!(private_accumulators), "int64")?;
                let advance = s.register_with_result(&IMulOp::new(s.ctx_mut(), step, count));
                let (loop_start, loop_end, loop_step) = if narrow_loop {
                    // Existing Stage5 runtime validation proves 0 <= E <= 2^20.
                    // t <= 511, step=2048, and the final increment is <2^21.
                    // Only this physical induction variable narrows. Semantic
                    // coordinates and every pointer expression remain signed i64.
                    let ty = cubecl_core::ir::ElemType::UInt(cubecl_core::ir::UIntKind::U32)
                        .to_type(s.ctx());
                    let cast = |v| s.register_with_result(&CastOp::new(s.ctx_mut(), ty, v));
                    (
                        cast(if uniform_loop { zero } else { unit }),
                        cast(reduction_extent),
                        cast(advance),
                    )
                } else {
                    (
                        if uniform_loop { zero } else { unit },
                        reduction_extent,
                        advance,
                    )
                };
                let loop_op = RangeLoopOp::new(s.ctx_mut(), loop_start, loop_end, loop_step);
                let iteration = loop_op.iter_var(s.ctx());
                let child =
                    s.branch_child(OpInserter::new_at_block_end(loop_op.loop_body(s.ctx())));
                let iteration = if narrow_loop {
                    child.register_with_result(&CastOp::new(child.ctx_mut(), ity, iteration))
                } else {
                    iteration
                };
                let base = if uniform_loop {
                    child.register_with_result(&IAddOp::new(child.ctx_mut(), iteration, unit))
                } else {
                    iteration
                };
                let value_ids = operand_ids(op);
                let body = closure(&value_ids, &bound_ops, &ready);
                // Reference-derived mechanism experiment: protect/collect all
                // scalar values before updating independent accumulators. The
                // semantic lazy bodies are unchanged and remain inside guards.
                // Inactive values combine the identity into accumulators that
                // start at the identity (+0 added to +0 for sum). For sum, keep
                // the RN/no-FTZ policy: the neutral additions rely on it (NaN
                // payloads are not a sum contract).
                let mut batch = Vec::new();
                for (i, accumulator) in accumulators.iter().enumerate() {
                    let offset = constant(&child, &json!(i as u32 * cooperation), "int64")?;
                    let coordinate =
                        child.register_with_result(&IAddOp::new(child.ctx_mut(), base, offset));
                    let live = child.register_with_result(&SLessThanOp::new(
                        child.ctx_mut(),
                        coordinate,
                        reduction_extent,
                    ));
                    // Inactive pending values stay at the identity, so the
                    // unconditional batch update leaves the accumulator as is.
                    let pending = batched_loads
                        .then(|| {
                            combiner
                                .identity_locals(&child, fty)
                                .map(|(locals, _)| locals)
                        })
                        .transpose()?;
                    // Guard the access itself, not a later select of its value.
                    branch(&child, live, |c| {
                        let mut args = scalars[1..].to_vec();
                        args.push(coordinate);
                        let mut values = ready.clone();
                        emit(c, &body, &arrays, outer, &args, &mut values)?;
                        let operands: Vec<_> = value_ids.iter().map(|id| values[*id]).collect();
                        if let Some(ptr) = &pending {
                            let lifted = combiner.lift(c, &operands)?;
                            for (p, v) in ptr.iter().zip(&lifted) {
                                c.register(&StoreOp::new(c.ctx_mut(), *p, *v));
                            }
                            return Ok(());
                        }
                        let old = load_all(c, accumulator);
                        let new = combiner.accumulate(c, &old, &operands)?;
                        store_all(c, accumulator, &new);
                        Ok(())
                    })?;
                    if let Some(ptr) = pending {
                        batch.push((accumulator.clone(), ptr));
                    }
                }
                let update = |c: &Scope| -> Result<(), Error> {
                    for (accumulator, ptr) in &batch {
                        let value = load_all(c, ptr);
                        let old = load_all(c, accumulator);
                        let new = combiner.combine(c, &old, &value);
                        store_all(c, accumulator, &new);
                    }
                    Ok(())
                };
                if uniform_loop {
                    // Preserve even the frozen plan's inactive-update set.
                    let live = child.register_with_result(&SLessThanOp::new(
                        child.ctx_mut(),
                        base,
                        reduction_extent,
                    ));
                    branch(&child, live, update)?;
                } else {
                    update(&child)?;
                }
                child.terminate_yield();
                s.register(&loop_op);
                let mut values: Vec<_> = accumulators.iter().map(|a| load_all(s, a)).collect();
                // Both orders are balanced combines (ordinary FP32 adds for
                // sum). The explicit adjacent plan changes only this physical
                // merge, not semantic replay, chain assignment, collective order
                // or FP permissions.
                if private_merge == "adjacent" {
                    let left = combiner.combine(s, &values[0], &values[1]);
                    let right = combiner.combine(s, &values[2], &values[3]);
                    values[0] = combiner.combine(s, &left, &right);
                } else {
                    let mut stride = private_accumulators / 2;
                    while stride != 0 {
                        for i in 0..stride {
                            values[i] = combiner.combine(s, &values[i], &values[i + stride]);
                        }
                        stride /= 2;
                    }
                }
                values.swap_remove(0)
            } else {
                let (accumulator, neutral) = combiner.identity_locals(s, fty)?;
                let width = constant(s, &json!(vector_width), "int64")?;
                let begin = s.register_with_result(&IMulOp::new(s.ctx_mut(), unit, width));
                let reduction_step = s.register_with_result(&IMulOp::new(s.ctx_mut(), step, width));
                let loop_op = RangeLoopOp::new(
                    s.ctx_mut(),
                    if final_load_batch { zero } else { begin },
                    reduction_extent,
                    reduction_step,
                );
                let coordinate = loop_op.iter_var(s.ctx());
                let child =
                    s.branch_child(OpInserter::new_at_block_end(loop_op.loop_body(s.ctx())));
                let coordinate = if final_load_batch {
                    child.register_with_result(&IAddOp::new(child.ctx_mut(), coordinate, unit))
                } else {
                    coordinate
                };
                let mut args = scalars[1..].to_vec();
                args.push(coordinate);
                let value_ids = operand_ids(op);
                let body = closure(&value_ids, &bound_ops, &ready);
                let operands = |values: &HashMap<String, IrValue>| -> Vec<IrValue> {
                    value_ids.iter().map(|id| values[*id]).collect()
                };
                let mut local = ready.clone();
                // `value` holds one element's operand values; the final-load
                // probe (sum only) passes its lifted pending value, the same
                // single component.
                let accumulate = |c: &Scope, value: Vec<IrValue>| -> Result<(), Error> {
                    let old = load_all(c, &accumulator);
                    let new = if let Some(ptr) = correction {
                        // Compensation is sum-only (one component).
                        let low = c.register_with_result(&LoadOp::new(c.ctx_mut(), ptr));
                        let (sum, low) = compensated_add(c, old[0], low, value[0])?;
                        c.register(&StoreOp::new(c.ctx_mut(), ptr, low));
                        vec![sum]
                    } else {
                        combiner.accumulate(c, &old, &value)?
                    };
                    store_all(c, &accumulator, &new);
                    Ok(())
                };
                if final_load_batch {
                    // The already-established uniform-loop/grouped-load
                    // expression, isolated to this one-chain final probe.
                    let pending: Vec<_> = neutral
                        .iter()
                        .map(|n| {
                            let p = child.create_local_mut(fty, None);
                            child.register(&StoreOp::new(child.ctx_mut(), p, *n));
                            p
                        })
                        .collect();
                    let live = child.register_with_result(&SLessThanOp::new(
                        child.ctx_mut(),
                        coordinate,
                        reduction_extent,
                    ));
                    branch(&child, live, |c| {
                        emit(c, &body, &arrays, outer, &args, &mut local)?;
                        let lifted = combiner.lift(c, &operands(&local))?;
                        store_all(c, &pending, &lifted);
                        Ok(())
                    })?;
                    let value = load_all(&child, &pending);
                    accumulate(&child, value)?;
                } else if column {
                    branch(&child, valid, |c| {
                        emit(c, &body, &arrays, outer, &args, &mut local)?;
                        accumulate(c, operands(&local))?;
                        Ok(())
                    })?;
                } else if vector_width == 1 {
                    emit(&child, &body, &arrays, outer, &args, &mut local)?;
                    accumulate(&child, operands(&local))?;
                } else {
                    let offset = constant(&child, &json!(vector_width - 1), "int64")?;
                    let last = child.register_with_result(&IAddOp::new(
                        child.ctx_mut(),
                        coordinate,
                        offset,
                    ));
                    let full = child.register_with_result(&SLessThanOp::new(
                        child.ctx_mut(),
                        last,
                        reduction_extent,
                    ));
                    let vector_branch = IfOp::new(child.ctx_mut(), full);
                    let vector = child.branch_child(OpInserter::new_at_block_end(
                        vector_branch.then_block(child.ctx()),
                    ));
                    let lanes = crate::descriptor5::reduction_vector_values(
                        &vector,
                        &body,
                        &arrays,
                        outer,
                        &args,
                        &ready,
                        (&value_ids, vector_width),
                    )?;
                    if combiner.arity() == 1 {
                        for value in lanes {
                            accumulate(&vector, value)?;
                        }
                    } else {
                        // Welford: the contiguous lanes combine as a balanced
                        // pair tree (Triton's per-thread lanes), so equal
                        // weights meet with ratio exactly 1/2; then one
                        // combine into the accumulator.
                        let mut level = lanes
                            .iter()
                            .map(|v| combiner.lift(&vector, v))
                            .collect::<Result<Vec<_>, _>>()?;
                        while level.len() > 1 {
                            level = level
                                .chunks(2)
                                .map(|pair| match pair {
                                    [a, b] => combiner.combine(&vector, a, b),
                                    [a] => a.clone(),
                                    _ => unreachable!(),
                                })
                                .collect();
                        }
                        let old = load_all(&vector, &accumulator);
                        let new = combiner.combine(&vector, &old, &level[0]);
                        store_all(&vector, &accumulator, &new);
                    }
                    vector.terminate_yield();
                    let tail = child.branch_child(OpInserter::new_at_block_end(
                        vector_branch.else_block(child.ctx()),
                    ));
                    for lane in 0..vector_width {
                        let offset = constant(&tail, &json!(lane), "int64")?;
                        let r = tail.register_with_result(&IAddOp::new(
                            tail.ctx_mut(),
                            coordinate,
                            offset,
                        ));
                        let valid = tail.register_with_result(&SLessThanOp::new(
                            tail.ctx_mut(),
                            r,
                            reduction_extent,
                        ));
                        branch(&tail, valid, |c| {
                            let mut args = args.clone();
                            args[defs.len()] = r;
                            let mut values = ready.clone();
                            emit(c, &body, &arrays, outer, &args, &mut values)?;
                            accumulate(c, operands(&values))?;
                            Ok(())
                        })?;
                    }
                    tail.terminate_yield();
                    child.register(&vector_branch);
                }
                child.terminate_yield();
                s.register(&loop_op);
                load_all(s, &accumulator)
            };
            let sum = if planes {
                if final_plane {
                    // CubeK's independent Plane body and lane-zero writer,
                    // using the retained Triton/ATen descending merge order.
                    // Generic FSum uses a different order on the CUDA target.
                    combiner.butterfly(s, partial, &[16, 8, 4, 2, 1])?
                } else {
                    combiner.plane(s, &partial)?
                }
            } else if plane_shared {
                let owner_value = owner_result
                    .then(|| combiner.identity_locals(s, fty).map(|(locals, _)| locals))
                    .transpose()?;
                // The reference-derived plan fixes the complete butterfly
                // order as well as private work assignment. Existing plans
                // retain the generic plane polyfill unchanged.
                let plane_value = if reference512 {
                    combiner.butterfly(s, partial, &[16, 8, 4, 2, 1])?
                } else {
                    combiner.plane(s, &partial)?
                };
                let divisor = constant(s, &json!(32), "int64")?;
                let plane_id = s.register_with_result(&SDivOp::new(s.ctx_mut(), unit, divisor));
                let plane_lane = s.register_with_result(&SRemOp::new(s.ctx_mut(), unit, divisor));
                let leader = s.register_with_result(&IEqualOp::new(s.ctx_mut(), plane_lane, zero));
                branch(s, leader, |c| {
                    for (array, value) in shared.iter().zip(&plane_value) {
                        c.register(&StoreOp::new(
                            c.ctx_mut(),
                            pointer(c, *array, plane_id),
                            *value,
                        ));
                    }
                    Ok(())
                })?;
                sync(s);
                let first_plane =
                    s.register_with_result(&IEqualOp::new(s.ctx_mut(), plane_id, zero));
                branch(s, first_plane, |c| {
                    // Lanes beyond the plane count keep the identity.
                    let (local, _) = combiner.identity_locals(c, fty)?;
                    let count = constant(c, &json!(shared_size), "int64")?;
                    let live =
                        c.register_with_result(&SLessThanOp::new(c.ctx_mut(), plane_lane, count));
                    branch(c, live, |a| {
                        for (array, l) in shared.iter().zip(&local) {
                            let value = a.register_with_result(&LoadOp::new(
                                a.ctx_mut(),
                                pointer(a, *array, plane_lane),
                            ));
                            a.register(&StoreOp::new(a.ctx_mut(), *l, value));
                        }
                        Ok(())
                    })?;
                    let value = load_all(c, &local);
                    let final_value = if reference512 {
                        // Only the first half-warp is read by the output
                        // owner; distances <=8 never mix it with identity lanes.
                        combiner.butterfly(c, value, &[8, 4, 2, 1])?
                    } else {
                        combiner.plane(c, &value)?
                    };
                    if let Some(local) = &owner_value {
                        // Every lane in plane 0 keeps its own result; only unit
                        // zero consumes it under the existing outer-store guard.
                        store_all(c, local, &final_value);
                    } else {
                        branch(c, leader, |a| {
                            for (array, value) in shared.iter().zip(&final_value) {
                                a.register(&StoreOp::new(
                                    a.ctx_mut(),
                                    pointer(a, *array, zero),
                                    *value,
                                ));
                            }
                            Ok(())
                        })?;
                    }
                    Ok(())
                })?;
                if let Some(local) = &owner_value {
                    load_all(s, local)
                } else {
                    sync(s);
                    shared
                        .iter()
                        .take(combiner.arity())
                        .map(|array| {
                            s.register_with_result(&LoadOp::new(
                                s.ctx_mut(),
                                pointer(s, *array, zero),
                            ))
                        })
                        .collect()
                }
            } else {
                for (array, value) in shared.iter().zip(&partial) {
                    let target = pointer(s, *array, shared_index(s, unit));
                    s.register(&StoreOp::new(s.ctx_mut(), target, *value));
                }
                // Compensation is sum-only: its residuals share component 0's
                // array, offset by the cooperation width.
                let correction_ptr = |c: &Scope, r: IrValue| {
                    let delta = constant(c, &json!(cooperation), "int64").unwrap();
                    let index = c.register_with_result(&IAddOp::new(c.ctx_mut(), r, delta));
                    pointer(c, shared[0], index)
                };
                if let Some(c) = correction {
                    let low = s.register_with_result(&LoadOp::new(s.ctx_mut(), c));
                    s.register(&StoreOp::new(s.ctx_mut(), correction_ptr(s, unit), low));
                }
                sync(s);
                let levels = cooperation.ilog2();
                for stride in (0..levels).rev().map(|i| 1 << i) {
                    let width = constant(s, &json!(stride), "int64")?;
                    let active =
                        s.register_with_result(&SLessThanOp::new(s.ctx_mut(), unit, width));
                    branch(s, active, |child| {
                        let rhs_unit =
                            child.register_with_result(&IAddOp::new(child.ctx_mut(), unit, width));
                        let arrays = &shared[..combiner.arity()];
                        let mut lhs_ptr = Vec::new();
                        let mut rhs_ptr = Vec::new();
                        for array in arrays {
                            lhs_ptr.push(pointer(child, *array, shared_index(child, unit)));
                            rhs_ptr.push(pointer(child, *array, shared_index(child, rhs_unit)));
                        }
                        let lhs: Vec<_> = lhs_ptr
                            .iter()
                            .map(|p| child.register_with_result(&LoadOp::new(child.ctx_mut(), *p)))
                            .collect();
                        let rhs: Vec<_> = rhs_ptr
                            .iter()
                            .map(|p| child.register_with_result(&LoadOp::new(child.ctx_mut(), *p)))
                            .collect();
                        let new = if compensated {
                            let lhs_low_ptr = correction_ptr(child, unit);
                            let lhs_low = child
                                .register_with_result(&LoadOp::new(child.ctx_mut(), lhs_low_ptr));
                            let rhs_low = child.register_with_result(&LoadOp::new(
                                child.ctx_mut(),
                                correction_ptr(child, rhs_unit),
                            ));
                            let (sum, low) = compensated_add(child, lhs[0], lhs_low, rhs[0])?;
                            let low = child.register_with_result(&FAddOp::new(
                                child.ctx_mut(),
                                low,
                                rhs_low,
                            ));
                            child.register(&StoreOp::new(child.ctx_mut(), lhs_low_ptr, low));
                            vec![sum]
                        } else {
                            combiner.combine(child, &lhs, &rhs)
                        };
                        for (p, v) in lhs_ptr.iter().zip(&new) {
                            child.register(&StoreOp::new(child.ctx_mut(), *p, *v));
                        }
                        Ok(())
                    })?;
                    sync(s);
                }
                let high: Vec<_> = shared[..combiner.arity()]
                    .iter()
                    .map(|array| {
                        let root = pointer(s, *array, lane_outer);
                        s.register_with_result(&LoadOp::new(s.ctx_mut(), root))
                    })
                    .collect();
                if compensated {
                    let low =
                        s.register_with_result(&LoadOp::new(s.ctx_mut(), correction_ptr(s, zero)));
                    vec![s.register_with_result(&FAddOp::new(s.ctx_mut(), high[0], low))]
                } else {
                    high
                }
            };
            let id = op["id"].as_str().unwrap();
            for component in bound_ops
                .iter()
                .filter(|c| c["op"] == "reduction_component" && c["args"][0].as_str() == Some(id))
            {
                let index = component["index"].as_u64().unwrap() as usize;
                ready.insert(component["id"].as_str().unwrap().into(), sum[index]);
            }
            ready.insert(id.into(), combiner.output(&sum));
            // All units must finish reading before a later reduction overwrites
            // scratch. The last reduction has no subsequent scratch writer:
            // epilogues only read the reduced SSA values and store to globals.
            if !planes && reduction_index + 1 < reductions.len() {
                sync(s);
            }
        }
        for store in bound_ops.iter().filter(|op| op["op"] == "store") {
            let mut body = closure(&operand_ids(store), &bound_ops, &ready);
            body.push(store.clone());
            if store["iteration_scope"] == "outer" {
                let leader = s.register_with_result(&IEqualOp::new(s.ctx_mut(), unit, zero));
                branch(s, leader, |child| {
                    let mut args = scalars[1..].to_vec();
                    args.push(zero);
                    if column {
                        branch(child, valid, |c| {
                            emit(c, &body, &arrays, outer, &args, &mut ready.clone())
                        })
                    } else {
                        emit(child, &body, &arrays, outer, &args, &mut ready.clone())
                    }
                })?;
            } else {
                let loop_op = RangeLoopOp::new(s.ctx_mut(), unit, reduction_extent, step);
                let coordinate = loop_op.iter_var(s.ctx());
                let child =
                    s.branch_child(OpInserter::new_at_block_end(loop_op.loop_body(s.ctx())));
                let mut args = scalars[1..].to_vec();
                args.push(coordinate);
                if column {
                    branch(&child, valid, |c| {
                        emit(c, &body, &arrays, outer, &args, &mut ready.clone())
                    })?;
                } else {
                    emit(&child, &body, &arrays, outer, &args, &mut ready.clone())?;
                }
                child.terminate_yield();
                s.register(&loop_op);
            }
        }
        Ok(())
    })?;
    b.register(&ReturnOp::new(b.ctx_mut()));
    Ok(b.build())
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

    /// The fixture's bf16 row sum: load, cast, reduce, round, store.
    fn sum_descriptor() -> Value {
        json!({"schema": 3, "index_dtype": "int64",
            "buffers": [
                {"access": "read", "dtype": "bfloat16", "offset": {"op": "constant", "value": 0},
                 "size": [{"op": "scalar", "slot": 1}, {"op": "scalar", "slot": 0}],
                 "stride": [{"op": "scalar", "slot": 0}, {"op": "constant", "value": 1}]},
                {"access": "write", "dtype": "bfloat16", "offset": {"op": "constant", "value": 0},
                 "size": [{"op": "scalar", "slot": 0}], "stride": [{"op": "constant", "value": 1}]}],
            "numel": {"op": "scalar", "slot": 0}, "outer": {"op": "scalar", "slot": 0},
            "reduction_extent": {"op": "scalar", "slot": 1},
            "scalars": [{"dtype": "int64", "id": 0, "positive": true},
                        {"dtype": "int64", "id": 1, "positive": true}],
            "ops": [
                {"op": "load", "id": "tmp0", "args": [], "buffer": 0, "dtype": "bfloat16",
                 "semantic_dtype": "bfloat16",
                 "index": {"op": "add", "args": [{"op": "linear"}, {"op": "mul",
                     "args": [{"op": "reduction"}, {"op": "scalar", "slot": 0}]}]}},
                {"op": "cast", "id": "tmp1", "args": ["tmp0"], "dtype": "float32",
                 "semantic_dtype": "bfloat16"},
                {"op": "reduction", "id": "tmp2", "args": ["tmp1"], "dtype": "float32",
                 "semantic_dtype": "bfloat16", "source_dtype": "bfloat16",
                 "accumulator_dtype": "float32", "reduction_type": "sum"},
                {"op": "cast", "id": "tmp3", "args": ["tmp2"], "dtype": "bfloat16",
                 "semantic_dtype": "bfloat16"},
                {"op": "store", "args": ["tmp3"], "buffer": 1, "dtype": "bfloat16",
                 "semantic_dtype": "bfloat16", "index": {"op": "linear"},
                 "iteration_scope": "outer"}]})
    }

    #[test]
    fn sum_descriptor_is_valid() {
        validate_description(&sum_descriptor()).unwrap();
    }

    #[test]
    fn reduction_operand_count_follows_its_type() {
        let mut d = sum_descriptor();
        d["ops"][2]["args"] = json!(["tmp1", "tmp1"]);
        let error = validate_description(&d).unwrap_err().to_string();
        assert_eq!(error, "reduction type sum takes 1 operand(s), got 2");
        d["ops"][2]["args"] = json!([]);
        let error = validate_description(&d).unwrap_err().to_string();
        assert_eq!(error, "reduction type sum takes 1 operand(s), got 0");
    }

    #[test]
    fn single_value_reductions_have_no_components() {
        let component = |args: Value, index: u64| {
            let mut d = sum_descriptor();
            let ops = d["ops"].as_array_mut().unwrap();
            ops.insert(
                3,
                json!({"op": "reduction_component", "id": "tmp9", "args": args,
                "index": index, "dtype": "float32", "semantic_dtype": "float32"}),
            );
            validate_description(&d).unwrap_err().to_string()
        };
        assert_eq!(
            component(json!(["tmp2"]), 0),
            "reduction type sum has no component 0"
        );
        assert_eq!(
            component(json!(["tmp1"]), 0),
            "reduction component of an unknown or later reduction"
        );
        assert_eq!(
            component(json!(["tmp2", "tmp2"]), 0),
            "reduction component requires one reduction operand"
        );
    }

    #[test]
    fn adjacent_merge_rejects_invalid_chain_count_before_ir_construction() {
        for count in [0, 1, 2, 3, 5] {
            let error = build_physical(
                &json!({}),
                [256, 1, 1],
                "plane_shared_load_batch",
                1,
                count,
                "adjacent",
            )
            .err()
            .expect("invalid adjacent chain count must fail locally");
            assert_eq!(
                error.to_string(),
                "adjacent private merge requires four accumulators"
            );
        }
    }
}
