//! Semantic sum DAG + independently validated cooperative physical plan.
//! No handwritten target source: all loops, shared accesses and barriers are IR.
use crate::descriptor2::{check_index, constant, elem, emit, index, validate_ops};
use cubecl_core::{
    compute::KernelBuilder,
    ir::{
        AddressType, Builtin, OpInserter, Scope,
        dialect::{
            branch::{IfOp, RangeLoopOp, ReturnOp},
            cmp::{FLessThanOrEqualOp, IEqualOp, SLessThanOp},
            general::{CastOp, ReadBuiltinOp, ReadScalarOp, SelectOp},
            math::{FAbsOp, FAddOp, FSubOp, IAddOp, IMulOp, SDivOp, SRemOp},
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

fn closure(id: &str, ops: &[Value], ready: &HashMap<String, IrValue>) -> Vec<Value> {
    let mut needed = HashSet::new();
    dependencies(id, ops, &ready.keys().cloned().collect(), &mut needed);
    ops.iter()
        .filter(|op| {
            op["op"] != "reduction" && op["id"].as_str().is_some_and(|id| needed.contains(id))
        })
        .cloned()
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
        if size.len() != stride.len() || size.len() > 4 {
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
    for op in &mut bound {
        if op["op"] == "reduction" {
            reductions += 1;
            if op["reduction_type"] != "sum"
                || op["accumulator_dtype"] != "float32"
                || op["dtype"] != "float32"
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
        return Err("one to four sums per cooperative schedule required".into());
    }
    let mut types = HashMap::new();
    for (original, checked) in ops.iter().zip(&bound) {
        if original["op"] == "reduction" {
            let argument = original["args"][0]
                .as_str()
                .ok_or("reduction operand missing")?;
            if types.get(argument).map(String::as_str) != Some("float32") {
                return Err("reduction requires explicit float32 input conversion".into());
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
        .filter(|op| op["op"] == "reduction")
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
    let shared = (!planes)
        .then(|| b.create_shared(ArrayType::get(b.ctx(), fty, shared_size as usize), None));
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
                let accumulators: Vec<_> = (0..private_accumulators)
                    .map(|_| {
                        let a = s.create_local_mut(fty, None);
                        s.register(&StoreOp::new(
                            s.ctx_mut(),
                            a,
                            constant(s, &json!(0.0), "float32").unwrap(),
                        ));
                        a
                    })
                    .collect();
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
                let value_id = op["args"][0].as_str().unwrap();
                let body = closure(value_id, &bound_ops, &ready);
                // Reference-derived mechanism experiment: protect/collect all
                // scalar values before updating independent accumulators. The
                // semantic lazy bodies are unchanged and remain inside guards.
                // Inactive values add +0 to accumulators initialized at +0.
                // Keep the RN/no-FTZ policy and that initialization: the neutral
                // additions rely on them (NaN payloads are not a sum contract).
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
                    let pending = batched_loads.then(|| {
                        let ptr = child.create_local_mut(fty, None);
                        child.register(&StoreOp::new(
                            child.ctx_mut(),
                            ptr,
                            constant(&child, &json!(0.0), "float32").unwrap(),
                        ));
                        ptr
                    });
                    // Guard the access itself, not a later select of its value.
                    branch(&child, live, |c| {
                        let mut args = scalars[1..].to_vec();
                        args.push(coordinate);
                        let mut values = ready.clone();
                        emit(c, &body, &arrays, outer, &args, &mut values)?;
                        if let Some(ptr) = pending {
                            c.register(&StoreOp::new(c.ctx_mut(), ptr, values[value_id]));
                            return Ok(());
                        }
                        let old = c.register_with_result(&LoadOp::new(c.ctx_mut(), *accumulator));
                        let sum = c.register_with_result(&FAddOp::new(
                            c.ctx_mut(),
                            old,
                            values[value_id],
                        ));
                        c.register(&StoreOp::new(c.ctx_mut(), *accumulator, sum));
                        Ok(())
                    })?;
                    if let Some(ptr) = pending {
                        batch.push((*accumulator, ptr));
                    }
                }
                let update = |c: &Scope| -> Result<(), Error> {
                    for (accumulator, ptr) in &batch {
                        let value = c.register_with_result(&LoadOp::new(c.ctx_mut(), *ptr));
                        let old = c.register_with_result(&LoadOp::new(c.ctx_mut(), *accumulator));
                        let sum = c.register_with_result(&FAddOp::new(c.ctx_mut(), old, value));
                        c.register(&StoreOp::new(c.ctx_mut(), *accumulator, sum));
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
                let mut values: Vec<_> = accumulators
                    .iter()
                    .map(|a| s.register_with_result(&LoadOp::new(s.ctx_mut(), *a)))
                    .collect();
                // Both orders are balanced, ordinary FP32 adds. The explicit
                // adjacent plan changes only this physical merge, not semantic
                // replay, chain assignment, collective order or FP permissions.
                if private_merge == "adjacent" {
                    let left =
                        s.register_with_result(&FAddOp::new(s.ctx_mut(), values[0], values[1]));
                    let right =
                        s.register_with_result(&FAddOp::new(s.ctx_mut(), values[2], values[3]));
                    values[0] = s.register_with_result(&FAddOp::new(s.ctx_mut(), left, right));
                } else {
                    let mut stride = private_accumulators / 2;
                    while stride != 0 {
                        for i in 0..stride {
                            values[i] = s.register_with_result(&FAddOp::new(
                                s.ctx_mut(),
                                values[i],
                                values[i + stride],
                            ));
                        }
                        stride /= 2;
                    }
                }
                values[0]
            } else {
                let accumulator = s.create_local_mut(fty, None);
                let neutral = constant(s, &json!(0.0), "float32")?;
                s.register(&StoreOp::new(s.ctx_mut(), accumulator, neutral));
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
                let value_id = op["args"][0].as_str().unwrap();
                let body = closure(value_id, &bound_ops, &ready);
                let mut local = ready.clone();
                let accumulate = |c: &Scope, value: IrValue| -> Result<(), Error> {
                    let old = c.register_with_result(&LoadOp::new(c.ctx_mut(), accumulator));
                    let sum = if let Some(ptr) = correction {
                        let low = c.register_with_result(&LoadOp::new(c.ctx_mut(), ptr));
                        let (sum, low) = compensated_add(c, old, low, value)?;
                        c.register(&StoreOp::new(c.ctx_mut(), ptr, low));
                        sum
                    } else {
                        c.register_with_result(&FAddOp::new(c.ctx_mut(), old, value))
                    };
                    c.register(&StoreOp::new(c.ctx_mut(), accumulator, sum));
                    Ok(())
                };
                if final_load_batch {
                    // The already-established uniform-loop/grouped-load
                    // expression, isolated to this one-chain final probe.
                    let pending = child.create_local_mut(fty, None);
                    child.register(&StoreOp::new(child.ctx_mut(), pending, neutral));
                    let live = child.register_with_result(&SLessThanOp::new(
                        child.ctx_mut(),
                        coordinate,
                        reduction_extent,
                    ));
                    branch(&child, live, |c| {
                        emit(c, &body, &arrays, outer, &args, &mut local)?;
                        c.register(&StoreOp::new(c.ctx_mut(), pending, local[value_id]));
                        Ok(())
                    })?;
                    let value = child.register_with_result(&LoadOp::new(child.ctx_mut(), pending));
                    accumulate(&child, value)?;
                } else if column {
                    branch(&child, valid, |c| {
                        emit(c, &body, &arrays, outer, &args, &mut local)?;
                        accumulate(c, local[value_id])?;
                        Ok(())
                    })?;
                } else if vector_width == 1 {
                    emit(&child, &body, &arrays, outer, &args, &mut local)?;
                    accumulate(&child, local[value_id])?;
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
                    for value in crate::descriptor5::reduction_vector_values(
                        &vector,
                        &body,
                        &arrays,
                        outer,
                        &args,
                        &ready,
                        (value_id, vector_width),
                    )? {
                        accumulate(&vector, value)?;
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
                            accumulate(c, values[value_id])?;
                            Ok(())
                        })?;
                    }
                    tail.terminate_yield();
                    child.register(&vector_branch);
                }
                child.terminate_yield();
                s.register(&loop_op);
                s.register_with_result(&LoadOp::new(s.ctx_mut(), accumulator))
            };
            let sum = if planes {
                if final_plane {
                    // CubeK's independent Plane body and lane-zero writer,
                    // using the retained Triton/ATen descending merge order.
                    // Generic FSum uses a different order on the CUDA target.
                    let mut value = partial;
                    for distance in [16, 8, 4, 2, 1] {
                        let mask = constant(s, &json!(distance), "int32")?;
                        let rhs = s.register_with_result(
                            &cubecl_core::ir::dialect::plane::ShuffleXorOp::new(
                                s.ctx_mut(),
                                value,
                                mask,
                            ),
                        );
                        value = s.register_with_result(&FAddOp::new(s.ctx_mut(), value, rhs));
                    }
                    value
                } else {
                    s.register_with_result(&cubecl_core::ir::dialect::plane::FSumOp::new(
                        s.ctx_mut(),
                        partial,
                    ))
                }
            } else if plane_shared {
                let shared = shared.unwrap();
                let owner_value = owner_result.then(|| {
                    let local = s.create_local_mut(fty, None);
                    s.register(&StoreOp::new(
                        s.ctx_mut(),
                        local,
                        constant(s, &json!(0.0), "float32").unwrap(),
                    ));
                    local
                });
                // The reference-derived plan fixes the complete butterfly
                // order as well as private work assignment. Existing plans
                // retain the generic plane polyfill unchanged.
                let reduce = |scope: &Scope,
                              mut value: IrValue,
                              distances: &[u32]|
                 -> Result<IrValue, Error> {
                    for distance in distances {
                        let mask = constant(scope, &json!(distance), "int32")?;
                        let rhs = scope.register_with_result(
                            &cubecl_core::ir::dialect::plane::ShuffleXorOp::new(
                                scope.ctx_mut(),
                                value,
                                mask,
                            ),
                        );
                        value =
                            scope.register_with_result(&FAddOp::new(scope.ctx_mut(), value, rhs));
                    }
                    Ok(value)
                };
                let plane_value = if reference512 {
                    reduce(s, partial, &[16, 8, 4, 2, 1])?
                } else {
                    s.register_with_result(&cubecl_core::ir::dialect::plane::FSumOp::new(
                        s.ctx_mut(),
                        partial,
                    ))
                };
                let divisor = constant(s, &json!(32), "int64")?;
                let plane_id = s.register_with_result(&SDivOp::new(s.ctx_mut(), unit, divisor));
                let plane_lane = s.register_with_result(&SRemOp::new(s.ctx_mut(), unit, divisor));
                let leader = s.register_with_result(&IEqualOp::new(s.ctx_mut(), plane_lane, zero));
                branch(s, leader, |c| {
                    c.register(&StoreOp::new(
                        c.ctx_mut(),
                        pointer(c, shared, plane_id),
                        plane_value,
                    ));
                    Ok(())
                })?;
                sync(s);
                let first_plane =
                    s.register_with_result(&IEqualOp::new(s.ctx_mut(), plane_id, zero));
                branch(s, first_plane, |c| {
                    let local = c.create_local_mut(fty, None);
                    c.register(&StoreOp::new(
                        c.ctx_mut(),
                        local,
                        constant(c, &json!(0.0), "float32")?,
                    ));
                    let count = constant(c, &json!(shared_size), "int64")?;
                    let live =
                        c.register_with_result(&SLessThanOp::new(c.ctx_mut(), plane_lane, count));
                    branch(c, live, |a| {
                        let value = a.register_with_result(&LoadOp::new(
                            a.ctx_mut(),
                            pointer(a, shared, plane_lane),
                        ));
                        a.register(&StoreOp::new(a.ctx_mut(), local, value));
                        Ok(())
                    })?;
                    let value = c.register_with_result(&LoadOp::new(c.ctx_mut(), local));
                    let final_value = if reference512 {
                        // Only the first half-warp is read by the output
                        // owner; distances <=8 never mix it with zero lanes.
                        reduce(c, value, &[8, 4, 2, 1])?
                    } else {
                        c.register_with_result(&cubecl_core::ir::dialect::plane::FSumOp::new(
                            c.ctx_mut(),
                            value,
                        ))
                    };
                    if let Some(local) = owner_value {
                        // Every lane in plane 0 keeps its own result; only unit
                        // zero consumes it under the existing outer-store guard.
                        c.register(&StoreOp::new(c.ctx_mut(), local, final_value));
                    } else {
                        branch(c, leader, |a| {
                            a.register(&StoreOp::new(
                                a.ctx_mut(),
                                pointer(a, shared, zero),
                                final_value,
                            ));
                            Ok(())
                        })?;
                    }
                    Ok(())
                })?;
                if let Some(local) = owner_value {
                    s.register_with_result(&LoadOp::new(s.ctx_mut(), local))
                } else {
                    sync(s);
                    s.register_with_result(&LoadOp::new(s.ctx_mut(), pointer(s, shared, zero)))
                }
            } else {
                let shared = shared.unwrap();
                let target = pointer(s, shared, shared_index(s, unit));
                s.register(&StoreOp::new(s.ctx_mut(), target, partial));
                let correction_ptr = |c: &Scope, r: IrValue| {
                    let delta = constant(c, &json!(cooperation), "int64").unwrap();
                    let index = c.register_with_result(&IAddOp::new(c.ctx_mut(), r, delta));
                    pointer(c, shared, index)
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
                        let lhs_ptr = pointer(child, shared, shared_index(child, unit));
                        let rhs_ptr = pointer(child, shared, shared_index(child, rhs_unit));
                        let lhs =
                            child.register_with_result(&LoadOp::new(child.ctx_mut(), lhs_ptr));
                        let rhs =
                            child.register_with_result(&LoadOp::new(child.ctx_mut(), rhs_ptr));
                        let sum = if compensated {
                            let lhs_low_ptr = correction_ptr(child, unit);
                            let lhs_low = child
                                .register_with_result(&LoadOp::new(child.ctx_mut(), lhs_low_ptr));
                            let rhs_low = child.register_with_result(&LoadOp::new(
                                child.ctx_mut(),
                                correction_ptr(child, rhs_unit),
                            ));
                            let (sum, low) = compensated_add(child, lhs, lhs_low, rhs)?;
                            let low = child.register_with_result(&FAddOp::new(
                                child.ctx_mut(),
                                low,
                                rhs_low,
                            ));
                            child.register(&StoreOp::new(child.ctx_mut(), lhs_low_ptr, low));
                            sum
                        } else {
                            child.register_with_result(&FAddOp::new(child.ctx_mut(), lhs, rhs))
                        };
                        child.register(&StoreOp::new(child.ctx_mut(), lhs_ptr, sum));
                        Ok(())
                    })?;
                    sync(s);
                }
                let root = pointer(s, shared, lane_outer);
                let high = s.register_with_result(&LoadOp::new(s.ctx_mut(), root));
                if compensated {
                    let low =
                        s.register_with_result(&LoadOp::new(s.ctx_mut(), correction_ptr(s, zero)));
                    s.register_with_result(&FAddOp::new(s.ctx_mut(), high, low))
                } else {
                    high
                }
            };
            ready.insert(op["id"].as_str().unwrap().into(), sum);
            // All units must finish reading before a later reduction overwrites
            // scratch. The last reduction has no subsequent scratch writer:
            // epilogues only read the reduced SSA values and store to globals.
            if !planes && reduction_index + 1 < reductions.len() {
                sync(s);
            }
        }
        for store in bound_ops.iter().filter(|op| op["op"] == "store") {
            let mut body = closure(store["args"][0].as_str().unwrap(), &bound_ops, &ready);
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
