//! Semantic sum DAG + independently validated cooperative physical plan.
//! No handwritten target source: all loops, shared accesses and barriers are IR.
use crate::descriptor2::{check_index, constant, elem, emit, index, validate_ops};
use cubecl_core::{
    compute::KernelBuilder,
    ir::{
        AddressType, Builtin, OpInserter, Scope,
        dialect::{
            branch::{IfOp, RangeLoopOp, ReturnOp},
            cmp::{IEqualOp, SLessThanOp},
            general::{CastOp, ReadBuiltinOp, ReadScalarOp},
            math::{FAddOp, IAddOp},
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
        if v["dtype"] != "int64" || v["id"] != i || !v["positive"].is_boolean() {
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

    let mut b = KernelBuilder::new(
        KernelSettings::new(
            Dim3::new_1d(256),
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
    let outer = builtin(Builtin::CubePosX);
    let unit = builtin(Builtin::UnitPosX);
    let zero = constant(&b, &json!(0), "int64")?;
    let step = constant(&b, &json!(256), "int64")?;
    let outer_extent = index(&b, &d["outer"], zero, &scalars[1..])?;
    let reduction_extent = index(&b, &d["reduction_extent"], zero, &scalars[1..])?;
    let shared = b.create_shared(ArrayType::get(b.ctx(), fty, 256), None);
    let valid = b.register_with_result(&SLessThanOp::new(b.ctx_mut(), outer, outer_extent));
    // CubePosX and runtime extents are workgroup-uniform; no unit predicate
    // encloses a barrier. Reduction loops and masked bodies contain no barriers.
    branch(&b, valid, |s| {
        let mut ready = HashMap::new();
        let bound_ops = bind_coordinate(&Value::Array(ops.clone()), defs.len())
            .as_array()
            .unwrap()
            .clone();
        for op in &bound_ops {
            if op["op"] != "reduction" {
                continue;
            }
            let accumulator = s.create_local_mut(fty, None);
            let neutral = constant(s, &json!(0.0), "float32")?;
            s.register(&StoreOp::new(s.ctx_mut(), accumulator, neutral));
            let loop_op = RangeLoopOp::new(s.ctx_mut(), unit, reduction_extent, step);
            let coordinate = loop_op.iter_var(s.ctx());
            let child = s.branch_child(OpInserter::new_at_block_end(loop_op.loop_body(s.ctx())));
            let mut args = scalars[1..].to_vec();
            args.push(coordinate);
            let value_id = op["args"][0].as_str().unwrap();
            let body = closure(value_id, &bound_ops, &ready);
            let mut local = ready.clone();
            emit(&child, &body, &arrays, outer, &args, &mut local)?;
            let old = child.register_with_result(&LoadOp::new(child.ctx_mut(), accumulator));
            let sum =
                child.register_with_result(&FAddOp::new(child.ctx_mut(), old, local[value_id]));
            child.register(&StoreOp::new(child.ctx_mut(), accumulator, sum));
            child.terminate_yield();
            s.register(&loop_op);
            let partial = s.register_with_result(&LoadOp::new(s.ctx_mut(), accumulator));
            let target = pointer(s, shared, unit);
            s.register(&StoreOp::new(s.ctx_mut(), target, partial));
            sync(s);
            for stride in [128, 64, 32, 16, 8, 4, 2, 1] {
                let width = constant(s, &json!(stride), "int64")?;
                let active = s.register_with_result(&SLessThanOp::new(s.ctx_mut(), unit, width));
                branch(s, active, |child| {
                    let rhs =
                        child.register_with_result(&IAddOp::new(child.ctx_mut(), unit, width));
                    let lhs_ptr = pointer(child, shared, unit);
                    let rhs_ptr = pointer(child, shared, rhs);
                    let lhs = child.register_with_result(&LoadOp::new(child.ctx_mut(), lhs_ptr));
                    let rhs = child.register_with_result(&LoadOp::new(child.ctx_mut(), rhs_ptr));
                    let sum = child.register_with_result(&FAddOp::new(child.ctx_mut(), lhs, rhs));
                    child.register(&StoreOp::new(child.ctx_mut(), lhs_ptr, sum));
                    Ok(())
                })?;
                sync(s);
            }
            let root = pointer(s, shared, zero);
            let sum = s.register_with_result(&LoadOp::new(s.ctx_mut(), root));
            ready.insert(op["id"].as_str().unwrap().into(), sum);
            // All units finish reading before a later reduction reuses scratch.
            sync(s);
        }
        for store in bound_ops.iter().filter(|op| op["op"] == "store") {
            let mut body = closure(store["args"][0].as_str().unwrap(), &bound_ops, &ready);
            body.push(store.clone());
            if store["iteration_scope"] == "outer" {
                let leader = s.register_with_result(&IEqualOp::new(s.ctx_mut(), unit, zero));
                branch(s, leader, |child| {
                    let mut args = scalars[1..].to_vec();
                    args.push(zero);
                    emit(child, &body, &arrays, outer, &args, &mut ready.clone())
                })?;
            } else {
                let loop_op = RangeLoopOp::new(s.ctx_mut(), unit, reduction_extent, step);
                let coordinate = loop_op.iter_var(s.ctx());
                let child =
                    s.branch_child(OpInserter::new_at_block_end(loop_op.loop_body(s.ctx())));
                let mut args = scalars[1..].to_vec();
                args.push(coordinate);
                emit(&child, &body, &arrays, outer, &args, &mut ready.clone())?;
                child.terminate_yield();
                s.register(&loop_op);
            }
        }
        Ok(())
    })?;
    b.register(&ReturnOp::new(b.ctx_mut()));
    Ok(b.build())
}
