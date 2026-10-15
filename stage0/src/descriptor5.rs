//! Stage5 physical adapter. Semantic operations remain the validated Stage2-4 DAG.
use crate::{descriptor2 as pointwise, descriptor3 as reduction, descriptor4 as effects};
use cubecl_core::{
    compute::KernelBuilder,
    ir::{
        AddressType, Builtin, OpInserter, Scope,
        dialect::{
            branch::{IfOp, ReturnOp},
            cmp::SLessThanOp,
            general::{CastOp, ReadBuiltinOp, ReadScalarOp, ReinterpretCastOp},
            math::{IAddOp, IMulOp},
            memory::{IndexOp, LoadOp, StoreOp},
            vector::{CompositeConstructOp, CompositeExtractOp},
        },
        pliron::{
            r#type::{TypeHandle, Typed},
            value::Value as IrValue,
        },
        settings::{Dim3, ExecutionMode, KernelSettings},
        types::{PointerType, VectorType, scalar::IndexType},
    },
};
use serde_json::{Value, json};
use std::collections::HashMap;
type Error = Box<dyn std::error::Error>;

fn branch(
    s: &Scope,
    condition: IrValue,
    body: impl FnOnce(&Scope) -> Result<(), Error>,
) -> Result<(), Error> {
    let op = IfOp::new(s.ctx_mut(), condition);
    let child = s.branch_child(OpInserter::new_at_block_end(op.then_block(s.ctx())));
    body(&child)?;
    child.terminate_yield();
    let child = s.branch_child(OpInserter::new_at_block_end(op.else_block(s.ctx())));
    child.terminate_yield();
    s.register(&op);
    Ok(())
}
fn pointer(s: &Scope, array: IrValue, coordinate: IrValue) -> IrValue {
    let i = s.register_with_result(&CastOp::new(
        s.ctx_mut(),
        IndexType::get(s.ctx()).to_handle(),
        coordinate,
    ));
    s.register_with_result(&IndexOp::new(s.ctx_mut(), array, i, None))
}
fn emit(
    s: &Scope,
    ops: &[Value],
    arrays: &[IrValue],
    linear: IrValue,
    scalars: &[IrValue],
    values: &mut HashMap<String, IrValue>,
    schema: u64,
) -> Result<(), Error> {
    if schema == 4 {
        effects::emit(s, ops, arrays, linear, scalars, values)
    } else {
        pointwise::emit(s, ops, arrays, linear, scalars, values)
    }
}
fn vector_type(s: &Scope, ty: &str, width: usize) -> Result<TypeHandle, Error> {
    Ok(VectorType::get(s.ctx(), pointwise::elem(ty)?.to_type(s.ctx()), width).to_handle())
}
fn vector_pointer(s: &Scope, p: IrValue, ty: &str, width: usize) -> Result<IrValue, Error> {
    let space = p
        .get_type(s.ctx())
        .deref(s.ctx())
        .downcast_ref::<PointerType>()
        .ok_or("vector reinterpret requires a pointer")?
        .address_space;
    let t = PointerType::get(s.ctx(), vector_type(s, ty, width)?, space).to_handle();
    Ok(s.register_with_result(&ReinterpretCastOp::new(s.ctx_mut(), t, p)))
}

pub fn build(d: &Value) -> Result<cubecl_runtime::kernel::KernelDefinition, Error> {
    let entry = d["entrypoint"].as_str().ok_or("entrypoint")?;
    if !entry.starts_with("cubecl5_")
        || !entry.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
    {
        return Err("Stage5 entrypoint".into());
    }
    let schema = d["semantic_schema"].as_u64().ok_or("semantic schema")?;
    let mut sem = d.clone();
    sem["schema"] = schema.into();
    match schema {
        3 => reduction::validate_description(&sem)?,
        4 => effects::validate_description(&sem)?,
        _ => return Err("unsupported Stage5 semantic schema".into()),
    }
    let p = &d["physical_plan"];
    if p["version"] != 2 {
        return Err("physical plan version".into());
    }
    let strategy = p["strategy"].as_str().ok_or("physical strategy")?;
    let accumulators = match p.get("private_accumulators") {
        None => 1,
        Some(value) => value.as_u64().ok_or("private accumulator count")?,
    };
    let independent = strategy == "shared_tree_multi_accumulator";
    let reference512 = strategy == "plane_shared_reference512";
    let batched = strategy == "plane_shared_load_batch" || reference512;
    let combined = (strategy == "plane_shared" || batched) && accumulators == 4;
    let private_merge = match p.get("private_merge") {
        None => "halving",
        Some(value) => value.as_str().ok_or("private merge order")?,
    };
    if private_merge != "halving" && !(private_merge == "adjacent" && batched && accumulators == 4)
    {
        return Err("unsupported private merge order".into());
    }
    if batched && accumulators != 4 {
        return Err("load-batch plan requires exactly four accumulators".into());
    }
    if (independent && !matches!(accumulators, 2 | 4))
        || (!independent && !combined && accumulators != 1)
    {
        return Err("unsupported private accumulator policy".into());
    }
    if !matches!(
        strategy,
        "plane_shared"
            | "plane_shared_load_batch"
            | "plane_shared_reference512"
            | "independent_planes"
    ) && p["plane_width"] != 0
    {
        return Err("non-plane physical plan must have plane_width=0".into());
    }
    let dims = p["workgroup"].as_array().ok_or("workgroup")?;
    if dims.len() != 3 {
        return Err("workgroup rank".into());
    }
    let mut workgroup = [0u32; 3];
    for (i, v) in dims.iter().enumerate() {
        workgroup[i] = u32::try_from(v.as_u64().ok_or("workgroup dimension")?)?;
    }
    let width = p["vector_width"].as_u64().ok_or("vector width")? as usize;
    let elements = p["elements_per_unit"].as_u64().ok_or("elements per unit")?;
    let outputs = p["outputs_per_workgroup"]
        .as_u64()
        .ok_or("outputs per workgroup")?;
    if elements != 1
        && !(strategy == "scalar_pointwise" && workgroup == [128, 1, 1] && elements == 4)
    {
        return Err("elements mapping".into());
    }
    if schema == 3 {
        if !matches!(
            strategy,
            "shared_tree_reduction_baseline"
                | "shared_tree_multi_accumulator"
                | "shared_tree_compensated"
                | "shared_tree"
                | "plane_shared"
                | "plane_shared_load_batch"
                | "plane_shared_reference512"
                | "independent_planes"
                | "coalesced_columns"
        ) {
            return Err("reduction physical strategy".into());
        }
        let legal = match strategy {
            "shared_tree_reduction_baseline" => {
                workgroup == [256, 1, 1] && outputs == 1 && width == 1
            }
            "shared_tree_multi_accumulator" => {
                workgroup == [256, 1, 1] && outputs == 1 && width == 1
            }
            "shared_tree_compensated" => workgroup == [256, 1, 1] && outputs == 1 && width == 1,
            "shared_tree" => {
                matches!(workgroup, [64, 1, 1] | [128, 1, 1]) && outputs == 1 && width == 1
            }
            "plane_shared" | "plane_shared_load_batch" => {
                matches!(workgroup, [64, 1, 1] | [128, 1, 1] | [256, 1, 1])
                    && outputs == 1
                    && (width == 1 || (workgroup[0] != 64 && matches!(width, 2 | 4)))
            }
            "plane_shared_reference512" => {
                workgroup == [512, 1, 1]
                    && outputs == 1
                    && width == 1
                    && accumulators == 4
                    && private_merge == "adjacent"
            }
            "independent_planes" => {
                matches!(workgroup, [32, 4, 1] | [32, 8, 1])
                    && outputs == workgroup[1] as u64
                    && width == 1
            }
            "coalesced_columns" => {
                matches!(workgroup, [32, 4, 1] | [32, 8, 1]) && outputs == 32 && width == 1
            }
            _ => false,
        };
        if !legal || elements != 1 {
            return Err("reduction dimensions".into());
        }
        if combined
            && (workgroup != [if reference512 { 512 } else { 256 }, 1, 1]
                || width != 1
                || outputs != 1)
        {
            return Err("combined private/plane topology".into());
        }
        if matches!(
            strategy,
            "plane_shared"
                | "plane_shared_load_batch"
                | "plane_shared_reference512"
                | "independent_planes"
        ) && p["plane_width"] != 32
        {
            return Err("plane width must match queried CUDA32 target".into());
        }
        if strategy == "shared_tree_reduction_baseline"
            && (workgroup != [256, 1, 1] || width != 1 || outputs != 1)
        {
            return Err("frozen tree plan".into());
        }
        if width > 1 {
            for op in sem["ops"].as_array().unwrap() {
                if op["op"] == "masked" {
                    return Err("reduction vector lazy region unsupported".into());
                }
                if op["op"] == "load" {
                    let ty = op["dtype"].as_str().unwrap();
                    if !matches!(ty, "float32" | "float16" | "bfloat16")
                        || width * if ty == "float32" { 4 } else { 2 } > 16
                    {
                        return Err("reduction vector storage width".into());
                    }
                    if !matches!(reduction_coefficient(&op["index"], None), Some(0 | 1)) {
                        return Err("reduction vector addressing not proven unit stride".into());
                    }
                }
            }
        }
        return reduction::build_physical(
            &sem,
            workgroup,
            strategy,
            width,
            accumulators as usize,
            private_merge,
        );
    }
    if outputs != 1
        || workgroup[1..] != [1, 1]
        || !matches!(workgroup[0], 64 | 128 | 256)
        || !matches!(
            strategy,
            "scalar_pointwise_baseline" | "scalar_pointwise" | "vector_pointwise"
        )
    {
        return Err("pointwise physical topology".into());
    }
    if strategy == "scalar_pointwise_baseline"
        && (workgroup != [256, 1, 1] || width != 1 || elements != 1)
    {
        return Err("frozen scalar plan".into());
    }
    if !matches!(width, 1 | 2 | 4 | 8)
        || (width != 1 && strategy != "vector_pointwise")
        || (strategy == "vector_pointwise" && width == 1)
    {
        return Err("vector plan".into());
    }
    if width == 8 && workgroup[0] == 64 {
        return Err("uncatalogued vector plan".into());
    }
    if strategy == "scalar_pointwise" && workgroup[0] == 256 {
        return Err("use the frozen named scalar256 plan".into());
    }
    let buffers = sem["buffers"].as_array().unwrap();
    let ops = sem["ops"].as_array().unwrap();
    if width > 1 {
        if ops.iter().filter(|op| op["op"] == "store").count() != 1 {
            return Err("vector adapter requires one observable store".into());
        }
        if sem["effects"]["rng"] != Value::Null || sem["effects"]["mutation"] != false {
            return Err("effectful vectors unsupported".into());
        }
        for op in ops {
            if matches!(
                op["op"].as_str(),
                Some("masked" | "indirect" | "atomic_add" | "rand" | "load_seed")
            ) {
                return Err("vector lazy/effect region unsupported".into());
            }
            if matches!(op["op"].as_str(), Some("load" | "store")) {
                let scalar_broadcast = op["op"] == "load" && op["index"]["op"] == "constant";
                if !scalar_broadcast && op["index"]["op"] != "linear" {
                    return Err("vector addressing requires proven direct linear access".into());
                }
                let ty = op["dtype"].as_str().unwrap();
                if !matches!(ty, "float32" | "float16" | "bfloat16")
                    || width * if ty == "float32" { 4 } else { 2 } > 16
                {
                    return Err("vector storage width".into());
                }
            }
        }
    }
    let entry = d["entrypoint"].as_str().ok_or("entrypoint")?;
    if !entry.starts_with("cubecl5_")
        || !entry.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
    {
        return Err("Stage5 entrypoint".into());
    }
    let mut b = KernelBuilder::new(
        KernelSettings::new(
            Dim3::new_1d(workgroup[0]),
            ExecutionMode::Unchecked,
            AddressType::U64,
        )
        .kernel_name(entry),
    );
    let arrays: Vec<_> = buffers
        .iter()
        .map(|v| {
            b.buffer(
                pointwise::elem(v["dtype"].as_str().unwrap())
                    .unwrap()
                    .to_type(b.ctx()),
            )
        })
        .collect();
    let ity = pointwise::elem("int64")?.to_type(b.ctx());
    let scalars: Vec<_> = (0..=sem["scalars"].as_array().unwrap().len())
        .map(|_| {
            let id = b.scalar(pointwise::elem("int64").unwrap());
            b.register_with_result(&ReadScalarOp::new(b.ctx_mut(), ity, id))
        })
        .collect();
    let builtin = |kind| {
        let v = b.register_with_result(&ReadBuiltinOp::new(
            b.ctx_mut(),
            IndexType::get(b.ctx()).to_handle(),
            kind,
        ));
        b.register_with_result(&CastOp::new(b.ctx_mut(), ity, v))
    };
    let block = builtin(Builtin::CubePosX);
    let unit = builtin(Builtin::UnitPosX);
    let units = pointwise::constant(&b, &json!(workgroup[0]), "int64")?;
    let block_base = b.register_with_result(&IMulOp::new(b.ctx_mut(), block, units));
    let stride = pointwise::constant(&b, &json!(width as u64 * elements), "int64")?;
    let block_base = b.register_with_result(&IMulOp::new(b.ctx_mut(), block_base, stride));
    for iteration in 0..elements {
        let offset = pointwise::constant(&b, &json!(iteration * workgroup[0] as u64), "int64")?;
        let u = b.register_with_result(&IAddOp::new(b.ctx_mut(), unit, offset));
        let w = pointwise::constant(&b, &json!(width), "int64")?;
        let u = b.register_with_result(&IMulOp::new(b.ctx_mut(), u, w));
        let base = b.register_with_result(&IAddOp::new(b.ctx_mut(), block_base, u));
        if width == 1 {
            let valid = b.register_with_result(&SLessThanOp::new(b.ctx_mut(), base, scalars[0]));
            branch(&b, valid, |s| {
                emit(
                    s,
                    ops,
                    &arrays,
                    base,
                    &scalars[1..],
                    &mut HashMap::new(),
                    schema,
                )
            })?;
            continue;
        }
        let last = pointwise::constant(&b, &json!(width - 1), "int64")?;
        let last = b.register_with_result(&IAddOp::new(b.ctx_mut(), base, last));
        let full = b.register_with_result(&SLessThanOp::new(b.ctx_mut(), last, scalars[0]));
        let op = IfOp::new(b.ctx_mut(), full);
        let s = b.branch_child(OpInserter::new_at_block_end(op.then_block(b.ctx())));
        let mut loaded = HashMap::new();
        for mem in ops.iter().filter(|v| v["op"] == "load") {
            let idx = pointwise::index(&s, &mem["index"], base, &scalars[1..])?;
            let ptr = pointer(&s, arrays[mem["buffer"].as_u64().unwrap() as usize], idx);
            let ptr = if mem["index"]["op"] == "constant" {
                ptr
            } else {
                vector_pointer(&s, ptr, mem["dtype"].as_str().unwrap(), width)?
            };
            loaded.insert(
                mem["id"].as_str().unwrap(),
                s.register_with_result(&LoadOp::new(s.ctx_mut(), ptr)),
            );
        }
        let mut stored: HashMap<&str, Vec<IrValue>> = HashMap::new();
        for lane in 0..width {
            let offset = pointwise::constant(&s, &json!(lane), "int64")?;
            let linear = s.register_with_result(&IAddOp::new(s.ctx_mut(), base, offset));
            let mut values = HashMap::new();
            for v in ops {
                let id = v["id"].as_str().unwrap_or("");
                if v["op"] == "load" {
                    let value = loaded[id];
                    let value = if v["index"]["op"] == "constant" {
                        value
                    } else {
                        s.register_with_result(&CompositeExtractOp::new(s.ctx_mut(), value, lane))
                    };
                    values.insert(id.to_string(), value);
                } else if v["op"] == "store" {
                    stored
                        .entry(id)
                        .or_default()
                        .push(values[v["args"][0].as_str().unwrap()]);
                } else {
                    emit(
                        &s,
                        std::slice::from_ref(v),
                        &arrays,
                        linear,
                        &scalars[1..],
                        &mut values,
                        schema,
                    )?;
                }
            }
        }
        for v in ops.iter().filter(|v| v["op"] == "store") {
            let ptr = vector_pointer(
                &s,
                pointer(&s, arrays[v["buffer"].as_u64().unwrap() as usize], base),
                v["dtype"].as_str().unwrap(),
                width,
            )?;
            let value = s.register_with_result(&CompositeConstructOp::new(
                s.ctx_mut(),
                vector_type(&s, v["dtype"].as_str().unwrap(), width)?,
                stored.remove(v["id"].as_str().unwrap_or("")).unwrap(),
            ));
            s.register(&StoreOp::new(s.ctx_mut(), ptr, value));
        }
        s.terminate_yield();
        let s = b.branch_child(OpInserter::new_at_block_end(op.else_block(b.ctx())));
        for lane in 0..width {
            let offset = pointwise::constant(&s, &json!(lane), "int64")?;
            let linear = s.register_with_result(&IAddOp::new(s.ctx_mut(), base, offset));
            let valid = s.register_with_result(&SLessThanOp::new(s.ctx_mut(), linear, scalars[0]));
            branch(&s, valid, |child| {
                emit(
                    child,
                    ops,
                    &arrays,
                    linear,
                    &scalars[1..],
                    &mut HashMap::new(),
                    schema,
                )
            })?;
        }
        s.terminate_yield();
        b.register(&op);
    }
    b.register(&ReturnOp::new(b.ctx_mut()));
    Ok(b.build())
}

// Vector memory + scalar semantic replay preserves every low-precision barrier.
// The physical adapter supplies a reduction coordinate; no IR handle is encoded
// in the semantic description. Uniform loads stay scalar broadcasts.
pub(crate) fn reduction_vector_values(
    s: &Scope,
    ops: &[Value],
    arrays: &[IrValue],
    outer: IrValue,
    args: &[IrValue],
    ready: &HashMap<String, IrValue>,
    request: (&str, usize),
) -> Result<Vec<IrValue>, Error> {
    let (result, width) = request;
    let slot = args.len() - 1;
    let mut loaded = HashMap::new();
    for op in ops.iter().filter(|v| v["op"] == "load") {
        let coordinate = pointwise::index(s, &op["index"], outer, args)?;
        let ptr = pointer(
            s,
            arrays[op["buffer"].as_u64().unwrap() as usize],
            coordinate,
        );
        let vector = reduction_coefficient(&op["index"], Some(slot))
            .ok_or("vector reduction-axis coefficient")?
            == 1;
        let ptr = if vector {
            vector_pointer(s, ptr, op["dtype"].as_str().unwrap(), width)?
        } else {
            ptr
        };
        loaded.insert(
            op["id"].as_str().unwrap(),
            (
                s.register_with_result(&LoadOp::new(s.ctx_mut(), ptr)),
                vector,
            ),
        );
    }
    let mut results = Vec::new();
    for lane in 0..width {
        let mut values = ready.clone();
        let offset = pointwise::constant(s, &json!(lane), "int64")?;
        let coordinate = s.register_with_result(&IAddOp::new(s.ctx_mut(), args[slot], offset));
        let mut args = args.to_vec();
        args[slot] = coordinate;
        for op in ops {
            if op["op"] == "load" {
                let (value, vector) = loaded[op["id"].as_str().unwrap()];
                let value = if vector {
                    s.register_with_result(&CompositeExtractOp::new(s.ctx_mut(), value, lane))
                } else {
                    value
                };
                values.insert(op["id"].as_str().unwrap().into(), value);
            } else {
                pointwise::emit(
                    s,
                    std::slice::from_ref(op),
                    arrays,
                    outer,
                    &args,
                    &mut values,
                )?;
            }
        }
        results.push(values[result]);
    }
    Ok(results)
}

fn reduction_coefficient(v: &Value, bound_slot: Option<usize>) -> Option<i64> {
    match v["op"].as_str()? {
        "reduction" => Some(1),
        "scalar" if bound_slot.is_some_and(|slot| v["slot"] == slot) => Some(1),
        "linear" | "scalar" | "constant" => Some(0),
        "add" => v["args"].as_array()?.iter().try_fold(0i64, |n, x| {
            n.checked_add(reduction_coefficient(x, bound_slot)?)
        }),
        "mul" => {
            let args = v["args"].as_array()?;
            let mut coefficient = 0i64;
            for (i, x) in args.iter().enumerate() {
                let c = reduction_coefficient(x, bound_slot)?;
                if c != 0 {
                    if coefficient != 0 {
                        return None;
                    }
                    coefficient = args.iter().enumerate().filter(|(j, _)| *j != i).try_fold(
                        c,
                        |n, (_, a)| {
                            if a["op"] != "constant" {
                                return None;
                            }
                            n.checked_mul(a["value"].as_i64()?)
                        },
                    )?;
                }
            }
            Some(coefficient)
        }
        _ => None,
    }
}
