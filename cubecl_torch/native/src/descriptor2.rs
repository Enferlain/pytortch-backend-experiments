//! Stage 2 typed semantics. Physical choices remain a separate, checked plan.
use cubecl_core::{
    compute::KernelBuilder,
    ir::{
        AddressType, Builtin, ConstantValue, ElemType, FloatKind, IntKind, OpInserter, Scope,
        dialect::{
            branch::{IfOp, ReturnOp},
            cmp::*,
            general::{CastOp, ReadBuiltinOp, ReadScalarOp, ReinterpretCastOp, SelectOp},
            math::*,
            memory::{IndexOp, LoadOp, StoreOp},
        },
        pliron::{
            attribute::boxed_attr_cast, builtin::ops::ConstantOp, r#type::Typed,
            value::Value as IrValue,
        },
        settings::{Dim3, ExecutionMode, KernelSettings},
        types::scalar::IndexType,
    },
};
use serde_json::Value;
use std::collections::HashMap;
type Error = Box<dyn std::error::Error>;
pub(crate) fn elem(name: &str) -> Result<ElemType, Error> {
    Ok(match name {
        "float32" => ElemType::Float(FloatKind::F32),
        "float16" => ElemType::Float(FloatKind::F16),
        "bfloat16" => ElemType::Float(FloatKind::BF16),
        // Runtime float scalar values only; never storage or arithmetic.
        "float64" => ElemType::Float(FloatKind::F64),
        "bool" => ElemType::Bool,
        "int64" => ElemType::Int(IntKind::I64),
        "int32" => ElemType::Int(IntKind::I32),
        _ => return Err(format!("unsupported dtype {name}").into()),
    })
}
fn dtype(v: &Value) -> Result<&str, Error> {
    v["dtype"].as_str().ok_or_else(|| "dtype missing".into())
}
pub(crate) fn constant(s: &Scope, v: &Value, ty: &str) -> Result<IrValue, Error> {
    let cv = match ty {
        "int64" | "int32" => ConstantValue::Int(v.as_i64().ok_or("integer constant missing")?),
        "bool" => ConstantValue::Bool(v.as_bool().ok_or("bool constant missing")?),
        _ => ConstantValue::Float(if let Some(f) = v.as_f64() {
            f
        } else {
            match v.as_str() {
                Some("nan") => f64::NAN,
                Some("inf") => f64::INFINITY,
                Some("-inf") => f64::NEG_INFINITY,
                _ => return Err("invalid float constant".into()),
            }
        }),
    };
    let attr = cv.as_attribute(s.ctx(), elem(ty)?);
    Ok(s.register_with_result(&ConstantOp::new(
        s.ctx_mut(),
        boxed_attr_cast(attr).ok_or("constant attr")?,
    )))
}
// Symbol positivity is a descriptor ABI contract, checked again on runtime arguments.
fn positive_index(v: &Value, scalars: &[Value]) -> bool {
    match v["op"].as_str() {
        Some("constant") => v["value"].as_i64().is_some_and(|x| x > 0),
        Some("scalar") => v["slot"]
            .as_u64()
            .and_then(|slot| scalars.get(slot as usize))
            .is_some_and(|s| s["positive"] == true),
        Some("add" | "mul") => v["args"]
            .as_array()
            .is_some_and(|a| !a.is_empty() && a.iter().all(|x| positive_index(x, scalars))),
        _ => false,
    }
}
pub(crate) fn check_index(v: &Value, scalars: &[Value], linear: bool) -> Result<(), Error> {
    match v["op"].as_str().ok_or("index op missing")? {
        "linear" if linear => {}
        "constant" => {
            v["value"].as_i64().ok_or("index constant missing")?;
        }
        "scalar" => {
            let slot = v["slot"].as_u64().ok_or("scalar slot missing")? as usize;
            if slot >= scalars.len() {
                return Err("invalid scalar slot".into());
            }
            if scalars[slot]["dtype"] != "int64" {
                return Err("index expression references a float scalar value".into());
            }
        }
        "add" | "mul" | "floor_div" | "mod" => {
            let a = v["args"].as_array().ok_or("index args")?;
            let op = v["op"].as_str().unwrap();
            if a.len() < 2 || ((op == "mod" || op == "floor_div") && a.len() != 2) {
                return Err("index arity".into());
            }
            if (op == "mod" || op == "floor_div") && !positive_index(&a[1], scalars) {
                return Err("index divisor requires positive constant/scalar domain".into());
            }
            for x in a {
                check_index(x, scalars, linear)?;
            }
        }
        _ => return Err("unsupported index expression".into()),
    };
    Ok(())
}
pub(crate) fn validate_ops(
    ops: &[Value],
    buffers: &[Value],
    scalars: &[Value],
    values: &mut HashMap<String, String>,
    depth: usize,
) -> Result<(), Error> {
    if depth > 8 || ops.len() > 4096 {
        return Err("descriptor region limit".into());
    }
    for op in ops {
        let ty = dtype(op)?;
        // A float64 value exists only as a runtime scalar read, before its cast.
        let scalar_value = op["op"] == "scalar_value";
        if !(matches!(ty, "float32" | "float16" | "bfloat16" | "bool" | "int64")
            || scalar_value && ty == "float64")
            || !(matches!(
                op["semantic_dtype"].as_str(),
                Some("float32" | "float16" | "bfloat16" | "bool" | "int64")
            ) || scalar_value && op["semantic_dtype"] == "float64")
        {
            return Err("unsupported Stage2 value dtype".into());
        }
        elem(ty)?;
        elem(
            op["semantic_dtype"]
                .as_str()
                .ok_or("semantic dtype missing")?,
        )?;
        let args = op["args"].as_array().ok_or("operand list")?;
        let types: Vec<_> = args
            .iter()
            .map(|a| {
                a.as_str()
                    .and_then(|n| values.get(n))
                    .map(String::as_str)
                    .ok_or("unknown SSA operand")
            })
            .collect::<Result<_, _>>()?;
        let name = op["op"].as_str().ok_or("op missing")?;
        let arity = match name {
            "load" | "constant" | "index" | "scalar_value" => 0,
            "store" | "cast" | "sin" | "cos" | "neg" | "exp" | "sqrt" | "rsqrt" | "reciprocal"
            | "masked" => 1,
            "where" => 3,
            "add" | "sub" | "mul" | "truediv" | "eq" | "ne" | "lt" | "le" | "gt" | "ge" => 2,
            _ => return Err(format!("unsupported semantic op {name}").into()),
        };
        if types.len() != arity {
            return Err("operation arity".into());
        }
        match name {
            "load" | "store" => {
                let index = op["buffer"].as_u64().ok_or("buffer missing")? as usize;
                let b = buffers.get(index).ok_or("buffer out of range")?;
                if b["dtype"] != ty || b["access"] != if name == "load" { "read" } else { "write" }
                {
                    return Err("buffer access/type mismatch".into());
                }
                check_index(&op["index"], scalars, true)?;
                if name == "store" && types[0] != ty {
                    return Err("store conversion missing".into());
                }
            }
            "constant" => match ty {
                "int64" => {
                    op["value"].as_i64().ok_or("constant bits")?;
                }
                "bool" => {
                    op["value"].as_bool().ok_or("constant bool")?;
                }
                _ => {
                    if !op["value"].is_number()
                        && !matches!(op["value"].as_str(), Some("nan" | "inf" | "-inf"))
                    {
                        return Err("constant float".into());
                    }
                }
            },
            "index" => {
                if ty != "int64" {
                    return Err("index dtype".into());
                }
                check_index(&op["index"], scalars, true)?;
            }
            "masked" => {
                if types[0] != "bool" {
                    return Err("masked predicate".into());
                }
                let mut inner = values.clone();
                validate_ops(
                    op["body"].as_array().ok_or("region missing")?,
                    buffers,
                    scalars,
                    &mut inner,
                    depth + 1,
                )?;
                if inner
                    .get(op["result"].as_str().ok_or("region result")?)
                    .map(String::as_str)
                    != Some(ty)
                {
                    return Err("region result type".into());
                }
                if !op["other"].is_number()
                    && !op["other"].is_boolean()
                    && !matches!(op["other"].as_str(), Some("nan" | "inf" | "-inf"))
                {
                    return Err("masked fill missing".into());
                }
            }
            "where" => {
                if types[0] != "bool" || types[1] != ty || types[2] != ty {
                    return Err("select operand type".into());
                }
            }
            "eq" | "ne" | "lt" | "le" | "gt" | "ge" => {
                if ty != "bool" || types[0] != types[1] || !matches!(types[0], "float32" | "int64")
                {
                    return Err("comparison operand type".into());
                }
            }
            "scalar_value" => {
                let slot = op["slot"].as_u64().ok_or("scalar value slot")? as usize;
                if scalars.get(slot).map(|s| &s["dtype"]) != Some(&Value::from("float64"))
                    || ty != "float64"
                {
                    return Err("scalar value requires a float64 scalar slot".into());
                }
            }
            "cast" => {
                if types[0] == "float64" && !matches!(ty, "float32" | "float16" | "bfloat16") {
                    return Err("float64 scalar values cast only to floating compute".into());
                }
            }
            _ => {
                if ty != "float32" || types.iter().any(|t| *t != ty) {
                    return Err("math requires explicit float32 compute conversion".into());
                }
            }
        }
        if name != "store" {
            let id = op["id"].as_str().ok_or("SSA result missing")?;
            if values.insert(id.into(), ty.into()).is_some() {
                return Err("duplicate SSA result".into());
            }
        }
    }
    Ok(())
}
pub(crate) fn index(
    s: &Scope,
    v: &Value,
    linear: IrValue,
    scalars: &[IrValue],
) -> Result<IrValue, Error> {
    Ok(match v["op"].as_str().unwrap() {
        "linear" => linear,
        "constant" => constant(s, &v["value"], "int64")?,
        "scalar" => scalars[v["slot"].as_u64().unwrap() as usize],
        op => {
            let args = v["args"].as_array().unwrap();
            let mut result = index(s, &args[0], linear, scalars)?;
            for arg in &args[1..] {
                let rhs = index(s, arg, linear, scalars)?;
                result = match op {
                    "add" => s.register_with_result(&IAddOp::new(s.ctx_mut(), result, rhs)),
                    "mul" => s.register_with_result(&IMulOp::new(s.ctx_mut(), result, rhs)),
                    "mod" => s.register_with_result(&SModFloorOp::new(s.ctx_mut(), result, rhs)),
                    "floor_div" => {
                        // Positive divisor: adjust a truncating quotient only for a
                        // negative remainder. Avoid a-rem, which can overflow near i64::MIN.
                        let quotient =
                            s.register_with_result(&SDivOp::new(s.ctx_mut(), result, rhs));
                        let remainder =
                            s.register_with_result(&SRemOp::new(s.ctx_mut(), result, rhs));
                        let zero = constant(s, &serde_json::json!(0), "int64")?;
                        let one = constant(s, &serde_json::json!(1), "int64")?;
                        let negative =
                            s.register_with_result(&SLessThanOp::new(s.ctx_mut(), remainder, zero));
                        let adjusted =
                            s.register_with_result(&ISubOp::new(s.ctx_mut(), quotient, one));
                        s.register_with_result(&SelectOp::new(
                            s.ctx_mut(),
                            negative,
                            adjusted,
                            quotient,
                        ))
                    }
                    _ => unreachable!(),
                };
            }
            result
        }
    })
}
pub(crate) fn emit(
    s: &Scope,
    ops: &[Value],
    arrays: &[IrValue],
    linear: IrValue,
    scalars: &[IrValue],
    values: &mut HashMap<String, IrValue>,
) -> Result<(), Error> {
    emit_with(s, ops, arrays, linear, scalars, values, emit)
}
pub(crate) fn emit_with(
    s: &Scope,
    ops: &[Value],
    arrays: &[IrValue],
    linear: IrValue,
    scalars: &[IrValue],
    values: &mut HashMap<String, IrValue>,
    region_emit: Emitter,
) -> Result<(), Error> {
    for op in ops {
        let name = op["op"].as_str().unwrap();
        let ty = dtype(op)?;
        let args: Vec<_> = op["args"]
            .as_array()
            .unwrap()
            .iter()
            .map(|a| values[a.as_str().unwrap()])
            .collect();
        macro_rules! unary {
            ($t:ty) => {
                s.register_with_result(&<$t>::new(s.ctx_mut(), args[0]))
            };
        }
        macro_rules! binary {
            ($t:ty) => {
                s.register_with_result(&<$t>::new(s.ctx_mut(), args[0], args[1]))
            };
        }
        let out = match name {
            "constant" => constant(s, &op["value"], ty)?,
            "index" => index(s, &op["index"], linear, scalars)?,
            // Float scalar values travel as their bit pattern in an i64 metadata slot.
            "scalar_value" => s.register_with_result(&ReinterpretCastOp::new(
                s.ctx_mut(),
                elem("float64")?.to_type(s.ctx()),
                scalars[op["slot"].as_u64().unwrap() as usize],
            )),
            "cast" => s.register_with_result(&CastOp::new(
                s.ctx_mut(),
                elem(ty)?.to_type(s.ctx()),
                args[0],
            )),
            "load" | "store" => {
                let idx = index(s, &op["index"], linear, scalars)?;
                let idx = s.register_with_result(&CastOp::new(
                    s.ctx_mut(),
                    IndexType::get(s.ctx()).to_handle(),
                    idx,
                ));
                let ptr = s.register_with_result(&IndexOp::new(
                    s.ctx_mut(),
                    arrays[op["buffer"].as_u64().unwrap() as usize],
                    idx,
                    None,
                ));
                if name == "store" {
                    s.register(&StoreOp::new(s.ctx_mut(), ptr, args[0]));
                    continue;
                }
                s.register_with_result(&LoadOp::new(s.ctx_mut(), ptr))
            }
            "sin" => unary!(SinOp),
            "cos" => unary!(CosOp),
            "neg" => unary!(FNegOp),
            "exp" => unary!(ExpOp),
            "sqrt" => unary!(SqrtOp),
            "rsqrt" => unary!(RsqrtOp),
            "reciprocal" => {
                let one = constant(s, &serde_json::json!(1.0), "float32")?;
                s.register_with_result(&FDivOp::new(s.ctx_mut(), one, args[0]))
            }
            "add" => binary!(FAddOp),
            "sub" => binary!(FSubOp),
            "mul" => binary!(FMulOp),
            "truediv" => binary!(FDivOp),
            "eq" => {
                if args[0].get_type(s.ctx()) == elem("int64")?.to_type(s.ctx()) {
                    binary!(IEqualOp)
                } else {
                    binary!(FEqualOp)
                }
            }
            "ne" => {
                if args[0].get_type(s.ctx()) == elem("int64")?.to_type(s.ctx()) {
                    binary!(INotEqualOp)
                } else {
                    binary!(FNotEqualOp)
                }
            }
            "lt" => {
                if args[0].get_type(s.ctx()) == elem("int64")?.to_type(s.ctx()) {
                    binary!(SLessThanOp)
                } else {
                    binary!(FLessThanOp)
                }
            }
            "le" => {
                if args[0].get_type(s.ctx()) == elem("int64")?.to_type(s.ctx()) {
                    binary!(SLessThanOrEqualOp)
                } else {
                    binary!(FLessThanOrEqualOp)
                }
            }
            "gt" => {
                if args[0].get_type(s.ctx()) == elem("int64")?.to_type(s.ctx()) {
                    binary!(SGreaterThanOp)
                } else {
                    binary!(FGreaterThanOp)
                }
            }
            "ge" => {
                if args[0].get_type(s.ctx()) == elem("int64")?.to_type(s.ctx()) {
                    binary!(SGreaterThanOrEqualOp)
                } else {
                    binary!(FGreaterThanOrEqualOp)
                }
            }
            "where" => {
                s.register_with_result(&SelectOp::new(s.ctx_mut(), args[0], args[1], args[2]))
            }
            "masked" => {
                let local = s.create_local_mut(elem(ty)?.to_type(s.ctx()), None);
                let branch = IfOp::new(s.ctx_mut(), args[0]);
                {
                    let child =
                        s.branch_child(OpInserter::new_at_block_end(branch.then_block(s.ctx())));
                    let mut inner = values.clone();
                    region_emit(
                        &child,
                        op["body"].as_array().unwrap(),
                        arrays,
                        linear,
                        scalars,
                        &mut inner,
                    )?;
                    child.register(&StoreOp::new(
                        child.ctx_mut(),
                        local,
                        inner[op["result"].as_str().unwrap()],
                    ));
                    child.terminate_yield();
                }
                {
                    let child =
                        s.branch_child(OpInserter::new_at_block_end(branch.else_block(s.ctx())));
                    let fill = constant(&child, &op["other"], ty)?;
                    child.register(&StoreOp::new(child.ctx_mut(), local, fill));
                    child.terminate_yield();
                }
                s.register(&branch);
                s.register_with_result(&LoadOp::new(s.ctx_mut(), local))
            }
            _ => unreachable!(),
        };
        values.insert(op["id"].as_str().unwrap().into(), out);
    }
    Ok(())
}
pub fn build(d: &Value) -> Result<cubecl_runtime::kernel::KernelDefinition, Error> {
    build_with(d, 2, validate_ops, emit)
}

type Validator =
    fn(&[Value], &[Value], &[Value], &mut HashMap<String, String>, usize) -> Result<(), Error>;
type Emitter = fn(
    &Scope,
    &[Value],
    &[IrValue],
    IrValue,
    &[IrValue],
    &mut HashMap<String, IrValue>,
) -> Result<(), Error>;
pub(crate) fn validate_description(
    d: &Value,
    schema: u64,
    validate: Validator,
) -> Result<(), Error> {
    if d["schema"] != schema || d["index_dtype"] != "int64" {
        return Err("unsupported semantic schema".into());
    }
    let buffers = d["buffers"].as_array().ok_or("buffers missing")?;
    if !(1..=if schema == 4 { 6 } else { 5 }).contains(&buffers.len()) {
        return Err("buffer count unsupported".into());
    }
    let scalar_defs = d["scalars"].as_array().ok_or("scalars missing")?;
    let ns = scalar_defs.len();
    if ns > 32 {
        return Err("scalar limit".into());
    }
    for (i, scalar) in d["scalars"].as_array().unwrap().iter().enumerate() {
        if !(scalar["dtype"] == "int64"
            || scalar["dtype"] == "float64" && scalar["positive"] != true)
            || scalar["id"] != i
            || (scalar.get("positive").is_some() && !scalar["positive"].is_boolean())
        {
            return Err("scalar schema".into());
        }
    }
    check_index(&d["numel"], scalar_defs, false)?;
    for (i, b) in buffers.iter().enumerate() {
        if b["access"]
            != if i == buffers.len() - 1 {
                "write"
            } else {
                "read"
            }
            || !(matches!(dtype(b)?, "float32" | "float16" | "bfloat16" | "bool")
                || schema == 4 && dtype(b)? == "int64")
        {
            return Err("storage access/dtype".into());
        }
        let sizes = b["size"].as_array().ok_or("sizes")?;
        let strides = b["stride"].as_array().ok_or("strides")?;
        if sizes.len() > 4 || sizes.len() != strides.len() {
            return Err("layout rank".into());
        }
        for x in sizes.iter().chain(strides) {
            check_index(x, scalar_defs, false)?;
        }
        check_index(&b["offset"], scalar_defs, false)?;
    }
    let ops = d["ops"].as_array().ok_or("ops missing")?;
    validate(ops, buffers, scalar_defs, &mut HashMap::new(), 0)?;
    Ok(())
}

pub(crate) fn build_with(
    d: &Value,
    schema: u64,
    validate: Validator,
    emit: Emitter,
) -> Result<cubecl_runtime::kernel::KernelDefinition, Error> {
    if d["schema"] != schema || d["index_dtype"] != "int64" {
        return Err("unsupported Stage 2 schema".into());
    }
    let entry = d["entrypoint"].as_str().ok_or("entry missing")?;
    if !entry.starts_with(&format!("cubecl{schema}_"))
        || !entry.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
    {
        return Err("entry invalid".into());
    }
    let plan = &d["physical_plan"];
    if plan["workgroup"] != serde_json::json!([256, 1, 1])
        || plan["iteration_mapping"] != "absolute_linear_unit"
        || plan["elements_per_unit"] != 1
        || plan["local_vector_width"] != 1
        || plan["bounds_strategy"] != "explicit_less_than_numel"
        || plan["reduction_mapping"] != "none"
    {
        return Err("unsupported physical plan".into());
    }
    validate_description(d, schema, validate)?;
    let ops = d["ops"].as_array().unwrap();
    let buffers = d["buffers"].as_array().unwrap();
    let ns = d["scalars"].as_array().unwrap().len();
    // IR construction starts only after structural/type validation.
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
        .map(|v| b.buffer(elem(dtype(v).unwrap()).unwrap().to_type(b.ctx())))
        .collect();
    let ity = elem("int64")?.to_type(b.ctx());
    let scalars: Vec<_> = (0..=ns)
        .map(|_| {
            let id = b.scalar(elem("int64").unwrap());
            b.register_with_result(&ReadScalarOp::new(b.ctx_mut(), ity, id))
        })
        .collect();
    // Widen each builtin BEFORE multiplication: AbsolutePos can overflow a u32 multiply.
    let block = b.register_with_result(&ReadBuiltinOp::new(
        b.ctx_mut(),
        IndexType::get(b.ctx()).to_handle(),
        Builtin::CubePosX,
    ));
    let block = b.register_with_result(&CastOp::new(b.ctx_mut(), ity, block));
    let unit = b.register_with_result(&ReadBuiltinOp::new(
        b.ctx_mut(),
        IndexType::get(b.ctx()).to_handle(),
        Builtin::UnitPosX,
    ));
    let unit = b.register_with_result(&CastOp::new(b.ctx_mut(), ity, unit));
    let width = constant(&b, &serde_json::json!(256), "int64")?;
    let base = b.register_with_result(&IMulOp::new(b.ctx_mut(), block, width));
    let linear = b.register_with_result(&IAddOp::new(b.ctx_mut(), base, unit));
    let valid = b.register_with_result(&SLessThanOp::new(b.ctx_mut(), linear, scalars[0]));
    let branch = IfOp::new(b.ctx_mut(), valid);
    {
        let child = b.branch_child(OpInserter::new_at_block_end(branch.then_block(b.ctx())));
        emit(
            &child,
            ops,
            &arrays,
            linear,
            &scalars[1..],
            &mut HashMap::new(),
        )?;
        child.terminate_yield();
    }
    {
        let child = b.branch_child(OpInserter::new_at_block_end(branch.else_block(b.ctx())));
        child.terminate_yield();
    }
    b.register(&branch);
    b.register(&ReturnOp::new(b.ctx_mut()));
    Ok(b.build())
}
