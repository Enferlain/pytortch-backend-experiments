//! Stage4 stateful/data-dependent semantics. Existing scalar physical builder is reused.
use crate::descriptor2::{self, constant, elem};
use cubecl_core::ir::{
    ConstantValue, ElemType, Scope, UIntKind,
    dialect::{
        bitwise::{BitwiseXorOp, ShiftRightOp},
        general::CastOp,
        math::{IAddOp, IMulOp},
    },
    pliron::{attribute::boxed_attr_cast, builtin::ops::ConstantOp, value::Value as IrValue},
};
use serde_json::{Value, json};
use std::collections::{HashMap, HashSet};
type Error = Box<dyn std::error::Error>;
fn cast(s: &Scope, v: IrValue, t: ElemType) -> IrValue {
    s.register_with_result(&CastOp::new(s.ctx_mut(), t.to_type(s.ctx()), v))
}
fn uconst(s: &Scope, v: u64) -> IrValue {
    let attr = ConstantValue::UInt(v).as_attribute(s.ctx(), ElemType::UInt(UIntKind::U64));
    s.register_with_result(&ConstantOp::new(
        s.ctx_mut(),
        boxed_attr_cast(attr).unwrap(),
    ))
}
fn shift(s: &Scope, v: IrValue, bits: u64) -> IrValue {
    let n = uconst(s, bits);
    s.register_with_result(&ShiftRightOp::new(s.ctx_mut(), v, n))
}
fn xor(s: &Scope, a: IrValue, b: IrValue) -> IrValue {
    s.register_with_result(&BitwiseXorOp::new(s.ctx_mut(), a, b))
}
fn low(s: &Scope, v: IrValue) -> IrValue {
    cast(
        s,
        cast(s, v, ElemType::UInt(UIntKind::U32)),
        ElemType::UInt(UIntKind::U64),
    )
}
// Random123 Philox4x32-10: key=seed bits, counter=(logical counter,0,0,0).
// Explicit modulo-2^32 operations; float conversion uses the top24 bits, [0,1).
fn philox(s: &Scope, seed: IrValue, counter: IrValue) -> Result<IrValue, Error> {
    let seed = cast(s, seed, ElemType::UInt(UIntKind::U64));
    // Inductor random lowering explicitly requests i32. The backend guards
    // its logical extent; this bit ABI is separate from signed-i64 addresses.
    // The replay counter is i32. Preserve its 32 bits before widening; a
    // negative bit pattern must not sign-extend into a second counter word.
    let counter = cast(
        s,
        cast(s, counter, ElemType::UInt(UIntKind::U32)),
        ElemType::UInt(UIntKind::U64),
    );
    let mut c = [low(s, counter), uconst(s, 0), uconst(s, 0), uconst(s, 0)];
    let mut k = [low(s, seed), shift(s, seed, 32)];
    for round in 0..10 {
        let m0 = uconst(s, 0xD2511F53);
        let m1 = uconst(s, 0xCD9E8D57);
        let p0 = s.register_with_result(&IMulOp::new(s.ctx_mut(), c[0], m0));
        let p1 = s.register_with_result(&IMulOp::new(s.ctx_mut(), c[2], m1));
        c = [
            xor(s, xor(s, shift(s, p1, 32), c[1]), k[0]),
            low(s, p1),
            xor(s, xor(s, shift(s, p0, 32), c[3]), k[1]),
            low(s, p0),
        ];
        if round < 9 {
            for (j, w) in [0x9E3779B9, 0xBB67AE85].iter().enumerate() {
                let w = uconst(s, *w);
                k[j] = low(
                    s,
                    s.register_with_result(&IAddOp::new(s.ctx_mut(), k[j], w)),
                );
            }
        }
    }
    let bits = shift(s, c[0], 8);
    let f = cast(s, bits, elem("float32")?);
    let scale = constant(s, &json!(1.0 / 16777216.0), "float32")?;
    Ok(
        s.register_with_result(&cubecl_core::ir::dialect::math::FMulOp::new(
            s.ctx_mut(),
            f,
            scale,
        )),
    )
}
fn validate(
    ops: &[Value],
    buffers: &[Value],
    scalars: &[Value],
    values: &mut HashMap<String, String>,
    depth: usize,
) -> Result<(), Error> {
    validate_region(ops, buffers, scalars, values, &mut HashSet::new(), depth)
}
fn validate_region(
    ops: &[Value],
    buffers: &[Value],
    scalars: &[Value],
    values: &mut HashMap<String, String>,
    checked: &mut HashSet<String>,
    depth: usize,
) -> Result<(), Error> {
    if depth > 8 || ops.len() > 4096 {
        return Err("Stage4 region limit".into());
    }
    for op in ops {
        let name = op["op"].as_str().ok_or("operation missing")?;
        if matches!(name, "load" | "store" | "atomic_add") {
            let buffer = op["buffer"].as_u64().ok_or("memory buffer")? as usize;
            let b = buffers.get(buffer).ok_or("memory buffer out of range")?;
            let ty = op["dtype"].as_str().ok_or("memory dtype")?;
            if b["dtype"] != ty
                || op["semantic_dtype"] != ty
                || b["access"] != if name == "load" { "read" } else { "write" }
            {
                return Err("memory access/type mismatch".into());
            }
            checked_index(&op["index"], scalars, values, checked)?;
            let args = op["args"].as_array().ok_or("memory operands")?;
            if name == "load" {
                if !args.is_empty() {
                    return Err("load arity".into());
                }
                let id = op["id"].as_str().ok_or("load SSA")?;
                if values.insert(id.into(), ty.into()).is_some() {
                    return Err("duplicate SSA".into());
                }
            } else {
                if args.len() != 1
                    || values
                        .get(args[0].as_str().ok_or("store SSA")?)
                        .map(String::as_str)
                        != Some(ty)
                {
                    return Err("store conversion/arity".into());
                }
                if name == "atomic_add" && (ty != "float32" || op["effect"] != "atomic_rmw") {
                    return Err("unsupported atomic dtype/effect".into());
                }
            }
        } else if name == "indirect" {
            let args = op["args"].as_array().ok_or("indirect operands")?;
            if args.len() != 1
                || values
                    .get(args[0].as_str().ok_or("index SSA")?)
                    .map(String::as_str)
                    != Some("int64")
                || op["dtype"] != "int64"
                || op["semantic_dtype"] != "int64"
                || op["bounds"] != "require_in_range"
            {
                return Err("unsupported indirect bounds/dtype".into());
            }
            descriptor2::check_index(&op["size"], scalars, false)?;
            let id = op["id"].as_str().ok_or("index result")?;
            if values.insert(id.into(), "int64".into()).is_some() {
                return Err("duplicate SSA".into());
            }
            checked.insert(id.into());
        } else if matches!(name, "iadd" | "imul" | "imin" | "imax" | "and") {
            let args = op["args"].as_array().ok_or("integer operands")?;
            let ty = if name == "and" { "bool" } else { "int64" };
            if args.len() != 2
                || op["dtype"] != ty
                || op["semantic_dtype"] != ty
                || args
                    .iter()
                    .any(|a| values.get(a.as_str().unwrap_or("")).map(String::as_str) != Some(ty))
            {
                return Err("unsupported internal integer operation".into());
            }
            let id = op["id"].as_str().ok_or("integer SSA")?;
            if values.insert(id.into(), ty.into()).is_some() {
                return Err("duplicate SSA".into());
            }
        } else if name == "masked" {
            let args = op["args"].as_array().ok_or("mask operands")?;
            if args.len() != 1
                || values
                    .get(args[0].as_str().unwrap_or(""))
                    .map(String::as_str)
                    != Some("bool")
            {
                return Err("mask predicate".into());
            }
            let mut inner = values.clone();
            let mut inner_checked = checked.clone();
            validate_region(
                op["body"].as_array().ok_or("mask body")?,
                buffers,
                scalars,
                &mut inner,
                &mut inner_checked,
                depth + 1,
            )?;
            let ty = op["dtype"].as_str().ok_or("mask dtype")?;
            elem(ty)?;
            if ty == "float64" {
                // float64 exists only as a scalar_value read before its cast.
                return Err("mask result dtype float64".into());
            }
            if inner
                .get(op["result"].as_str().ok_or("mask result")?)
                .map(String::as_str)
                != Some(ty)
            {
                return Err("mask result type".into());
            }
            constant_fill_check(op, ty)?;
            let id = op["id"].as_str().ok_or("mask SSA")?;
            if values.insert(id.into(), ty.into()).is_some() {
                return Err("duplicate SSA".into());
            }
        } else if name == "rand" {
            let args = op["args"].as_array().ok_or("rand operands")?;
            if args.len() != 2
                || values
                    .get(args[0].as_str().ok_or("seed")?)
                    .map(String::as_str)
                    != Some("int64")
                || values
                    .get(args[1].as_str().ok_or("counter")?)
                    .map(String::as_str)
                    != Some("int32")
                || op["dtype"] != "float32"
                || op["semantic_dtype"] != "float32"
                || op["algorithm"] != "philox4x32_10_top24"
            {
                return Err("unsupported RNG contract".into());
            }
            let id = op["id"].as_str().ok_or("RNG SSA")?;
            if values.insert(id.into(), "float32".into()).is_some() {
                return Err("duplicate RNG SSA".into());
            }
        } else if name == "constant" && op["dtype"] == "int32" {
            let v = op["value"].as_i64().ok_or("counter constant bits")?;
            if !op["args"].as_array().is_some_and(Vec::is_empty)
                || op["semantic_dtype"] != "int32"
                || !(i32::MIN as i64..=i32::MAX as i64).contains(&v)
            {
                return Err("unsupported counter constant width".into());
            }
            let id = op["id"].as_str().ok_or("counter SSA")?;
            if values.insert(id.into(), "int32".into()).is_some() {
                return Err("duplicate counter SSA".into());
            }
        } else if name == "cast" && op["dtype"] == "int32" {
            let args = op["args"].as_array().ok_or("counter cast operands")?;
            if args.len() != 1
                || op["semantic_dtype"] != "int32"
                || values
                    .get(args[0].as_str().ok_or("counter operand")?)
                    .map(String::as_str)
                    != Some("int64")
            {
                return Err("unsupported counter conversion".into());
            }
            let id = op["id"].as_str().ok_or("counter SSA")?;
            if values.insert(id.into(), "int32".into()).is_some() {
                return Err("duplicate counter SSA".into());
            }
        } else if name == "index" && op["dtype"] == "int32" {
            descriptor2::check_index(&op["index"], scalars, true)?;
            if op["semantic_dtype"] != "int32" {
                return Err("index semantic dtype".into());
            }
            let id = op["id"].as_str().ok_or("index SSA")?;
            if values.insert(id.into(), "int32".into()).is_some() {
                return Err("duplicate SSA".into());
            }
        } else {
            descriptor2::validate_ops(std::slice::from_ref(op), buffers, scalars, values, depth)?;
        }
    }
    Ok(())
}
pub(crate) fn emit(
    s: &Scope,
    ops: &[Value],
    arrays: &[IrValue],
    linear: IrValue,
    scalars: &[IrValue],
    values: &mut HashMap<String, IrValue>,
) -> Result<(), Error> {
    for op in ops {
        let value = if matches!(op["op"].as_str(), Some("load" | "store" | "atomic_add")) {
            use cubecl_core::ir::{
                dialect::{
                    atomic::AtomicFAddOp,
                    memory::{IndexOp, LoadOp, StoreOp},
                },
                types::scalar::IndexType,
            };
            let idx = address(s, &op["index"], linear, scalars, values)?;
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
            if op["op"] == "load" {
                Some(s.register_with_result(&LoadOp::new(s.ctx_mut(), ptr)))
            } else {
                let arg = values[op["args"][0].as_str().unwrap()];
                if op["op"] == "atomic_add" {
                    s.register_with_result(&AtomicFAddOp::new(s.ctx_mut(), ptr, arg));
                } else {
                    s.register(&StoreOp::new(s.ctx_mut(), ptr, arg));
                }
                None
            }
        } else if op["op"] == "indirect" {
            use cubecl_core::ir::dialect::{
                bitwise::BitwiseAndOp,
                cmp::{SGreaterThanOrEqualOp, SLessThanOp},
            };
            let arg = values[op["args"][0].as_str().unwrap()];
            let size = descriptor2::index(s, &op["size"], linear, scalars)?;
            let zero = constant(s, &json!(0), "int64")?;
            let lower = s.register_with_result(&SGreaterThanOrEqualOp::new(s.ctx_mut(), arg, zero));
            let upper = s.register_with_result(&SLessThanOp::new(s.ctx_mut(), arg, size));
            let valid = s.register_with_result(&BitwiseAndOp::new(s.ctx_mut(), lower, upper));
            s.register(&crate::require::RequireOp::new(s.ctx_mut(), valid));
            Some(arg)
        } else if matches!(
            op["op"].as_str(),
            Some("iadd" | "imul" | "imin" | "imax" | "and")
        ) {
            let a = values[op["args"][0].as_str().unwrap()];
            let b = values[op["args"][1].as_str().unwrap()];
            Some(match op["op"].as_str().unwrap() {
                "iadd" => s.register_with_result(&IAddOp::new(s.ctx_mut(), a, b)),
                "imul" => s.register_with_result(&IMulOp::new(s.ctx_mut(), a, b)),
                "imin" => s.register_with_result(&cubecl_core::ir::dialect::cmp::SMinOp::new(
                    s.ctx_mut(),
                    a,
                    b,
                )),
                "imax" => s.register_with_result(&cubecl_core::ir::dialect::cmp::SMaxOp::new(
                    s.ctx_mut(),
                    a,
                    b,
                )),
                "and" => s.register_with_result(
                    &cubecl_core::ir::dialect::bitwise::BitwiseAndOp::new(s.ctx_mut(), a, b),
                ),
                _ => unreachable!(),
            })
        } else if op["op"] == "rand" {
            let args = op["args"].as_array().unwrap();
            Some(philox(
                s,
                values[args[0].as_str().unwrap()],
                values[args[1].as_str().unwrap()],
            )?)
        } else if op["op"] == "index" && op["dtype"] == "int32" {
            let v = descriptor2::index(s, &op["index"], linear, scalars)?;
            Some(cast(s, v, elem("int32")?))
        } else {
            descriptor2::emit_with(
                s,
                std::slice::from_ref(op),
                arrays,
                linear,
                scalars,
                values,
                emit,
            )?;
            None
        };
        if let Some(v) = value {
            values.insert(op["id"].as_str().unwrap().into(), v);
        }
    }
    Ok(())
}
pub(crate) fn validate_description(d: &Value) -> Result<(), Error> {
    let ops = d["ops"].as_array().ok_or("operation list")?;
    fn contains(ops: &[Value], name: &str) -> bool {
        ops.iter().any(|op| {
            op["op"] == name
                || op["body"]
                    .as_array()
                    .is_some_and(|body| contains(body, name))
        })
    }
    let rng = contains(ops, "rand");
    if rng
        && d["numel"]["op"] == "constant"
        && d["numel"]["value"]
            .as_i64()
            .is_some_and(|n| n > i32::MAX as i64)
    {
        return Err("unsupported RNG physical extent".into());
    }
    let atomic = contains(ops, "atomic_add");
    if d["effects"]
        != json!({"rng":if rng {Some("explicit_seed_counter")} else {None},"mutation":atomic})
    {
        return Err("unsupported state/effect declaration".into());
    }
    descriptor2::validate_description(d, 4, validate)
}

pub fn build(d: &Value) -> Result<cubecl_runtime::kernel::KernelDefinition, Error> {
    validate_description(d)?;
    descriptor2::build_with(d, 4, validate, emit)
}

fn checked_index(
    v: &Value,
    scalars: &[Value],
    values: &HashMap<String, String>,
    checked: &HashSet<String>,
) -> Result<(), Error> {
    if v["op"] == "value" {
        if !checked.contains(v["id"].as_str().ok_or("indirect reference")?) {
            return Err("address requires a checked indirect SSA".into());
        }
        if values
            .get(v["id"].as_str().ok_or("indirect reference")?)
            .map(String::as_str)
            != Some("int64")
        {
            return Err("indirect reference requires i64 SSA".into());
        }
        return Ok(());
    }
    if matches!(v["op"].as_str(), Some("add" | "mul" | "floor_div" | "mod")) {
        let args = v["args"].as_array().ok_or("address operands")?;
        if args.len() < 2 {
            return Err("address arity".into());
        }
        if matches!(v["op"].as_str(), Some("floor_div" | "mod")) {
            fn has_value(v: &Value) -> bool {
                v["op"] == "value"
                    || v["args"]
                        .as_array()
                        .is_some_and(|a| a.iter().any(has_value))
            }
            if has_value(v) {
                return Err("indirect quotient/modulo unsupported".into());
            }
            let mut direct = v.clone();
            direct["args"][0] = json!({"op":"constant","value":0});
            descriptor2::check_index(&direct, scalars, true)?;
        }
        for a in args {
            checked_index(a, scalars, values, checked)?
        }
        return Ok(());
    }
    descriptor2::check_index(v, scalars, true)
}
fn address(
    s: &Scope,
    v: &Value,
    linear: IrValue,
    scalars: &[IrValue],
    values: &HashMap<String, IrValue>,
) -> Result<IrValue, Error> {
    if v["op"] == "value" {
        return Ok(values[v["id"].as_str().unwrap()]);
    }
    if matches!(v["op"].as_str(), Some("add" | "mul")) {
        let args = v["args"].as_array().unwrap();
        let mut result = address(s, &args[0], linear, scalars, values)?;
        for a in &args[1..] {
            let rhs = address(s, a, linear, scalars, values)?;
            result = if v["op"] == "add" {
                s.register_with_result(&IAddOp::new(s.ctx_mut(), result, rhs))
            } else {
                s.register_with_result(&IMulOp::new(s.ctx_mut(), result, rhs))
            };
        }
        return Ok(result);
    }
    descriptor2::index(s, v, linear, scalars)
}

fn constant_fill_check(op: &Value, ty: &str) -> Result<(), Error> {
    if !op["other"].is_number()
        && !op["other"].is_boolean()
        && !matches!(op["other"].as_str(), Some("nan" | "inf" | "-inf"))
    {
        return Err("mask fill".into());
    }
    if matches!(ty, "int64" | "int32") {
        let value = op["other"].as_i64().ok_or("integer fill")?;
        if ty == "int32" && !(i32::MIN as i64..=i32::MAX as i64).contains(&value) {
            return Err("integer fill width".into());
        }
    }
    if ty == "bool" && !op["other"].is_boolean() {
        return Err("bool fill".into());
    }
    Ok(())
}
