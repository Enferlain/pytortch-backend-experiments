//! The semantic op layer shared by every descriptor schema: index
//! expressions, op validation and scalar replay into IR. Physical choices
//! remain a separate, checked plan.
use crate::ir::{constant, constant_value, elem, integer};
use cubecl_core::ir::{
    ElemType, OpInserter, Scope, UIntKind,
    dialect::{
        bitwise::{
            BitwiseAndOp, BitwiseNotOp, BitwiseOrOp, BitwiseXorOp, ShiftLeftOp, ShiftRightOp,
        },
        branch::IfOp,
        cmp::*,
        general::{BoolNotOp, CastOp, ReinterpretCastOp, SelectOp},
        math::*,
        memory::{IndexOp, LoadOp, StoreOp},
    },
    pliron::{r#type::Typed, value::Value as IrValue},
    types::scalar::IndexType,
};
use serde_json::Value;
use std::collections::HashMap;
type Error = Box<dyn std::error::Error>;
// A descriptor-size sanity bound, not a kernel limit (eager CUDA OffsetCalculator MAX_DIMS).
pub(crate) const MAX_RANK: usize = 25;
pub(crate) fn dtype(v: &Value) -> Result<&str, Error> {
    v["dtype"].as_str().ok_or_else(|| "dtype missing".into())
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
/// A value dtype's size in bytes.
fn size(ty: &str) -> Result<u32, Error> {
    Ok(match (ty, integer(ty)) {
        ("float32", _) => 4,
        ("float16" | "bfloat16", _) => 2,
        (_, Some((_, width))) => width / 8,
        _ => return Err(format!("no size for {ty}").into()),
    })
}
const FLOAT: &[&str] = &["float32"];
// Numbers exclude bool; INT also excludes floats; ANY is every value dtype.
const INT: &[&str] = &["int8", "int16", "int32", "int64", "uint8"];
const NUMBER: &[&str] = &["float32", "int8", "int16", "int32", "int64", "uint8"];
const ANY: &[&str] = &[
    "float32", "bool", "int8", "int16", "int32", "int64", "uint8",
];
// Integers with bool: the bitwise ops.
const INTEGER: &[&str] = &["bool", "int8", "int16", "int32", "int64", "uint8"];
/// Value-op typing: (op, arity, operand dtypes, result dtype; None is the
/// operands' dtype). Operands share one dtype. Mirrors Python's
/// `admission.OP_RULES`, whose float16/bfloat16 compute here as float32.
const VALUE_OPS: &[(&str, usize, &[&str], Option<&str>)] = &[
    ("sin", 1, FLOAT, None),
    ("cos", 1, FLOAT, None),
    ("neg", 1, NUMBER, None),
    ("abs", 1, NUMBER, None),
    ("exp", 1, FLOAT, None),
    ("sqrt", 1, FLOAT, None),
    ("rsqrt", 1, FLOAT, None),
    ("reciprocal", 1, FLOAT, None),
    ("add", 2, NUMBER, None),
    ("sub", 2, NUMBER, None),
    ("mul", 2, NUMBER, None),
    ("truediv", 2, FLOAT, None),
    // Explicit rounding: one rounding, a product never contracted, IEEE division.
    ("fma", 3, FLOAT, None),
    ("mul_rn", 2, FLOAT, None),
    ("div_rn", 2, FLOAT, None),
    ("maximum", 2, ANY, None),
    ("minimum", 2, ANY, None),
    ("bitwise_and", 2, INTEGER, None),
    ("bitwise_or", 2, INTEGER, None),
    ("bitwise_xor", 2, INTEGER, None),
    ("bitwise_not", 1, INTEGER, None),
    ("truncdiv", 2, INT, None),
    ("floordiv", 2, INT, None),
    ("mod", 2, INT, None),
    ("bitwise_left_shift", 2, INT, None),
    ("bitwise_right_shift", 2, INT, None),
    ("isnan", 1, FLOAT, Some("bool")),
    ("eq", 2, ANY, Some("bool")),
    ("ne", 2, ANY, Some("bool")),
    ("lt", 2, ANY, Some("bool")),
    ("le", 2, ANY, Some("bool")),
    ("gt", 2, ANY, Some("bool")),
    ("ge", 2, ANY, Some("bool")),
];
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
        let value = |t: &str| {
            matches!(t, "float32" | "float16" | "bfloat16")
                || integer(t).is_some()
                || scalar_value && t == "float64"
        };
        if !value(ty) || !value(op["semantic_dtype"].as_str().unwrap_or("")) {
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
        let rule = VALUE_OPS.iter().find(|(n, ..)| *n == name);
        let arity = match (name, rule) {
            (_, Some((_, arity, ..))) => *arity,
            ("load" | "constant" | "index" | "scalar_value", _) => 0,
            ("store" | "cast" | "bitcast" | "masked", _) => 1,
            ("where", _) => 3,
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
            "constant" => {
                constant_value(&op["value"], ty)?;
            }
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
                constant_value(&op["other"], ty).map_err(|e| format!("masked fill: {e}"))?;
            }
            "where" => {
                if types[0] != "bool" || types[1] != ty || types[2] != ty {
                    return Err("select operand type".into());
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
            "bitcast" => {
                // Never to or from bool: a reinterpreted byte need not be 0 or 1.
                if types[0] == "float64"
                    || types[0] == "bool"
                    || ty == "bool"
                    || size(types[0])? != size(ty)?
                {
                    return Err("bitcast requires equal sizes".into());
                }
            }
            "cast" => {
                if types[0] == "float64" && !matches!(ty, "float32" | "float16" | "bfloat16") {
                    return Err("float64 scalar values cast only to floating compute".into());
                }
            }
            _ => {
                let (_, _, operands, result) = rule.ok_or("value op without a typing rule")?;
                if !operands.contains(&types[0])
                    || types.iter().any(|t| *t != types[0])
                    || ty != result.unwrap_or(types[0])
                {
                    return Err(format!(
                        "{name} over {} requires explicit {} compute conversion",
                        types[0],
                        operands.join("/")
                    )
                    .into());
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
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Kind {
    Float,
    Signed,
    Unsigned,
    Bool,
}
/// How a value dtype computes: it selects each op's instruction.
pub(crate) fn kind(ty: &str) -> Kind {
    match integer(ty) {
        None => Kind::Float,
        Some((_, 1)) => Kind::Bool,
        Some((true, _)) => Kind::Signed,
        Some((false, _)) => Kind::Unsigned,
    }
}
/// The unsigned type of the same width for a signed dtype. C++ signed overflow
/// and left shifts into or of the sign bit are undefined, and NVRTC folds
/// compares on them, so signed add, sub, mul, neg and left shift run in this
/// type, where they wrap as eager's do.
fn wrapping_type(ty: &str) -> Option<ElemType> {
    match ty {
        "int8" => Some(ElemType::UInt(UIntKind::U8)),
        "int16" => Some(ElemType::UInt(UIntKind::U16)),
        "int32" => Some(ElemType::UInt(UIntKind::U32)),
        "int64" => Some(ElemType::UInt(UIntKind::U64)),
        _ => None,
    }
}
/// `op(a, b)` for integers of `ty`, computed in its wrapping type when it has
/// one. The reinterpretations are free on the GPU.
fn wrapping(
    s: &Scope,
    ty: &str,
    a: IrValue,
    b: IrValue,
    op: impl Fn(&Scope, IrValue, IrValue) -> IrValue,
) -> Result<IrValue, Error> {
    let Some(unsigned) = wrapping_type(ty) else {
        return Ok(op(s, a, b));
    };
    let reinterpret = |to: ElemType, v: IrValue| {
        let to = to.to_type(s.ctx());
        s.register_with_result(&ReinterpretCastOp::new(s.ctx_mut(), to, v))
    };
    let result = op(s, reinterpret(unsigned, a), reinterpret(unsigned, b));
    Ok(reinterpret(elem(ty)?, result))
}
/// The kind of an emitted value, from its IR element type.
fn value_kind(s: &Scope, v: IrValue) -> Result<Kind, Error> {
    let ty = v.get_type(s.ctx());
    for name in [
        "float32", "bool", "int8", "int16", "int32", "int64", "uint8",
    ] {
        if elem(name)?.to_type(s.ctx()) == ty {
            return Ok(kind(name));
        }
    }
    Err("comparison of an unsupported value type".into())
}
/// `a <name> b` for two values of one kind, with a bool result. Bool orders
/// False < True through an unsigned comparison of 0/1.
fn compare(s: &Scope, name: &str, kind: Kind, a: IrValue, b: IrValue) -> Result<IrValue, Error> {
    macro_rules! op {
        ($t:ty) => {
            s.register_with_result(&<$t>::new(s.ctx_mut(), a, b))
        };
    }
    Ok(match (kind, name) {
        (Kind::Bool, "eq") => op!(BoolEqualOp),
        (Kind::Bool, "ne") => op!(BoolNotEqualOp),
        (Kind::Bool, _) => {
            let byte = elem("uint8")?.to_type(s.ctx());
            let a = s.register_with_result(&CastOp::new(s.ctx_mut(), byte, a));
            let b = s.register_with_result(&CastOp::new(s.ctx_mut(), byte, b));
            return compare(s, name, Kind::Unsigned, a, b);
        }
        (Kind::Float, "eq") => op!(FEqualOp),
        (Kind::Float, "ne") => op!(FNotEqualOp),
        (Kind::Float, "lt") => op!(FLessThanOp),
        (Kind::Float, "le") => op!(FLessThanOrEqualOp),
        (Kind::Float, "gt") => op!(FGreaterThanOp),
        (Kind::Float, "ge") => op!(FGreaterThanOrEqualOp),
        (_, "eq") => op!(IEqualOp),
        (_, "ne") => op!(INotEqualOp),
        (Kind::Signed, "lt") => op!(SLessThanOp),
        (Kind::Signed, "le") => op!(SLessThanOrEqualOp),
        (Kind::Signed, "gt") => op!(SGreaterThanOp),
        (Kind::Signed, "ge") => op!(SGreaterThanOrEqualOp),
        (Kind::Unsigned, "lt") => op!(ULessThanOp),
        (Kind::Unsigned, "le") => op!(ULessThanOrEqualOp),
        (Kind::Unsigned, "gt") => op!(UGreaterThanOp),
        (Kind::Unsigned, "ge") => op!(UGreaterThanOrEqualOp),
        _ => return Err(format!("unsupported comparison {name}").into()),
    })
}
/// Integer division and C remainder as eager CUDA for every nonzero divisor.
/// A zero divisor is replaced by 1 before dividing, as Triton's floordiv does:
/// the result is unspecified, but NVRTC cannot exploit a division by zero.
fn divide(s: &Scope, name: &str, ty: &str, a: IrValue, b: IrValue) -> Result<IrValue, Error> {
    let signed = kind(ty) == Kind::Signed;
    let int = |v: i64| constant(s, &serde_json::json!(v), ty);
    let zero = int(0)?;
    let is_zero = compare(s, "eq", kind(ty), b, zero)?;
    let b = s.register_with_result(&SelectOp::new(s.ctx_mut(), is_zero, int(1)?, b));
    let quotient = || {
        if signed {
            s.register_with_result(&SDivOp::new(s.ctx_mut(), a, b))
        } else {
            s.register_with_result(&UDivOp::new(s.ctx_mut(), a, b))
        }
    };
    let remainder = || {
        if signed {
            s.register_with_result(&SRemOp::new(s.ctx_mut(), a, b))
        } else {
            s.register_with_result(&URemOp::new(s.ctx_mut(), a, b))
        }
    };
    Ok(match name {
        "mod" => remainder(),
        "floordiv" if signed => {
            // c10::div_floor_integer: one less when the remainder is nonzero
            // and its sign differs from the divisor's.
            let q = quotient();
            let r = remainder();
            let inexact = compare(s, "ne", Kind::Signed, r, zero)?;
            let r_negative = compare(s, "lt", Kind::Signed, r, zero)?;
            let b_negative = compare(s, "lt", Kind::Signed, b, zero)?;
            let signs = compare(s, "ne", Kind::Bool, r_negative, b_negative)?;
            let adjust = s.register_with_result(&BitwiseAndOp::new(s.ctx_mut(), inexact, signs));
            let lower = s.register_with_result(&ISubOp::new(s.ctx_mut(), q, int(1)?));
            s.register_with_result(&SelectOp::new(s.ctx_mut(), adjust, lower, q))
        }
        // truncdiv; unsigned floor division truncates too.
        _ => quotient(),
    })
}
/// `a << b` or `a >> b` as eager CUDA: an amount below 0 or at least the
/// width gives 0, or the sign fill for a signed right shift. The amount is
/// clamped before shifting, so no shift is out of range for NVRTC to exploit.
fn shift(s: &Scope, left: bool, ty: &str, a: IrValue, b: IrValue) -> Result<IrValue, Error> {
    let (signed, width) = integer(ty).ok_or("shift of a non-integer")?;
    let int = |v: i64| constant(s, &serde_json::json!(v), ty);
    let zero = int(0)?;
    let in_range = {
        let below = compare(s, "lt", kind(ty), b, int(i64::from(width))?)?;
        if signed {
            let nonnegative = compare(s, "ge", Kind::Signed, b, zero)?;
            s.register_with_result(&BitwiseAndOp::new(s.ctx_mut(), nonnegative, below))
        } else {
            below
        }
    };
    let amount = s.register_with_result(&SelectOp::new(s.ctx_mut(), in_range, b, zero));
    let (shifted, outside) = if left {
        let shifted = wrapping(s, ty, a, amount, |s, a, b| {
            s.register_with_result(&ShiftLeftOp::new(s.ctx_mut(), a, b))
        })?;
        (shifted, zero)
    } else {
        let shifted = s.register_with_result(&ShiftRightOp::new(s.ctx_mut(), a, amount));
        let fill = if signed {
            let last = int(i64::from(width) - 1)?;
            s.register_with_result(&ShiftRightOp::new(s.ctx_mut(), a, last))
        } else {
            zero
        };
        (shifted, fill)
    };
    Ok(s.register_with_result(&SelectOp::new(s.ctx_mut(), in_range, shifted, outside)))
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
        macro_rules! wrapping {
            ($t:ty, $a:expr, $b:expr) => {
                wrapping(s, ty, $a, $b, |s, a, b| {
                    s.register_with_result(&<$t>::new(s.ctx_mut(), a, b))
                })?
            };
        }
        // Integer and bool results select the integer instruction of an op.
        let int = kind(ty) != Kind::Float;
        let out = match name {
            "constant" => constant(s, &op["value"], ty)?,
            "index" => index(s, &op["index"], linear, scalars)?,
            // Float scalar values travel as their bit pattern in an i64 metadata slot.
            "scalar_value" => s.register_with_result(&ReinterpretCastOp::new(
                s.ctx_mut(),
                elem("float64")?.to_type(s.ctx()),
                scalars[op["slot"].as_u64().unwrap() as usize],
            )),
            "bitcast" => s.register_with_result(&ReinterpretCastOp::new(
                s.ctx_mut(),
                elem(ty)?.to_type(s.ctx()),
                args[0],
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
            "exp" => unary!(ExpOp),
            "sqrt" => unary!(SqrtOp),
            "rsqrt" => unary!(RsqrtOp),
            "reciprocal" => {
                let one = constant(s, &serde_json::json!(1.0), "float32")?;
                s.register_with_result(&FDivOp::new(s.ctx_mut(), one, args[0]))
            }
            "add" if int => wrapping!(IAddOp, args[0], args[1]),
            "add" => binary!(FAddOp),
            "sub" if int => wrapping!(ISubOp, args[0], args[1]),
            "sub" => binary!(FSubOp),
            "mul" if int => wrapping!(IMulOp, args[0], args[1]),
            "mul" => binary!(FMulOp),
            "neg" if int => {
                // Two's complement 0 - a, wrapping at the minimum as eager.
                let zero = constant(s, &serde_json::json!(0), ty)?;
                wrapping!(ISubOp, zero, args[0])
            }
            "neg" => unary!(FNegOp),
            "abs" => match kind(ty) {
                // fabs (PTX abs.f32): clears the sign bit; NaN payload open.
                Kind::Float => unary!(FAbsOp),
                // a < 0 ? 0 - a : a, wrapping at the minimum as eager; C++ abs on
                // int is undefined there.
                Kind::Signed => {
                    let zero = constant(s, &serde_json::json!(0), ty)?;
                    let negated = wrapping!(ISubOp, zero, args[0]);
                    let negative = compare(s, "lt", Kind::Signed, args[0], zero)?;
                    s.register_with_result(&SelectOp::new(s.ctx_mut(), negative, negated, args[0]))
                }
                _ => args[0],
            },
            "truediv" => binary!(FDivOp),
            // fma(), __fmul_rn and __fdiv_rn in CUDA C++: unaffected by --fmad and
            // --prec-div.
            "fma" => s.register_with_result(&FmaOp::new(s.ctx_mut(), args[0], args[1], args[2])),
            "mul_rn" => binary!(FMulRnOp),
            "div_rn" => binary!(FDivRnOp),
            // NaN-propagating, -0 < +0: PTX max/min.NaN.f32 on sm_80+, as
            // Triton's tl.maximum/minimum(..., PropagateNan.ALL).
            "maximum" | "minimum" => {
                let max = name == "maximum";
                match kind(ty) {
                    Kind::Float if max => binary!(FMaxNanOp),
                    Kind::Float => binary!(FMinNanOp),
                    Kind::Signed if max => binary!(SMaxOp),
                    Kind::Signed => binary!(SMinOp),
                    Kind::Unsigned if max => binary!(UMaxOp),
                    Kind::Unsigned => binary!(UMinOp),
                    // False < True.
                    Kind::Bool if max => binary!(BitwiseOrOp),
                    Kind::Bool => binary!(BitwiseAndOp),
                }
            }
            "bitwise_and" => binary!(BitwiseAndOp),
            "bitwise_or" => binary!(BitwiseOrOp),
            "bitwise_xor" if kind(ty) == Kind::Bool => binary!(BoolNotEqualOp),
            "bitwise_xor" => binary!(BitwiseXorOp),
            // Never C++ `~` on a bool: it promotes to int and is always nonzero.
            "bitwise_not" if kind(ty) == Kind::Bool => unary!(BoolNotOp),
            "bitwise_not" => unary!(BitwiseNotOp),
            "truncdiv" | "floordiv" | "mod" => divide(s, name, ty, args[0], args[1])?,
            "bitwise_left_shift" | "bitwise_right_shift" => {
                shift(s, name == "bitwise_left_shift", ty, args[0], args[1])?
            }
            "isnan" => unary!(IsNanOp),
            "eq" | "ne" | "lt" | "le" | "gt" | "ge" => {
                compare(s, name, value_kind(s, args[0])?, args[0], args[1])?
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

pub(crate) type Validator =
    fn(&[Value], &[Value], &[Value], &mut HashMap<String, String>, usize) -> Result<(), Error>;
pub(crate) type Emitter = fn(
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
        if sizes.len() > MAX_RANK || sizes.len() != strides.len() {
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn check(ops: Value) -> Result<(), Error> {
        validate_ops(ops.as_array().unwrap(), &[], &[], &mut HashMap::new(), 0)
    }

    fn constant_op(id: &str, ty: &str, value: Value) -> Value {
        json!({"id": id, "op": "constant", "args": [], "dtype": ty, "semantic_dtype": ty, "value": value})
    }

    #[test]
    fn bitcast_requires_equal_sizes() {
        let bitcast = |ty: &str| {
            json!([
                constant_op("c", "int32", json!(7)),
                {"id": "b", "op": "bitcast", "args": ["c"], "dtype": ty, "semantic_dtype": ty},
            ])
        };
        check(bitcast("float32")).unwrap();
        assert!(check(json!([
            constant_op("c", "uint8", json!(2)),
            {"id": "b", "op": "bitcast", "args": ["c"], "dtype": "bool", "semantic_dtype": "bool"},
        ]))
        .is_err());
        let error = check(bitcast("int16")).unwrap_err().to_string();
        assert_eq!(error, "bitcast requires equal sizes");
    }

    #[test]
    fn integer_ops_require_one_operand_dtype() {
        let add = |a: &str, b: &str, op: &str| {
            json!([
                constant_op("a", a, json!(1)),
                constant_op("b", b, json!(2)),
                {"id": "c", "op": op, "args": ["a", "b"], "dtype": a, "semantic_dtype": a},
            ])
        };
        check(add("int8", "int8", "add")).unwrap();
        check(add("uint8", "uint8", "bitwise_right_shift")).unwrap();
        assert!(check(add("int8", "int16", "add")).is_err());
        assert!(check(add("float32", "float32", "floordiv")).is_err());
        let bools = json!([
            constant_op("a", "bool", json!(true)),
            {"id": "c", "op": "add", "args": ["a", "a"], "dtype": "bool", "semantic_dtype": "bool"},
        ]);
        assert!(check(bools).is_err());
    }

    #[test]
    fn abs_takes_numbers_but_not_bool() {
        let abs = |ty: &str, value: Value| {
            json!([
                constant_op("a", ty, value),
                {"id": "b", "op": "abs", "args": ["a"], "dtype": ty, "semantic_dtype": ty},
            ])
        };
        for ty in ["float32", "int8", "int16", "int32", "int64", "uint8"] {
            check(abs(ty, json!(1))).unwrap();
        }
        let error = check(abs("bool", json!(true))).unwrap_err().to_string();
        assert!(error.starts_with("abs over bool requires"), "{error}");
    }

    #[test]
    fn explicit_rounding_ops_take_floats_only() {
        let float = |v: f64| constant_op("x", "float32", json!(v));
        check(json!([
            float(1.5),
            {"id": "f", "op": "fma", "args": ["x", "x", "x"], "dtype": "float32", "semantic_dtype": "float32"},
            {"id": "m", "op": "mul_rn", "args": ["x", "x"], "dtype": "float32", "semantic_dtype": "float32"},
            {"id": "d", "op": "div_rn", "args": ["x", "x"], "dtype": "float32", "semantic_dtype": "float32"},
        ]))
        .unwrap();
        let int_fma = json!([
            constant_op("a", "int32", json!(2)),
            {"id": "f", "op": "fma", "args": ["a", "a", "a"], "dtype": "int32", "semantic_dtype": "int32"},
        ]);
        let error = check(int_fma).unwrap_err().to_string();
        assert!(error.contains("fma") && error.contains("int32"), "{error}");
        let two = json!([
            float(1.5),
            {"id": "f", "op": "fma", "args": ["x", "x"], "dtype": "float32", "semantic_dtype": "float32"},
        ]);
        assert_eq!(check(two).unwrap_err().to_string(), "operation arity");
    }
}
