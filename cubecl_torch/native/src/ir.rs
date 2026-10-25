//! IR-building helpers shared by the kernel builders: storage dtypes,
//! constants, guarded branches and indexed or vector pointers.
use cubecl_core::ir::{
    ConstantValue, ElemType, FloatKind, IntKind, OpInserter, Scope, UIntKind,
    dialect::{
        branch::IfOp,
        general::{CastOp, ReinterpretCastOp},
        memory::IndexOp,
    },
    pliron::{
        attribute::boxed_attr_cast,
        builtin::ops::ConstantOp,
        r#type::{TypeHandle, Typed},
        value::Value as IrValue,
    },
    types::{PointerType, VectorType, scalar::IndexType},
};
use serde_json::Value;
type Error = Box<dyn std::error::Error>;
/// Integer and bool value dtypes: (name, signed, width in bits; bool is 1).
/// Floats are separate: their constants and arithmetic follow the numerical policy.
pub(crate) const INTEGERS: [(&str, bool, u32); 6] = [
    ("bool", false, 1),
    ("int8", true, 8),
    ("int16", true, 16),
    ("int32", true, 32),
    ("int64", true, 64),
    ("uint8", false, 8),
];
pub(crate) fn integer(name: &str) -> Option<(bool, u32)> {
    INTEGERS
        .iter()
        .find(|(n, _, _)| *n == name)
        .map(|&(_, signed, width)| (signed, width))
}
pub(crate) fn elem(name: &str) -> Result<ElemType, Error> {
    Ok(match name {
        "float32" => ElemType::Float(FloatKind::F32),
        "float16" => ElemType::Float(FloatKind::F16),
        "bfloat16" => ElemType::Float(FloatKind::BF16),
        // Runtime float scalar values only; never storage or arithmetic.
        "float64" => ElemType::Float(FloatKind::F64),
        "bool" => ElemType::Bool,
        "int8" => ElemType::Int(IntKind::I8),
        "int16" => ElemType::Int(IntKind::I16),
        "int32" => ElemType::Int(IntKind::I32),
        "int64" => ElemType::Int(IntKind::I64),
        "uint8" => ElemType::UInt(UIntKind::U8),
        _ => return Err(format!("unsupported dtype {name}").into()),
    })
}
/// The descriptor constant `v` as a value of `ty`, range-checked for integers.
pub(crate) fn constant_value(v: &Value, ty: &str) -> Result<ConstantValue, Error> {
    Ok(match integer(ty) {
        Some((_, 1)) => ConstantValue::Bool(v.as_bool().ok_or("bool constant missing")?),
        Some((true, width)) => {
            let x = v.as_i64().ok_or("integer constant missing")?;
            if width < 64 && !(-(1i64 << (width - 1))..1i64 << (width - 1)).contains(&x) {
                return Err(format!("constant {x} outside {ty}").into());
            }
            ConstantValue::Int(x)
        }
        Some((false, width)) => {
            let x = v.as_u64().ok_or("unsigned constant missing")?;
            if x >= 1u64 << width {
                return Err(format!("constant {x} outside {ty}").into());
            }
            ConstantValue::UInt(x)
        }
        None => ConstantValue::Float(if let Some(f) = v.as_f64() {
            f
        } else {
            match v.as_str() {
                Some("nan") => f64::NAN,
                Some("inf") => f64::INFINITY,
                Some("-inf") => f64::NEG_INFINITY,
                _ => return Err("invalid float constant".into()),
            }
        }),
    })
}
pub(crate) fn constant(s: &Scope, v: &Value, ty: &str) -> Result<IrValue, Error> {
    let attr = constant_value(v, ty)?.as_attribute(s.ctx(), elem(ty)?);
    Ok(s.register_with_result(&ConstantOp::new(
        s.ctx_mut(),
        boxed_attr_cast(attr).ok_or("constant attr")?,
    )))
}

pub(crate) fn pointer(s: &Scope, array: IrValue, coordinate: IrValue) -> IrValue {
    let index = s.register_with_result(&CastOp::new(
        s.ctx_mut(),
        IndexType::get(s.ctx()).to_handle(),
        coordinate,
    ));
    s.register_with_result(&IndexOp::new(s.ctx_mut(), array, index, None))
}

pub(crate) fn branch<F>(s: &Scope, cond: IrValue, body: F) -> Result<(), Error>
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
pub(crate) fn vector_type(s: &Scope, ty: &str, width: usize) -> Result<TypeHandle, Error> {
    Ok(VectorType::get(s.ctx(), elem(ty)?.to_type(s.ctx()), width).to_handle())
}
pub(crate) fn vector_pointer(
    s: &Scope,
    p: IrValue,
    ty: &str,
    width: usize,
) -> Result<IrValue, Error> {
    let space = p
        .get_type(s.ctx())
        .deref(s.ctx())
        .downcast_ref::<PointerType>()
        .ok_or("vector reinterpret requires a pointer")?
        .address_space;
    let t = PointerType::get(s.ctx(), vector_type(s, ty, width)?, space).to_handle();
    Ok(s.register_with_result(&ReinterpretCastOp::new(s.ctx_mut(), t, p)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn integer_constants_are_range_checked_per_width() {
        for (ty, low, high) in [
            ("int8", json!(-128), json!(127)),
            ("int16", json!(-32768), json!(32767)),
            ("int32", json!(i32::MIN), json!(i32::MAX)),
            ("int64", json!(i64::MIN), json!(i64::MAX)),
            ("uint8", json!(0), json!(255)),
        ] {
            elem(ty).unwrap();
            constant_value(&low, ty).unwrap();
            constant_value(&high, ty).unwrap();
        }
        for (ty, v) in [
            ("int8", json!(-129)),
            ("int8", json!(128)),
            ("int16", json!(32768)),
            ("int32", json!(i32::MAX as i64 + 1)),
            ("uint8", json!(256)),
            ("uint8", json!(-1)),
            ("int64", json!(1.5)),
            ("bool", json!(1)),
        ] {
            assert!(constant_value(&v, ty).is_err(), "{ty} {v}");
        }
        assert!(elem("uint16").is_err());
        assert_eq!(integer("uint8"), Some((false, 8)));
        assert_eq!(integer("float32"), None);
    }
}
