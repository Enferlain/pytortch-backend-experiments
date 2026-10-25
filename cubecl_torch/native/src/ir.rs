//! IR-building helpers shared by the kernel builders: storage dtypes,
//! constants, guarded branches and indexed or vector pointers.
use cubecl_core::ir::{
    ConstantValue, ElemType, FloatKind, IntKind, OpInserter, Scope,
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
