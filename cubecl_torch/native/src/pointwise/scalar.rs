//! The frozen scalar pointwise kernel (256 units, one element each) for
//! schema 2, and for schema 4 through `effects`.
use crate::ir::{constant, elem};
use crate::semantic::{Emitter, Validator, dtype, emit, validate_description, validate_ops};
use cubecl_core::{
    compute::KernelBuilder,
    ir::{
        AddressType, Builtin, OpInserter,
        dialect::{
            branch::{IfOp, ReturnOp},
            cmp::*,
            general::{CastOp, ReadBuiltinOp, ReadScalarOp},
            math::*,
        },
        settings::{Dim3, ExecutionMode, KernelSettings},
        types::scalar::IndexType,
    },
};
use serde_json::Value;
use std::collections::HashMap;
type Error = Box<dyn std::error::Error>;
pub fn build(d: &Value) -> Result<cubecl_runtime::kernel::KernelDefinition, Error> {
    build_with(d, 2, validate_ops, emit)
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
