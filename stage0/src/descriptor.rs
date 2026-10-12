//! Versioned experimental semantic descriptor produced by Inductor replay.
use cubecl_core::{
    compute::KernelBuilder,
    ir::{
        AddressType, Builtin, ConstantValue, ElemType, FloatKind, OpInserter,
        dialect::{
            branch::{IfOp, ReturnOp},
            cmp::ULessThanOp,
            general::{BufferLenOp, ReadBuiltinOp},
            math::{CosOp, FAddOp, FMulOp, SinOp},
            memory::{IndexOp, LoadOp, StoreOp},
        },
        pliron::{attribute::boxed_attr_cast, builtin::ops::ConstantOp},
        settings::{Dim3, ExecutionMode, KernelSettings},
        types::scalar::{Float32Type, IndexType},
    },
};
use serde_json::Value;
use std::collections::HashMap;

pub fn build(
    d: &Value,
) -> Result<cubecl_runtime::kernel::KernelDefinition, Box<dyn std::error::Error>> {
    if d["schema"] != 1 || d["dtype"] != "float32" || d["physical"] != "scalar_256" {
        return Err("unsupported Stage 1 descriptor".into());
    }
    let entry = d["entrypoint"].as_str().ok_or("entrypoint missing")?;
    if !entry.starts_with("cubecl_")
        || !entry.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
    {
        return Err("invalid entrypoint".into());
    }
    let buffers = d["buffers"].as_array().ok_or("buffers missing")?;
    if !(2..=3).contains(&buffers.len())
        || buffers.last().unwrap()["access"] != "write"
        || buffers[..buffers.len() - 1]
            .iter()
            .any(|b| b["access"] != "read")
    {
        return Err("expected one or two inputs and one output".into());
    }
    let mut b = KernelBuilder::new(
        KernelSettings::new(
            Dim3::new_1d(256),
            ExecutionMode::Unchecked,
            AddressType::U64,
        )
        .kernel_name(entry),
    );
    let ty = Float32Type::get(b.ctx()).to_handle();
    let arrays: Vec<_> = buffers.iter().map(|_| b.buffer(ty)).collect();
    let index_ty = IndexType::get(b.ctx()).to_handle();
    let i = b.register_with_result(&ReadBuiltinOp::new(
        b.ctx_mut(),
        index_ty,
        Builtin::AbsolutePos,
    ));
    let n = b.register_with_result(&BufferLenOp::new(b.ctx_mut(), 0usize));
    let valid = b.register_with_result(&ULessThanOp::new(b.ctx_mut(), i, n));
    let branch = IfOp::new(b.ctx_mut(), valid);
    {
        let s = b.branch_child(OpInserter::new_at_block_end(branch.then_block(b.ctx())));
        let mut values = HashMap::new();
        for op in d["ops"].as_array().ok_or("ops missing")? {
            if op["dtype"] != "float32" {
                return Err("unsupported operation dtype".into());
            }
            let name = op["op"].as_str().ok_or("operation missing")?;
            let args = op["args"].as_array().ok_or("args missing")?;
            let get = |j: usize| -> Result<_, Box<dyn std::error::Error>> {
                let id = args
                    .get(j)
                    .and_then(Value::as_str)
                    .ok_or("operand missing")?;
                values
                    .get(id)
                    .copied()
                    .ok_or_else(|| "unknown SSA operand".into())
            };
            let value = match name {
                "load" => {
                    let buffer = op["buffer"].as_u64().ok_or("buffer missing")? as usize;
                    if buffer >= arrays.len() - 1 || op["index"] != "i" {
                        return Err("unsupported load".into());
                    }
                    let ptr =
                        s.register_with_result(&IndexOp::new(s.ctx_mut(), arrays[buffer], i, None));
                    s.register_with_result(&LoadOp::new(s.ctx_mut(), ptr))
                }
                "constant" => {
                    let v = op["value"].as_f64().ok_or("float constant missing")?;
                    let attr = ConstantValue::Float(v)
                        .as_attribute(s.ctx(), ElemType::Float(FloatKind::F32));
                    s.register_with_result(&ConstantOp::new(
                        s.ctx_mut(),
                        boxed_attr_cast(attr).unwrap(),
                    ))
                }
                "sin" => s.register_with_result(&SinOp::new(s.ctx_mut(), get(0)?)),
                "cos" => s.register_with_result(&CosOp::new(s.ctx_mut(), get(0)?)),
                "add" => s.register_with_result(&FAddOp::new(s.ctx_mut(), get(0)?, get(1)?)),
                "mul" => s.register_with_result(&FMulOp::new(s.ctx_mut(), get(0)?, get(1)?)),
                "store" => {
                    let buffer = op["buffer"].as_u64().ok_or("buffer missing")? as usize;
                    if buffer != arrays.len() - 1 || op["index"] != "i" {
                        return Err("unsupported store".into());
                    }
                    let ptr =
                        s.register_with_result(&IndexOp::new(s.ctx_mut(), arrays[buffer], i, None));
                    s.register(&StoreOp::new(s.ctx_mut(), ptr, get(0)?));
                    continue;
                }
                _ => return Err(format!("unsupported CubeCL operation {name}").into()),
            };
            let id = op["id"].as_str().ok_or("result identity missing")?;
            if values.insert(id.to_owned(), value).is_some() {
                return Err("duplicate SSA identity".into());
            }
        }
        s.terminate_yield();
    }
    {
        let s = b.branch_child(OpInserter::new_at_block_end(branch.else_block(b.ctx())));
        s.terminate_yield();
    }
    b.register(&branch);
    b.register(&ReturnOp::new(b.ctx_mut()));
    Ok(b.build())
}
