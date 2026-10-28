//! Schema 5 pointwise plans: scalar or vector units replaying the validated
//! semantic ops (`scalar_pointwise_baseline`, `scalar_pointwise`, `vector_pointwise`).
use crate::ir::{self, branch, pointer, vector_pointer, vector_type};
use crate::{effects, semantic};
use cubecl_core::{
    compute::KernelBuilder,
    ir::{
        AddressType, Builtin, OpInserter, Scope,
        dialect::{
            branch::{IfOp, ReturnOp},
            cmp::SLessThanOp,
            general::{CastOp, ReadBuiltinOp, ReadScalarOp},
            math::{IAddOp, IMulOp},
            memory::{LoadOp, StoreOp},
            vector::{CompositeConstructOp, CompositeExtractOp},
        },
        pliron::value::Value as IrValue,
        settings::{Dim3, ExecutionMode, KernelSettings},
        types::scalar::IndexType,
    },
};
use serde_json::{Value, json};
use std::collections::HashMap;
type Error = Box<dyn std::error::Error>;
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
        semantic::emit(s, ops, arrays, linear, scalars, values)
    }
}

/// Emission for a plan `plan::build` has validated.
pub(crate) fn build(
    d: &Value,
    sem: &Value,
    workgroup: [u32; 3],
    width: usize,
    elements: u64,
    schema: u64,
) -> Result<cubecl_runtime::kernel::KernelDefinition, Error> {
    let buffers = sem["buffers"].as_array().unwrap();
    let ops = sem["ops"].as_array().unwrap();
    let entry = d["entrypoint"].as_str().ok_or("entrypoint")?;
    if !entry.starts_with("cubecl5_")
        || !entry.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
    {
        return Err("invalid entrypoint".into());
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
                ir::elem(v["dtype"].as_str().unwrap())
                    .unwrap()
                    .to_type(b.ctx()),
            )
        })
        .collect();
    let ity = ir::elem("int64")?.to_type(b.ctx());
    let scalars: Vec<_> = (0..=sem["scalars"].as_array().unwrap().len())
        .map(|_| {
            let id = b.scalar(ir::elem("int64").unwrap());
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
    let units = ir::constant(&b, &json!(workgroup[0]), "int64")?;
    let block_base = b.register_with_result(&IMulOp::new(b.ctx_mut(), block, units));
    let stride = ir::constant(&b, &json!(width as u64 * elements), "int64")?;
    let block_base = b.register_with_result(&IMulOp::new(b.ctx_mut(), block_base, stride));
    for iteration in 0..elements {
        let offset = ir::constant(&b, &json!(iteration * workgroup[0] as u64), "int64")?;
        let u = b.register_with_result(&IAddOp::new(b.ctx_mut(), unit, offset));
        let w = ir::constant(&b, &json!(width), "int64")?;
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
        let last = ir::constant(&b, &json!(width - 1), "int64")?;
        let last = b.register_with_result(&IAddOp::new(b.ctx_mut(), base, last));
        let full = b.register_with_result(&SLessThanOp::new(b.ctx_mut(), last, scalars[0]));
        let op = IfOp::new(b.ctx_mut(), full);
        let s = b.branch_child(OpInserter::new_at_block_end(op.then_block(b.ctx())));
        let mut loaded = HashMap::new();
        for mem in ops.iter().filter(|v| v["op"] == "load") {
            let idx = semantic::index(&s, &mem["index"], base, &scalars[1..])?;
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
        // Lane values per store, keyed by the store's position (stores have no id).
        let mut stored: HashMap<usize, Vec<IrValue>> = HashMap::new();
        for lane in 0..width {
            let offset = ir::constant(&s, &json!(lane), "int64")?;
            let linear = s.register_with_result(&IAddOp::new(s.ctx_mut(), base, offset));
            let mut values = HashMap::new();
            for (position, v) in ops.iter().enumerate() {
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
                        .entry(position)
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
        for (position, v) in ops.iter().enumerate().filter(|(_, v)| v["op"] == "store") {
            let ptr = vector_pointer(
                &s,
                pointer(&s, arrays[v["buffer"].as_u64().unwrap() as usize], base),
                v["dtype"].as_str().unwrap(),
                width,
            )?;
            let value = s.register_with_result(&CompositeConstructOp::new(
                s.ctx_mut(),
                vector_type(&s, v["dtype"].as_str().unwrap(), width)?,
                stored.remove(&position).unwrap(),
            ));
            s.register(&StoreOp::new(s.ctx_mut(), ptr, value));
        }
        s.terminate_yield();
        let s = b.branch_child(OpInserter::new_at_block_end(op.else_block(b.ctx())));
        for lane in 0..width {
            let offset = ir::constant(&s, &json!(lane), "int64")?;
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
