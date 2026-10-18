//! Existing FastDivmod::div + PlainLayout for the proved bounded column coordinate.
#[allow(dead_code)]
#[path = "../../../stage0/src/descriptor2.rs"]
mod descriptor2;
use cubecl::Compiler;
use cubecl::ir::{
    AddressType, Builtin, ElemType, OpInserter, UIntKind,
    dialect::{
        branch::IfOp,
        cmp::SLessThanOp,
        general::{CastOp, ReadBuiltinOp, ReadScalarOp},
        math::{IAddOp, IMulOp},
        memory::{IndexOp, LoadOp},
    },
    pliron::{op::Op, printable::Printable, value::Value as IrValue},
    settings::{Dim3, ExecutionMode, KernelSettings},
    types::scalar::IndexType,
};
use cubecl::prelude::*;
use cubecl::std::{
    FastDivmod, FastDivmodExpand,
    tensor::layout::{
        LayoutExpand, linear::LinearViewLayoutExpand, permuted::PermutedLayout, plain::PlainLayout,
    },
};
use cubecl_cpp::{
    shared::{CompilationOptions, CppCompiler},
    target::Cuda,
};
use serde_json::{Value as Json, json};
use std::{collections::HashMap, fs, path::PathBuf};

fn c(s: &Scope, v: i64) -> IrValue {
    descriptor2::constant(s, &json!(v), "int64").unwrap()
}
fn idx(s: &Scope, v: IrValue) -> IrValue {
    s.register_with_result(&CastOp::new(
        s.ctx_mut(),
        IndexType::get(s.ctx()).to_handle(),
        v,
    ))
}
fn signed(s: &Scope, v: IrValue) -> IrValue {
    s.register_with_result(&CastOp::new(
        s.ctx_mut(),
        descriptor2::elem("int64").unwrap().to_type(s.ctx()),
        v,
    ))
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out = PathBuf::from(std::env::args().nth(1).ok_or("output")?);
    let packet: Json = serde_json::from_str(&fs::read_to_string(
        std::env::args().nth(2).ok_or("descriptor")?,
    )?)?;
    let sem = &packet["semantic"];
    let templates: Vec<Json> = serde_json::from_str(include_str!("templates.json"))?;
    let plan = json!({"version":2,"strategy":"scalar_pointwise_baseline","workgroup":[256,1,1],"vector_width":1,"elements_per_unit":1,"outputs_per_workgroup":1,"plane_width":0});
    let entry = packet["entrypoint"].as_str().ok_or("entrypoint")?;
    if packet["schema"] != 5
        || packet["mechanism"] != "cubecl-column-plain-fast64-v1"
        || packet["physical_plan"] != plan
        || !templates.contains(sem)
        || !entry.starts_with("cubecl5_column_")
        || !entry.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
    {
        return Err("layout diagnostic admission".into());
    }
    let buffers = sem["buffers"].as_array().ok_or("buffers")?;
    let reference = buffers.last().unwrap()["size"]
        .as_array()
        .ok_or("reference shape")?;
    let rank = reference.len();
    if rank != 2 {
        return Err("reference rank".into());
    }
    let original_count = sem["scalars"].as_array().ok_or("scalars")?.len();
    let total_scalars = 1 + original_count + 2 * rank;
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
        .map(|v| {
            b.buffer(
                descriptor2::elem(v["dtype"].as_str().unwrap())
                    .unwrap()
                    .to_type(b.ctx()),
            )
        })
        .collect();
    let scalars: Vec<_> = (0..total_scalars)
        .map(|_| {
            let id = b.scalar(descriptor2::elem("int64").unwrap());
            b.register_with_result(&ReadScalarOp::new(
                b.ctx_mut(),
                descriptor2::elem("int64").unwrap().to_type(b.ctx()),
                id,
            ))
        })
        .collect();
    let semantic_scalars = &scalars[1..1 + original_count];
    let mut shape = Sequence::<FastDivmod<usize>>::__expand_new(&b);
    let mut row_divisor = None;
    for (axis, dim) in reference.iter().enumerate() {
        let divisor = descriptor2::index(&b, dim, c(&b, 0), semantic_scalars)?;
        let base = 1 + original_count + 2 * axis;
        let shift = b.register_with_result(&CastOp::new(
            b.ctx_mut(),
            ElemType::UInt(UIntKind::U32).to_type(b.ctx()),
            scalars[base + 1],
        ));
        let fast = FastDivmodExpand::Fast {
            divisor: idx(&b, divisor).into(),
            multiplier: idx(&b, scalars[base]).into(),
            shift_right: shift.into(),
        };
        if axis == rank - 1 {
            row_divisor = Some(fast.clone());
        }
        shape.__expand_push_method(&b, fast);
    }
    let mut layouts = HashMap::new();
    for op in sem["ops"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|o| o["op"] == "load")
    {
        let id = op["buffer"].as_u64().unwrap() as usize;
        if id == 1 {
            continue;
        }
        if op["index"] == json!({"op":"linear"}) {
            layouts.insert(
                id,
                LinearViewLayoutExpand::Plain(PlainLayout::__expand_new(
                    &b,
                    idx(&b, scalars[0]).into(),
                )),
            );
            continue;
        }
        let sizes = buffers[id]["size"].as_array().unwrap();
        let strides = buffers[id]["stride"].as_array().unwrap();
        if sizes.len() > rank {
            return Err("input rank".into());
        }
        let pad = rank - sizes.len();
        let mut stride = Sequence::<usize>::__expand_new(&b);
        for axis in 0..rank {
            let value = if axis < pad {
                c(&b, 0)
            } else if sizes[axis - pad] == reference[axis] {
                descriptor2::index(&b, &strides[axis - pad], c(&b, 0), semantic_scalars)?
            } else if sizes[axis - pad] == json!({"op":"constant","value":1}) {
                c(&b, 0)
            } else {
                return Err("broadcast shape not structurally proved".into());
            };
            stride.__expand_push_method(&b, idx(&b, value).into());
        }
        layouts.insert(
            id,
            LinearViewLayoutExpand::Permuted(PermutedLayout::__expand_new(
                &b,
                shape.clone(),
                stride,
                idx(&b, scalars[0]).into(),
                1,
            )),
        );
    }
    let column_extent = descriptor2::index(&b, &buffers[1]["size"][0], c(&b, 0), semantic_scalars)?;
    let column_layout = PlainLayout::__expand_new(&b, idx(&b, column_extent).into());
    let row_divisor = row_divisor.ok_or("column divisor")?;
    let builtin = |kind| {
        let v = b.register_with_result(&ReadBuiltinOp::new(
            b.ctx_mut(),
            IndexType::get(b.ctx()).to_handle(),
            kind,
        ));
        signed(&b, v)
    };
    let cube = builtin(Builtin::CubePosX);
    let unit = builtin(Builtin::UnitPosX);
    let base = b.register_with_result(&IMulOp::new(b.ctx_mut(), cube, c(&b, 256)));
    let linear = b.register_with_result(&IAddOp::new(b.ctx_mut(), base, unit));
    let condition = b.register_with_result(&SLessThanOp::new(b.ctx_mut(), linear, scalars[0]));
    let branch = IfOp::new(b.ctx_mut(), condition);
    let s = b.branch_child(OpInserter::new_at_block_end(branch.then_block(b.ctx())));
    let mut values = HashMap::new();
    for op in sem["ops"].as_array().unwrap() {
        if op["op"] == "load" {
            let id = op["buffer"].as_u64().unwrap() as usize;
            let coordinate = if id == 1 {
                let row = row_divisor.__expand_div_method(&s, idx(&s, linear).into());
                column_layout
                    .__expand_to_source_pos_method(&s, row)
                    .value(&s)
            } else {
                layouts[&id]
                    .__expand_to_source_pos_method(&s, idx(&s, linear).into())
                    .value(&s)
            };
            let pointer =
                s.register_with_result(&IndexOp::new(s.ctx_mut(), arrays[id], coordinate, None));
            let value = s.register_with_result(&LoadOp::new(s.ctx_mut(), pointer));
            values.insert(op["id"].as_str().unwrap().to_owned(), value);
        } else {
            descriptor2::emit(
                &s,
                std::slice::from_ref(op),
                &arrays,
                linear,
                semantic_scalars,
                &mut values,
            )?;
        }
    }
    s.terminate_yield();
    let other = b.branch_child(OpInserter::new_at_block_end(branch.else_block(b.ctx())));
    other.terminate_yield();
    b.register(&branch);
    drop(s);
    drop(other);
    let kernel = b.build();
    fs::create_dir_all(&out)?;
    fs::write(
        out.join("cubecl.ir"),
        kernel
            .body
            .state()
            .module
            .get_operation()
            .disp(kernel.body.ctx())
            .to_string(),
    )?;
    let repr = CppCompiler::<Cuda>::default().compile(kernel, &CompilationOptions::default())?;
    fs::write(out.join("kernel.cu"), repr.source)?;
    fs::write(
        out.join("manifest.json"),
        serde_json::to_string_pretty(
            &json!({"schema":1,"descriptor_schema":5,"semantic_schema":4,"entrypoint":entry,"block":[256,1,1],"physical_plan":plan,"dynamic_shared_bytes":repr.shared_memory_size,"route":"cpp","mechanism":packet["mechanism"],"metadata":{"scalar_count":total_scalars,"bytes":8*(buffers.len()+total_scalars)},"arguments":buffers.iter().map(|v|format!("{} device pointer",v["dtype"].as_str().unwrap())).chain(std::iter::once("info device pointer".into())).collect::<Vec<_>>(),"grid_constants":false,"vector_width":1,"address_bits":64,"layout_rank":rank}),
        )?,
    )?;
    Ok(())
}
