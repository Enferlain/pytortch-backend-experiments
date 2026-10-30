extern crate cubecl_core as cubecl;

mod effects;
mod ir;
mod plan;
mod pointwise;
mod reduction;
mod require;
mod semantic;
use cubecl_core::{
    compute::KernelBuilder,
    ir::{
        AddressType, Builtin, ConstantValue, ElemType, FloatKind, OpInserter,
        dialect::{
            branch::{IfOp, ReturnOp},
            cmp::ULessThanOp,
            general::{BufferLenOp, ReadBuiltinOp},
            math::{FAddOp, SinOp},
            memory::{IndexOp, LoadOp, StoreOp},
        },
        pliron::{
            attribute::boxed_attr_cast, builtin::ops::ConstantOp, op::Op, printable::Printable,
        },
        settings::{Dim3, ExecutionMode, KernelSettings},
        types::scalar::{Float32Type, IndexType},
    },
};
use cubecl_cpp::{
    shared::{CompilationOptions, CppCompiler},
    target::Cuda,
};
use cubecl_runtime::compiler::Compiler;
use std::{path::PathBuf, time::Instant};

fn build() -> cubecl_runtime::kernel::KernelDefinition {
    let settings = KernelSettings::new(
        Dim3::new_1d(256),
        ExecutionMode::Unchecked,
        AddressType::U64,
    )
    .kernel_name("cubecl_sin_add");
    let mut b = KernelBuilder::new(settings);
    let f32_ty = Float32Type::get(b.ctx()).to_handle();
    let x = b.buffer(f32_ty);
    let y = b.buffer(f32_ty);
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
        let xp = s.register_with_result(&IndexOp::new(s.ctx_mut(), x, i, None));
        let val = s.register_with_result(&LoadOp::new(s.ctx_mut(), xp));
        let sin = s.register_with_result(&SinOp::new(s.ctx_mut(), val));
        let one_attr =
            ConstantValue::Float(1.0).as_attribute(s.ctx(), ElemType::Float(FloatKind::F32));
        let one = s.register_with_result(&ConstantOp::new(
            s.ctx_mut(),
            boxed_attr_cast(one_attr).unwrap(),
        ));
        let result = s.register_with_result(&FAddOp::new(s.ctx_mut(), sin, one));
        let yp = s.register_with_result(&IndexOp::new(s.ctx_mut(), y, i, None));
        s.register(&StoreOp::new(s.ctx_mut(), yp, result));
        s.terminate_yield();
    }
    {
        let s = b.branch_child(OpInserter::new_at_block_end(branch.else_block(b.ctx())));
        s.terminate_yield();
    }
    b.register(&branch);
    b.register(&ReturnOp::new(b.ctx_mut()));
    b.build()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out = PathBuf::from(std::env::args().nth(1).unwrap_or("artifacts/cpp".into()));
    let route = std::env::args().nth(2).unwrap_or("cpp".into());
    std::fs::create_dir_all(&out)?;
    let descriptor_start = Instant::now();
    let descriptor = std::env::args()
        .nth(3)
        .map(
            |path| -> Result<serde_json::Value, Box<dyn std::error::Error>> {
                Ok(serde_json::from_str(&std::fs::read_to_string(path)?)?)
            },
        )
        .transpose()?;
    let descriptor = descriptor
        .map(
            |d| -> Result<serde_json::Value, Box<dyn std::error::Error>> {
                if d["schema"] == 5 {
                    let mut semantic = d["semantic"].clone();
                    if !semantic.is_object() || !d["physical_plan"].is_object() {
                        return Err("invalid descriptor envelope".into());
                    }
                    semantic["semantic_schema"] = semantic["schema"].clone();
                    semantic["schema"] = 5.into();
                    semantic["physical_plan"] = d["physical_plan"].clone();
                    semantic["entrypoint"] = d["entrypoint"].clone();
                    Ok(semantic)
                } else {
                    Ok(d)
                }
            },
        )
        .transpose()?;
    let entrypoint = descriptor
        .as_ref()
        .and_then(|d| d["entrypoint"].as_str())
        .unwrap_or("cubecl_sin_add");
    let buffer_count = descriptor
        .as_ref()
        .and_then(|d| d["buffers"].as_array())
        .map_or(2, Vec::len);
    let descriptor_ms = descriptor_start.elapsed().as_secs_f64() * 1000.0;
    let ir_start = Instant::now();
    let kernel = if let Some(d) = &descriptor {
        if d["schema"] == 5 {
            plan::build(d)?
        } else if d["schema"] == 4 {
            effects::build(d)?
        } else if d["schema"] == 3 {
            reduction::build(d)?
        } else if d["schema"] == 2 {
            pointwise::scalar::build(d)?
        } else {
            pointwise::minimal::build(d)?
        }
    } else {
        build()
    };
    // Explicit offline probe route; never selected by the backend/physical planner.
    let kernel = if let Some(mechanism) = route.strip_prefix("cpp-probe-") {
        let d = descriptor.as_ref().ok_or("probe requires a descriptor")?;
        let final_plane = matches!(mechanism, "final-plane" | "final-plane-uniform");
        let expected = if final_plane {
            "independent_planes"
        } else {
            "plane_shared_reference512"
        };
        if d["schema"] != 5 || d["physical_plan"]["strategy"] != expected {
            return Err(format!("probe requires a validated {expected} descriptor").into());
        }
        if final_plane && d["physical_plan"]["workgroup"] != serde_json::json!([32, 4, 1]) {
            return Err("final-plane probe requires the descriptor's four-plane mapping".into());
        }
        let mut sem = d.clone();
        sem["schema"] = 3.into();
        let options = match mechanism {
            "final-plane" | "final-plane-uniform" => reduction::CodegenOptions {
                final_plane: true,
                final_load_batch: mechanism == "final-plane-uniform",
                ..Default::default()
            },
            "frozen" => reduction::CodegenOptions::default(),
            "uniform" => reduction::CodegenOptions {
                uniform_loop: true,
                ..Default::default()
            },
            "owner" => reduction::CodegenOptions {
                owner_result: true,
                ..Default::default()
            },
            "narrow" => reduction::CodegenOptions {
                narrow_loop: true,
                ..Default::default()
            },
            "uniform_narrow" => reduction::CodegenOptions {
                uniform_loop: true,
                narrow_loop: true,
                ..Default::default()
            },
            _ => return Err("unknown codegen probe mechanism".into()),
        };
        reduction::build_physical_codegen(
            &sem,
            if final_plane { [32, 4, 1] } else { [512, 1, 1] },
            expected,
            1,
            if final_plane { 1 } else { 4 },
            if final_plane { "halving" } else { "adjacent" },
            options,
        )?
    } else {
        kernel
    };
    let ir_ms = ir_start.elapsed().as_secs_f64() * 1000.0;
    std::fs::write(
        out.join("cubecl.ir"),
        kernel
            .body
            .state()
            .module
            .get_operation()
            .disp(kernel.body.ctx())
            .to_string(),
    )?;
    std::fs::write(
        out.join("info.txt"),
        format!("{:#?}\n{:#?}", kernel.info, kernel.settings),
    )?;
    let start = Instant::now();
    let (source, shared, io) = if route == "cpp" || route.starts_with("cpp-probe-") {
        let repr =
            CppCompiler::<Cuda>::default().compile(kernel, &CompilationOptions::default())?;
        (
            repr.source,
            repr.shared_memory_size,
            format!("{:?}", repr.io),
        )
    } else if route == "hip-probe" {
        #[cfg(feature = "hip-probe")]
        {
            let repr = CppCompiler::<cubecl_cpp::target::Hip>::default()
                .compile(kernel, &CompilationOptions::default())?;
            (
                repr.source,
                repr.shared_memory_size,
                format!("{:?}", repr.io),
            )
        }
        #[cfg(not(feature = "hip-probe"))]
        return Err("build with --features hip-probe".into());
    } else {
        #[cfg(feature = "llvm")]
        {
            use cubecl_core::ir::nvidia::SmArch;
            use cubecl_llvm::{LlvmTarget, PlironArtifact, PlironCompiler, PlironOptions};
            let mut c = PlironCompiler {
                target: LlvmTarget::Nvptx,
            };
            let options = PlironOptions {
                sm_arch: Some(SmArch::new(86, true)),
                grid_constants: false,
                ..Default::default()
            };
            let PlironArtifact::NvptxCode(repr) = c.compile(kernel, &options)? else {
                unreachable!()
            };
            assert_eq!(repr.entrypoint, entrypoint);
            std::fs::write(
                out.join("kernel.ptx"),
                bytemuck::cast_slice::<i8, u8>(&repr.ptx),
            )?;
            (repr.ir, repr.shared_memory_size, format!("{:?}", repr.io))
        }
        #[cfg(not(feature = "llvm"))]
        return Err("build with --features llvm for route llvm".into());
    };
    let ms = start.elapsed().as_secs_f64() * 1000.0;
    std::fs::write(
        out.join(if route == "cpp" || route.starts_with("cpp-probe-") {
            "kernel.cu"
        } else if route == "hip-probe" {
            "kernel.hip"
        } else {
            "kernel.ll"
        }),
        source,
    )?;
    let mut info = cubecl_core::InfoBuilder::default();
    if let Some(d) = &descriptor
        && (d["schema"] == 2 || d["schema"] == 3 || d["schema"] == 4 || d["schema"] == 5)
    {
        // Inspectable ABI template only; actual launch values come from stage2_metadata.
        let column =
            d["schema"] == 5 && d["physical_plan"]["strategy"] == "bounded_column_pointwise";
        let count = d["scalars"].as_array().unwrap().len() + 1 + if column { 4 } else { 0 };
        for _ in 0..count {
            info.scalars.push(257i64);
        }
    }
    for _ in 0..buffer_count {
        info.metadata.register_buffer(257, AddressType::U64);
    }
    let words = info.finish(AddressType::U64).data;
    std::fs::write(out.join("metadata-257.bin"), bytemuck::cast_slice(&words))?;
    let arguments: Vec<String> = if descriptor.is_some() {
        (0..buffer_count)
            .map(|i| format!("buffer {i} float32 device pointer"))
            .chain(std::iter::once("info device pointer".into()))
            .collect()
    } else {
        vec![
            "input float32 device pointer".into(),
            "output float32 device pointer".into(),
            "info device pointer".into(),
        ]
    };
    let layout = if descriptor.is_some() {
        "buffer lengths in descriptor order, in elements"
    } else {
        "input length in elements, output length in elements"
    };
    let mut manifest = serde_json::json!({
        "schema": 1, "cubecl_commit": "907676dca8c4cd407237892cc2915ff6aadc4f1e",
        "route": route, "entrypoint": entrypoint, "block": [256,1,1],
        "grid": "ceil_div(N,256),1,1; skip N=0", "dynamic_shared_bytes": shared,
        "grid_constants": false, "address_bits": 64, "vector_width": 1,
        "arguments": arguments,
        "metadata": {"producer": "InfoBuilder", "words_u64": words, "bytes": buffer_count * 8,
            "layout": layout, "dynamic": false},
        "buffer_io": io, "cubecl_compile_ms": ms,
        "descriptor_parse_ms": descriptor_ms, "cubecl_ir_build_ms": ir_ms
    });
    if let Some(d) = &descriptor
        && (d["schema"] == 2 || d["schema"] == 3 || d["schema"] == 4 || d["schema"] == 5)
    {
        let column =
            d["schema"] == 5 && d["physical_plan"]["strategy"] == "bounded_column_pointwise";
        let scalar_count = d["scalars"].as_array().unwrap().len() + 1 + if column { 4 } else { 0 };
        manifest["descriptor_schema"] = d["schema"].clone();
        manifest["metadata"]["example_only"] = true.into();
        manifest["metadata"]["bytes"] = ((buffer_count + scalar_count) * 8).into();
        manifest["metadata"]["layout"] =
            "InfoBuilder: signed i64 numel and runtime symbols; independent buffer lengths".into();
        manifest["metadata"]["scalar_count"] = scalar_count.into();
        if column {
            manifest["mechanism"] = "cubecl-column-plain-fast64-v1".into();
            manifest["layout_rank"] = 2.into();
            manifest["metadata"]["layout"] = "InfoBuilder: i64 numel, N, B, then U64 FastDivmod multiplier bits and shift for B,N".into();
        }
        manifest["arguments"] = serde_json::json!(
            d["buffers"]
                .as_array()
                .unwrap()
                .iter()
                .map(|b| format!("{} device pointer", b["dtype"].as_str().unwrap()))
                .chain(std::iter::once("info device pointer".into()))
                .collect::<Vec<_>>()
        );
    }
    if let Some(d) = &descriptor
        && d["schema"] == 3
    {
        manifest["physical_plan"] = d["physical_plan"].clone();
        manifest["grid"] = "outer_extent,1,1; skip outer=0".into();
        manifest["metadata"]["layout"] = "InfoBuilder: reserved signed i64 launch_units=outer_extent*256, then runtime logical symbols; independent buffer lengths".into();
    }
    if let Some(d) = &descriptor
        && d["schema"] == 5
    {
        manifest["physical_plan"] = d["physical_plan"].clone();
        manifest["block"] = d["physical_plan"]["workgroup"].clone();
        manifest["vector_width"] = d["physical_plan"]["vector_width"].clone();
        manifest["semantic_schema"] = d["semantic_schema"].clone();
        manifest["grid"] = "runtime semantic extent mapped by validated PhysicalCandidate".into();
    }
    std::fs::write(
        out.join("manifest.json"),
        serde_json::to_string_pretty(&manifest)?,
    )?;
    println!("{}", serde_json::to_string(&manifest)?);
    Ok(())
}
