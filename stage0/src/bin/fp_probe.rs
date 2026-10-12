//! Fixed audit fixtures at the existing compiler boundary, not new backend ops.
use cubecl_core::{
    compute::KernelBuilder,
    ir::{
        AddressType, Builtin, ConstantValue, ElemType, FloatKind, IntKind, OpInserter, UIntKind,
        dialect::{
            branch::{IfOp, ReturnOp},
            cmp::ULessThanOp,
            general::{BufferLenOp, CastOp, ReadBuiltinOp},
            math::*,
            memory::{IndexOp, LoadOp, StoreOp},
        },
        pliron::{
            attribute::boxed_attr_cast, builtin::ops::ConstantOp, op::Op, printable::Printable,
        },
        settings::{Dim3, ExecutionMode, KernelSettings},
        types::scalar::IndexType,
    },
};
use cubecl_cpp::{
    shared::{CompilationOptions, CppCompiler},
    target::Cuda,
};
use cubecl_runtime::{compiler::Compiler, kernel::KernelDefinition};
use std::path::Path;

fn build(case: &str, storage: ElemType, name: &str) -> KernelDefinition {
    let settings = KernelSettings::new(
        Dim3::new_1d(256),
        ExecutionMode::Unchecked,
        AddressType::U64,
    )
    .kernel_name(name);
    let mut b = KernelBuilder::new(settings);
    let ty = storage.to_type(b.ctx());
    let arrays = [b.buffer(ty), b.buffer(ty), b.buffer(ty)];
    let i = b.register_with_result(&ReadBuiltinOp::new(
        b.ctx_mut(),
        IndexType::get(b.ctx()).to_handle(),
        Builtin::AbsolutePos,
    ));
    let n = b.register_with_result(&BufferLenOp::new(b.ctx_mut(), 0));
    let mask = b.register_with_result(&ULessThanOp::new(b.ctx_mut(), i, n));
    let branch = IfOp::new(b.ctx_mut(), mask);
    {
        let s = b.branch_child(OpInserter::new_at_block_end(branch.then_block(b.ctx())));
        let mut operands = Vec::new();
        for array in arrays.iter().take(2) {
            let ptr = s.register_with_result(&IndexOp::new(s.ctx_mut(), *array, i, None));
            let value = s.register_with_result(&LoadOp::new(s.ctx_mut(), ptr));
            let value = if matches!(storage, ElemType::Float(FloatKind::F16 | FloatKind::BF16)) {
                s.register_with_result(&CastOp::new(
                    s.ctx_mut(),
                    ElemType::Float(FloatKind::F32).to_type(s.ctx()),
                    value,
                ))
            } else {
                value
            };
            operands.push(value);
        }
        let (x, y) = (operands[0], operands[1]);
        let constant = |v: f64| {
            let attr =
                ConstantValue::Float(v).as_attribute(s.ctx(), ElemType::Float(FloatKind::F32));
            s.register_with_result(&ConstantOp::new(
                s.ctx_mut(),
                boxed_attr_cast(attr).unwrap(),
            ))
        };
        macro_rules! bin {
            ($op:ty, $a:expr, $b:expr) => {
                s.register_with_result(&<$op>::new(s.ctx_mut(), $a, $b))
            };
        }
        let result = match case {
            "add_positive_zero_left" => bin!(FAddOp, constant(0.0), x),
            "add_positive_zero_right" => bin!(FAddOp, x, constant(0.0)),
            "add_negative_zero_left" => bin!(FAddOp, constant(-0.0), x),
            "add_negative_zero_right" => bin!(FAddOp, x, constant(-0.0)),
            "sub_self" => bin!(FSubOp, x, x),
            "mul_positive_zero_left" => bin!(FMulOp, constant(0.0), x),
            "mul_positive_zero_right" => bin!(FMulOp, x, constant(0.0)),
            "mul_negative_zero_left" => bin!(FMulOp, constant(-0.0), x),
            "mul_negative_zero_right" => bin!(FMulOp, x, constant(-0.0)),
            "div_positive_zero" => bin!(FDivOp, constant(0.0), y),
            "div_negative_zero" => bin!(FDivOp, constant(-0.0), y),
            "rem_by_one" => bin!(FRemOp, x, constant(1.0)),
            "rem_positive_zero" => bin!(FRemOp, constant(0.0), y),
            "rem_negative_zero" => bin!(FRemOp, constant(-0.0), y),
            "floor_positive_zero" => bin!(FModFloorOp, constant(0.0), y),
            "floor_negative_zero" => bin!(FModFloorOp, constant(-0.0), y),
            "loaded_add" => bin!(FAddOp, x, y),
            "loaded_mul" => bin!(FMulOp, x, y),
            "loaded_div" => bin!(FDivOp, x, y),
            "loaded_rem" => bin!(FRemOp, x, y),
            "loaded_floor" => bin!(FModFloorOp, x, y),
            "contract_add" => {
                let p = bin!(FMulOp, x, y);
                bin!(FAddOp, p, constant(-1.0))
            }
            "contract_sub" => {
                let p = bin!(FMulOp, x, y);
                bin!(FSubOp, p, constant(1.0))
            }
            "contract_rsub" => {
                let p = bin!(FMulOp, x, y);
                bin!(FSubOp, constant(1.0), p)
            }
            "srem_by_one" => {
                let attr = ConstantValue::Int(1).as_attribute(s.ctx(), storage);
                let one = s.register_with_result(&ConstantOp::new(
                    s.ctx_mut(),
                    boxed_attr_cast(attr).unwrap(),
                ));
                bin!(SRemOp, x, one)
            }
            "urem_by_one" => {
                let attr = ConstantValue::UInt(1).as_attribute(s.ctx(), storage);
                let one = s.register_with_result(&ConstantOp::new(
                    s.ctx_mut(),
                    boxed_attr_cast(attr).unwrap(),
                ));
                bin!(URemOp, x, one)
            }
            _ => panic!("unknown fixed audit fixture {case}"),
        };
        let result = if matches!(storage, ElemType::Float(FloatKind::F16 | FloatKind::BF16)) {
            s.register_with_result(&CastOp::new(s.ctx_mut(), storage.to_type(s.ctx()), result))
        } else {
            result
        };
        let ptr = s.register_with_result(&IndexOp::new(s.ctx_mut(), arrays[2], i, None));
        s.register(&StoreOp::new(s.ctx_mut(), ptr, result));
        s.terminate_yield();
    }
    b.branch_child(OpInserter::new_at_block_end(branch.else_block(b.ctx())))
        .terminate_yield();
    b.register(&branch);
    b.register(&ReturnOp::new(b.ctx_mut()));
    b.build()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 4 {
        return Err("usage: fp_probe OUT cpp|llvm CASE DTYPE".into());
    }
    let (out, route, case, dtype) = (&args[0], &args[1], &args[2], &args[3]);
    let storage = match dtype.as_str() {
        "float32" => ElemType::Float(FloatKind::F32),
        "float16" => ElemType::Float(FloatKind::F16),
        "bfloat16" => ElemType::Float(FloatKind::BF16),
        "int64" => ElemType::Int(IntKind::I64),
        "uint64" => ElemType::UInt(UIntKind::U64),
        _ => return Err("audit storage type unsupported".into()),
    };
    let name = format!("cubecl_fp_{case}_{dtype}");
    let out = Path::new(out);
    std::fs::create_dir_all(out)?;
    let kernel = build(case, storage, &name);
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
    if route == "cpp" {
        let source =
            CppCompiler::<Cuda>::default().compile(kernel, &CompilationOptions::default())?;
        std::fs::write(out.join("kernel.cu"), source.source)?;
    } else if route == "llvm" {
        #[cfg(feature = "llvm")]
        {
            use cubecl_core::ir::nvidia::SmArch;
            use cubecl_llvm::{LlvmTarget, PlironArtifact, PlironCompiler, PlironOptions};
            let mut compiler = PlironCompiler {
                target: LlvmTarget::Nvptx,
            };
            let options = PlironOptions {
                sm_arch: Some(SmArch::new(86, true)),
                grid_constants: false,
                ..Default::default()
            };
            let PlironArtifact::NvptxCode(source) = compiler.compile(kernel, &options)? else {
                unreachable!()
            };
            std::fs::write(out.join("kernel.ll"), source.ir)?;
            std::fs::write(
                out.join("kernel.ptx"),
                bytemuck::cast_slice::<i8, u8>(&source.ptx),
            )?;
        }
        #[cfg(not(feature = "llvm"))]
        return Err("build audit compiler with llvm feature".into());
    } else {
        return Err("audit route unsupported".into());
    }
    let manifest = serde_json::json!({
        "schema": 1, "entrypoint": name, "address_bits": 64, "vector_width": 1,
        "arguments": [format!("x {dtype} device pointer"), format!("y {dtype} device pointer"), format!("output {dtype} device pointer"), "info device pointer".to_string()],
        "metadata": {"bytes": 24}, "block": [256,1,1], "dynamic_shared_bytes": 0,
        "grid_constants": false, "fixture": case, "storage_dtype": dtype,
        "implicit_fp_contraction": false
    });
    std::fs::write(
        out.join("manifest.json"),
        serde_json::to_string_pretty(&manifest)?,
    )?;
    Ok(())
}
