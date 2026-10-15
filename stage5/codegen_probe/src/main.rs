//! Compile actual CubeK readers/components without its runtime or allocator.
use cubecl::Compiler;
use cubecl::ir::pliron::{op::Op, printable::Printable};
use cubecl::{
    ir::{
        AddressType,
        dialect::general::{BufferLenOp, ReadScalarOp},
        settings::{Dim3, ExecutionMode, KernelSettings},
        types::scalar::IndexType,
    },
    prelude::*,
    std::tensor::{View, layout::plain::PlainLayout},
};
use cubecl_cpp::{
    shared::{CompilationOptions, CppCompiler},
    target::Cuda,
};
use cubek_reduce::components::instructions::Sum;
use cubek_reduce::{BoundChecks, components::readers::bound_checks::ReaderBoundChecks};
use std::{fs, path::PathBuf};

type One = Const<1>;
type P = (f32, One, f32);

#[cube]
fn reader_probe(
    input: &[Vector<f32, One>],
    output: &mut [f32],
    n: usize,
    rows: usize,
    #[comptime] checks: BoundChecks,
) {
    let inst = Sum {};
    let bounds = ReaderBoundChecks::<P>::new::<Sum>(&inst, n, ComptimeOption::new_None(), checks);
    let view = View::new::<&[Vector<f32, One>], usize>(input, PlainLayout::new(n * rows));
    let row = CUBE_POS_X as usize;
    let t = UNIT_POS as usize;
    if row < rows {
        let mut a0 = 0.0f32;
        let mut a1 = 0.0f32;
        let mut a2 = 0.0f32;
        let mut a3 = 0.0f32;
        for i in 0..n.div_ceil(2048) {
            let r = i * 2048 + t;
            let v0 = bounds.read(r, row * n + r, &view);
            let v1 = bounds.read(r + 512, row * n + r + 512, &view);
            let v2 = bounds.read(r + 1024, row * n + r + 1024, &view);
            let v3 = bounds.read(r + 1536, row * n + r + 1536, &view);
            a0 += v0.extract(0usize);
            a1 += v1.extract(0usize);
            a2 += v2.extract(0usize);
            a3 += v3.extract(0usize);
        }
        // Diagnostic per-thread chains, no new reduction topology.
        output[row * 512 + t] = (a0 + a1) + (a2 + a3);
    }
}

#[cube]
fn component_probe(
    input: Box<[Vector<f32, One>]>,
    output: Box<[Vector<f32, One>]>,
    n: usize,
    rows: usize,
    #[comptime] blueprint: cubek_reduce::routines::ReduceBlueprint,
) {
    let imeta = TensorMeta {
        len: n * rows,
        rank: 2usize,
    };
    let ometa = TensorMeta {
        len: rows,
        rank: 2usize,
    };
    let input = OwnedTensor::from_parts(imeta, input);
    let mut output = OwnedTensor::from_parts(ometa, output);
    let (input, mut output) = cubek_reduce::components::args::init_tensors::<
        cubek_reduce::components::args::TensorArgs,
        f32,
        One,
        f32,
        One,
    >(&input, &mut output);
    cubek_reduce::launch::reduce_kernel_virtual::<f32, One, f32, One, f32>(
        &input,
        &mut output,
        1usize,
        0usize,
        blueprint,
        cubek_reduce::components::instructions::ReduceOperationConfig::Sum,
    );
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out = PathBuf::from(std::env::args().nth(1).ok_or("output path")?);
    let mode = std::env::args().nth(2).unwrap_or("branch".into());
    if out.to_str() == Some("--metadata") {
        let n = std::env::args().nth(3).ok_or("n")?.parse::<usize>()?;
        let rows = std::env::args().nth(4).ok_or("rows")?.parse::<usize>()?;
        let component = mode.starts_with("cube") || mode.starts_with("plane");
        let mut info = cubecl::InfoBuilder::default();
        info.scalars.push(n as u64);
        info.scalars.push(rows as u64);
        if component {
            info.metadata.register_tensor(
                (n * rows) as u64,
                vec![rows, n].into(),
                vec![n, 1].into(),
                AddressType::U64,
            );
            info.metadata.register_tensor(
                rows as u64,
                vec![rows, 1].into(),
                vec![1, 1].into(),
                AddressType::U64,
            );
        } else {
            info.metadata
                .register_buffer((n * rows) as u64, AddressType::U64);
            info.metadata
                .register_buffer((rows * 512) as u64, AddressType::U64);
        }
        println!(
            "{}",
            serde_json::to_string(&info.finish(AddressType::U64).data)?
        );
        return Ok(());
    }
    fs::create_dir_all(&out)?;
    let is_component = mode.starts_with("cube") || mode.starts_with("plane");
    let block = if mode.starts_with("plane") {
        Dim3::new_2d(32, 8)
    } else if is_component {
        Dim3::new_2d(32, 16)
    } else {
        Dim3::new_1d(512)
    };
    let mut b = KernelBuilder::new(
        KernelSettings::new(block, ExecutionMode::Unchecked, AddressType::U64)
            .kernel_name(format!("cubek_reader_{}", mode.replace("-", "_"))),
    );
    let vty = <Vector<f32, One>>::__expand_as_type(&b);
    let input = if is_component {
        b.tensor(vty)
    } else {
        b.buffer(vty)
    };
    let output = if is_component {
        b.tensor(vty)
    } else {
        b.buffer(vty)
    };
    let mut scalar = || {
        let id = b.scalar(ElemType::Index);
        b.register_with_result(&ReadScalarOp::new(
            b.ctx_mut(),
            IndexType::get(b.ctx()).to_handle(),
            id,
        ))
        .into()
    };
    let n = scalar();
    let rows = scalar();
    let ilen = b.register_with_result(&BufferLenOp::new(b.ctx_mut(), 0usize));
    let olen = b.register_with_result(&BufferLenOp::new(b.ctx_mut(), 1usize));
    let input = cubecl::frontend::slice::from_raw_parts::<Vector<f32, One>>(
        &b,
        input,
        0usize.into_expand(&b),
        ilen.into(),
    );
    let mut output = cubecl::frontend::slice::from_raw_parts::<f32>(
        &b,
        output,
        0usize.into_expand(&b),
        olen.into(),
    );
    if is_component {
        use cubek_reduce::routines::*;
        let checks = if mode.ends_with("mask") {
            BoundChecks::Mask
        } else {
            BoundChecks::Branch
        };
        let global = if mode.starts_with("plane") {
            GlobalReduceBlueprint::Plane(PlaneReduceBlueprint {
                plane_idle: cubek_reduce::IdleMode::None,
                bound_checks: checks,
                plane_merge_strategy: PlaneMergeStrategy::Lazy,
                plane_dim_ceil: false,
            })
        } else {
            GlobalReduceBlueprint::Cube(CubeBlueprint {
                cube_idle: cubek_reduce::IdleMode::None,
                bound_checks: checks,
                num_shared_accumulators: 16,
                use_planes: true,
            })
        };
        component_probe::expand(
            &b,
            input.expand.into(),
            output.expand.into(),
            n,
            rows,
            ReduceBlueprint {
                vectorization_mode: cubek_reduce::VectorizationMode::Parallel,
                global,
            },
        );
    } else {
        reader_probe::expand(
            &b,
            &input,
            &mut output,
            n,
            rows,
            if mode == "mask" {
                BoundChecks::Mask
            } else {
                BoundChecks::Branch
            },
        );
    }
    let kernel = b.build();
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
    fs::write(out.join("info.txt"), format!("{:#?}", kernel.info))?;
    let repr = CppCompiler::<Cuda>::default().compile(kernel, &CompilationOptions::default())?;
    fs::write(out.join("kernel.cu"), repr.source)?;
    fs::write(
        out.join("resources.json"),
        serde_json::to_string_pretty(
            &serde_json::json!({"shared":repr.shared_memory_size,"io":format!("{:?}",repr.io)}),
        )?,
    )?;
    Ok(())
}
