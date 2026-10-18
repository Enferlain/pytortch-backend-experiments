//! One runtime-free translation of CubeCL's perpendicular contiguous component.
#[allow(dead_code)]
#[path = "../../../stage0/src/descriptor2.rs"]
mod descriptor2;
use cubecl::Compiler;
use cubecl::ir::{
    AddressType,
    dialect::{
        branch::ReturnOp,
        cmp::IEqualOp,
        general::{CastOp, ReadScalarOp, ReinterpretCastOp, SelectOp},
        math::{IAddOp, IMulOp, SDivOp},
        memory::{IndexOp, LoadOp, StoreOp},
        vector::{CompositeConstructOp, CompositeExtractOp},
    },
    pliron::{op::Op, printable::Printable, r#type::Typed, value::Value as IrValue},
    settings::{Dim3, ExecutionMode, KernelSettings},
    types::{PointerType, VectorType, scalar::IndexType},
};
use cubecl::prelude::*;
use cubecl::std::tensor::r#virtual::{
    VirtualTensor, VirtualTensorOperations, VirtualTensorOperationsExpand,
};
use cubecl_cpp::{
    shared::{CompilationOptions, CppCompiler},
    target::Cuda,
};
use half::{bf16, f16};
use serde_json::{Value as Json, json};
use std::{collections::HashMap, fs, path::PathBuf};

include!("component.rs");

struct Replay;
#[derive(Clone)]
struct ReplayExpand {
    arrays: Vec<IrValue>,
    scalars: Vec<IrValue>,
    ops: Vec<Json>,
    dtype: String,
    width: usize,
    output: bool,
}
impl CubeType for Replay {
    type ExpandType = ReplayExpand;
}
impl IntoExpand for ReplayExpand {
    type Expand = Self;
    fn into_expand(self, _: &Scope) -> Self {
        self
    }
}
impl ExpandTypeClone for ReplayExpand {
    fn clone_unchecked(&self) -> Self {
        self.clone()
    }
}
impl IntoMut for ReplayExpand {
    fn into_mut(self, _: &Scope) -> Self {
        self
    }
}
impl CubeDebug for ReplayExpand {}
impl AsRefExpand for ReplayExpand {
    fn __expand_ref_method(&self, _: &Scope) -> &Self {
        self
    }
}
impl AsMutExpand for ReplayExpand {
    fn __expand_ref_mut_method(&mut self, _: &Scope) -> &mut Self {
        self
    }
}
impl Vectorized for Replay {}
impl VectorizedExpand for ReplayExpand {
    fn __expand_vector_size_method(&self, _: &Scope) -> VectorSize {
        self.width
    }
}
impl<T: Numeric, N: Size> VirtualTensorOperations<T, N> for Replay {}
fn c(s: &Scope, v: i64) -> IrValue {
    descriptor2::constant(s, &json!(v), "int64").unwrap()
}
fn cast(s: &Scope, v: IrValue, ty: cubecl::ir::pliron::r#type::TypeHandle) -> IrValue {
    s.register_with_result(&CastOp::new(s.ctx_mut(), ty, v))
}
fn signed(s: &Scope, v: IrValue) -> IrValue {
    cast(s, v, descriptor2::elem("int64").unwrap().to_type(s.ctx()))
}
fn index(s: &Scope, v: IrValue) -> IrValue {
    cast(s, v, IndexType::get(s.ctx()).to_handle())
}
impl ReplayExpand {
    fn vector_type(&self, s: &Scope) -> cubecl::ir::pliron::r#type::TypeHandle {
        if self.width == 1 {
            return descriptor2::elem(&self.dtype).unwrap().to_type(s.ctx());
        }
        VectorType::get(
            s.ctx(),
            descriptor2::elem(&self.dtype).unwrap().to_type(s.ctx()),
            self.width,
        )
        .to_handle()
    }
    fn pointer(&self, s: &Scope, buffer: usize, scalar: IrValue) -> IrValue {
        let p = s.register_with_result(&IndexOp::new(
            s.ctx_mut(),
            self.arrays[buffer],
            index(s, scalar),
            None,
        ));
        let space = p
            .get_type(s.ctx())
            .deref(s.ctx())
            .downcast_ref::<PointerType>()
            .unwrap()
            .address_space;
        let ty = PointerType::get(s.ctx(), self.vector_type(s), space).to_handle();
        s.register_with_result(&ReinterpretCastOp::new(s.ctx_mut(), ty, p))
    }
    fn meta(&self, s: &Scope, axis: NativeExpand<usize>, stride: bool) -> NativeExpand<usize> {
        let is0 = s.register_with_result(&IEqualOp::new(
            s.ctx_mut(),
            signed(s, axis.value(s)),
            c(s, 0),
        ));
        let (a, b) = if !stride {
            (self.scalars[0], self.scalars[1])
        } else if self.output {
            (self.scalars[1], c(s, 1))
        } else {
            (c(s, 1), self.scalars[0])
        };
        index(
            s,
            s.register_with_result(&SelectOp::new(s.ctx_mut(), is0, a, b)),
        )
        .into()
    }
}
impl<T: Numeric, N: Size> VirtualTensorOperationsExpand<T, N> for ReplayExpand {
    fn __expand_read_method(
        &self,
        s: &Scope,
        pos: NativeExpand<usize>,
    ) -> NativeExpand<Vector<T, N>> {
        assert!(!self.output);
        let scalar = s.register_with_result(&IMulOp::new(
            s.ctx_mut(),
            signed(s, pos.value(s)),
            c(s, self.width as i64),
        ));
        s.register_with_result(&LoadOp::new(s.ctx_mut(), self.pointer(s, 0, scalar)))
            .into()
    }
    fn __expand_write_method(
        &mut self,
        s: &Scope,
        pos: NativeExpand<usize>,
        v: NativeExpand<Vector<T, N>>,
    ) {
        assert!(self.output);
        let scalar = s.register_with_result(&IMulOp::new(
            s.ctx_mut(),
            signed(s, pos.value(s)),
            c(s, self.width as i64),
        ));
        let mut elements = Vec::new();
        for component in 0..self.width {
            let input = if self.width == 1 {
                v.value(s)
            } else {
                s.register_with_result(&CompositeExtractOp::new(s.ctx_mut(), v.value(s), component))
            };
            let linear =
                s.register_with_result(&IAddOp::new(s.ctx_mut(), scalar, c(s, component as i64)));
            let mut values =
                HashMap::from([(self.ops[0]["id"].as_str().unwrap().to_owned(), input)]);
            descriptor2::emit(
                s,
                &self.ops[1..self.ops.len() - 1],
                &self.arrays,
                linear,
                &self.scalars,
                &mut values,
            )
            .unwrap();
            elements.push(values[self.ops.last().unwrap()["args"][0].as_str().unwrap()]);
        }
        let packed = if self.width == 1 {
            elements[0]
        } else {
            s.register_with_result(&CompositeConstructOp::new(
                s.ctx_mut(),
                self.vector_type(s),
                elements,
            ))
        };
        s.register(&StoreOp::new(
            s.ctx_mut(),
            self.pointer(s, 1, scalar),
            packed,
        ));
    }
    fn __expand_shape_method(&self, s: &Scope, axis: NativeExpand<usize>) -> NativeExpand<usize> {
        self.meta(s, axis, false)
    }
    fn __expand_stride_method(&self, s: &Scope, axis: NativeExpand<usize>) -> NativeExpand<usize> {
        self.meta(s, axis, true)
    }
    fn __expand_rank_method(&self, s: &Scope) -> NativeExpand<usize> {
        2usize.into_expand(s)
    }
    fn __expand_len_method(&self, s: &Scope) -> NativeExpand<usize> {
        let n = s.register_with_result(&IMulOp::new(s.ctx_mut(), self.scalars[0], self.scalars[1]));
        index(
            s,
            s.register_with_result(&SDivOp::new(s.ctx_mut(), n, c(s, self.width as i64))),
        )
        .into()
    }
    fn __expand_buffer_len_method(&self, s: &Scope) -> NativeExpand<usize> {
        <Self as VirtualTensorOperationsExpand<T, N>>::__expand_len_method(self, s)
    }
    fn __expand_read_window_method(
        &self,
        _: &Scope,
        _: NativeExpand<usize>,
        _: NativeExpand<usize>,
    ) -> &SliceExpand<Vector<T, N>> {
        panic!("no windows in this component")
    }
    fn __expand_as_tensor_map_method(
        &self,
        s: &Scope,
    ) -> ComptimeOptionExpand<TensorMap<T, Tiled>> {
        ComptimeOption::__expand_new_None(s)
    }
}
fn expand<T: Numeric, N: Size>(b: &Scope, input: ReplayExpand, output: ReplayExpand) {
    let elem = descriptor2::elem(&input.dtype).unwrap();
    let input = VirtualTensor::<T, N>::__expand_new::<Replay>(b, input);
    let mut output = VirtualTensor::<T, N, ReadWrite>::__expand_new::<Replay>(b, output);
    copy_perpendicular::expand::<T, N>(b, &input, &mut output, 0usize.into_expand(b), elem);
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out = PathBuf::from(std::env::args().nth(1).ok_or("output")?);
    let packet: Json = serde_json::from_str(&fs::read_to_string(
        std::env::args().nth(2).ok_or("descriptor")?,
    )?)?;
    let sem = &packet["semantic"];
    let templates: Vec<Json> = serde_json::from_str(include_str!("templates.json"))?;
    let plan = json!({"version":2,"strategy":"scalar_pointwise_baseline","workgroup":[256,1,1],"vector_width":1,"elements_per_unit":1,"outputs_per_workgroup":1,"plane_width":0});
    let width = packet["storage_vector_width"].as_u64().ok_or("width")? as usize;
    let entry = packet["entrypoint"].as_str().ok_or("entrypoint")?;
    if packet["schema"] != 5
        || packet["mechanism"] != "cubecl-perpendicular-virtual-v1"
        || packet["physical_plan"] != plan
        || !templates.contains(sem)
        || !matches!(width, 1 | 4 | 8)
        || !entry.starts_with("cubecl5_transpose_")
        || !entry.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
    {
        return Err("transpose diagnostic admission".into());
    }
    let dtype = sem["buffers"][0]["dtype"].as_str().ok_or("dtype")?;
    if dtype == "float32" && width > 4 {
        return Err("FP32 I/O width".into());
    }
    let mut b = KernelBuilder::new(
        KernelSettings::new(
            Dim3::new_1d(256),
            ExecutionMode::Unchecked,
            AddressType::U64,
        )
        .kernel_name(entry),
    );
    let arrays: Vec<_> = sem["buffers"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| {
            b.buffer(
                descriptor2::elem(v["dtype"].as_str().unwrap())
                    .unwrap()
                    .to_type(b.ctx()),
            )
        })
        .collect();
    let scalars: Vec<_> = (0..3)
        .map(|_| {
            let id = b.scalar(descriptor2::elem("int64").unwrap());
            b.register_with_result(&ReadScalarOp::new(
                b.ctx_mut(),
                descriptor2::elem("int64").unwrap().to_type(b.ctx()),
                id,
            ))
        })
        .collect();
    let input = ReplayExpand {
        arrays,
        scalars: scalars[1..].to_vec(),
        ops: sem["ops"].as_array().unwrap().clone(),
        dtype: dtype.into(),
        width,
        output: false,
    };
    let output = ReplayExpand {
        output: true,
        ..input.clone()
    };
    macro_rules! run {
        ($t:ty,$n:expr) => {
            expand::<$t, Const<$n>>(&b, input, output)
        };
    }
    match (dtype, width) {
        ("float32", 1) => run!(f32, 1),
        ("float32", 4) => run!(f32, 4),
        ("float16", 1) => run!(f16, 1),
        ("float16", 8) => run!(f16, 8),
        ("bfloat16", 1) => run!(bf16, 1),
        ("bfloat16", 8) => run!(bf16, 8),
        _ => return Err("unsupported storage width".into()),
    }
    b.register(&ReturnOp::new(b.ctx_mut()));
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
            &json!({"schema":1,"descriptor_schema":5,"semantic_schema":4,"entrypoint":entry,"block":[256,1,1],"physical_plan":plan,"dynamic_shared_bytes":repr.shared_memory_size,"route":"cpp","mechanism":packet["mechanism"],"metadata":{"scalar_count":3,"bytes":40},"arguments":[format!("{dtype} device pointer"),format!("{dtype} device pointer"),"info device pointer"],"grid_constants":false,"vector_width":1,"storage_vector_width":width,"address_bits":64}),
        )?,
    )?;
    Ok(())
}
