//! One diagnostic RMS composition. Semantic replay and upstream bodies are reused.
#[allow(dead_code)]
#[path = "../../../stage0/src/descriptor2.rs"]
mod descriptor2;

use cubecl::Compiler;
use cubecl::ir::{
    AddressType, Builtin,
    dialect::{
        branch::ReturnOp,
        cmp::IEqualOp,
        general::{CastOp, ReadBuiltinOp, ReadScalarOp, ReinterpretCastOp, SelectOp},
        math::{IAddOp, IMulOp, ISubOp},
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
use cubek_reduce::{
    BoundChecks, ReduceInstruction, VectorizationMode,
    components::{
        instructions::{ReduceStep, Sum},
        readers::{Reader, plane::PlaneReader},
    },
};
use serde_json::{Value as Json, json};
use std::{collections::HashMap, fs, path::PathBuf};

type One = Const<1>;
type Four = Const<4>;
type P = (f32, Four, f32);

// Expanded handles only: no CubeK tensor bindings, runtime client or allocator.
struct Replay;
#[derive(Clone)]
struct ReplayExpand {
    ops: Vec<Json>,
    arrays: Vec<IrValue>,
    scalars: Vec<IrValue>,
    row: IrValue,
    n: IrValue,
    rows: IrValue,
    mode: u8,
    sum_id: String,
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
        if self.mode == 0 { 4 } else { 1 }
    }
}
impl VirtualTensorOperations<f32, One> for Replay {}

// The reader sees vectors of FP32 semantic operands; storage and epilogue stay
// scalar and retain the unchanged descriptor emitter's precision barriers.
struct Operand;
impl CubeType for Operand {
    type ExpandType = ReplayExpand;
}
impl Vectorized for Operand {}
impl VirtualTensorOperations<f32, Four> for Operand {}

impl VirtualTensorOperationsExpand<f32, Four> for ReplayExpand {
    fn __expand_read_method(
        &self,
        s: &Scope,
        pos: NativeExpand<usize>,
    ) -> NativeExpand<Vector<f32, Four>> {
        assert_eq!(self.mode, 0);
        let start =
            s.register_with_result(&IMulOp::new(s.ctx_mut(), signed(s, pos.value(s)), c(s, 4)));
        // Adapt descriptor5::reduction_vector_values locally: storage-vector
        // load, scalar extraction, then unchanged non-load semantic replay.
        let load = &self.ops[0];
        assert_eq!(load["op"], "load");
        let args = self.coordinates(s, index_type(s, start).into());
        let coordinate = descriptor2::index(s, &load["index"], self.row, &args).unwrap();
        let coordinate = index_type(s, coordinate);
        let ptr =
            s.register_with_result(&IndexOp::new(s.ctx_mut(), self.arrays[0], coordinate, None));
        let space = ptr
            .get_type(s.ctx())
            .deref(s.ctx())
            .downcast_ref::<PointerType>()
            .unwrap()
            .address_space;
        let storage = descriptor2::elem(load["dtype"].as_str().unwrap())
            .unwrap()
            .to_type(s.ctx());
        let storage4 = VectorType::get(s.ctx(), storage, 4).to_handle();
        let pointer4 = PointerType::get(s.ctx(), storage4, space).to_handle();
        let ptr = s.register_with_result(&ReinterpretCastOp::new(s.ctx_mut(), pointer4, ptr));
        let packed = s.register_with_result(&LoadOp::new(s.ctx_mut(), ptr));
        let mut components = Vec::new();
        for component in 0..4 {
            let value =
                s.register_with_result(&CompositeExtractOp::new(s.ctx_mut(), packed, component));
            let scalar_pos =
                s.register_with_result(&IAddOp::new(s.ctx_mut(), start, c(s, component as i64)));
            let mut values = HashMap::from([(load["id"].as_str().unwrap().to_owned(), value)]);
            descriptor2::emit(
                s,
                &self.ops[1..],
                &self.arrays,
                self.row,
                &self.coordinates(s, index_type(s, scalar_pos).into()),
                &mut values,
            )
            .unwrap();
            components.push(values[self.sum_id.as_str()]);
        }
        s.register_with_result(&CompositeConstructOp::new(
            s.ctx_mut(),
            VectorType::get(
                s.ctx(),
                descriptor2::elem("float32").unwrap().to_type(s.ctx()),
                4,
            )
            .to_handle(),
            components,
        ))
        .into()
    }
    fn __expand_write_method(
        &mut self,
        _: &Scope,
        _: NativeExpand<usize>,
        _: NativeExpand<Vector<f32, Four>>,
    ) {
        panic!("operand is read-only")
    }
    fn __expand_shape_method(&self, s: &Scope, axis: NativeExpand<usize>) -> NativeExpand<usize> {
        self.shape(s, axis, false)
    }
    fn __expand_stride_method(&self, s: &Scope, axis: NativeExpand<usize>) -> NativeExpand<usize> {
        self.shape(s, axis, true)
    }
    fn __expand_rank_method(&self, s: &Scope) -> NativeExpand<usize> {
        2usize.into_expand(s)
    }
    fn __expand_len_method(&self, s: &Scope) -> NativeExpand<usize> {
        use cubecl::ir::dialect::math::SDivOp;
        let elements = s.register_with_result(&IMulOp::new(s.ctx_mut(), self.n, self.rows));
        let vectors = s.register_with_result(&SDivOp::new(s.ctx_mut(), elements, c(s, 4)));
        index_type(s, vectors).into()
    }
    fn __expand_buffer_len_method(&self, s: &Scope) -> NativeExpand<usize> {
        <Self as VirtualTensorOperationsExpand<f32, Four>>::__expand_len_method(self, s)
    }
    fn __expand_read_window_method(
        &self,
        _: &Scope,
        _: NativeExpand<usize>,
        _: NativeExpand<usize>,
    ) -> &SliceExpand<Vector<f32, Four>> {
        panic!("vector operand has no read windows")
    }
    fn __expand_as_tensor_map_method(
        &self,
        s: &Scope,
    ) -> ComptimeOptionExpand<TensorMap<f32, Tiled>> {
        ComptimeOption::__expand_new_None(s)
    }
}

fn index_type(s: &Scope, v: IrValue) -> IrValue {
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
fn c(s: &Scope, v: i64) -> IrValue {
    descriptor2::constant(s, &json!(v), "int64").unwrap()
}

impl ReplayExpand {
    fn coordinates(&self, s: &Scope, pos: NativeExpand<usize>) -> Vec<IrValue> {
        let pos = signed(s, pos.value(s));
        let base = s.register_with_result(&IMulOp::new(s.ctx_mut(), self.row, self.n));
        let column = s.register_with_result(&ISubOp::new(s.ctx_mut(), pos, base));
        let mut scalars = self.scalars.clone();
        scalars.push(column);
        scalars
    }
    fn shape(&self, s: &Scope, axis: NativeExpand<usize>, stride: bool) -> NativeExpand<usize> {
        let first = s.register_with_result(&IEqualOp::new(
            s.ctx_mut(),
            signed(s, axis.value(s)),
            c(s, 0),
        ));
        let (a, b) = if stride {
            (if self.mode == 1 { c(s, 1) } else { self.n }, c(s, 1))
        } else {
            (self.rows, if self.mode == 1 { c(s, 1) } else { self.n })
        };
        let value = s.register_with_result(&SelectOp::new(s.ctx_mut(), first, a, b));
        index_type(s, value).into()
    }
}
impl VirtualTensorOperationsExpand<f32, One> for ReplayExpand {
    fn __expand_read_method(
        &self,
        s: &Scope,
        pos: NativeExpand<usize>,
    ) -> NativeExpand<Vector<f32, One>> {
        assert_eq!(self.mode, 0, "only the operand virtual tensor may be read");
        let mut values = HashMap::new();
        descriptor2::emit(
            s,
            &self.ops,
            &self.arrays,
            self.row,
            &self.coordinates(s, pos),
            &mut values,
        )
        .unwrap();
        values[self.sum_id.as_str()].into()
    }
    fn __expand_write_method(
        &mut self,
        s: &Scope,
        pos: NativeExpand<usize>,
        sum: NativeExpand<Vector<f32, One>>,
    ) {
        if self.mode == 3 {
            let ptr = s.register_with_result(&IndexOp::new(s.ctx_mut(), self.arrays[0], pos.value(s), None));
            s.register(&StoreOp::new(s.ctx_mut(), ptr, sum.read_value(s)));
            return;
        }
        assert_eq!(
            self.mode, 2,
            "metadata-only reduced output cannot be written"
        );
        let mut values = HashMap::from([(self.sum_id.clone(), sum.read_value(s))]);
        descriptor2::emit(
            s,
            &self.ops,
            &self.arrays,
            self.row,
            &self.coordinates(s, pos),
            &mut values,
        )
        .unwrap();
    }
    fn __expand_shape_method(&self, s: &Scope, axis: NativeExpand<usize>) -> NativeExpand<usize> {
        self.shape(s, axis, false)
    }
    fn __expand_stride_method(&self, s: &Scope, axis: NativeExpand<usize>) -> NativeExpand<usize> {
        self.shape(s, axis, true)
    }
    fn __expand_rank_method(&self, s: &Scope) -> NativeExpand<usize> {
        2usize.into_expand(s)
    }
    fn __expand_len_method(&self, s: &Scope) -> NativeExpand<usize> {
        let n = if self.mode == 1 {
            self.rows
        } else {
            s.register_with_result(&IMulOp::new(s.ctx_mut(), self.n, self.rows))
        };
        index_type(s, n).into()
    }
    fn __expand_buffer_len_method(&self, s: &Scope) -> NativeExpand<usize> {
        <Self as VirtualTensorOperationsExpand<f32, One>>::__expand_len_method(self, s)
    }
    fn __expand_read_window_method(
        &self,
        _: &Scope,
        _: NativeExpand<usize>,
        _: NativeExpand<usize>,
    ) -> &SliceExpand<Vector<f32, One>> {
        panic!("RMS adapter has no read windows")
    }
    fn __expand_as_tensor_map_method(
        &self,
        s: &Scope,
    ) -> ComptimeOptionExpand<TensorMap<f32, Tiled>> {
        ComptimeOption::__expand_new_None(s)
    }
}

#[cube]
fn rms_components(
    input: &VirtualTensor<f32, Four>,
    meta: &mut VirtualTensor<f32, One, ReadWrite>,
    sink: &mut VirtualTensor<f32, One, ReadWrite>,
    debug: &mut VirtualTensor<f32, One, ReadWrite>,
    row: usize,
    n: usize,
    rows: usize,
) {
    if row < rows {
        let inst = Sum {};
        let reader = Reader::<P>::new::<Sum, (f32, One)>(
            input,
            meta,
            &inst,
            1usize,
            row,
            ComptimeOption::new_None(),
            BoundChecks::Branch,
            VectorizationMode::Parallel,
            false,
        );
        let reader = PlaneReader::<P>::new(reader);
        let mut accumulator = <Sum as ReduceInstruction<P>>::null_accumulator(&inst);
        for i in 0..reader.length() {
            let item = reader.read(i);
            for component in 0..4usize {
                let coordinate = row * (4usize * n + 32usize) + (32usize * i + UNIT_POS_X as usize) * 4usize + component;
                debug.write(coordinate, Vector::<f32, One>::new(item.elements[component]));
            }
            <Sum as ReduceInstruction<P>>::reduce(
                &inst,
                &mut accumulator,
                item,
                ReduceStep::Identity,
            );
        }
        <Sum as ReduceInstruction<P>>::plane_reduce_inplace(&inst, &mut accumulator);
        let (sum, _) =
            <Sum as ReduceInstruction<P>>::to_output_parallel::<f32, u32>(&inst, accumulator, n);
        debug.write(row * (4usize * n + 32usize) + 4usize * n + UNIT_POS_X as usize, Vector::<f32, One>::new(sum.item()));
        let sum = Vector::<f32, One>::new(sum.item());
        for r in range_stepped(UNIT_POS_X as usize, n, 32usize) {
            sink.write(row * n + r, sum);
        }
    }
}

fn bind(v: &Json) -> Json {
    if v["op"] == "reduction" && v.get("dtype").is_none() {
        return json!({"op":"scalar","slot":2});
    }
    match v {
        Json::Array(a) => Json::Array(a.iter().map(bind).collect()),
        Json::Object(o) => Json::Object(o.iter().map(|(k, v)| (k.clone(), bind(v))).collect()),
        _ => v.clone(),
    }
}
pub fn generate() -> Result<(), Box<dyn std::error::Error>> {
    let out = PathBuf::from(std::env::args().nth(1).ok_or("output directory")?);
    let packet: Json = serde_json::from_str(&fs::read_to_string(
        std::env::args().nth(2).ok_or("descriptor")?,
    )?)?;
    let sem = &packet["semantic"];
    let entry = packet["entrypoint"].as_str().ok_or("entrypoint")?;
    if packet["schema"] != 5
        || !entry.starts_with("cubecl5_rms_cubek_pack4_")
        || !entry.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
    {
        return Err("RMS diagnostic descriptor/entrypoint mismatch".into());
    }
    // Exact, reviewed semantic templates; the diagnostic does not admit a generic DAG.
    let templates = [
        include_str!(
            "../../artifacts/reuse-first-selection/kernels/177394e178b0928b66264afbde55a1a3cc808be7a32e2890b0f8ee898e2074f4/semantic.json"
        ),
        include_str!(
            "../../artifacts/reuse-first-selection/kernels/55f3919b96c6b5b78e23396ad9b7e33909e983423e950d64a544defdab7e45bd/semantic.json"
        ),
        include_str!(
            "../../artifacts/reuse-first-selection/kernels/48767364e45478057a7a64181bf625c7f3f162b113c4b9cb347eb4343ddfaffb/semantic.json"
        ),
    ];
    let dtype = sem["buffers"][0]["dtype"].as_str().ok_or("dtype")?;
    if !templates
        .iter()
        .any(|s| serde_json::from_str::<Json>(s).is_ok_and(|t| sem == &t))
        || !matches!(dtype, "float32" | "float16" | "bfloat16")
    {
        return Err("RMS semantic contract differs from qualified template".into());
    }
    if packet["physical_plan"]
        != json!({"version":2,"strategy":"independent_planes","workgroup":[32,8,1],
       "vector_width":1,"elements_per_unit":1,"outputs_per_workgroup":8,"plane_width":32})
        || packet["mechanism"] != "actual-cubek-rms-plane-packed-storage4-v1"
    {
        return Err("RMS diagnostic mechanism mismatch".into());
    }
    let mut b = KernelBuilder::new(
        KernelSettings::new(
            Dim3::new_2d(32, 8),
            ExecutionMode::Unchecked,
            AddressType::U64,
        )
        .kernel_name(packet["entrypoint"].as_str().ok_or("entrypoint")?),
    );
    let x = b.buffer(descriptor2::elem(dtype).unwrap().to_type(b.ctx()));
    let trace = b.buffer(descriptor2::elem("float32").unwrap().to_type(b.ctx()));
    let y = b.buffer(descriptor2::elem(dtype).unwrap().to_type(b.ctx()));
    let arrays = vec![x, y];
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
    let builtin = |kind| {
        let v = b.register_with_result(&ReadBuiltinOp::new(
            b.ctx_mut(),
            IndexType::get(b.ctx()).to_handle(),
            kind,
        ));
        signed(&b, v)
    };
    let cube = builtin(Builtin::CubePosX);
    let plane = builtin(Builtin::UnitPosY);
    let base = b.register_with_result(&IMulOp::new(b.ctx_mut(), cube, c(&b, 8)));
    let row = b.register_with_result(&IAddOp::new(b.ctx_mut(), base, plane));
    let ops = bind(&sem["ops"]).as_array().ok_or("ops")?.clone();
    let cut = ops
        .iter()
        .position(|o| o["op"] == "reduction")
        .ok_or("sum")?;
    let sum_id = ops[cut]["id"].as_str().ok_or("sum id")?.to_owned();
    let input = ReplayExpand {
        ops: ops[..cut].to_vec(),
        arrays: arrays.clone(),
        scalars: scalars[1..].to_vec(),
        row,
        n: scalars[1],
        rows: scalars[2],
        mode: 0,
        sum_id: ops[cut]["args"][0].as_str().unwrap().to_owned(),
    };
    let meta = ReplayExpand {
        mode: 1,
        ..input.clone()
    };
    let sink = ReplayExpand {
        mode: 2,
        sum_id,
        ops: ops.into_iter().filter(|o| o["op"] != "reduction").collect(),
        ..input.clone()
    };
    let debug = ReplayExpand { mode: 3, arrays: vec![trace], ..input.clone() };
    let mut debug = VirtualTensor::<f32, One, ReadWrite>::__expand_new::<Replay>(&b, debug);
    let input = VirtualTensor::<f32, Four>::__expand_new::<Operand>(&b, input);
    let mut meta = VirtualTensor::<f32, One, ReadWrite>::__expand_new::<Replay>(&b, meta);
    let mut sink = VirtualTensor::<f32, One, ReadWrite>::__expand_new::<Replay>(&b, sink);
    rms_components::expand(
        &b,
        &input,
        &mut meta,
        &mut sink,
        &mut debug,
        index_type(&b, row).into(),
        index_type(&b, scalars[1]).into(),
        index_type(&b, scalars[2]).into(),
    );
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
        serde_json::to_string_pretty(&json!({
            "schema":1,"descriptor_schema":5,"semantic_schema":3,"entrypoint":packet["entrypoint"],
            "block":[32,8,1],"physical_plan":packet["physical_plan"],"dynamic_shared_bytes":repr.shared_memory_size,
            "route":"cpp","mechanism":"actual-cubek-rms-plane-packed-storage4-v1","metadata":{"scalar_count":3,"bytes":48},
            "arguments":[format!("{dtype} device pointer"),"float32 observation pointer",format!("{dtype} device pointer"),"info device pointer"],
            "grid_constants":false,"vector_width":1,"operand_vector_width":4,"storage_vector_width":4,"address_bits":64
        }))?,
    )?;
    Ok(())
}
