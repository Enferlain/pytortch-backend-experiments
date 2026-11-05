//! The fused ecosystem reduce (openspec fused-reduction-bridge 3.2): a reduction
//! kernel's recorded body run inside CubeK's reduce.
//!
//! CubeK's reduce reads its input and writes its output through
//! [`VirtualTensor`]s. Here both are our own: the input's `read` replays the
//! ops the reduction's operand is made of (loads and per-element ops), and the
//! output's `write` replays the per-output ops and stores, each with the
//! descriptor compiler's own emitter (`cubecl_stage0::replay`). Burn's
//! `FusedReduceArgs` does the same one level up, through `ReduceArgs`; it needs
//! that level because its state is a cube type. CubeK plans the launch from the
//! layout Inductor's block analysis recorded (`prepare_reduce_launch_for_layout`),
//! with `fuse_on_read` set, and infers its blueprint.
//!
//! Index arithmetic: CubeK addresses the input by scalar offsets through the
//! layout's strides (the reference loads' memory order). A lane's offset is
//! decoded back into coordinates, the output dimensions folded into the
//! output's linear index (Inductor's `xindex`) and the reduction coordinate
//! (Inductor's `r0_index`) given to the recorded indices as scalar slot 0. The
//! recorded indices then address every buffer as written.
use cubecl::ir::{
    AddressType, ElemType, Scope,
    dialect::{
        cmp::IEqualOp,
        general::{CastOp, SelectOp},
        math::{IAddOp, IMulOp, SDivOp, SRemOp},
        vector::{CompositeConstructOp, CompositeExtractOp},
    },
    pliron::value::Value as IrValue,
    types::VectorType,
};
use cubecl::prelude::*;
use cubecl::std::tensor::r#virtual::{
    VirtualTensor, VirtualTensorOperations, VirtualTensorOperationsExpand,
};
use cubecl_stage0::replay;
use cubek_reduce::{
    ReduceDtypes, ReduceStrategy,
    components::instructions::ReduceOperationConfig,
    launch::{VectorizationStrategy, prepare_reduce_launch_for_layout, reduce_kernel_virtual},
    routines::ReduceBlueprint,
};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

type Error = Box<dyn std::error::Error>;

/// One recorded buffer as a kernel argument: a global buffer whose element
/// type is its compilation argument, so buffers of different dtypes sit in one
/// [`Sequence`]. (Burn's `GlobalTensor` is the upstream form; it needs a newer
/// CubeCL than ours, bead yo9.)
pub struct RecordedBuffer;

#[derive(Clone)]
pub struct RecordedBufferExpand {
    value: IrValue,
}

impl CubeType for RecordedBuffer {
    type ExpandType = RecordedBufferExpand;
}

impl LaunchArg for RecordedBuffer {
    type RuntimeArg = (BufferArg, ElemType);
    type CompilationArg = ElemType;

    fn register((arg, elem): Self::RuntimeArg, launcher: &mut KernelLauncher) -> ElemType {
        launcher.register_buffer(arg, elem.size());
        elem
    }

    fn expand(elem: &ElemType, builder: &mut KernelBuilder) -> RecordedBufferExpand {
        let ty = elem.to_type(builder.scope.ctx());
        RecordedBufferExpand {
            value: builder.buffer(ty),
        }
    }
}

/// The expand-type traits every cube type's expansion carries, for a type that
/// is only ever passed along.
macro_rules! plain_expand {
    ($t:ty) => {
        impl IntoExpand for $t {
            type Expand = Self;
            fn into_expand(self, _: &Scope) -> Self {
                self
            }
        }
        impl ExpandTypeClone for $t {
            fn clone_unchecked(&self) -> Self {
                self.clone()
            }
        }
        impl IntoMut for $t {
            fn into_mut(self, _: &Scope) -> Self {
                self
            }
        }
        impl CubeDebug for $t {}
        impl AsRefExpand for $t {
            fn __expand_ref_method(&self, _: &Scope) -> &Self {
                self
            }
        }
        impl AsMutExpand for $t {
            fn __expand_ref_mut_method(&mut self, _: &Scope) -> &mut Self {
                self
            }
        }
    };
}

plain_expand!(RecordedBufferExpand);

/// What the kernel is compiled for: the body split at its reduction, and the
/// input and output layouts CubeK reduces over. Part of the kernel's identity.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FusedBody {
    /// The operand's ops, reduction coordinate bound to scalar slot 0 (JSON).
    prologue: String,
    /// The ops after the reduction, the same binding (JSON).
    epilogue: String,
    /// The operand's id (`read` returns it) and the reduction's id (`write`
    /// binds it).
    operand: String,
    reduction: String,
    /// Output dimensions, outermost first, as Inductor's `xindex` decomposes.
    outer: Vec<usize>,
    /// The input: the output dimensions then the reduction's, with strides.
    input_shape: Vec<usize>,
    input_strides: Vec<usize>,
    /// The output: the output dimensions then a unit dimension, dense in the
    /// input's memory order.
    output_strides: Vec<usize>,
}

impl FusedBody {
    /// The body of `descriptor` (a schema-3 reduction descriptor CubeK's sum
    /// covers, `legality.fused_reduce_reason`) over `layout` (the agreed
    /// operand layout, `operand_layout.operand_layout`).
    pub fn new(descriptor: &Value, layout: &Value) -> Result<Self, Error> {
        replay::validate(descriptor)?;
        if descriptor["scalars"]
            .as_array()
            .is_some_and(|s| !s.is_empty())
        {
            return Err("fused reduce: runtime scalars are not covered".into());
        }
        let ops = descriptor["ops"].as_array().ok_or("operations missing")?;
        if let Some(op) = ops.iter().find(|op| {
            matches!(
                op["op"].as_str(),
                Some("masked" | "indirect" | "reduction_component")
            )
        }) {
            return Err(format!("fused reduce: {} is not covered", op["op"]).into());
        }
        let reductions: Vec<_> = ops.iter().filter(|op| op["op"] == "reduction").collect();
        let [reduction] = reductions[..] else {
            return Err("fused reduce: exactly one reduction".into());
        };
        if reduction["reduction_type"] != "sum" {
            return Err("fused reduce: a sum".into());
        }
        let id = |op: &Value| op["id"].as_str().map(String::from);
        let by_id: HashMap<String, &Value> =
            ops.iter().filter_map(|op| Some((id(op)?, op))).collect();
        let operand = reduction["args"][0].as_str().ok_or("reduction operand")?;
        let mut closure = HashSet::new();
        let mut pending = vec![operand.to_string()];
        while let Some(name) = pending.pop() {
            if let Some(op) = by_id.get(&name)
                && closure.insert(name)
            {
                for a in op["args"].as_array().into_iter().flatten() {
                    pending.extend(a.as_str().map(String::from));
                }
            }
        }
        let reduction_id = id(reduction).ok_or("reduction id")?;
        let args = |op: &Value| -> Vec<String> {
            op["args"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|a| a.as_str().map(String::from))
                .collect()
        };
        // Ops of the operand that an op after the reduction reads too (Inductor
        // emits a shared constant or load once): replayed in the write as
        // well, unless they vary along the reduction.
        let mut shared = HashSet::new();
        let mut pending: Vec<String> = ops
            .iter()
            .filter(|op| id(op).is_none_or(|i| !closure.contains(&i) && i != reduction_id))
            .flat_map(args)
            .filter(|a| closure.contains(a))
            .collect();
        while let Some(name) = pending.pop() {
            if shared.insert(name.clone()) {
                pending.extend(args(by_id[&name]));
            }
        }
        if let Some(op) = shared.iter().map(|i| by_id[i]).find(|op| {
            op.get("index")
                .is_some_and(|i| i.to_string().contains(r#""op":"reduction""#))
        }) {
            return Err(format!(
                "fused reduce: an op after the reduction reads {}, which varies along it",
                op["id"]
            )
            .into());
        }
        let (mut prologue, mut epilogue) = (Vec::new(), Vec::new());
        for op in ops {
            match id(op) {
                Some(i) if closure.contains(&i) => {
                    if shared.contains(&i) {
                        epilogue.push(op.clone());
                    }
                    prologue.push(op.clone());
                }
                Some(i) if i == reduction_id => {}
                _ => epilogue.push(op.clone()),
            }
        }
        let dims = |part: &str, key: &str| -> Result<Vec<usize>, Error> {
            layout[part][key]
                .as_array()
                .ok_or("layout missing")?
                .iter()
                .map(|v| v.as_u64().map(|v| v as usize).ok_or("layout entry".into()))
                .collect()
        };
        let outer = dims("outer", "dims")?;
        let outer_strides = dims("outer", "strides")?;
        let (reduction_dims, reduction_strides) =
            (dims("reduction", "dims")?, dims("reduction", "strides")?);
        let ([extent], [step]) = (&reduction_dims[..], &reduction_strides[..]) else {
            return Err("fused reduce: one reduction dimension".into());
        };
        let input_shape: Vec<usize> = outer.iter().copied().chain([*extent]).collect();
        let input_strides: Vec<usize> = outer_strides.iter().copied().chain([*step]).collect();
        // Decoding an offset needs dimensions that don't overlap in memory.
        let mut order: Vec<usize> = (0..input_shape.len()).collect();
        order.sort_by_key(|&a| input_strides[a]);
        let mut span = 1;
        for &a in &order {
            if input_shape[a] > 1 && input_strides[a] < span {
                return Err("fused reduce: the layout's dimensions overlap".into());
            }
            span = span.max(input_strides[a] * input_shape[a]);
        }
        let mut output_strides = vec![0; input_shape.len()];
        let mut size = 1;
        for &a in order.iter().filter(|&&a| a != input_shape.len() - 1) {
            output_strides[a] = size;
            size *= input_shape[a];
        }
        *output_strides.last_mut().unwrap() = size;
        Ok(Self {
            prologue: Value::Array(replay::bind_reduction(&prologue, 0)).to_string(),
            epilogue: Value::Array(replay::bind_reduction(&epilogue, 0)).to_string(),
            operand: operand.into(),
            reduction: reduction_id,
            outer,
            input_shape,
            input_strides,
            output_strides,
        })
    }

    pub fn input_layout(&self) -> (Vec<usize>, Vec<usize>) {
        (self.input_shape.clone(), self.input_strides.clone())
    }

    pub fn output_layout(&self) -> (Vec<usize>, Vec<usize>) {
        let mut shape = self.outer.clone();
        shape.push(1);
        (shape, self.output_strides.clone())
    }
}

fn int64(s: &Scope, v: usize) -> IrValue {
    replay::constant(s, v as i64, "int64").expect("an int64 constant")
}

fn to_int64(s: &Scope, v: IrValue) -> IrValue {
    let ty = replay::elem("int64").expect("int64").to_type(s.ctx());
    s.register_with_result(&CastOp::new(s.ctx_mut(), ty, v))
}

/// The coordinates of scalar `offset` (int64) in a layout of `shape` and
/// `strides` whose dimensions don't overlap: outermost stride first.
fn decode(s: &Scope, offset: IrValue, shape: &[usize], strides: &[usize]) -> Vec<IrValue> {
    let mut order: Vec<usize> = (0..shape.len()).collect();
    order.sort_by_key(|&a| std::cmp::Reverse(strides[a]));
    let mut coords = vec![int64(s, 0); shape.len()];
    let mut rest = offset;
    for a in order {
        if shape[a] == 1 {
            continue;
        }
        let stride = int64(s, strides[a]);
        coords[a] = s.register_with_result(&SDivOp::new(s.ctx_mut(), rest, stride));
        rest = s.register_with_result(&SRemOp::new(s.ctx_mut(), rest, stride));
    }
    coords
}

/// The row-major linear index of `coords` over `dims` (Inductor's `xindex`).
fn linear(s: &Scope, coords: &[IrValue], dims: &[usize]) -> IrValue {
    let mut index = int64(s, 0);
    for (c, d) in coords.iter().zip(dims) {
        let scaled = s.register_with_result(&IMulOp::new(s.ctx_mut(), index, int64(s, *d)));
        index = s.register_with_result(&IAddOp::new(s.ctx_mut(), scaled, *c));
    }
    index
}

/// `values[axis]`, `axis` an index expansion: a constant when it is one, else
/// a select chain.
fn pick(s: &Scope, axis: NativeExpand<usize>, values: &[usize]) -> NativeExpand<usize> {
    if let Some(c) = axis.constant() {
        let a = usize::from_const_value(c);
        return values[a].into_expand(s);
    }
    let axis = axis.read_value(s);
    let index = |v: usize| -> IrValue { v.into_expand(s).read_value(s) };
    let mut out = index(*values.last().unwrap());
    for (a, v) in values.iter().enumerate().rev().skip(1) {
        let hit = s.register_with_result(&IEqualOp::new(s.ctx_mut(), axis, index(a)));
        out = s.register_with_result(&SelectOp::new(s.ctx_mut(), hit, index(*v), out));
    }
    out.into()
}

fn span(shape: &[usize], strides: &[usize]) -> usize {
    1 + shape
        .iter()
        .zip(strides)
        .map(|(d, s)| d.saturating_sub(1) * s)
        .sum::<usize>()
}

/// The parsed body and buffers one expansion shares.
struct Expansion {
    body: FusedBody,
    prologue: Vec<Value>,
    epilogue: Vec<Value>,
    arrays: Vec<IrValue>,
}

/// CubeK's input: `read` replays the operand per lane.
pub struct RecordedInput;
#[derive(Clone)]
pub struct RecordedInputExpand {
    x: Rc<Expansion>,
    vector_size: usize,
}

/// CubeK's output: `write` replays the per-output ops and stores per lane.
pub struct RecordedOutput;
#[derive(Clone)]
pub struct RecordedOutputExpand {
    x: Rc<Expansion>,
    vector_size: usize,
}

plain_expand!(RecordedInputExpand);
plain_expand!(RecordedOutputExpand);

impl CubeType for RecordedInput {
    type ExpandType = RecordedInputExpand;
}
impl CubeType for RecordedOutput {
    type ExpandType = RecordedOutputExpand;
}
impl Vectorized for RecordedInput {}
impl Vectorized for RecordedOutput {}
impl VectorizedExpand for RecordedInputExpand {
    fn __expand_vector_size_method(&self, _: &Scope) -> VectorSize {
        self.vector_size
    }
}
impl VectorizedExpand for RecordedOutputExpand {
    fn __expand_vector_size_method(&self, _: &Scope) -> VectorSize {
        self.vector_size
    }
}
impl<E: Numeric, N: Size> VirtualTensorOperations<E, N> for RecordedInput {}
impl<E: Numeric, N: Size> VirtualTensorOperations<E, N> for RecordedOutput {}

/// Builds one value per lane from `lane` and gathers them into a vector of
/// `elem` (the value itself for one lane).
fn gather(s: &Scope, lanes: usize, elem: ElemType, lane: impl Fn(usize) -> IrValue) -> IrValue {
    let values: Vec<IrValue> = (0..lanes).map(lane).collect();
    if lanes == 1 {
        return values[0];
    }
    let ty = VectorType::get(s.ctx(), elem.to_type(s.ctx()), lanes).to_handle();
    s.register_with_result(&CompositeConstructOp::new(s.ctx_mut(), ty, values))
}

impl<E: Numeric, N: Size> VirtualTensorOperationsExpand<E, N> for RecordedInputExpand {
    fn __expand_read_method(
        &self,
        scope: &Scope,
        index: NativeExpand<usize>,
    ) -> NativeExpand<Vector<E, N>> {
        let x = &self.x;
        let base = to_int64(scope, index.read_value(scope));
        let base = scope.register_with_result(&IMulOp::new(
            scope.ctx_mut(),
            base,
            int64(scope, self.vector_size),
        ));
        let elem = E::elem_type(scope);
        gather(scope, self.vector_size, elem, |k| {
            let offset =
                scope.register_with_result(&IAddOp::new(scope.ctx_mut(), base, int64(scope, k)));
            let coords = decode(scope, offset, &x.body.input_shape, &x.body.input_strides);
            let rank = x.body.outer.len();
            let output = linear(scope, &coords[..rank], &x.body.outer);
            let mut values = HashMap::new();
            replay::emit(
                scope,
                &x.prologue,
                &x.arrays,
                output,
                &coords[rank..],
                &mut values,
            )
            .expect("the operand's ops replay");
            values[&x.body.operand]
        })
        .into()
    }

    fn __expand_read_window_method(
        &self,
        _: &Scope,
        _: NativeExpand<usize>,
        _: NativeExpand<usize>,
    ) -> &'static SliceExpand<Vector<E, N>> {
        unreachable!("the fused input is read element by element")
    }

    fn __expand_write_method(
        &mut self,
        _: &Scope,
        _: NativeExpand<usize>,
        _: NativeExpand<Vector<E, N>>,
    ) {
        unreachable!("the fused input is never written")
    }

    fn __expand_shape_method(&self, s: &Scope, axis: NativeExpand<usize>) -> NativeExpand<usize> {
        pick(s, axis, &self.x.body.input_shape)
    }

    fn __expand_stride_method(&self, s: &Scope, axis: NativeExpand<usize>) -> NativeExpand<usize> {
        pick(s, axis, &self.x.body.input_strides)
    }

    fn __expand_rank_method(&self, s: &Scope) -> NativeExpand<usize> {
        self.x.body.input_shape.len().into_expand(s)
    }

    fn __expand_len_method(&self, s: &Scope) -> NativeExpand<usize> {
        let b = &self.x.body;
        span(&b.input_shape, &b.input_strides)
            .div_ceil(self.vector_size)
            .into_expand(s)
    }

    fn __expand_buffer_len_method(&self, s: &Scope) -> NativeExpand<usize> {
        <Self as VirtualTensorOperationsExpand<E, N>>::__expand_len_method(self, s)
    }

    fn __expand_as_tensor_map_method(
        &self,
        s: &Scope,
    ) -> ComptimeOptionExpand<TensorMap<E, Tiled>> {
        ComptimeOption::__expand_new_None(s)
    }
}

impl<E: Numeric, N: Size> VirtualTensorOperationsExpand<E, N> for RecordedOutputExpand {
    fn __expand_read_method(
        &self,
        _: &Scope,
        _: NativeExpand<usize>,
    ) -> NativeExpand<Vector<E, N>> {
        unreachable!("a sum never reads its output")
    }

    fn __expand_read_window_method(
        &self,
        _: &Scope,
        _: NativeExpand<usize>,
        _: NativeExpand<usize>,
    ) -> &'static SliceExpand<Vector<E, N>> {
        unreachable!("the fused output is written element by element")
    }

    fn __expand_write_method(
        &mut self,
        scope: &Scope,
        index: NativeExpand<usize>,
        value: NativeExpand<Vector<E, N>>,
    ) {
        let x = &self.x;
        let (shape, strides) = x.body.output_layout();
        let value = value.read_value(scope);
        let base = to_int64(scope, index.read_value(scope));
        let base = scope.register_with_result(&IMulOp::new(
            scope.ctx_mut(),
            base,
            int64(scope, self.vector_size),
        ));
        let zero = int64(scope, 0);
        for k in 0..self.vector_size {
            let offset =
                scope.register_with_result(&IAddOp::new(scope.ctx_mut(), base, int64(scope, k)));
            let coords = decode(scope, offset, &shape, &strides);
            let rank = x.body.outer.len();
            let output = linear(scope, &coords[..rank], &x.body.outer);
            let reduced = if self.vector_size == 1 {
                value
            } else {
                scope.register_with_result(&CompositeExtractOp::new(scope.ctx_mut(), value, k))
            };
            let mut values = HashMap::from([(x.body.reduction.clone(), reduced)]);
            replay::emit(scope, &x.epilogue, &x.arrays, output, &[zero], &mut values)
                .expect("the per-output ops replay");
        }
    }

    fn __expand_shape_method(&self, s: &Scope, axis: NativeExpand<usize>) -> NativeExpand<usize> {
        pick(s, axis, &self.x.body.output_layout().0)
    }

    fn __expand_stride_method(&self, s: &Scope, axis: NativeExpand<usize>) -> NativeExpand<usize> {
        pick(s, axis, &self.x.body.output_strides)
    }

    fn __expand_rank_method(&self, s: &Scope) -> NativeExpand<usize> {
        self.x.body.input_shape.len().into_expand(s)
    }

    fn __expand_len_method(&self, s: &Scope) -> NativeExpand<usize> {
        let (shape, strides) = self.x.body.output_layout();
        span(&shape, &strides)
            .div_ceil(self.vector_size)
            .into_expand(s)
    }

    fn __expand_buffer_len_method(&self, s: &Scope) -> NativeExpand<usize> {
        <Self as VirtualTensorOperationsExpand<E, N>>::__expand_len_method(self, s)
    }

    fn __expand_as_tensor_map_method(
        &self,
        s: &Scope,
    ) -> ComptimeOptionExpand<TensorMap<E, Tiled>> {
        ComptimeOption::__expand_new_None(s)
    }
}

#[cube(launch_unchecked, address_type = "dynamic")]
#[allow(clippy::too_many_arguments)]
#[allow(
    clippy::extra_unused_type_parameters,
    reason = "used in the intrinsic, which the unexpanded function never runs"
)]
pub fn fused_reduce_kernel<In: Numeric, SizeIn: Size, Out: Numeric, SizeOut: Size, Acc: Numeric>(
    buffers: &Sequence<RecordedBuffer>,
    reduce_axis: usize,
    out_vec_axis: usize,
    #[comptime] body: FusedBody,
    #[comptime] blueprint: ReduceBlueprint,
    #[define(In)] _input_dtype: ElemType,
    #[define(Out)] _output_dtype: ElemType,
    #[define(Acc)] _acc_dtype: ElemType,
) {
    intrinsic!(|scope| {
        let x = Rc::new(Expansion {
            prologue: serde_json::from_str::<Vec<Value>>(&body.prologue).unwrap(),
            epilogue: serde_json::from_str::<Vec<Value>>(&body.epilogue).unwrap(),
            arrays: buffers.iter_cloned().map(|b| b.value).collect(),
            body,
        });
        let input = VirtualTensor::<In, SizeIn>::__expand_new::<RecordedInput>(
            scope,
            RecordedInputExpand {
                x: x.clone(),
                vector_size: SizeIn::__expand_value(scope),
            },
        );
        let mut output = VirtualTensor::<Out, SizeOut, ReadWrite>::__expand_new::<RecordedOutput>(
            scope,
            RecordedOutputExpand {
                x,
                vector_size: SizeOut::__expand_value(scope),
            },
        );
        reduce_kernel_virtual::expand::<In, SizeIn, Out, SizeOut, Acc>(
            scope,
            &input,
            &mut output,
            reduce_axis,
            out_vec_axis,
            blueprint,
            ReduceOperationConfig::Sum,
        )
    })
}

/// The dtypes CubeK reduces with: the operand's (after the body's casts), the
/// reduction's result and its accumulator.
pub fn dtypes(descriptor: &Value, body: &FusedBody) -> Result<ReduceDtypes, Error> {
    let ops = descriptor["ops"].as_array().ok_or("operations missing")?;
    let find = |id: &str| {
        ops.iter()
            .find(|op| op["id"] == id)
            .ok_or_else(|| Error::from(format!("op {id} missing")))
    };
    let operand = find(&body.operand)?;
    let reduction = find(&body.reduction)?;
    let elem = |v: &Value| replay::elem(v.as_str().ok_or("dtype missing")?);
    Ok(ReduceDtypes {
        input: elem(&operand["dtype"])?,
        output: elem(&reduction["dtype"])?,
        accumulation: elem(&reduction["accumulator_dtype"])?,
    })
}

/// Whether `CUBECL_TORCH_FUSED_TRACE` asks for each launch's inferred plan on
/// stderr (read once).
fn trace() -> bool {
    static TRACE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *TRACE.get_or_init(|| std::env::var_os("CUBECL_TORCH_FUSED_TRACE").is_some())
}

/// Plans and launches the fused reduce of `body` over `buffers` (each with its
/// element type and element count), with CubeK's routine `routine`.
pub fn launch(
    client: &Client,
    body: FusedBody,
    dtypes: ReduceDtypes,
    buffers: Vec<(BufferArg, ElemType, usize)>,
    routine: cubek_reduce::launch::RoutineStrategy,
) -> Result<(), String> {
    let (in_shape, in_strides) = body.input_layout();
    let (out_shape, out_strides) = body.output_layout();
    let axis = in_shape.len() - 1;
    let largest = buffers.iter().map(|(_, _, len)| *len).max().unwrap_or(0);
    let address_type = if largest < i32::MAX as usize {
        AddressType::U32
    } else {
        AddressType::U64
    };
    let described = trace().then(|| format!("{routine:?}"));
    let strategy = ReduceStrategy {
        routine,
        vectorization: VectorizationStrategy {
            parallel_output_vectorization: false,
        },
        // Each launch is planned from its own static shape (coverage refuses
        // runtime scalars) and the blueprint is part of the kernel's identity,
        // so every raw shape is its own key: CubeK's level for a caller outside
        // autotune. It lets CubeK drop bound checks a shape doesn't need, and
        // with them the branch that kept the read's last multiply from
        // contracting into the accumulation (fused-reduction-bridge 5.1).
        autotune_level: cubecl::config::autotune::AutotuneLevel::Full,
    };
    let (blueprint, settings, out_vec_axis) = prepare_reduce_launch_for_layout(
        client,
        (&in_shape.clone().into(), &in_strides.clone().into()),
        (&out_shape.into(), &out_strides.into()),
        axis,
        strategy,
        dtypes,
        ReduceOperationConfig::Sum,
        address_type,
        None,
        true,
    )
    .map_err(|e| format!("{e:?}"))?;
    if let Some(routine) = described {
        // A diagnostic: what CubeK inferred (fused-reduction-bridge task 5.2).
        eprintln!(
            "cubecl_torch fused_reduce: input {in_shape:?}/{in_strides:?} routine \
             {routine} blueprint {blueprint:?} cube_count {:?} cube_dim {:?} \
             vector {}/{}",
            settings.cube_count,
            settings.cube_dim,
            settings.vector.vector_size_input,
            settings.vector.vector_size_output,
        );
    }
    let mut args = SequenceArg::<RecordedBuffer>::new();
    for (buffer, elem, _) in buffers {
        args.push((buffer, elem));
    }
    unsafe {
        fused_reduce_kernel::launch_unchecked(
            client,
            settings.cube_count,
            settings.cube_dim,
            settings.address_type,
            settings.vector.vector_size_input,
            settings.vector.vector_size_output,
            args,
            axis,
            out_vec_axis,
            body,
            blueprint,
            dtypes.input,
            dtypes.output,
            dtypes.accumulation,
        )
    };
    Ok(())
}
