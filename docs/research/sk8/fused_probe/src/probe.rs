//! An Inductor reduction kernel whose fused loop body runs inside cubecl-tile loops (bead sk8). The body is Inductor's own op list (a schema-3
//! `semantic.json` with one sum), emitted by `semantic::emit` from inside two tile recipes: the
//! value the reduction reads, and the value each output element stores. Only for measurement;
//! nothing in the compiler route uses it.
//!
use crate::semantic;
use cubecl::ir::types::scalar::IndexType;
use cubecl::{
    frontend::{AsMutExpand, AsRefExpand, CubeDebug, ExpandTypeClone, IntoExpand, IntoMut},
    ir::{
        AddressSpace, Scope,
        dialect::{general::CastOp, memory::LoadOp},
        interfaces::SizedType,
        pliron::{op::Op, value::Value as IrValue},
        types::PointerType,
    },
    prelude::*,
    std::tensor::TensorHandle,
    tile::{
        launch::Grid,
        procedural::{Procedural, Reads, Recipe, RecipeCoords, RecipeExpand},
        *,
    },
};
use cubecl_runtime::compiler::Compiler;
use serde_json::{Value, json};
use std::{
    collections::HashMap,
    sync::OnceLock,
    time::{Duration, Instant},
};

type Error = Box<dyn std::error::Error>;

const M: Axis = Axis(0);
const K: Axis = Axis(1);
const N: Axis = Axis(2);
const REGISTER_BLOCK: RegisterBlock = RegisterBlock::new(16);

/// The kernel's op list, its coordinate bound to scalar slot 0 as the reduction emitter binds it.
static OPS: OnceLock<Vec<Value>> = OnceLock::new();
/// `reduction::analysis::bind_coordinate`, copied: the reduction coordinate as scalar slot `slot`.
fn bind_coordinate(v: &Value, slot: usize) -> Value {
    if v["op"] == "reduction" && v.get("dtype").is_none() {
        return json!({"op": "scalar", "slot": slot});
    }
    match v {
        Value::Array(a) => Value::Array(a.iter().map(|x| bind_coordinate(x, slot)).collect()),
        Value::Object(o) => Value::Object(
            o.iter()
                .map(|(k, v)| (k.clone(), bind_coordinate(v, slot)))
                .collect(),
        ),
        _ => v.clone(),
    }
}

/// `reduction::analysis::closure`, reduced to plain ops (no mask regions): the ops `ids` need,
/// in order, minus reductions and what is already in `ready`.
fn closure(ids: &[&str], ops: &[Value], ready: &HashMap<String, IrValue>) -> Vec<Value> {
    fn visit(id: &str, ops: &[Value], ready: &HashMap<String, IrValue>, needed: &mut Vec<String>) {
        if ready.contains_key(id) || needed.iter().any(|n| n == id) {
            return;
        }
        needed.push(id.into());
        if let Some(op) = ops.iter().find(|op| op["id"] == id) {
            for a in op["args"].as_array().unwrap() {
                visit(a.as_str().unwrap(), ops, ready, needed);
            }
        }
    }
    let mut needed = Vec::new();
    for id in ids {
        visit(id, ops, ready, &mut needed);
    }
    ops.iter()
        .filter(|op| op["op"] != "reduction")
        .filter(|op| {
            op["id"]
                .as_str()
                .is_some_and(|id| needed.iter().any(|n| n == id))
        })
        .cloned()
        .collect()
}

/// Time spent inside `semantic::emit` while the kernel expands.
static EMIT: std::sync::Mutex<Duration> = std::sync::Mutex::new(Duration::ZERO);
/// The row length, when the summing recipe should validate its row index (`validate` argument):
/// CubeCL's `validate_index` prints every index past `x`'s logical length, the length CubeCL's
/// metadata holds for the binding, not the pooled block it sits in.
static VALIDATE_COLS: OnceLock<usize> = OnceLock::new();

/// The reduction's operand id, the reduction's id, and the stored value's id.
fn roles(ops: &[Value]) -> (String, String, String) {
    let red = ops.iter().find(|o| o["op"] == "reduction").unwrap();
    let store = ops.iter().find(|o| o["op"] == "store").unwrap();
    let s = |v: &Value| v.as_str().unwrap().to_string();
    (s(&red["args"][0]), s(&red["id"]), s(&store["args"][0]))
}

/// A recipe evaluating Inductor's ops at a tile coordinate: before the reduction (the value it
/// sums) or after it (the value stored, reading the row's sum back from `sums`).
#[derive(Clone)]
pub struct Body;

#[derive(Clone)]
pub struct BodyExpand {
    /// The kernel's buffers, in the descriptor's order, then the row sums.
    arrays: Vec<IrValue>,
    /// `x`'s logical length, as the slice over the whole binding holds it.
    x_len: IrValue,
    post: bool,
}

impl CubeType for Body {
    type ExpandType = BodyExpand;
}

impl IntoExpand for BodyExpand {
    type Expand = Self;
    fn into_expand(self, _: &Scope) -> Self {
        self
    }
}
impl ExpandTypeClone for BodyExpand {
    fn clone_unchecked(&self) -> Self {
        self.clone()
    }
}
impl IntoMut for BodyExpand {
    fn into_mut(self, _: &Scope) -> Self {
        self
    }
}
impl CubeDebug for BodyExpand {}
impl AsRefExpand for BodyExpand {
    fn __expand_ref_method(&self, _: &Scope) -> &Self {
        self
    }
}
impl AsMutExpand for BodyExpand {
    fn __expand_ref_mut_method(&mut self, _: &Scope) -> &mut Self {
        self
    }
}

impl Recipe<f32> for Body {
    fn evaluate(&self, _: &RecipeCoords) -> f32 {
        cubecl::unexpanded!()
    }
}

impl Reads for BodyExpand {
    fn reads(&self, _: &Scope, _: Axis) -> bool {
        true
    }
}

/// Whether an index expression moves with the row: the row data, not a per-column weight.
fn reads_row(v: &Value) -> bool {
    match v {
        Value::Object(o) => {
            o.get("op").is_some_and(|op| op == "linear") || o.values().any(reads_row)
        }
        Value::Array(a) => a.iter().any(reads_row),
        _ => false,
    }
}

fn int64(s: &Scope, v: IrValue) -> IrValue {
    let ty = crate::ir::elem("int64").unwrap().to_type(s.ctx());
    s.register_with_result(&CastOp::new(s.ctx_mut(), ty, v))
}

impl RecipeExpand<f32> for BodyExpand {
    fn __expand_evaluate_method(
        &self,
        scope: &Scope,
        coordinates: &<RecipeCoords as CubeType>::ExpandType,
    ) -> NativeExpand<f32> {
        let start = Instant::now();
        let row = coordinates.__expand_along_method(scope, M).value(scope);
        let col = coordinates.__expand_along_method(scope, K).value(scope);
        let (linear, slot) = (int64(scope, row), int64(scope, col));
        if let (Some(&cols), false) = (VALIDATE_COLS.get(), self.post) {
            let as_usize = |v: IrValue| {
                let ty = IndexType::get(scope.ctx()).to_handle();
                NativeExpand::<usize>::from(scope.register_with_result(&CastOp::new(
                    scope.ctx_mut(),
                    ty,
                    v,
                )))
            };
            let cols = NativeExpand::<usize>::from(
                crate::ir::constant(scope, &json!(cols), "int64")
                    .map(|c| {
                        let ty = IndexType::get(scope.ctx()).to_handle();
                        scope.register_with_result(&CastOp::new(scope.ctx_mut(), ty, c))
                    })
                    .unwrap(),
            );
            let flat = as_usize(row)
                .__expand_mul_method(scope, cols)
                .__expand_add_method(scope, as_usize(col));
            let len = NativeExpand::<usize>::from(self.x_len);
            cubecl::io::validate_index::expand(scope, "x", flat, len, "fused_row");
        }
        let ops = OPS.get().unwrap();
        let (operand, reduced, stored) = roles(ops);
        let mut values = HashMap::new();
        let target = if self.post {
            // The row's sum, read back from the sums buffer (the last array).
            let sum = json!({"op": "load", "id": reduced, "args": [], "dtype": "float32",
                "buffer": self.arrays.len() - 1, "index": {"op": "linear"}});
            semantic::emit(scope, &[sum], &self.arrays, linear, &[slot], &mut values).unwrap();
            stored
        } else {
            operand
        };
        let mut body = closure(&[&target], ops, &values);
        // Triton's hints: the summing pass keeps the row in L1, the storing pass streams it.
        for op in body.iter_mut().filter(|o| o["op"] == "load") {
            if reads_row(&op["index"]) {
                op[if self.post { "streaming" } else { "retained" }] = true.into();
            }
        }
        semantic::emit(scope, &body, &self.arrays, linear, &[slot], &mut values).unwrap();
        *EMIT.lock().unwrap() += start.elapsed();
        NativeExpand::from(values[&target])
    }
}

/// The recipes, holding the kernel's raw buffers: the value summed, and the value stored.
#[allow(dead_code)]
fn summed(_x: &[f32], _w: &[f32], _sums: &[f32]) -> Body {
    cubecl::unexpanded!()
}
#[allow(dead_code)]
fn stored(_x: &[f32], _w: &[f32], _sums: &[f32]) -> Body {
    cubecl::unexpanded!()
}
fn recipe(scope: &Scope, buffers: [IrValue; 3], x_len: IrValue, post: bool) -> BodyExpand {
    let [x, w, sums] = buffers;
    // The descriptor's buffers are x, w and the output; the output is never read here.
    BodyExpand {
        arrays: vec![x, w, x, sums],
        x_len,
        post,
    }
    .into_expand(scope)
}
mod summed {
    use super::*;
    pub fn expand(
        scope: &Scope,
        x: &NativeExpand<[f32]>,
        w: &NativeExpand<[f32]>,
        s: &NativeExpand<[f32]>,
    ) -> BodyExpand {
        let x_len = x.__extract_length(scope).value(scope);
        recipe(
            scope,
            [x, w, s].map(|b| b.__extract_list(scope)),
            x_len,
            false,
        )
    }
}
mod stored {
    use super::*;
    pub fn expand(
        scope: &Scope,
        x: &NativeExpand<[f32]>,
        w: &NativeExpand<[f32]>,
        s: &NativeExpand<[f32]>,
    ) -> BodyExpand {
        let x_len = x.__extract_length(scope).value(scope);
        recipe(
            scope,
            [x, w, s].map(|b| b.__extract_list(scope)),
            x_len,
            true,
        )
    }
}

/// Mark the global loads emitted so far, not yet hinted, as read again later (the reduction's
/// shared-memory and sum reads are not global, or are hinted by the body itself).
#[allow(dead_code)]
fn retain_rest() {}
mod retain_rest {
    use super::*;
    use cubecl::ir::pliron::{
        builtin::op_interfaces::OneResultInterface,
        graph::walkers::{
            IRNode, WALKCONFIG_PREORDER_FORWARD,
            interruptible::{WalkResult, immutable::walk_op, walk_advance},
        },
        operation::Operation,
        r#type::{Typed, type_cast},
    };
    fn visit(
        ctx: &cubecl::ir::pliron::context::Context,
        n: &mut usize,
        node: IRNode,
    ) -> WalkResult<()> {
        if let IRNode::Operation(op) = node
            && let Some(load) = Operation::get_op::<LoadOp>(op, ctx)
        {
            let ptr = load.ptr(ctx).get_type(ctx);
            let global = ptr
                .deref(ctx)
                .downcast_ref::<PointerType>()
                .is_some_and(|p| matches!(p.address_space, AddressSpace::Global(_)));
            let ty = load.get_result(ctx).get_type(ctx);
            let sized = type_cast::<dyn SizedType>(&*ty.deref(ctx)).is_some();
            if global && sized && load.is_retained(ctx) {
                *n += 1;
            }
        }
        walk_advance()
    }
    pub fn expand(scope: &Scope) {
        let root = scope.state().entry_func.get_operation();
        let mut n = 0;
        let _ = walk_op(
            scope.ctx(),
            &mut n,
            &WALKCONFIG_PREORDER_FORWARD,
            root,
            visit,
        );
        eprintln!("{n} retained row loads in the summing pass");
    }
}

/// One cube per row, 16 planes × 32 units splitting it (any row length: the planes take the steps
/// in runs, a partial last step masked); the reduction reads the `pre` recipe and the output is
/// copied from the `post` recipe.
#[cube(launch, create_dummy_kernel)]
fn fused_row<E: Float>(
    s: &TileArg<'_>,
    out: &TileArg<'_>,
    x: &Tensor<f32>,
    w: &Tensor<f32>,
    sums: &mut Tensor<f32>,
    space: Partitioning,
    #[comptime] planes: usize,
    #[define(E)] _dtype: ElemType,
) {
    let s = s.tile(&space);
    let out = out.tile(&space);
    let pre = Procedural::new::<f32, Body>(
        comptime!(space.space().clone()),
        summed(x.as_slice(), w.as_slice(), sums.as_slice()),
    )
    .tile::<f32>();
    let post = Procedural::new::<f32, Body>(
        comptime!(space.space().clone()),
        stored(x.as_slice(), w.as_slice(), sums.as_slice()),
    )
    .tile::<f32>();
    for cube in &space {
        let mut s_cube = s.at(&cube);
        // CubeK's `InSlots` shape: the planes take the row's steps in runs (`planes_across`), each
        // into a cube-level accumulator, and meet in a slot each.
        let slots = s_cube.smem_slots_accumulation::<E>(planes);
        let pre_cube = pre.at(&cube);
        let partial = s_cube.block_reducer::<E>(&pre_cube, REGISTER_BLOCK, Monoid::Sum);
        let steps = cube.walk();
        let writer = steps.position(K);
        for step in steps {
            for unit in &step {
                let mut acc = partial.at(&unit);
                acc.reduce_axis_accumulate(&pre.at(&unit), Monoid::Sum);
            }
        }
        slots.drain(&partial, writer);
        retain_rest();
        s_cube.copy_from(&slots.source);
        sync_cube();
        let mut out_cube = out.at(&cube);
        out_cube.copy_from(&post.at(&cube));
    }
}

/// Deterministic normal samples (Box–Muller over a 64-bit LCG).
fn normals(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    let mut uniform = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    (0..n)
        .map(|_| {
            let (u, v) = (uniform(), uniform());
            ((-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()) as f32
        })
        .collect()
}

pub fn run(args: &[String]) -> Result<(), Error> {
    let semantic: Value = serde_json::from_str(&std::fs::read_to_string(&args[2])?)?;
    let rows: usize = args[3].parse()?;
    let cols: usize = args[4].parse()?;
    let reps: usize = args.get(5).map_or(Ok(3), |r| r.parse())?;
    let width: usize = args.get(6).map_or(Ok(1), |r| r.parse())?;
    if args.get(7).is_some_and(|a| a == "validate") {
        VALIDATE_COLS.set(cols).map_err(|_| "validate set twice")?;
    }
    let ops: Vec<Value> = semantic["ops"]
        .as_array()
        .ok_or("no ops")?
        .iter()
        .map(|op| bind_coordinate(op, 0))
        .collect();
    OPS.set(ops).map_err(|_| "ops set twice")?;
    let (planes, units) = (16usize, 32usize);

    let client = cubecl::Device::Cuda(Default::default()).client();
    let dtype = f32::elem_type_native();
    let x_host = normals(rows * cols, 1);
    let w_host = normals(cols, 2);
    let upload = |v: &[f32], shape: Vec<usize>| {
        TensorHandle::new_contiguous(shape, client.create_from_slice(f32::as_bytes(v)), dtype)
    };
    let x = upload(&x_host, vec![rows, 1, cols]);
    let w = upload(&w_host, vec![cols]);
    let s = TensorHandle::new_contiguous([rows, 1], client.empty(rows * 4), dtype);
    let out = TensorHandle::new_contiguous([rows, 1, cols], client.empty(rows * cols * 4), dtype);

    let partitioning = Partitioning::new(
        Space::new(&[(M, rows), (N, 1), (K, cols)]),
        Levels::leaf(&[(M, 1), (N, 1), (K, width)])
            .units(&[(K, units)])
            .planes_across(K, planes)
            .cubes(&[M, N])
            .build(),
    );
    let concrete = partitioning.space().clone();
    let launcher = Launcher::new(&client, partitioning, &concrete, Grid::FromLevels)?;
    let arg = |h: &TensorHandle, axes: &[Axis], width: usize| {
        launcher
            .arg(h.clone().binding(), h.dtype)
            .axes(axes)
            .vectorize(width)
            .build()
            .unwrap()
    };
    let raw = |h: &TensorHandle| h.clone().binding().into_tensor_arg();

    // Expansion and C++ generation alone, as our compiler process would run them.
    let start = Instant::now();
    let kernel = fused_row::create_dummy_kernel(
        client.properties_shared(),
        client.target_properties_shared(),
        launcher.cube_count(),
        launcher.cube_dim(),
        arg(&s, &[M, N], 1).arg(),
        arg(&out, &[M, N, K], 1).arg(),
        raw(&x),
        raw(&w),
        raw(&s),
        launcher.partitioning_arg(),
        planes,
        dtype,
    );
    let define = kernel.define();
    let expand_ms = start.elapsed().as_secs_f64() * 1e3;
    let emit_ms = EMIT.lock().unwrap().as_secs_f64() * 1e3;
    let start = Instant::now();
    let repr = cubecl_cpp::shared::CppCompiler::<cubecl_cpp::target::Cuda>::default()
        .compile(define, &cubecl_cpp::shared::CompilationOptions::default())?;
    let cpp_ms = start.elapsed().as_secs_f64() * 1e3;
    std::fs::create_dir_all(&args[1])?;
    std::fs::write(format!("{}/tile_kernel.cu", args[1]), &repr.source)?;

    let mut first_ms = 0.0;
    for rep in 0..reps {
        let start = Instant::now();
        fused_row::launch(
            &client,
            launcher.cube_count(),
            launcher.cube_dim(),
            arg(&s, &[M, N], 1).arg(),
            arg(&out, &[M, N, K], 1).arg(),
            raw(&x),
            raw(&w),
            raw(&s),
            launcher.partitioning_arg(),
            planes,
            dtype,
        );
        if rep == 0 {
            client.read_one(s.handle.clone())?;
            first_ms = start.elapsed().as_secs_f64() * 1e3;
        }
    }
    let got: Vec<f32> = f32::from_bytes(&client.read_one(out.handle.clone())?).to_vec();
    let mut max_rel = 0f64;
    for r in 0..rows {
        let row = &x_host[r * cols..(r + 1) * cols];
        let ms = row.iter().map(|&v| v as f64 * v as f64).sum::<f64>() / cols as f64;
        let scale = 1.0 / (ms + 1e-6).sqrt();
        for c in 0..cols {
            let want = row[c] as f64 * scale * w_host[c] as f64;
            let err = (got[r * cols + c] as f64 - want).abs() / want.abs().max(1e-30);
            max_rel = max_rel.max(err);
        }
    }
    println!(
        "tile-probe {rows}x{cols} width {width}: expand {expand_ms:.1} ms (emit {emit_ms:.2} ms), \
         C++ {cpp_ms:.1} ms, first launch {first_ms:.1} ms, {} source lines, max rel err {max_rel:.3e}",
        repr.source.lines().count()
    );
    Ok(())
}
