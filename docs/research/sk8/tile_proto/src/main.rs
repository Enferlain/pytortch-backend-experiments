//! Prototype: `out = x * x.sum(-1, keepdim=True)` as a cubecl-tile kernel (bead sk8).
//!
//! Usage: tile_proto <in.bin> <out.bin> <rows> <cols> [reps]
//! Reads float32 row-major `x`, writes float32 `out`.

use cubecl::prelude::*;
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::Grid;
use cubecl::tile::*;

const M: Axis = Axis(0);
const K: Axis = Axis(1);
const N: Axis = Axis(2);

const REGISTER_BLOCK: RegisterBlock = RegisterBlock::new(16);

/// Cache hints the tile DSL cannot state, set on the IR as it is expanded: every 16-byte global
/// load emitted so far and not yet hinted gets the hint (the CubeCL patch's `LoadOp` flags, which
/// the C++ backend emits as `ld.global.L1::evict_last` / `ld.global.nc.L1::evict_first`). Only
/// the row loads are 16-byte global loads in these kernels.
mod hints {
    use cubecl::ir::{
        AddressSpace, Scope, dialect::memory::LoadOp, interfaces::SizedType, types::PointerType,
    };
    use pliron::{
        builtin::op_interfaces::OneResultInterface,
        context::{Context, Ptr},
        graph::walkers::{
            IRNode, WALKCONFIG_PREORDER_FORWARD,
            interruptible::{WalkResult, immutable::walk_op, walk_advance},
        },
        op::Op,
        operation::Operation,
        r#type::{Typed, type_cast},
    };

    fn row_loads(ctx: &Context, loads: &mut Vec<Ptr<Operation>>, node: IRNode) -> WalkResult<()> {
        if let IRNode::Operation(op) = node
            && let Some(load) = Operation::get_op::<LoadOp>(op, ctx)
            && !load.is_retained(ctx)
            && !load.is_streaming(ctx)
        {
            let ty = load.get_result(ctx).get_type(ctx);
            let size = type_cast::<dyn SizedType>(&*ty.deref(ctx)).map(|t| t.size(ctx));
            let ptr_ty = load.ptr(ctx).get_type(ctx);
            let global = ptr_ty
                .deref(ctx)
                .downcast_ref::<PointerType>()
                .is_some_and(|p| matches!(p.address_space, AddressSpace::Global(_)));
            if global && size == Some(16) {
                loads.push(op);
            }
        }
        walk_advance()
    }

    pub fn mark(scope: &Scope, retained: bool) {
        let root = scope.state().entry_func.get_operation();
        let mut loads = Vec::new();
        let _ = walk_op(
            scope.ctx(),
            &mut loads,
            &WALKCONFIG_PREORDER_FORWARD,
            root,
            row_loads,
        );
        let ctx = scope.ctx();
        for op in &loads {
            let load = Operation::get_op::<LoadOp>(*op, ctx).unwrap();
            match retained {
                true => load.set_retained(ctx),
                false => load.set_streaming(ctx),
            }
        }
        eprintln!(
            "hinted {} row loads {}",
            loads.len(),
            if retained { "retained" } else { "streaming" }
        );
    }
}

/// Mark the row loads emitted so far as read again later.
#[allow(dead_code)]
fn retain_loads() {}
mod retain_loads {
    pub fn expand(scope: &cubecl::ir::Scope) {
        super::hints::mark(scope, true)
    }
}

/// Mark the row loads emitted so far as their data's last use.
#[allow(dead_code)]
fn stream_loads() {}
mod stream_loads {
    pub fn expand(scope: &cubecl::ir::Scope) {
        super::hints::mark(scope, false)
    }
}

/// One cube per row. Its planes split the row, each walks its slice in steps the plane's units
/// share; the planes meet in shared memory; then the whole cube scales the row by the sum.
#[cube(launch)]
fn sum_consumer<E: Float>(
    x: &TileArg<'_>,
    s: &TileArg<'_>,
    out: &TileArg<'_>,
    space: Partitioning,
    #[comptime] planes: usize,
    #[comptime] hinted: bool,
    #[define(E)] _dtype: ElemType,
) {
    let x = x.tile(&space);
    let s = s.tile(&space);
    let out = out.tile(&space);
    for cube in &space {
        let mut s_cube = s.at(&cube);
        let sum = s_cube.smem_cyclic_accumulation::<E>(planes);
        for plane in &cube {
            let x_plane = x.at(&plane);
            let partial = s
                .at(&plane)
                .block_reducer::<E>(&x_plane, REGISTER_BLOCK, Monoid::Sum);
            for step in &plane {
                for unit in &step {
                    let mut acc = partial.at(&unit);
                    acc.reduce_axis_accumulate(&x_plane.at(&unit), Monoid::Sum);
                }
            }
            sum.drain_at(&partial, &plane, plane.coord(K));
        }
        if comptime!(hinted) {
            retain_loads();
        }
        sync_cube();
        s_cube.copy_from(&sum.source);
        sync_cube();
        let mut out_cube = out.at(&cube);
        out_cube.product(&x.at(&cube), &s_cube);
        if comptime!(hinted) {
            stream_loads();
        }
    }
}

/// One plane per row, the cube's planes on consecutive rows: the plane's units split the row
/// and the plane's sum drains straight into `s`; then the cube scales its rows.
#[cube(launch)]
fn sum_consumer_row_plane<E: Float>(
    x: &TileArg<'_>,
    s: &TileArg<'_>,
    out: &TileArg<'_>,
    space: Partitioning,
    #[define(E)] _dtype: ElemType,
) {
    let x = x.tile(&space);
    let s = s.tile(&space);
    let out = out.tile(&space);
    for cube in &space {
        for plane in &cube {
            let x_plane = x.at(&plane);
            let partial = s
                .at(&plane)
                .block_reducer::<E>(&x_plane, REGISTER_BLOCK, Monoid::Sum);
            for step in &plane {
                for unit in &step {
                    let mut acc = partial.at(&unit);
                    acc.reduce_axis_accumulate(&x_plane.at(&unit), Monoid::Sum);
                }
            }
            partial.drained_into(&s.at(&plane));
        }
        sync_cube();
        let mut out_cube = out.at(&cube);
        out_cube.product(&x.at(&cube), &s.at(&cube));
    }
}

/// [`sum_consumer`] with a sum per vector lane: the row is read as `[K, V]` with `V` the four
/// lanes of a load, so each unit keeps four running sums, the planes meet lane by lane, and the
/// lanes are added only at the end (Triton keeps a value per register and adds them last).
#[cube(launch)]
fn sum_consumer_lanes<E: Float>(
    x: &TileArg<'_>,
    s4: &TileArg<'_>,
    s: &TileArg<'_>,
    out: &TileArg<'_>,
    space: Partitioning,
    #[comptime] planes: usize,
    #[define(E)] _dtype: ElemType,
) {
    let x = x.tile(&space);
    let s4 = s4.tile(&space);
    let s = s.tile(&space);
    let out = out.tile(&space);
    for cube in &space {
        let mut s4_cube = s4.at(&cube);
        let sum = s4_cube.smem_cyclic_accumulation::<E>(planes);
        for plane in &cube {
            let x_plane = x.at(&plane);
            let partial = s4
                .at(&plane)
                .block_reducer::<E>(&x_plane, REGISTER_BLOCK, Monoid::Sum);
            for step in &plane {
                for unit in &step {
                    let mut acc = partial.at(&unit);
                    acc.reduce_axis_accumulate(&x_plane.at(&unit), Monoid::Sum);
                }
            }
            sum.drain_at(&partial, &plane, plane.coord(K));
        }
        retain_loads();
        sync_cube();
        s4_cube.copy_from(&sum.source);
        sync_cube();
        let mut s_cube = s.at(&cube);
        s_cube.reduce_axis(&sum.source, Monoid::Sum);
        sync_cube();
        let mut out_cube = out.at(&cube);
        out_cube.product(&x.at(&cube), &s_cube);
        stream_loads();
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (inp, outp) = (&args[1], &args[2]);
    let rows: usize = args[3].parse().unwrap();
    let cols: usize = args[4].parse().unwrap();
    let reps: usize = args.get(5).map(|r| r.parse().unwrap()).unwrap_or(1);
    let mode = args.get(6).map(String::as_str).unwrap_or("hinted");
    let row_plane = mode == "row_plane";
    let hinted = mode == "hinted";
    let (planes, units, vw) = (16usize, 32usize, 4usize);

    let client = cubecl::Device::Cuda(Default::default()).client();
    let dtype = f32::elem_type_native();

    let host: Vec<f32> = bytemuck::cast_slice(&std::fs::read(inp).unwrap()).to_vec();
    assert_eq!(host.len(), rows * cols);
    let x = TensorHandle::new_contiguous(
        [rows, 1, cols],
        client.create_from_slice(bytemuck::cast_slice(&host)),
        dtype,
    );
    let s = TensorHandle::new_contiguous([rows, 1], client.empty(rows * 4), dtype);
    let out = TensorHandle::new_contiguous([rows, 1, cols], client.empty(rows * cols * 4), dtype);

    if mode == "lanes" {
        const V: Axis = Axis(3);
        let quads = cols / vw;
        let as4 = |h: &TensorHandle| {
            TensorHandle::new_contiguous([rows, 1, quads, vw], h.handle.clone(), dtype)
        };
        let (x4, out4) = (as4(&x), as4(&out));
        let s4 = TensorHandle::new_contiguous([rows, 1, vw], client.empty(rows * vw * 4), dtype);
        let partitioning = Partitioning::new(
            Space::new(&[(M, rows), (N, 1), (K, quads), (V, vw)]),
            Levels::leaf(&[(M, 1), (N, 1), (K, 1), (V, vw)])
                .units(&[(K, units)])
                .walk(&[(K, quads / (planes * units))])
                .planes(&[(K, planes)])
                .cubes(&[M, N])
                .build(),
        );
        let concrete = partitioning.space().clone();
        let launcher = Launcher::new(&client, partitioning, &concrete, Grid::FromLevels).unwrap();
        let arg = |h: &TensorHandle, axes: &[Axis], width: usize| {
            launcher
                .arg(h.clone().binding(), h.dtype)
                .axes(axes)
                .vectorize(width)
                .build()
                .unwrap()
        };
        for _ in 0..reps {
            sum_consumer_lanes::launch(
                &client,
                launcher.cube_count(),
                launcher.cube_dim(),
                arg(&x4, &[M, N, K, V], vw).arg(),
                arg(&s4, &[M, N, V], vw).arg(),
                arg(&s, &[M, N], 1).arg(),
                arg(&out4, &[M, N, K, V], vw).arg(),
                launcher.partitioning_arg(),
                planes,
                dtype,
            );
        }
        let bytes = client.read_one(out.handle.clone()).unwrap();
        std::fs::write(outp, &bytes[..]).unwrap();
        let sb = client.read_one(s.handle.clone()).unwrap();
        std::fs::write(format!("{outp}.sums"), &sb[..]).unwrap();
        let s4b = client.read_one(s4.handle.clone()).unwrap();
        std::fs::write(format!("{outp}.sums4"), &s4b[..]).unwrap();
        return;
    }

    let levels = if row_plane {
        Levels::leaf(&[(M, 1), (N, 1), (K, vw)])
            .units(&[(K, units)])
            .walk(&[(K, cols / (units * vw))])
            .planes(&[(M, planes)])
            .cubes(&[M, N])
    } else {
        Levels::leaf(&[(M, 1), (N, 1), (K, vw)])
            .units(&[(K, units)])
            .walk(&[(K, cols / (planes * units * vw))])
            .planes(&[(K, planes)])
            .cubes(&[M, N])
    };
    let partitioning =
        Partitioning::new(Space::new(&[(M, rows), (N, 1), (K, cols)]), levels.build());
    let concrete = partitioning.space().clone();
    let launcher = Launcher::new(&client, partitioning, &concrete, Grid::FromLevels).unwrap();
    println!(
        "cube_count {:?} cube_dim {:?}",
        launcher.cube_count(),
        launcher.cube_dim()
    );

    let arg = |h: &TensorHandle, axes: &[Axis], width: usize| {
        launcher
            .arg(h.clone().binding(), h.dtype)
            .axes(axes)
            .vectorize(width)
            .build()
            .unwrap()
    };
    for _ in 0..reps {
        if row_plane {
            sum_consumer_row_plane::launch(
                &client,
                launcher.cube_count(),
                launcher.cube_dim(),
                arg(&x, &[M, N, K], vw).arg(),
                arg(&s, &[M, N], 1).arg(),
                arg(&out, &[M, N, K], vw).arg(),
                launcher.partitioning_arg(),
                dtype,
            );
            continue;
        }
        sum_consumer::launch(
            &client,
            launcher.cube_count(),
            launcher.cube_dim(),
            arg(&x, &[M, N, K], vw).arg(),
            arg(&s, &[M, N], 1).arg(),
            arg(&out, &[M, N, K], vw).arg(),
            launcher.partitioning_arg(),
            planes,
            hinted,
            dtype,
        );
    }
    let bytes = client.read_one(out.handle.clone()).unwrap();
    std::fs::write(outp, &bytes[..]).unwrap();
    let sb = client.read_one(s.handle.clone()).unwrap();
    std::fs::write(format!("{outp}.sums"), &sb[..]).unwrap();
}
