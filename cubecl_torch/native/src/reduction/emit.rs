//! The cooperative reduction emitter: one builder for every reduction
//! strategy, plus the offline codegen probes selected by `CodegenOptions`.
use super::analysis::{bind_coordinate, closure, operand_ids, validate_description};
use super::combiner::{Combiner, broadcast, compensated_add};
use super::vector::reduction_vector_values;
use crate::ir::{branch, constant, elem, pointer};
use crate::semantic::{emit, index};
use cubecl_core::{
    compute::KernelBuilder,
    ir::{
        AddressType, Builtin, OpInserter, Scope,
        dialect::{
            branch::{IfOp, RangeLoopOp, ReturnOp},
            cmp::{IEqualOp, SLessThanOp},
            general::{CastOp, ReadBuiltinOp, ReadScalarOp},
            math::{FAddOp, IAddOp, IMulOp, SDivOp, SRemOp},
            memory::{LoadOp, StoreOp},
            synchronization::{SyncOp, SyncScope, SyncScopeAttr},
        },
        pliron::{r#type::TypeHandle, value::Value as IrValue},
        settings::{Dim3, ExecutionMode, KernelSettings},
        types::{ArrayType, scalar::IndexType},
    },
};
use serde_json::{Value, json};
use std::collections::{HashMap, HashSet};
type Error = Box<dyn std::error::Error>;

fn sync(s: &Scope) {
    s.register(&SyncOp::new(
        s.ctx_mut(),
        SyncScopeAttr::new(SyncScope::Cube),
    ));
}

fn load_all(s: &Scope, locals: &[IrValue]) -> Vec<IrValue> {
    locals
        .iter()
        .map(|l| s.register_with_result(&LoadOp::new(s.ctx_mut(), *l)))
        .collect()
}

fn store_all(s: &Scope, locals: &[IrValue], values: &[IrValue]) {
    for (l, v) in locals.iter().zip(values) {
        s.register(&StoreOp::new(s.ctx_mut(), *l, *v));
    }
}

pub fn build(d: &Value) -> Result<cubecl_runtime::kernel::KernelDefinition, Error> {
    if d["schema"] != 3 || d["index_dtype"] != "int64" {
        return Err("unsupported reduction schema".into());
    }
    let expected = json!({"iteration_mapping":"one_workgroup_per_outer", "workgroup":[256,1,1],
        "elements_per_unit":"strided_loop_step_workgroup", "local_vector_width":1,
        "reduction_mapping":"shared_binary_tree", "accumulator_dtype":"float32", "shared_elements":256,
        "barriers":"uniform_after_publish_and_each_tree_level", "result_owner":"unit_zero_store_all_units_read",
        "bounds_strategy":"loop_reduction_coordinate_less_than_extent", "phases":1});
    if d["physical_plan"] != expected {
        return Err(
            "unsupported physical reduction topology (including divergent barriers)".into(),
        );
    }
    let entry = d["entrypoint"].as_str().ok_or("entry missing")?;
    if !entry.starts_with("cubecl3_")
        || !entry.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
    {
        return Err("entry invalid".into());
    }
    validate_description(d)?;
    build_physical(d, [256, 1, 1], "shared_tree", 1, 1, "halving")
}

pub(crate) fn build_physical(
    d: &Value,
    workgroup: [u32; 3],
    strategy: &str,
    vector_width: usize,
    private_accumulators: usize,
    private_merge: &str,
) -> Result<cubecl_runtime::kernel::KernelDefinition, Error> {
    build_physical_codegen(
        d,
        workgroup,
        strategy,
        vector_width,
        private_accumulators,
        private_merge,
        CodegenOptions {
            uniform_loop: strategy == "plane_shared_reference512",
            ..Default::default()
        },
    )
}

#[derive(Clone, Copy, Default)]
pub(crate) struct CodegenOptions {
    pub uniform_loop: bool,
    pub owner_result: bool,
    pub narrow_loop: bool,
    pub final_plane: bool,
    pub final_load_batch: bool,
}

/// The plan's legality for this emitter, checked before any IR is built.
/// Returns the scratch arity (the widest combiner).
fn check_plan(
    d: &Value,
    workgroup: [u32; 3],
    strategy: &str,
    vector_width: usize,
    private_accumulators: usize,
    private_merge: &str,
    codegen: CodegenOptions,
) -> Result<usize, Error> {
    let CodegenOptions {
        uniform_loop,
        owner_result,
        narrow_loop,
        final_plane,
        final_load_batch,
    } = codegen;
    if (uniform_loop || owner_result || narrow_loop) && strategy != "plane_shared_reference512" {
        return Err("codegen probe requires the frozen reference512 plan".into());
    }
    if final_load_batch && !final_plane {
        return Err("final load batching requires the final-plane probe".into());
    }
    if final_plane
        && (strategy != "independent_planes"
            || workgroup != [32, 4, 1]
            || vector_width != 1
            || private_accumulators != 1
            || d["reduction_extent"]["op"] != "constant"
            || d["reduction_extent"]["value"]
                .as_u64()
                .is_none_or(|extent| !(1..=32).contains(&extent))
            || d["ops"].as_array().is_none_or(|ops| {
                ops.iter().filter(|op| op["op"] == "reduction").count() != 1
                    || ops
                        .iter()
                        .any(|op| op["op"] == "store" && op["iteration_scope"] != "outer")
            }))
    {
        return Err(
            "final-plane probe requires the small owner-only independent-plane plan (extent 1..32)"
                .into(),
        );
    }
    if private_merge == "adjacent" && private_accumulators != 4 {
        return Err("adjacent private merge requires four accumulators".into());
    }
    if private_merge == "lanes"
        && !(strategy == "plane_shared"
            && vector_width > 1
            && private_accumulators == 1
            && d["ops"].as_array().is_some_and(|ops| {
                ops.iter().filter(|op| op["op"] == "reduction").all(|op| {
                    op["reduction_type"]
                        .as_str()
                        .is_some_and(|t| t.starts_with("welford"))
                })
            }))
    {
        return Err("lane accumulators require a Welford vector plane plan".into());
    }
    validate_description(d)?;
    let ops = d["ops"].as_array().unwrap();
    let mut shared_arity = 1;
    for op in ops.iter().filter(|op| op["op"] == "reduction") {
        let combiner = Combiner::from_descriptor(op)?;
        if !combiner.supports(strategy, final_plane) {
            return Err(format!(
                "strategy {} does not support reduction type {}",
                if final_plane {
                    "independent_planes_final9"
                } else {
                    strategy
                },
                combiner.name()
            )
            .into());
        }
        shared_arity = shared_arity.max(combiner.arity());
    }
    if owner_result
        && (ops.iter().filter(|op| op["op"] == "reduction").count() != 1
            || ops
                .iter()
                .any(|op| op["op"] == "store" && op["iteration_scope"] != "outer"))
    {
        return Err("owner-result probe requires one reduction and only owner stores".into());
    }
    if vector_width > 1 && private_accumulators > 1 {
        // plan.rs admits vector chains only here; keep the emitter honest.
        let welford_only = d["ops"].as_array().is_some_and(|ops| {
            ops.iter().filter(|op| op["op"] == "reduction").all(|op| {
                op["reduction_type"]
                    .as_str()
                    .is_some_and(|t| t.starts_with("welford"))
            })
        });
        if strategy != "independent_planes"
            || !welford_only
            || !private_accumulators.is_power_of_two()
        {
            return Err(
                "vector chains require Welford independent planes and a power-of-two count".into(),
            );
        }
    }
    Ok(shared_arity)
}

/// The qualified reference512 path translates CubeK's uniform chunk loop.
/// Other options are offline probes, never selected by the physical planner.
/// No chain coordinate, protected body, or accumulator-update order changes.
pub(crate) fn build_physical_codegen(
    d: &Value,
    workgroup: [u32; 3],
    strategy: &str,
    vector_width: usize,
    private_accumulators: usize,
    private_merge: &str,
    codegen: CodegenOptions,
) -> Result<cubecl_runtime::kernel::KernelDefinition, Error> {
    let shared_arity = check_plan(
        d,
        workgroup,
        strategy,
        vector_width,
        private_accumulators,
        private_merge,
        codegen,
    )?;
    let entry = d["entrypoint"].as_str().ok_or("entry missing")?;
    let buffers = d["buffers"].as_array().unwrap();
    let defs = d["scalars"].as_array().unwrap();
    let ops = d["ops"].as_array().unwrap();
    let mut b = KernelBuilder::new(
        KernelSettings::new(
            Dim3::new_3d(workgroup[0], workgroup[1], workgroup[2]),
            ExecutionMode::Unchecked,
            AddressType::U64,
        )
        .kernel_name(entry),
    );
    let arrays: Vec<_> = buffers
        .iter()
        .map(|v| b.buffer(elem(v["dtype"].as_str().unwrap()).unwrap().to_type(b.ctx())))
        .collect();
    let ity = elem("int64")?.to_type(b.ctx());
    let fty = elem("float32")?.to_type(b.ctx());
    let scalars: Vec<_> = (0..=defs.len())
        .map(|_| {
            let id = b.scalar(elem("int64").unwrap());
            b.register_with_result(&ReadScalarOp::new(b.ctx_mut(), ity, id))
        })
        .collect();
    let builtin = |kind| {
        let x = b.register_with_result(&ReadBuiltinOp::new(
            b.ctx_mut(),
            IndexType::get(b.ctx()).to_handle(),
            kind,
        ));
        b.register_with_result(&CastOp::new(b.ctx_mut(), ity, x))
    };
    let cube = builtin(Builtin::CubePosX);
    let lane = builtin(Builtin::UnitPosX);
    let column = strategy == "coalesced_columns";
    let planes = strategy == "independent_planes";
    // Vector chains: Welford independent planes with several whole-vector
    // accumulators per lane (plan.rs admits them only there).
    let chained = planes && vector_width > 1 && private_accumulators > 1;
    let unit = if column {
        builtin(Builtin::UnitPosY)
    } else {
        lane
    };
    let outputs = if column {
        workgroup[0]
    } else if planes {
        workgroup[1]
    } else {
        1
    };
    let lane_outer = if column {
        lane
    } else if planes {
        builtin(Builtin::UnitPosY)
    } else {
        constant(&b, &json!(0), "int64")?
    };
    let scale = constant(&b, &json!(outputs), "int64")?;
    let outer = b.register_with_result(&IMulOp::new(b.ctx_mut(), cube, scale));
    let outer = b.register_with_result(&IAddOp::new(b.ctx_mut(), outer, lane_outer));
    let cooperation = if column { workgroup[1] } else { workgroup[0] };
    let zero = constant(&b, &json!(0), "int64")?;
    let step = constant(&b, &json!(cooperation), "int64")?;
    let outer_extent = index(&b, &d["outer"], zero, &scalars[1..])?;
    let reduction_extent = index(&b, &d["reduction_extent"], zero, &scalars[1..])?;
    let reference512 = strategy == "plane_shared_reference512";
    let batched_loads = strategy == "plane_shared_load_batch" || reference512;
    let plane_shared = strategy == "plane_shared" || batched_loads;
    let compensated = strategy == "shared_tree_compensated";
    let shared_size = if plane_shared {
        workgroup[0] / 32
    } else {
        workgroup[0] * workgroup[1] * if compensated { 2 } else { 1 }
    };
    // One scratch array per accumulator component.
    let shared: Vec<_> = if planes {
        Vec::new()
    } else {
        (0..shared_arity)
            .map(|_| b.create_shared(ArrayType::get(b.ctx(), fty, shared_size as usize), None))
            .collect()
    };
    let valid = b.register_with_result(&SLessThanOp::new(b.ctx_mut(), outer, outer_extent));
    // CubePosX and runtime extents are workgroup-uniform; no unit predicate
    // encloses a barrier. Reduction loops and masked bodies contain no barriers.
    let collective_guard = if column {
        constant(&b, &json!(true), "bool")?
    } else {
        valid
    };
    let kernel = Kernel {
        vector_width,
        private_accumulators,
        private_merge,
        codegen,
        column,
        planes,
        chained,
        reference512,
        batched_loads,
        plane_shared,
        compensated,
        outputs,
        cooperation,
        shared_size,
        ops: bind_coordinate(&Value::Array(ops.clone()), defs.len())
            .as_array()
            .unwrap()
            .clone(),
        slot: defs.len(),
        arrays,
        scalars,
        shared,
        fty,
        ity,
        unit,
        lane_outer,
        outer,
        zero,
        step,
        reduction_extent,
        valid,
    };
    branch(&b, collective_guard, |s| {
        let mut ready = HashMap::new();
        let reductions: Vec<_> = kernel
            .ops
            .iter()
            .filter(|op| op["op"] == "reduction")
            .collect();
        for (reduction_index, op) in reductions.iter().enumerate() {
            let combiner = Combiner::from_descriptor(op)?;
            let correction = compensated.then(|| s.create_local_mut(fty, None));
            if let Some(c) = correction {
                s.register(&StoreOp::new(
                    s.ctx_mut(),
                    c,
                    constant(s, &json!(0.0), "float32")?,
                ));
            }
            let partial = if private_accumulators > 1 && vector_width == 1 {
                kernel.private_chains(s, combiner, op, &ready)?
            } else {
                kernel.strided_chains(s, combiner, op, &ready, correction)?
            };
            let sum = kernel.cross_unit(s, combiner, partial, correction)?;
            let id = op["id"].as_str().unwrap();
            for component in kernel
                .ops
                .iter()
                .filter(|c| c["op"] == "reduction_component" && c["args"][0].as_str() == Some(id))
            {
                let index = component["index"].as_u64().unwrap() as usize;
                ready.insert(component["id"].as_str().unwrap().into(), sum[index]);
            }
            ready.insert(id.into(), combiner.output(&sum));
            // All units must finish reading before a later reduction overwrites
            // scratch. The last reduction has no subsequent scratch writer:
            // epilogues only read the reduced SSA values and store to globals.
            if !planes && reduction_index + 1 < reductions.len() {
                sync(s);
            }
        }
        kernel.stores(s, &ready)
    })?;
    b.register(&ReturnOp::new(b.ctx_mut()));
    Ok(b.build())
}

/// What the per-reduction phases share: the physical plan, and the IR values
/// fixed before the reduction loop (unit mapping, extents, scratch).
struct Kernel<'a> {
    vector_width: usize,
    private_accumulators: usize,
    private_merge: &'a str,
    codegen: CodegenOptions,
    /// `coalesced_columns`: units along y cooperate on one of 32 columns.
    column: bool,
    /// `independent_planes`: one plane per output row, no shared scratch.
    planes: bool,
    /// Welford independent planes with several whole-vector chains per lane.
    chained: bool,
    /// `plane_shared_reference512`: CubeK's uniform chunk loop and fixed butterflies.
    reference512: bool,
    /// Load-batch plans: collect every chain's value, then update the accumulators.
    batched_loads: bool,
    /// Plane collectives with one shared slot per plane (all `plane_shared*`).
    plane_shared: bool,
    /// `shared_tree_compensated`: TwoSum residuals beside the shared tree.
    compensated: bool,
    /// Outer rows per workgroup.
    outputs: u32,
    /// Units cooperating on one row: the reduction loop's stride.
    cooperation: u32,
    shared_size: u32,
    /// Semantic ops with the reduction coordinate bound to emitter slot `slot`.
    ops: Vec<Value>,
    /// The coordinate's position in `args`: the runtime symbol count.
    slot: usize,
    arrays: Vec<IrValue>,
    /// Slot 0 (launch units), then the runtime symbols.
    scalars: Vec<IrValue>,
    /// One scratch array per accumulator component (none for planes).
    shared: Vec<IrValue>,
    fty: TypeHandle,
    ity: TypeHandle,
    unit: IrValue,
    lane_outer: IrValue,
    outer: IrValue,
    zero: IrValue,
    step: IrValue,
    reduction_extent: IrValue,
    valid: IrValue,
}

impl Kernel<'_> {
    /// The emitter's scalars: the runtime symbols, then the reduction coordinate.
    fn args(&self, coordinate: IrValue) -> Vec<IrValue> {
        let mut args = self.scalars[1..].to_vec();
        args.push(coordinate);
        args
    }

    fn shared_index(&self, s: &Scope, r: IrValue) -> IrValue {
        let stride = constant(s, &json!(self.outputs), "int64").unwrap();
        let scaled = s.register_with_result(&IMulOp::new(s.ctx_mut(), r, stride));
        s.register_with_result(&IAddOp::new(s.ctx_mut(), scaled, self.lane_outer))
    }

    /// ATen's independent strided chains, scalar loads, followed by a
    /// balanced private merge. This is a physical choice only; logical replay
    /// and all semantic/storage barriers are reused.
    fn private_chains(
        &self,
        s: &Scope,
        combiner: Combiner,
        op: &Value,
        ready: &HashMap<String, IrValue>,
    ) -> Result<Vec<IrValue>, Error> {
        let &Kernel {
            private_accumulators,
            private_merge,
            codegen:
                CodegenOptions {
                    uniform_loop,
                    narrow_loop,
                    ..
                },
            batched_loads,
            cooperation,
            fty,
            ity,
            unit,
            outer,
            zero,
            step,
            reduction_extent,
            ref arrays,
            ..
        } = self;
        let accumulators = (0..private_accumulators)
            .map(|_| combiner.identity_locals(s, fty).map(|(locals, _)| locals))
            .collect::<Result<Vec<_>, _>>()?;
        let count = constant(s, &json!(private_accumulators), "int64")?;
        let advance = s.register_with_result(&IMulOp::new(s.ctx_mut(), step, count));
        let (loop_start, loop_end, loop_step) = if narrow_loop {
            // Runtime launch validation proves 0 <= E <= 2^20.
            // t <= 511, step=2048, and the final increment is <2^21.
            // Only this physical induction variable narrows. Semantic
            // coordinates and every pointer expression remain signed i64.
            let ty =
                cubecl_core::ir::ElemType::UInt(cubecl_core::ir::UIntKind::U32).to_type(s.ctx());
            let cast = |v| s.register_with_result(&CastOp::new(s.ctx_mut(), ty, v));
            (
                cast(if uniform_loop { zero } else { unit }),
                cast(reduction_extent),
                cast(advance),
            )
        } else {
            (
                if uniform_loop { zero } else { unit },
                reduction_extent,
                advance,
            )
        };
        let loop_op = RangeLoopOp::new(s.ctx_mut(), loop_start, loop_end, loop_step);
        let iteration = loop_op.iter_var(s.ctx());
        let child = s.branch_child(OpInserter::new_at_block_end(loop_op.loop_body(s.ctx())));
        let iteration = if narrow_loop {
            child.register_with_result(&CastOp::new(child.ctx_mut(), ity, iteration))
        } else {
            iteration
        };
        let base = if uniform_loop {
            child.register_with_result(&IAddOp::new(child.ctx_mut(), iteration, unit))
        } else {
            iteration
        };
        let value_ids = operand_ids(op);
        let body = closure(&value_ids, &self.ops, ready);
        // Reference-derived mechanism experiment: protect/collect all
        // scalar values before updating independent accumulators. The
        // semantic lazy bodies are unchanged and remain inside guards.
        // Inactive values combine the identity into accumulators that
        // start at the identity (+0 added to +0 for sum). For sum, keep
        // the RN/no-FTZ policy: the neutral additions rely on it (NaN
        // payloads are not a sum contract).
        let mut batch = Vec::new();
        for (i, accumulator) in accumulators.iter().enumerate() {
            let offset = constant(&child, &json!(i as u32 * cooperation), "int64")?;
            let coordinate =
                child.register_with_result(&IAddOp::new(child.ctx_mut(), base, offset));
            let live = child.register_with_result(&SLessThanOp::new(
                child.ctx_mut(),
                coordinate,
                reduction_extent,
            ));
            // Inactive pending values stay at the identity, so the
            // unconditional batch update leaves the accumulator as is.
            let pending = batched_loads
                .then(|| {
                    combiner
                        .identity_locals(&child, fty)
                        .map(|(locals, _)| locals)
                })
                .transpose()?;
            // Guard the access itself, not a later select of its value.
            branch(&child, live, |c| {
                let args = self.args(coordinate);
                let mut values = ready.clone();
                emit(c, &body, arrays, outer, &args, &mut values)?;
                let operands: Vec<_> = value_ids.iter().map(|id| values[*id]).collect();
                if let Some(ptr) = &pending {
                    let lifted = combiner.lift(c, &operands)?;
                    for (p, v) in ptr.iter().zip(&lifted) {
                        c.register(&StoreOp::new(c.ctx_mut(), *p, *v));
                    }
                    return Ok(());
                }
                let old = load_all(c, accumulator);
                let new = combiner.accumulate(c, &old, &operands)?;
                store_all(c, accumulator, &new);
                Ok(())
            })?;
            if let Some(ptr) = pending {
                batch.push((accumulator.clone(), ptr));
            }
        }
        let update = |c: &Scope| -> Result<(), Error> {
            for (accumulator, ptr) in &batch {
                let value = load_all(c, ptr);
                let old = load_all(c, accumulator);
                let new = combiner.combine(c, &old, &value);
                store_all(c, accumulator, &new);
            }
            Ok(())
        };
        if uniform_loop {
            // Preserve even the frozen plan's inactive-update set.
            let live = child.register_with_result(&SLessThanOp::new(
                child.ctx_mut(),
                base,
                reduction_extent,
            ));
            branch(&child, live, update)?;
        } else {
            update(&child)?;
        }
        child.terminate_yield();
        s.register(&loop_op);
        let mut values: Vec<_> = accumulators.iter().map(|a| load_all(s, a)).collect();
        // Both orders are balanced combines (ordinary FP32 adds for
        // sum). The explicit adjacent plan changes only this physical
        // merge, not semantic replay, chain assignment, collective order
        // or FP permissions.
        if private_merge == "adjacent" {
            let left = combiner.combine(s, &values[0], &values[1]);
            let right = combiner.combine(s, &values[2], &values[3]);
            values[0] = combiner.combine(s, &left, &right);
        } else {
            let mut stride = private_accumulators / 2;
            while stride != 0 {
                for i in 0..stride {
                    values[i] = combiner.combine(s, &values[i], &values[i + stride]);
                }
                stride /= 2;
            }
        }
        Ok(values.swap_remove(0))
    }

    /// One accumulator per unit, strided by the cooperation width; or, for
    /// Welford independent planes, several vector chains per lane reading
    /// chain i at offset i * cooperation * width. Compensation applies here.
    fn strided_chains(
        &self,
        s: &Scope,
        combiner: Combiner,
        op: &Value,
        ready: &HashMap<String, IrValue>,
        correction: Option<IrValue>,
    ) -> Result<Vec<IrValue>, Error> {
        let &Kernel {
            vector_width,
            private_accumulators,
            private_merge,
            codegen: CodegenOptions {
                final_load_batch, ..
            },
            column,
            chained,
            fty,
            unit,
            outer,
            zero,
            step,
            reduction_extent,
            valid,
            ref arrays,
            ..
        } = self;
        let (accumulator, neutral) = combiner.identity_locals(s, fty)?;
        // Welford "lanes": one accumulator per vector lane across the loop,
        // merged in lane order after it (Triton's R0_BLOCK accumulators;
        // CubeK keeps one per vector lane too).
        let mut lane_accumulators = Vec::new();
        if private_merge == "lanes" {
            lane_accumulators.push(accumulator.clone());
            for _ in 1..vector_width {
                lane_accumulators.push(combiner.identity_locals(s, fty)?.0);
            }
        }
        let mut chains = vec![accumulator.clone()];
        for _ in 1..private_accumulators {
            chains.push(combiner.identity_locals(s, fty)?.0);
        }
        let width = constant(s, &json!(vector_width), "int64")?;
        let begin = s.register_with_result(&IMulOp::new(s.ctx_mut(), unit, width));
        let chunk = s.register_with_result(&IMulOp::new(s.ctx_mut(), step, width));
        let reduction_step = if private_accumulators > 1 {
            let count = constant(s, &json!(private_accumulators), "int64")?;
            s.register_with_result(&IMulOp::new(s.ctx_mut(), chunk, count))
        } else {
            chunk
        };
        let loop_op = RangeLoopOp::new(
            s.ctx_mut(),
            if final_load_batch { zero } else { begin },
            reduction_extent,
            reduction_step,
        );
        let coordinate = loop_op.iter_var(s.ctx());
        let child = s.branch_child(OpInserter::new_at_block_end(loop_op.loop_body(s.ctx())));
        let coordinate = if final_load_batch {
            child.register_with_result(&IAddOp::new(child.ctx_mut(), coordinate, unit))
        } else {
            coordinate
        };
        let args = self.args(coordinate);
        let value_ids = operand_ids(op);
        let body = closure(&value_ids, &self.ops, ready);
        let operands = |values: &HashMap<String, IrValue>| -> Vec<IrValue> {
            value_ids.iter().map(|id| values[*id]).collect()
        };
        let mut local = ready.clone();
        // `value` holds one element's operand values; the final-load
        // probe (sum only) passes its lifted pending value, the same
        // single component.
        let accumulate =
            |c: &Scope, accumulator: &[IrValue], value: Vec<IrValue>| -> Result<(), Error> {
                let old = load_all(c, accumulator);
                let new = if let Some(ptr) = correction {
                    // Compensation is sum-only (one component).
                    let low = c.register_with_result(&LoadOp::new(c.ctx_mut(), ptr));
                    let (sum, low) = compensated_add(c, old[0], low, value[0])?;
                    c.register(&StoreOp::new(c.ctx_mut(), ptr, low));
                    vec![sum]
                } else {
                    combiner.accumulate(c, &old, &value)?
                };
                store_all(c, accumulator, &new);
                Ok(())
            };
        if final_load_batch {
            // The already-established uniform-loop/grouped-load
            // expression, isolated to this one-chain final probe.
            let pending: Vec<_> = neutral
                .iter()
                .map(|n| {
                    let p = child.create_local_mut(fty, None);
                    child.register(&StoreOp::new(child.ctx_mut(), p, *n));
                    p
                })
                .collect();
            let live = child.register_with_result(&SLessThanOp::new(
                child.ctx_mut(),
                coordinate,
                reduction_extent,
            ));
            branch(&child, live, |c| {
                emit(c, &body, arrays, outer, &args, &mut local)?;
                let lifted = combiner.lift(c, &operands(&local))?;
                store_all(c, &pending, &lifted);
                Ok(())
            })?;
            let value = load_all(&child, &pending);
            accumulate(&child, &accumulator, value)?;
        } else if column {
            branch(&child, valid, |c| {
                emit(c, &body, arrays, outer, &args, &mut local)?;
                accumulate(c, &accumulator, operands(&local))?;
                Ok(())
            })?;
        } else if vector_width == 1 {
            emit(&child, &body, arrays, outer, &args, &mut local)?;
            accumulate(&child, &accumulator, operands(&local))?;
        } else if !lane_accumulators.is_empty() {
            self.lane_step(
                &child,
                (&body, &value_ids, ready),
                &lane_accumulators,
                coordinate,
                &accumulate,
            )?;
        } else {
            for (chain, accumulator) in chains.iter().enumerate() {
                let coordinate = if chain == 0 {
                    coordinate
                } else {
                    let offset = constant(&child, &json!(chain), "int64")?;
                    let offset =
                        child.register_with_result(&IMulOp::new(child.ctx_mut(), chunk, offset));
                    child.register_with_result(&IAddOp::new(child.ctx_mut(), coordinate, offset))
                };
                self.vector_step(
                    &child,
                    combiner,
                    (&body, &value_ids, ready),
                    accumulator,
                    coordinate,
                    &accumulate,
                )?;
            }
        }
        child.terminate_yield();
        s.register(&loop_op);
        if !lane_accumulators.is_empty() {
            let mut merged = load_all(s, &lane_accumulators[0]);
            for lane in &lane_accumulators[1..] {
                merged = combiner.combine(s, &merged, &load_all(s, lane));
            }
            return Ok(merged);
        }
        let mut values: Vec<_> = chains.iter().map(|a| load_all(s, a)).collect();
        if chained {
            // Triton's order: each chain (Triton's warp) reduces
            // across the plane first, then the chains merge.
            for value in &mut values {
                *value = combiner.butterfly(s, value.clone(), &[16, 8, 4, 2, 1])?;
            }
        }
        let mut stride = private_accumulators / 2;
        while stride != 0 {
            for i in 0..stride {
                values[i] = combiner.combine(s, &values[i], &values[i + stride]);
            }
            stride /= 2;
        }
        Ok(values.swap_remove(0))
    }

    /// One vector at `coordinate`, each lane into its own accumulator when the
    /// vector lies wholly inside the extent; the guarded scalar tail, element
    /// by element into its lane's accumulator, otherwise.
    fn lane_step(
        &self,
        child: &Scope,
        (body, value_ids, ready): (&[Value], &[&str], &HashMap<String, IrValue>),
        accumulators: &[Vec<IrValue>],
        coordinate: IrValue,
        accumulate: &impl Fn(&Scope, &[IrValue], Vec<IrValue>) -> Result<(), Error>,
    ) -> Result<(), Error> {
        let &Kernel {
            vector_width,
            outer,
            reduction_extent,
            slot,
            ref arrays,
            ..
        } = self;
        let args = self.args(coordinate);
        let offset = constant(child, &json!(vector_width - 1), "int64")?;
        let last = child.register_with_result(&IAddOp::new(child.ctx_mut(), coordinate, offset));
        let full =
            child.register_with_result(&SLessThanOp::new(child.ctx_mut(), last, reduction_extent));
        let vector_branch = IfOp::new(child.ctx_mut(), full);
        let vector = child.branch_child(OpInserter::new_at_block_end(
            vector_branch.then_block(child.ctx()),
        ));
        let lanes = reduction_vector_values(
            &vector,
            body,
            arrays,
            outer,
            &args,
            ready,
            (value_ids, vector_width),
        )?;
        for (value, accumulator) in lanes.into_iter().zip(accumulators) {
            accumulate(&vector, accumulator, value)?;
        }
        vector.terminate_yield();
        let tail = child.branch_child(OpInserter::new_at_block_end(
            vector_branch.else_block(child.ctx()),
        ));
        for (lane, accumulator) in accumulators.iter().enumerate() {
            let offset = constant(&tail, &json!(lane), "int64")?;
            let r = tail.register_with_result(&IAddOp::new(tail.ctx_mut(), coordinate, offset));
            let valid =
                tail.register_with_result(&SLessThanOp::new(tail.ctx_mut(), r, reduction_extent));
            branch(&tail, valid, |c| {
                let mut args = args.clone();
                args[slot] = r;
                let mut values = ready.clone();
                emit(c, body, arrays, outer, &args, &mut values)?;
                accumulate(
                    c,
                    accumulator,
                    value_ids.iter().map(|id| values[*id]).collect(),
                )
            })?;
        }
        tail.terminate_yield();
        child.register(&vector_branch);
        Ok(())
    }

    /// One vector of `vector_width` elements at `coordinate` into one chain's
    /// accumulator when it lies wholly inside the extent; the guarded scalar
    /// tail otherwise.
    fn vector_step(
        &self,
        child: &Scope,
        combiner: Combiner,
        (body, value_ids, ready): (&[Value], &[&str], &HashMap<String, IrValue>),
        accumulator: &[IrValue],
        coordinate: IrValue,
        accumulate: &impl Fn(&Scope, &[IrValue], Vec<IrValue>) -> Result<(), Error>,
    ) -> Result<(), Error> {
        let &Kernel {
            vector_width,
            outer,
            reduction_extent,
            slot,
            ref arrays,
            ..
        } = self;
        let args = self.args(coordinate);
        let offset = constant(child, &json!(vector_width - 1), "int64")?;
        let last = child.register_with_result(&IAddOp::new(child.ctx_mut(), coordinate, offset));
        let full =
            child.register_with_result(&SLessThanOp::new(child.ctx_mut(), last, reduction_extent));
        let vector_branch = IfOp::new(child.ctx_mut(), full);
        let vector = child.branch_child(OpInserter::new_at_block_end(
            vector_branch.then_block(child.ctx()),
        ));
        let lanes = reduction_vector_values(
            &vector,
            body,
            arrays,
            outer,
            &args,
            ready,
            (value_ids, vector_width),
        )?;
        if combiner.arity() == 1 {
            for value in lanes {
                accumulate(&vector, accumulator, value)?;
            }
        } else {
            // Welford: the contiguous lanes combine as a balanced
            // pair tree (Triton's per-thread lanes), so equal
            // weights meet with ratio exactly 1/2; then one
            // combine into the accumulator.
            let mut level = lanes
                .iter()
                .map(|v| combiner.lift(&vector, v))
                .collect::<Result<Vec<_>, _>>()?;
            while level.len() > 1 {
                level = level
                    .chunks(2)
                    .map(|pair| match pair {
                        [a, b] => combiner.combine(&vector, a, b),
                        [a] => a.clone(),
                        _ => unreachable!(),
                    })
                    .collect();
            }
            let old = load_all(&vector, accumulator);
            let new = combiner.combine(&vector, &old, &level[0]);
            store_all(&vector, accumulator, &new);
        }
        vector.terminate_yield();
        let tail = child.branch_child(OpInserter::new_at_block_end(
            vector_branch.else_block(child.ctx()),
        ));
        for lane in 0..vector_width {
            let offset = constant(&tail, &json!(lane), "int64")?;
            let r = tail.register_with_result(&IAddOp::new(tail.ctx_mut(), coordinate, offset));
            let valid =
                tail.register_with_result(&SLessThanOp::new(tail.ctx_mut(), r, reduction_extent));
            branch(&tail, valid, |c| {
                let mut args = args.clone();
                args[slot] = r;
                let mut values = ready.clone();
                emit(c, body, arrays, outer, &args, &mut values)?;
                accumulate(
                    c,
                    accumulator,
                    value_ids.iter().map(|id| values[*id]).collect(),
                )?;
                Ok(())
            })?;
        }
        tail.terminate_yield();
        child.register(&vector_branch);
        Ok(())
    }

    /// Combine every unit's partial into the reduced value all units read.
    fn cross_unit(
        &self,
        s: &Scope,
        combiner: Combiner,
        partial: Vec<IrValue>,
        correction: Option<IrValue>,
    ) -> Result<Vec<IrValue>, Error> {
        if self.planes {
            Ok(if self.chained {
                // The chains already reduced across the plane.
                broadcast(s, &partial)
            } else if self.codegen.final_plane {
                // CubeK's independent Plane body and lane-zero writer,
                // using the retained Triton/ATen descending merge order.
                // Generic FSum uses a different order on the CUDA target.
                combiner.butterfly(s, partial, &[16, 8, 4, 2, 1])?
            } else {
                combiner.plane(s, &partial)?
            })
        } else if self.plane_shared {
            self.plane_shared(s, combiner, partial)
        } else {
            self.shared_tree(s, combiner, partial, correction)
        }
    }

    /// Plane collectives, one shared slot per plane, then plane 0 combines
    /// the slots and its leader publishes the result.
    fn plane_shared(
        &self,
        s: &Scope,
        combiner: Combiner,
        partial: Vec<IrValue>,
    ) -> Result<Vec<IrValue>, Error> {
        let &Kernel {
            codegen: CodegenOptions { owner_result, .. },
            reference512,
            shared_size,
            fty,
            unit,
            zero,
            ref shared,
            ..
        } = self;
        let owner_value = owner_result
            .then(|| combiner.identity_locals(s, fty).map(|(locals, _)| locals))
            .transpose()?;
        // The reference-derived plan fixes the complete butterfly
        // order as well as private work assignment. Existing plans
        // retain the generic plane polyfill unchanged.
        let plane_value = if reference512 {
            combiner.butterfly(s, partial, &[16, 8, 4, 2, 1])?
        } else {
            combiner.plane(s, &partial)?
        };
        let divisor = constant(s, &json!(32), "int64")?;
        let plane_id = s.register_with_result(&SDivOp::new(s.ctx_mut(), unit, divisor));
        let plane_lane = s.register_with_result(&SRemOp::new(s.ctx_mut(), unit, divisor));
        let leader = s.register_with_result(&IEqualOp::new(s.ctx_mut(), plane_lane, zero));
        branch(s, leader, |c| {
            for (array, value) in shared.iter().zip(&plane_value) {
                c.register(&StoreOp::new(
                    c.ctx_mut(),
                    pointer(c, *array, plane_id),
                    *value,
                ));
            }
            Ok(())
        })?;
        sync(s);
        let first_plane = s.register_with_result(&IEqualOp::new(s.ctx_mut(), plane_id, zero));
        branch(s, first_plane, |c| {
            // Lanes beyond the plane count keep the identity.
            let (local, _) = combiner.identity_locals(c, fty)?;
            let count = constant(c, &json!(shared_size), "int64")?;
            let live = c.register_with_result(&SLessThanOp::new(c.ctx_mut(), plane_lane, count));
            branch(c, live, |a| {
                for (array, l) in shared.iter().zip(&local) {
                    let value = a.register_with_result(&LoadOp::new(
                        a.ctx_mut(),
                        pointer(a, *array, plane_lane),
                    ));
                    a.register(&StoreOp::new(a.ctx_mut(), *l, value));
                }
                Ok(())
            })?;
            let value = load_all(c, &local);
            let final_value = if reference512 {
                // Only the first half-warp is read by the output
                // owner; distances <=8 never mix it with identity lanes.
                combiner.butterfly(c, value, &[8, 4, 2, 1])?
            } else {
                combiner.plane(c, &value)?
            };
            if let Some(local) = &owner_value {
                // Every lane in plane 0 keeps its own result; only unit
                // zero consumes it under the existing outer-store guard.
                store_all(c, local, &final_value);
            } else {
                branch(c, leader, |a| {
                    for (array, value) in shared.iter().zip(&final_value) {
                        a.register(&StoreOp::new(a.ctx_mut(), pointer(a, *array, zero), *value));
                    }
                    Ok(())
                })?;
            }
            Ok(())
        })?;
        Ok(if let Some(local) = &owner_value {
            load_all(s, local)
        } else {
            sync(s);
            shared
                .iter()
                .take(combiner.arity())
                .map(|array| {
                    s.register_with_result(&LoadOp::new(s.ctx_mut(), pointer(s, *array, zero)))
                })
                .collect()
        })
    }

    /// Every unit publishes its partial, then a halving tree over shared
    /// scratch with a barrier per level; compensated residuals ride along.
    fn shared_tree(
        &self,
        s: &Scope,
        combiner: Combiner,
        partial: Vec<IrValue>,
        correction: Option<IrValue>,
    ) -> Result<Vec<IrValue>, Error> {
        let &Kernel {
            compensated,
            cooperation,
            unit,
            lane_outer,
            zero,
            ref shared,
            ..
        } = self;
        for (array, value) in shared.iter().zip(&partial) {
            let target = pointer(s, *array, self.shared_index(s, unit));
            s.register(&StoreOp::new(s.ctx_mut(), target, *value));
        }
        // Compensation is sum-only: its residuals share component 0's
        // array, offset by the cooperation width.
        let correction_ptr = |c: &Scope, r: IrValue| {
            let delta = constant(c, &json!(cooperation), "int64").unwrap();
            let index = c.register_with_result(&IAddOp::new(c.ctx_mut(), r, delta));
            pointer(c, shared[0], index)
        };
        if let Some(c) = correction {
            let low = s.register_with_result(&LoadOp::new(s.ctx_mut(), c));
            s.register(&StoreOp::new(s.ctx_mut(), correction_ptr(s, unit), low));
        }
        sync(s);
        let levels = cooperation.ilog2();
        for stride in (0..levels).rev().map(|i| 1 << i) {
            let width = constant(s, &json!(stride), "int64")?;
            let active = s.register_with_result(&SLessThanOp::new(s.ctx_mut(), unit, width));
            branch(s, active, |child| {
                let rhs_unit =
                    child.register_with_result(&IAddOp::new(child.ctx_mut(), unit, width));
                let arrays = &shared[..combiner.arity()];
                let mut lhs_ptr = Vec::new();
                let mut rhs_ptr = Vec::new();
                for array in arrays {
                    lhs_ptr.push(pointer(child, *array, self.shared_index(child, unit)));
                    rhs_ptr.push(pointer(child, *array, self.shared_index(child, rhs_unit)));
                }
                let lhs: Vec<_> = lhs_ptr
                    .iter()
                    .map(|p| child.register_with_result(&LoadOp::new(child.ctx_mut(), *p)))
                    .collect();
                let rhs: Vec<_> = rhs_ptr
                    .iter()
                    .map(|p| child.register_with_result(&LoadOp::new(child.ctx_mut(), *p)))
                    .collect();
                let new = if compensated {
                    let lhs_low_ptr = correction_ptr(child, unit);
                    let lhs_low =
                        child.register_with_result(&LoadOp::new(child.ctx_mut(), lhs_low_ptr));
                    let rhs_low = child.register_with_result(&LoadOp::new(
                        child.ctx_mut(),
                        correction_ptr(child, rhs_unit),
                    ));
                    let (sum, low) = compensated_add(child, lhs[0], lhs_low, rhs[0])?;
                    let low =
                        child.register_with_result(&FAddOp::new(child.ctx_mut(), low, rhs_low));
                    child.register(&StoreOp::new(child.ctx_mut(), lhs_low_ptr, low));
                    vec![sum]
                } else {
                    combiner.combine(child, &lhs, &rhs)
                };
                for (p, v) in lhs_ptr.iter().zip(&new) {
                    child.register(&StoreOp::new(child.ctx_mut(), *p, *v));
                }
                Ok(())
            })?;
            sync(s);
        }
        let high: Vec<_> = shared[..combiner.arity()]
            .iter()
            .map(|array| {
                let root = pointer(s, *array, lane_outer);
                s.register_with_result(&LoadOp::new(s.ctx_mut(), root))
            })
            .collect();
        Ok(if compensated {
            let low = s.register_with_result(&LoadOp::new(s.ctx_mut(), correction_ptr(s, zero)));
            vec![s.register_with_result(&FAddOp::new(s.ctx_mut(), high[0], low))]
        } else {
            high
        })
    }

    /// Outer-scope (per-row) stores together under unit zero, then
    /// reduction-scope (per-element) stores together in one strided loop over
    /// the extent; each group replays the union of its stores' dependency
    /// closures once (Triton's single second loop; CubeK's paired writers).
    /// A group of one store emits exactly that store's former body.
    fn stores(&self, s: &Scope, ready: &HashMap<String, IrValue>) -> Result<(), Error> {
        let &Kernel {
            column,
            unit,
            outer,
            zero,
            step,
            reduction_extent,
            valid,
            ref arrays,
            ..
        } = self;
        let group = |per_row: bool| -> Option<Vec<Value>> {
            let stores: Vec<&Value> = self
                .ops
                .iter()
                .filter(|op| op["op"] == "store" && (op["iteration_scope"] == "outer") == per_row)
                .collect();
            if stores.is_empty() {
                return None;
            }
            let ids: Vec<&str> = stores.iter().flat_map(|op| operand_ids(op)).collect();
            let mut body = closure(&ids, &self.ops, ready);
            body.extend(stores.into_iter().cloned());
            Some(body)
        };
        if let Some(body) = group(true) {
            let leader = s.register_with_result(&IEqualOp::new(s.ctx_mut(), unit, zero));
            branch(s, leader, |child| {
                let args = self.args(zero);
                if column {
                    branch(child, valid, |c| {
                        emit(c, &body, arrays, outer, &args, &mut ready.clone())
                    })
                } else {
                    emit(child, &body, arrays, outer, &args, &mut ready.clone())
                }
            })?;
        }
        if let Some(mut body) = group(false) {
            // The per-element pass re-reads what the reduction pass read: its
            // last use, loaded as streaming (Triton's evict_first second loop;
            // without it L1 drops part of the row first, bead 4z8.33).
            let reductions: Vec<&str> = self
                .ops
                .iter()
                .filter(|op| op["op"] == "reduction")
                .flat_map(operand_ids)
                .collect();
            let reduced: HashSet<u64> = closure(&reductions, &self.ops, &HashMap::new())
                .iter()
                .filter(|op| op["op"] == "load")
                .filter_map(|op| op["buffer"].as_u64())
                .collect();
            for op in body.iter_mut().filter(|op| {
                op["op"] == "load" && op["buffer"].as_u64().is_some_and(|b| reduced.contains(&b))
            }) {
                op["streaming"] = json!(true);
            }
            let loop_op = RangeLoopOp::new(s.ctx_mut(), unit, reduction_extent, step);
            let coordinate = loop_op.iter_var(s.ctx());
            let child = s.branch_child(OpInserter::new_at_block_end(loop_op.loop_body(s.ctx())));
            let args = self.args(coordinate);
            if column {
                branch(&child, valid, |c| {
                    emit(c, &body, arrays, outer, &args, &mut ready.clone())
                })?;
            } else {
                emit(&child, &body, arrays, outer, &args, &mut ready.clone())?;
            }
            child.terminate_yield();
            s.register(&loop_op);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adjacent_merge_rejects_invalid_chain_count_before_ir_construction() {
        for count in [0, 1, 2, 3, 5] {
            let error = build_physical(
                &json!({}),
                [256, 1, 1],
                "plane_shared_load_batch",
                1,
                count,
                "adjacent",
            )
            .err()
            .expect("invalid adjacent chain count must fail locally");
            assert_eq!(
                error.to_string(),
                "adjacent private merge requires four accumulators"
            );
        }
    }
}
