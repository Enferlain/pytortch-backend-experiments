//! The reduction phase of a `tile_row_sum` kernel on `cubecl-tile` loops: one
//! row per cube, 16 planes of 32 units taking the row's steps in runs, each
//! plane landing its partial in a shared slot (CubeK's `InSlots` layout). The
//! kernel is ours otherwise: extents come from our scalars, nothing is bound
//! through a tile `Launcher`, and the element value is Inductor's loop body
//! from `semantic::emit`, inside a recipe.
use crate::ir::{constant, elem};
use crate::semantic::emit;
use cubecl::{
    frontend::{AsMutExpand, AsRefExpand, CubeDebug, ExpandTypeClone, IntoExpand, IntoMut},
    ir::{
        Scope,
        dialect::{general::CastOp, memory::LoadOp},
        pliron::value::Value as IrValue,
    },
    prelude::*,
};
use cubecl_tile::{
    procedural::{Constant, Procedural, Reads, Recipe, RecipeCoords, RecipeExpand},
    *,
};
use serde_json::{Value, json};
use std::{collections::HashMap, sync::Arc};

/// The row; a second axis that the launch grid fills but no kernel reads; the
/// reduction axis.
const M: Axis = Axis(0);
const N: Axis = Axis(1);
const K: Axis = Axis(2);
const PLANES: usize = 16;
const UNITS: usize = 32;
const REGISTER_BLOCK: RegisterBlock = RegisterBlock::new(16);

type Zero = <Constant<f32> as CubeType>::ExpandType;

/// What `Kernel::tile_row` hands over: the reduced value's ops, and the values
/// that replay them.
pub(super) struct RowSum {
    pub body: Vec<Value>,
    pub operand: String,
    pub arrays: Vec<IrValue>,
    pub scalars: Vec<IrValue>,
    pub outer: IrValue,
    pub ready: HashMap<String, IrValue>,
    pub rows: IrValue,
    pub columns: IrValue,
    pub vector_width: usize,
}

/// The element a reduction reads: Inductor's ops at a tile's column.
#[derive(Clone)]
pub struct RowBody;

#[derive(Clone)]
pub struct RowBodyExpand {
    body: Arc<Vec<Value>>,
    operand: Arc<str>,
    arrays: Vec<IrValue>,
    scalars: Vec<IrValue>,
    outer: IrValue,
    ready: HashMap<String, IrValue>,
}

impl CubeType for RowBody {
    type ExpandType = RowBodyExpand;
}

impl IntoExpand for RowBodyExpand {
    type Expand = Self;
    fn into_expand(self, _: &Scope) -> Self {
        self
    }
}
impl ExpandTypeClone for RowBodyExpand {
    fn clone_unchecked(&self) -> Self {
        self.clone()
    }
}
impl IntoMut for RowBodyExpand {
    fn into_mut(self, _: &Scope) -> Self {
        self
    }
}
impl CubeDebug for RowBodyExpand {}
impl AsRefExpand for RowBodyExpand {
    fn __expand_ref_method(&self, _: &Scope) -> &Self {
        self
    }
}
impl AsMutExpand for RowBodyExpand {
    fn __expand_ref_mut_method(&mut self, _: &Scope) -> &mut Self {
        self
    }
}

impl Recipe<f32> for RowBody {
    fn evaluate(&self, _: &RecipeCoords) -> f32 {
        cubecl::unexpanded!()
    }
}

impl Reads for RowBodyExpand {
    fn reads(&self, _: &Scope, _: Axis) -> bool {
        true
    }
}

impl RecipeExpand<f32> for RowBodyExpand {
    fn __expand_evaluate_method(
        &self,
        scope: &Scope,
        coordinates: &<RecipeCoords as CubeType>::ExpandType,
    ) -> NativeExpand<f32> {
        let column = coordinates.__expand_along_method(scope, K).value(scope);
        let int64 = elem("int64").unwrap().to_type(scope.ctx());
        let column = scope.register_with_result(&CastOp::new(scope.ctx_mut(), int64, column));
        let mut args = self.scalars.clone();
        args.push(column);
        let mut values = self.ready.clone();
        emit(
            scope,
            &self.body,
            &self.arrays,
            self.outer,
            &args,
            &mut values,
        )
        .expect("validated reduction body");
        NativeExpand::from(values[&*self.operand])
    }
}

/// The sum of `recipe` over this cube's row, on every unit.
#[cube]
fn reduce_row(
    rows: i64,
    columns: i64,
    recipe: RowBody,
    zero: Constant<f32>,
    #[comptime] form: Space,
    #[comptime] levels: Vec<Level>,
) -> f32 {
    let mut sizes = Sequence::<usize>::new();
    sizes.push(usize::cast_from(rows));
    sizes.push(1usize.runtime());
    sizes.push(usize::cast_from(columns));
    let space = Partitioning::with_levels(
        Space::with_sizes(comptime!(form.clone()), sizes),
        comptime!(levels.clone()),
    );
    let values = Procedural::new::<f32, RowBody>(comptime!(form.clone()), recipe).tile::<f32>();
    // No memory behind the sum: the tile only shapes the reducer and its slots.
    let row = Procedural::new::<f32, Constant<f32>>(comptime!(form.subspace(&[M, N])), zero)
        .tile_in::<f32>(&space);
    let mut sum = 0.0f32;
    for cube in &space {
        let row_cube = row.at(&cube);
        let slots = row_cube.smem_slots_accumulation::<f32>(PLANES);
        let values_cube = values.at(&cube);
        let partial = row_cube.block_reducer::<f32>(&values_cube, REGISTER_BLOCK, Monoid::Sum);
        let steps = cube.walk();
        let writer = steps.position(K);
        for step in steps {
            for unit in &step {
                let mut accumulator = partial.at(&unit);
                accumulator.reduce_axis_accumulate(&values.at(&unit), Monoid::Sum);
            }
        }
        slots.drain(&partial, writer);
        sum = slots.sum(PLANES);
    }
    sum
}

pub(super) fn row_sum(s: &Scope, sum: RowSum) -> IrValue {
    let form = Space::new(&[(M, 1), (N, 1), (K, 1)]).with_dynamic(&[M, K]);
    let levels = Levels::leaf(&[(M, 1), (N, 1), (K, sum.vector_width)])
        .units(&[(K, UNITS)])
        .planes_across(K, PLANES)
        .cubes(&[M, N])
        .build();
    let recipe = RowBodyExpand {
        body: Arc::new(sum.body),
        operand: sum.operand.into(),
        arrays: sum.arrays,
        scalars: sum.scalars,
        outer: sum.outer,
        ready: sum.ready,
    };
    let value = reduce_row::expand(
        s,
        NativeExpand::from(sum.rows),
        NativeExpand::from(sum.columns),
        recipe,
        Zero {
            value: NativeExpand::from(constant(s, &json!(0.0), "float32").unwrap()),
        },
        form,
        levels,
    );
    // A mutable local comes back as its pointer.
    s.register_with_result(&LoadOp::new(s.ctx_mut(), value.value(s)))
}
