use super::{ReduceFamily, ReduceInstruction, ReduceRequirements};
use crate::components::{
    instructions::{Accumulator, AccumulatorFormat, Item, ReduceOutputMode, ReduceStep, Value},
    precision::ReducePrecision,
};
use cubecl::prelude::*;

#[derive(Debug, CubeType, Clone)]
pub struct Sum {}

impl ReduceFamily for Sum {
    type Instruction<P: ReducePrecision> = Self;
    type Config = ();
}

#[cube]
impl<P: ReducePrecision> ReduceInstruction<P> for Sum {
    type SharedAccumulator = Shared<[Vector<P::EA, P::SI>]>;
    type Config = ();

    fn requirements(_this: &Self) -> ReduceRequirements {
        ReduceRequirements { coordinates: false }
    }

    fn accumulator_format(_this: &Self) -> comptime_type!(AccumulatorFormat) {
        AccumulatorFormat::Single
    }

    fn from_config(_config: Self::Config) -> Self {
        Sum {}
    }
    fn null_input(_this: &Self) -> Vector<P::EI, P::SI> {
        Vector::empty().fill(P::EI::from_int(0))
    }

    fn null_accumulator(_this: &Self) -> Accumulator<P> {
        Accumulator::<P> {
            elements: Value::new_single(Vector::empty().fill(P::EA::from_int(0))),
            args: Value::new_None(),
        }
    }

    fn reduce(
        _this: &Self,
        accumulator: &mut Accumulator<P>,
        item: Item<P>,
        #[comptime] reduce_step: ReduceStep,
    ) {
        let accumulator_item = accumulator.elements.item();
        let item = item.elements;
        let elements = match reduce_step {
            ReduceStep::Plane => accumulator_item + plane_sum(Vector::cast_from(item)),
            ReduceStep::Identity => accumulator_item + Vector::cast_from(item),
        };

        accumulator.elements.assign(&Value::new_single(elements));
    }

    fn plane_reduce_inplace(_this: &Self, accumulator: &mut Accumulator<P>) {
        let sum = plane_sum(Vector::cast_from(accumulator.elements.item()));
        accumulator.elements.assign(&Value::new_single(sum));
    }

    fn fold_lanes(_this: &Self, accumulator: &mut Accumulator<P>) {
        let lanes = accumulator.elements.item();
        let mut folded = Vector::empty().fill(P::EA::from_int(0));
        folded.insert(0usize, pairwise_sum(lanes, 0usize, lanes.vector_size()));
        accumulator.elements.assign(&Value::new_single(folded));
    }

    fn plane_reduce_folded(_this: &Self, accumulator: &mut Accumulator<P>) {
        let folded = accumulator.elements.item().extract(0usize);
        let mut lanes = Vector::empty().fill(P::EA::from_int(0));
        lanes.insert(0usize, plane_sum(folded));
        accumulator.elements.assign(&Value::new_single(lanes));
    }

    fn fuse_accumulators(_this: &Self, accumulator: &mut Accumulator<P>, other: &Accumulator<P>) {
        let accumulator_item = accumulator.elements.item();
        let other_item = other.elements.item();

        accumulator
            .elements
            .assign(&Value::new_single(accumulator_item + other_item));
    }

    fn output_mode(_this: &Self) -> comptime_type!(ReduceOutputMode) {
        ReduceOutputMode::Values
    }

    fn to_output_parallel<Out: Numeric, Idx: Numeric>(
        _this: &Self,
        accumulator: Accumulator<P>,
        _shape_axis_reduce: usize,
    ) -> (Value<Out>, Value<Idx>) {
        let sum = Vector::vector_sum(accumulator.elements.item());

        (Value::new_single(Out::cast_from(sum)), Value::new_None())
    }

    fn to_output_perpendicular<Out: Numeric, Idx: Numeric>(
        _this: &Self,
        accumulator: Accumulator<P>,
        _shape_axis_reduce: usize,
    ) -> (Value<Vector<Out, P::SI>>, Value<Vector<Idx, P::SI>>) {
        (
            Value::new_single(Vector::cast_from(accumulator.elements.item())),
            Value::new_None(),
        )
    }
}

/// The sum of `len` lanes from `start`, as a balanced tree: `(l0 + l1) + (l2 + l3)`.
#[cube]
fn pairwise_sum<E: Numeric, N: Size>(
    lanes: Vector<E, N>,
    #[comptime] start: usize,
    #[comptime] len: usize,
) -> E {
    if comptime!(len == 1) {
        lanes.extract(start)
    } else {
        let half = comptime!(len / 2);
        pairwise_sum(lanes, start, half) + pairwise_sum(lanes, comptime!(start + half), half)
    }
}
