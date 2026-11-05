//! The recorded body's emitter, for the runtime bridge's fused reduce
//! (fused-reduction-bridge 3.1): the descriptor compiler's own op emitter,
//! called from inside CubeK's kernel expansion instead of a kernel of ours.
//!
//! A reduction descriptor's ops split at its one reduction into the operand's
//! ops (replayed per element in `read_input`) and the per-output ops and stores
//! (replayed in `write_output`, with the reduced value bound to the reduction's
//! id). Index expressions name the output coordinate `linear` and the reduction
//! coordinate `{"op": "reduction"}`, which [`bind_reduction`] turns into scalar
//! slot `slot` for [`emit`].
use crate::reduction::analysis;
use crate::semantic;
use cubecl_core::ir::{Scope, pliron::value::Value as IrValue};
use serde_json::Value;
use std::collections::HashMap;

pub type Error = Box<dyn std::error::Error>;

/// Checks a reduction descriptor (schema 3) as the descriptor compiler does
/// before building any kernel: buffers, scalars, indices and ops.
pub fn validate(descriptor: &Value) -> Result<(), Error> {
    analysis::validate_description(descriptor)
}

/// `ops` with every reduction coordinate in their indices read from scalar
/// slot `slot` instead.
pub fn bind_reduction(ops: &[Value], slot: usize) -> Vec<Value> {
    match analysis::bind_coordinate(&Value::Array(ops.to_vec()), slot) {
        Value::Array(bound) => bound,
        _ => unreachable!("bind_coordinate keeps an array an array"),
    }
}

/// Emits `ops` into `scope`: loads and stores through `arrays` (indexed by
/// each op's `buffer`), indices over `linear` and `scalars`, every result
/// inserted into `values` by its id (which may hold values defined before).
pub fn emit(
    scope: &Scope,
    ops: &[Value],
    arrays: &[IrValue],
    linear: IrValue,
    scalars: &[IrValue],
    values: &mut HashMap<String, IrValue>,
) -> Result<(), Error> {
    semantic::emit(scope, ops, arrays, linear, scalars, values)
}
