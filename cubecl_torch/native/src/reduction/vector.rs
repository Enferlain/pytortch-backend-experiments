//! Vector loads for reduction plans, and the unit-stride proof they rely on.
use crate::ir::{constant, pointer, vector_pointer};
use crate::semantic;
use cubecl_core::ir::{
    Scope,
    dialect::{math::IAddOp, memory::LoadOp, vector::CompositeExtractOp},
    pliron::value::Value as IrValue,
};
use serde_json::{Value, json};
use std::collections::HashMap;
type Error = Box<dyn std::error::Error>;

// Vector memory + scalar semantic replay preserves every low-precision barrier.
// The physical adapter supplies a reduction coordinate; no IR handle is encoded
// in the semantic description. Uniform loads stay scalar broadcasts.
pub(crate) fn reduction_vector_values(
    s: &Scope,
    ops: &[Value],
    arrays: &[IrValue],
    outer: IrValue,
    args: &[IrValue],
    ready: &HashMap<String, IrValue>,
    request: (&[&str], usize),
) -> Result<Vec<Vec<IrValue>>, Error> {
    let (result, width) = request;
    let slot = args.len() - 1;
    let mut loaded = HashMap::new();
    for op in ops.iter().filter(|v| v["op"] == "load") {
        let coordinate = semantic::index(s, &op["index"], outer, args)?;
        let ptr = pointer(
            s,
            arrays[op["buffer"].as_u64().unwrap() as usize],
            coordinate,
        );
        let vector = reduction_coefficient(&op["index"], Some(slot))
            .ok_or("vector reduction-axis coefficient")?
            == 1;
        let ptr = if vector {
            vector_pointer(s, ptr, op["dtype"].as_str().unwrap(), width)?
        } else {
            ptr
        };
        let load = LoadOp::new(s.ctx_mut(), ptr);
        if op["streaming"] == true {
            load.set_streaming(s.ctx());
        } else if op["retained"] == true {
            load.set_retained(s.ctx());
        }
        loaded.insert(
            op["id"].as_str().unwrap(),
            (s.register_with_result(&load), vector),
        );
    }
    let mut results = Vec::new();
    for lane in 0..width {
        let mut values = ready.clone();
        let offset = constant(s, &json!(lane), "int64")?;
        let coordinate = s.register_with_result(&IAddOp::new(s.ctx_mut(), args[slot], offset));
        let mut args = args.to_vec();
        args[slot] = coordinate;
        for op in ops {
            if op["op"] == "load" {
                let (value, vector) = loaded[op["id"].as_str().unwrap()];
                let value = if vector {
                    s.register_with_result(&CompositeExtractOp::new(s.ctx_mut(), value, lane))
                } else {
                    value
                };
                values.insert(op["id"].as_str().unwrap().into(), value);
            } else {
                semantic::emit(
                    s,
                    std::slice::from_ref(op),
                    arrays,
                    outer,
                    &args,
                    &mut values,
                )?;
            }
        }
        results.push(result.iter().map(|id| values[*id]).collect());
    }
    Ok(results)
}

pub(crate) fn reduction_coefficient(v: &Value, bound_slot: Option<usize>) -> Option<i64> {
    match v["op"].as_str()? {
        "reduction" => Some(1),
        "scalar" if bound_slot.is_some_and(|slot| v["slot"] == slot) => Some(1),
        "linear" | "scalar" | "constant" => Some(0),
        "add" => v["args"].as_array()?.iter().try_fold(0i64, |n, x| {
            n.checked_add(reduction_coefficient(x, bound_slot)?)
        }),
        "mul" => {
            let args = v["args"].as_array()?;
            let mut coefficient = 0i64;
            for (i, x) in args.iter().enumerate() {
                let c = reduction_coefficient(x, bound_slot)?;
                if c != 0 {
                    if coefficient != 0 {
                        return None;
                    }
                    coefficient = args.iter().enumerate().filter(|(j, _)| *j != i).try_fold(
                        c,
                        |n, (_, a)| {
                            if a["op"] != "constant" {
                                return None;
                            }
                            n.checked_mul(a["value"].as_i64()?)
                        },
                    )?;
                }
            }
            Some(coefficient)
        }
        _ => None,
    }
}
