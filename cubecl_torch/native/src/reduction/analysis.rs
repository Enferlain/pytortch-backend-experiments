//! Schema 3 validation and dataflow: binding the reduction coordinate,
//! dependency closures and coordinate/effect queries.
use super::combiner::Combiner;
use crate::semantic::{MAX_RANK, check_buffers, check_index, validate_ops};
use cubecl_core::ir::pliron::value::Value as IrValue;
use serde_json::{Value, json};
use std::collections::{HashMap, HashSet};
type Error = Box<dyn std::error::Error>;

// Physical coordinates are bound only here. A semantic reduction coordinate is
// mapped to an extra emitter value slot, never to a runtime metadata argument.
pub(crate) fn bind_coordinate(v: &Value, slot: usize) -> Value {
    if v["op"] == "reduction" && v.get("dtype").is_none() {
        return json!({"op":"scalar", "slot":slot});
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

pub(crate) fn dependencies(
    id: &str,
    ops: &[Value],
    ready: &HashSet<String>,
    needed: &mut HashSet<String>,
) {
    if ready.contains(id) || !needed.insert(id.into()) {
        return;
    }
    if let Some(op) = ops.iter().find(|op| op["id"] == id) {
        for a in op["args"].as_array().unwrap() {
            dependencies(a.as_str().unwrap(), ops, ready, needed);
        }
        if let Some(result) = op["result"].as_str() {
            dependencies(result, ops, ready, needed);
        }
        // Mask bodies may refer to enclosing SSA values. Local body ids are
        // emitted by the shared lazy-region emitter, not extracted from it.
        fn region(v: &Value, ops: &[Value], ready: &HashSet<String>, needed: &mut HashSet<String>) {
            if let Some(a) = v["args"].as_array() {
                for x in a {
                    dependencies(x.as_str().unwrap(), ops, ready, needed);
                }
            }
            if let Some(result) = v["result"].as_str() {
                dependencies(result, ops, ready, needed);
            }
            if let Some(body) = v["body"].as_array() {
                for x in body {
                    region(x, ops, ready, needed);
                }
            }
        }
        if let Some(body) = op["body"].as_array() {
            for v in body {
                region(v, ops, ready, needed);
            }
        }
    }
}

/// The ops computing `ids`, in descriptor order. Reductions and their
/// components are values in `ready` once reduced, never re-emitted.
pub(crate) fn closure(ids: &[&str], ops: &[Value], ready: &HashMap<String, IrValue>) -> Vec<Value> {
    let mut needed = HashSet::new();
    let ready = ready.keys().cloned().collect();
    for id in ids {
        dependencies(id, ops, &ready, &mut needed);
    }
    ops.iter()
        .filter(|op| {
            !matches!(op["op"].as_str(), Some("reduction" | "reduction_component"))
                && op["id"].as_str().is_some_and(|id| needed.contains(id))
        })
        .cloned()
        .collect()
}

pub(crate) fn operand_ids(op: &Value) -> Vec<&str> {
    op["args"]
        .as_array()
        .unwrap()
        .iter()
        .map(|a| a.as_str().unwrap())
        .collect()
}

pub(crate) fn contains_coordinate(v: &Value) -> bool {
    if v["op"] == "reduction" && v.get("dtype").is_none() {
        return true;
    }
    match v {
        Value::Array(a) => a.iter().any(contains_coordinate),
        Value::Object(o) => o.values().any(contains_coordinate),
        _ => false,
    }
}

pub(crate) fn region_has_effect(v: &Value) -> bool {
    if matches!(v["op"].as_str(), Some("store" | "reduction")) && v.get("dtype").is_some() {
        return true;
    }
    v["body"]
        .as_array()
        .is_some_and(|a| a.iter().any(region_has_effect))
}

pub(crate) fn validate_description(d: &Value) -> Result<(), Error> {
    if d["schema"] != 3 || d["index_dtype"] != "int64" {
        return Err("unsupported reduction schema".into());
    }
    let buffers = d["buffers"].as_array().ok_or("buffers missing")?;
    check_buffers(
        buffers,
        d["ops"].as_array().ok_or("operations missing")?,
        true,
    )?;
    if buffers.len() < 2 {
        return Err("a reduction reads at least one buffer".into());
    }
    let defs = d["scalars"]
        .as_array()
        .ok_or("scalar definitions missing")?;
    if defs.len() > 32 {
        return Err("scalar limit".into());
    }
    for (i, v) in defs.iter().enumerate() {
        if !(v["dtype"] == "int64" || v["dtype"] == "float64" && v["positive"] == false)
            || v["id"] != i
            || !v["positive"].is_boolean()
        {
            return Err("scalar schema".into());
        }
    }
    check_index(&d["outer"], defs, false)?;
    check_index(&d["reduction_extent"], defs, false)?;
    if d["numel"] != d["outer"] {
        return Err("outer logical domain mismatch".into());
    }
    for b in buffers {
        if !matches!(
            b["dtype"].as_str(),
            Some("float32" | "float16" | "bfloat16" | "bool")
        ) {
            return Err("storage contract".into());
        }
        let size = b["size"].as_array().ok_or("size missing")?;
        let stride = b["stride"].as_array().ok_or("stride missing")?;
        if size.len() != stride.len() || size.len() > MAX_RANK {
            return Err("layout rank".into());
        }
        for v in size
            .iter()
            .chain(stride)
            .chain(std::iter::once(&b["offset"]))
        {
            check_index(v, defs, false)?;
        }
    }
    let ops = d["ops"].as_array().ok_or("operations missing")?;
    // Runtime symbols and the private reduction coordinate are distinct even
    // before the physical adapter binds the latter into an emitter argument.
    fn runtime_slots(v: &Value, count: usize) -> Result<(), Error> {
        if v["op"] == "scalar" && v["slot"].as_u64().ok_or("scalar slot")? >= count as u64 {
            return Err(
                "invalid runtime scalar slot; reduction coordinate must be explicit".into(),
            );
        }
        match v {
            Value::Array(a) => {
                for x in a {
                    runtime_slots(x, count)?;
                }
            }
            Value::Object(o) => {
                for x in o.values() {
                    runtime_slots(x, count)?;
                }
            }
            _ => {}
        }
        Ok(())
    }
    runtime_slots(&d["ops"], defs.len())?;
    let mut bound = bind_coordinate(&Value::Array(ops.clone()), defs.len())
        .as_array()
        .unwrap()
        .clone();
    let mut extended_defs = defs.clone();
    extended_defs.push(json!({"dtype":"int64","positive":false}));
    let mut reductions = 0;
    let mut combiners = HashMap::new();
    for op in &mut bound {
        if op["op"] == "reduction" {
            reductions += 1;
            let combiner = Combiner::from_descriptor(op)?;
            if !matches!(combiner.arity(), 1 | 3) {
                return Err("multi-value reduction accumulators unsupported".into());
            }
            let operands = op["args"].as_array().ok_or("operand list")?.len();
            if operands != combiner.operands() {
                return Err(format!(
                    "reduction type {} takes {} operand(s), got {}",
                    combiner.name(),
                    combiner.operands(),
                    operands
                )
                .into());
            }
            combiners.insert(
                op["id"].as_str().ok_or("reduction id")?.to_string(),
                combiner,
            );
            if op["dtype"] != "float32"
                || !matches!(
                    op["source_dtype"].as_str(),
                    Some("float32" | "float16" | "bfloat16")
                )
            {
                return Err("unsupported reduction operator/accumulator".into());
            }
            if !matches!(
                op["semantic_dtype"].as_str(),
                Some("float32" | "float16" | "bfloat16")
            ) {
                return Err("reduction result semantic dtype".into());
            }
            // Type-check the same SSA edge as a float32 identity; no IR is built
            // until every scalar op and every reduction record is validated.
            // Every operand is checked as an explicit float32 conversion below.
            op["op"] = "cast".into();
            op["args"] = json!([op["args"][0]]);
        } else if op["op"] == "reduction_component" {
            // One component of an earlier multi-value reduction, typed float32.
            let source = op["args"]
                .as_array()
                .filter(|a| a.len() == 1)
                .and_then(|a| a[0].as_str())
                .ok_or("reduction component requires one reduction operand")?;
            let combiner = combiners
                .get(source)
                .ok_or("reduction component of an unknown or later reduction")?;
            let index = op["index"].as_u64().ok_or("reduction component index")?;
            if combiner.arity() == 1 || index >= combiner.arity() as u64 {
                return Err(format!(
                    "reduction type {} has no component {index}",
                    combiner.name()
                )
                .into());
            }
            if op["dtype"] != "float32" || op["semantic_dtype"] != "float32" {
                return Err("reduction component dtype".into());
            }
            op["op"] = "cast".into();
        } else if op["op"] == "store"
            && !matches!(op["iteration_scope"].as_str(), Some("outer" | "reduction"))
        {
            return Err("store iteration scope missing".into());
        }
        if let Some(body) = op["body"].as_array()
            && body.iter().any(region_has_effect)
        {
            return Err("masked reduction/effect unsupported".into());
        }
    }
    if !(1..=4).contains(&reductions) {
        return Err("one to four reductions per cooperative schedule required".into());
    }
    let mut types = HashMap::new();
    for (original, checked) in ops.iter().zip(&bound) {
        if original["op"] == "reduction" {
            for argument in original["args"].as_array().ok_or("operand list")? {
                let argument = argument.as_str().ok_or("reduction operand missing")?;
                if types.get(argument).map(String::as_str) != Some("float32") {
                    return Err("reduction requires explicit float32 input conversion".into());
                }
            }
        }
        validate_ops(
            std::slice::from_ref(checked),
            buffers,
            &extended_defs,
            &mut types,
            0,
        )?;
    }
    let reduced_ids: HashSet<_> = ops
        .iter()
        .filter(|op| matches!(op["op"].as_str(), Some("reduction" | "reduction_component")))
        .map(|op| op["id"].as_str().unwrap().to_string())
        .collect();
    for store in ops
        .iter()
        .filter(|op| op["op"] == "store" && op["iteration_scope"] == "outer")
    {
        let mut needed = HashSet::new();
        dependencies(
            store["args"][0].as_str().unwrap(),
            ops,
            &reduced_ids,
            &mut needed,
        );
        if contains_coordinate(&store["index"])
            || ops.iter().any(|op| {
                op["id"].as_str().is_some_and(|id| needed.contains(id)) && contains_coordinate(op)
            })
        {
            return Err("outer store depends on a varying reduction coordinate".into());
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The fixture's bf16 row sum: load, cast, reduce, round, store.
    fn sum_descriptor() -> Value {
        json!({"schema": 3, "index_dtype": "int64",
            "buffers": [
                {"access": "read", "dtype": "bfloat16", "offset": {"op": "constant", "value": 0},
                 "size": [{"op": "scalar", "slot": 1}, {"op": "scalar", "slot": 0}],
                 "stride": [{"op": "scalar", "slot": 0}, {"op": "constant", "value": 1}]},
                {"access": "write", "dtype": "bfloat16", "offset": {"op": "constant", "value": 0},
                 "size": [{"op": "scalar", "slot": 0}], "stride": [{"op": "constant", "value": 1}]}],
            "numel": {"op": "scalar", "slot": 0}, "outer": {"op": "scalar", "slot": 0},
            "reduction_extent": {"op": "scalar", "slot": 1},
            "scalars": [{"dtype": "int64", "id": 0, "positive": true},
                        {"dtype": "int64", "id": 1, "positive": true}],
            "ops": [
                {"op": "load", "id": "tmp0", "args": [], "buffer": 0, "dtype": "bfloat16",
                 "semantic_dtype": "bfloat16",
                 "index": {"op": "add", "args": [{"op": "linear"}, {"op": "mul",
                     "args": [{"op": "reduction"}, {"op": "scalar", "slot": 0}]}]}},
                {"op": "cast", "id": "tmp1", "args": ["tmp0"], "dtype": "float32",
                 "semantic_dtype": "bfloat16"},
                {"op": "reduction", "id": "tmp2", "args": ["tmp1"], "dtype": "float32",
                 "semantic_dtype": "bfloat16", "source_dtype": "bfloat16",
                 "accumulator_dtype": "float32", "reduction_type": "sum"},
                {"op": "cast", "id": "tmp3", "args": ["tmp2"], "dtype": "bfloat16",
                 "semantic_dtype": "bfloat16"},
                {"op": "store", "args": ["tmp3"], "buffer": 1, "dtype": "bfloat16",
                 "semantic_dtype": "bfloat16", "index": {"op": "linear"},
                 "iteration_scope": "outer"}]})
    }

    #[test]
    fn sum_descriptor_is_valid() {
        validate_description(&sum_descriptor()).unwrap();
    }

    #[test]
    fn reduction_operand_count_follows_its_type() {
        let mut d = sum_descriptor();
        d["ops"][2]["args"] = json!(["tmp1", "tmp1"]);
        let error = validate_description(&d).unwrap_err().to_string();
        assert_eq!(error, "reduction type sum takes 1 operand(s), got 2");
        d["ops"][2]["args"] = json!([]);
        let error = validate_description(&d).unwrap_err().to_string();
        assert_eq!(error, "reduction type sum takes 1 operand(s), got 0");
    }

    #[test]
    fn single_value_reductions_have_no_components() {
        let component = |args: Value, index: u64| {
            let mut d = sum_descriptor();
            let ops = d["ops"].as_array_mut().unwrap();
            ops.insert(
                3,
                json!({"op": "reduction_component", "id": "tmp9", "args": args,
                "index": index, "dtype": "float32", "semantic_dtype": "float32"}),
            );
            validate_description(&d).unwrap_err().to_string()
        };
        assert_eq!(
            component(json!(["tmp2"]), 0),
            "reduction type sum has no component 0"
        );
        assert_eq!(
            component(json!(["tmp1"]), 0),
            "reduction component of an unknown or later reduction"
        );
        assert_eq!(
            component(json!(["tmp2", "tmp2"]), 0),
            "reduction component requires one reduction operand"
        );
    }
}
