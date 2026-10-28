//! Physical adapter: validates the physical plan and dispatches to the
//! reduction, pointwise or column builder. Semantic operations remain the
//! validated schema 2-4 DAG.
use crate::reduction::reduction_coefficient;
use crate::{effects, pointwise, reduction};
use serde_json::{Value, json};
type Error = Box<dyn std::error::Error>;

pub fn build(d: &Value) -> Result<cubecl_runtime::kernel::KernelDefinition, Error> {
    let entry = d["entrypoint"].as_str().ok_or("entrypoint")?;
    if !entry.starts_with("cubecl5_")
        || !entry.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
    {
        return Err("invalid entrypoint".into());
    }
    let schema = d["semantic_schema"].as_u64().ok_or("semantic schema")?;
    let mut sem = d.clone();
    sem["schema"] = schema.into();
    match schema {
        3 => reduction::validate_description(&sem)?,
        4 => effects::validate_description(&sem)?,
        _ => return Err("unsupported semantic schema".into()),
    }
    let p = &d["physical_plan"];
    if p["version"] != 2 {
        return Err("physical plan version".into());
    }
    let strategy = p["strategy"].as_str().ok_or("physical strategy")?;
    if strategy == "bounded_column_pointwise" {
        let expected = json!({"version":2,"strategy":"bounded_column_pointwise","workgroup":[256,1,1],"vector_width":1,"elements_per_unit":1,"outputs_per_workgroup":1,"plane_width":0});
        if p != &expected {
            return Err("qualified column physical plan".into());
        }
        let mut original = sem.clone();
        for key in ["semantic_schema", "physical_plan", "entrypoint"] {
            original.as_object_mut().unwrap().remove(key);
        }
        return pointwise::column::build(&original, entry);
    }
    let accumulators = match p.get("private_accumulators") {
        None => 1,
        Some(value) => value.as_u64().ok_or("private accumulator count")?,
    };
    let independent = strategy == "shared_tree_multi_accumulator";
    // Welford rows may run independent planes with vector chains: several
    // accumulators per lane, each reading whole vectors (welford-reductions 7).
    // Only when every reduction is Welford: a fused sum must keep its
    // qualified plans.
    let welford = d["ops"].as_array().is_some_and(|ops| {
        let mut kinds = ops
            .iter()
            .filter(|op| op["op"] == "reduction")
            .map(|op| op["reduction_type"].as_str().unwrap_or(""))
            .peekable();
        kinds.peek().is_some() && kinds.all(|t| t.starts_with("welford"))
    });
    let vector_chains = strategy == "independent_planes"
        && welford
        && p["vector_width"].as_u64().is_some_and(|w| w > 1);
    let reference512 = strategy == "plane_shared_reference512";
    let batched = strategy == "plane_shared_load_batch" || reference512;
    let combined = (strategy == "plane_shared" || batched) && accumulators == 4;
    let private_merge = match p.get("private_merge") {
        None => "halving",
        Some(value) => value.as_str().ok_or("private merge order")?,
    };
    if private_merge != "halving" && !(private_merge == "adjacent" && batched && accumulators == 4)
    {
        return Err("unsupported private merge order".into());
    }
    if batched && accumulators != 4 {
        return Err("load-batch plan requires exactly four accumulators".into());
    }
    if (independent && !matches!(accumulators, 2 | 4))
        || (vector_chains && !matches!(accumulators, 1 | 2 | 4 | 8 | 16))
        || (!independent && !combined && !vector_chains && accumulators != 1)
    {
        return Err("unsupported private accumulator policy".into());
    }
    if !matches!(
        strategy,
        "plane_shared"
            | "plane_shared_load_batch"
            | "plane_shared_reference512"
            | "independent_planes"
            | "independent_planes_final9"
    ) && p["plane_width"] != 0
    {
        return Err("non-plane physical plan must have plane_width=0".into());
    }
    let dims = p["workgroup"].as_array().ok_or("workgroup")?;
    if dims.len() != 3 {
        return Err("workgroup rank".into());
    }
    let mut workgroup = [0u32; 3];
    for (i, v) in dims.iter().enumerate() {
        workgroup[i] = u32::try_from(v.as_u64().ok_or("workgroup dimension")?)?;
    }
    let width = p["vector_width"].as_u64().ok_or("vector width")? as usize;
    let elements = p["elements_per_unit"].as_u64().ok_or("elements per unit")?;
    let outputs = p["outputs_per_workgroup"]
        .as_u64()
        .ok_or("outputs per workgroup")?;
    if elements != 1
        && !(strategy == "scalar_pointwise" && workgroup == [128, 1, 1] && elements == 4)
    {
        return Err("elements mapping".into());
    }
    if schema == 3 {
        if !matches!(
            strategy,
            "shared_tree_reduction_baseline"
                | "shared_tree_multi_accumulator"
                | "shared_tree_compensated"
                | "shared_tree"
                | "plane_shared"
                | "plane_shared_load_batch"
                | "plane_shared_reference512"
                | "independent_planes"
                | "independent_planes_final9"
                | "coalesced_columns"
        ) {
            return Err("reduction physical strategy".into());
        }
        let legal = match strategy {
            "shared_tree_reduction_baseline" => {
                workgroup == [256, 1, 1] && outputs == 1 && width == 1
            }
            "shared_tree_multi_accumulator" => {
                workgroup == [256, 1, 1] && outputs == 1 && width == 1
            }
            "shared_tree_compensated" => workgroup == [256, 1, 1] && outputs == 1 && width == 1,
            "shared_tree" => {
                matches!(workgroup, [64, 1, 1] | [128, 1, 1]) && outputs == 1 && width == 1
            }
            "plane_shared" | "plane_shared_load_batch" => {
                matches!(workgroup, [64, 1, 1] | [128, 1, 1] | [256, 1, 1])
                    && outputs == 1
                    && (width == 1 || (workgroup[0] != 64 && matches!(width, 2 | 4 | 8)))
            }
            "plane_shared_reference512" => {
                workgroup == [512, 1, 1]
                    && outputs == 1
                    && width == 1
                    && accumulators == 4
                    && private_merge == "adjacent"
            }
            "independent_planes" => {
                matches!(workgroup, [32, 4, 1] | [32, 8, 1])
                    && outputs == workgroup[1] as u64
                    && (width == 1 || (vector_chains && matches!(width, 2 | 4 | 8)))
            }
            "independent_planes_final9" => workgroup == [32, 4, 1] && outputs == 4 && width == 1,
            "coalesced_columns" => {
                matches!(workgroup, [32, 4, 1] | [32, 8, 1]) && outputs == 32 && width == 1
            }
            _ => false,
        };
        if !legal || elements != 1 {
            return Err("reduction dimensions".into());
        }
        if combined
            && (workgroup != [if reference512 { 512 } else { 256 }, 1, 1]
                || width != 1
                || outputs != 1)
        {
            return Err("combined private/plane topology".into());
        }
        if matches!(
            strategy,
            "plane_shared"
                | "plane_shared_load_batch"
                | "plane_shared_reference512"
                | "independent_planes"
                | "independent_planes_final9"
        ) && p["plane_width"] != 32
        {
            return Err("plane width must match queried CUDA32 target".into());
        }
        if strategy == "shared_tree_reduction_baseline"
            && (workgroup != [256, 1, 1] || width != 1 || outputs != 1)
        {
            return Err("frozen tree plan".into());
        }
        if width > 1 {
            for op in sem["ops"].as_array().unwrap() {
                if op["op"] == "masked" {
                    return Err("reduction vector lazy region unsupported".into());
                }
                if op["op"] == "load" {
                    let ty = op["dtype"].as_str().unwrap();
                    if !matches!(ty, "float32" | "float16" | "bfloat16")
                        || width * if ty == "float32" { 4 } else { 2 } > 16
                    {
                        return Err("reduction vector storage width".into());
                    }
                    if !matches!(reduction_coefficient(&op["index"], None), Some(0 | 1)) {
                        return Err("reduction vector addressing not proven unit stride".into());
                    }
                }
            }
        }
        if strategy == "independent_planes_final9" {
            validate_final9(&sem)?;
            // Reuse the qualified probe emitter, with a distinct normal plan
            // identity. Generic FSum and all partial emitters stay unchanged.
            return reduction::build_physical_codegen(
                &sem,
                workgroup,
                "independent_planes",
                width,
                accumulators as usize,
                private_merge,
                reduction::CodegenOptions {
                    final_plane: true,
                    ..Default::default()
                },
            );
        }
        return reduction::build_physical(
            &sem,
            workgroup,
            strategy,
            width,
            accumulators as usize,
            private_merge,
        );
    }
    if outputs != 1
        || workgroup[1..] != [1, 1]
        || !matches!(workgroup[0], 64 | 128 | 256)
        || !matches!(
            strategy,
            "scalar_pointwise_baseline" | "scalar_pointwise" | "vector_pointwise"
        )
    {
        return Err("pointwise physical topology".into());
    }
    if strategy == "scalar_pointwise_baseline"
        && (workgroup != [256, 1, 1] || width != 1 || elements != 1)
    {
        return Err("frozen scalar plan".into());
    }
    if !matches!(width, 1 | 2 | 4 | 8)
        || (width != 1 && strategy != "vector_pointwise")
        || (strategy == "vector_pointwise" && width == 1)
    {
        return Err("vector plan".into());
    }
    if width == 8 && workgroup[0] == 64 {
        return Err("uncatalogued vector plan".into());
    }
    if strategy == "scalar_pointwise" && workgroup[0] == 256 {
        return Err("use the frozen named scalar256 plan".into());
    }
    let ops = sem["ops"].as_array().unwrap();
    if width > 1 {
        if sem["effects"]["rng"] != Value::Null || sem["effects"]["mutation"] != false {
            return Err("effectful vectors unsupported".into());
        }
        for op in ops {
            if matches!(
                op["op"].as_str(),
                Some("masked" | "indirect" | "atomic_add" | "rand" | "load_seed")
            ) {
                return Err("vector lazy/effect region unsupported".into());
            }
            if matches!(op["op"].as_str(), Some("load" | "store")) {
                let scalar_broadcast = op["op"] == "load" && op["index"]["op"] == "constant";
                if !scalar_broadcast && op["index"]["op"] != "linear" {
                    return Err("vector addressing requires proven direct linear access".into());
                }
                // 16-byte transactions at most; int64 vectors have two lanes.
                let bytes = match op["dtype"].as_str().unwrap() {
                    "float32" => 4,
                    "float16" | "bfloat16" => 2,
                    "int64" => 8,
                    _ => return Err("vector storage width".into()),
                };
                if width * bytes > 16 {
                    return Err("vector storage width".into());
                }
            }
        }
    }
    pointwise::physical::build(d, &sem, workgroup, width, elements, schema)
}

/// Native enforcement of stage5.final_plan's intentionally narrow contract.
fn validate_final9(d: &Value) -> Result<(), Error> {
    let c = |n: i64| json!({"op": "constant", "value": n});
    let outer = json!({"op": "scalar", "slot": 0});
    if d["schema"] != 3
        || d["index_dtype"] != "int64"
        || d["scalars"] != json!([{"dtype": "int64", "id": 0, "positive": true}])
        || d["outer"] != outer
        || d["numel"] != outer
        || d["reduction_extent"] != c(9)
        || d.get("effects").is_some_and(|e| {
            e.as_object()
                .is_none_or(|e| e.values().any(|v| v != &json!(false)))
        })
    {
        return Err("qualified final requires pure constant9 FP32 sum with scalar outer".into());
    }
    let ops = d["ops"].as_array().ok_or("final ops")?;
    let buffers = d["buffers"].as_array().ok_or("final buffers")?;
    if buffers.len() != 2 {
        return Err("qualified final requires exactly two buffers".into());
    }
    let ty = buffers[1]["dtype"].as_str().ok_or("final output dtype")?;
    let expected_buffers = json!([
        {"access": "read", "dtype": "float32", "offset": c(0),
         "size": [outer, c(9)], "stride": [c(9), c(1)]},
        {"access": "write", "dtype": ty, "offset": c(0),
         "size": [outer], "stride": [c(1)]}
    ]);
    if !matches!(ty, "float32" | "float16" | "bfloat16") || d["buffers"] != expected_buffers {
        return Err(
            "qualified final requires contiguous FP32 partials and supported output storage".into(),
        );
    }
    if ops.len() != if ty == "float32" { 3 } else { 7 } {
        return Err(
            "qualified final requires the qualified load, sum, cast and owner sequence".into(),
        );
    }
    let ids = ops[..ops.len() - 1]
        .iter()
        .map(|o| o["id"].as_str().ok_or("final SSA id"))
        .collect::<Result<Vec<_>, _>>()?;
    if ids.iter().collect::<std::collections::HashSet<_>>().len() != ids.len() {
        return Err("qualified final requires unique SSA values".into());
    }
    let load = ids[0];
    let total = ids[if ty == "float32" { 1 } else { 2 }];
    let mut expected_ops = vec![json!(
        {"op": "load", "id": load, "args": [], "buffer": 0,
         "dtype": "float32", "semantic_dtype": "float32",
         "index": {"op": "add", "args": [{"op": "reduction"},
                   {"op": "mul", "args": [c(9), {"op": "linear"}]}]}}
    )];
    if ty != "float32" {
        expected_ops.push(json!({"op": "cast", "id": ids[1], "args": [load],
                                "dtype": "float32", "semantic_dtype": ty}));
    }
    expected_ops.push(json!({"op": "reduction", "id": total,
        "args": [if ty == "float32" { load } else { ids[1] }],
        "dtype": "float32", "semantic_dtype": ty,
        "accumulator_dtype": "float32", "source_dtype": ty,
        "reduction_type": "sum"}));
    if ty != "float32" {
        for (i, cast_type) in [(3, ty), (4, "float32"), (5, ty)] {
            expected_ops.push(json!({"op": "cast", "id": ids[i], "args": [ids[i-1]],
                                    "dtype": cast_type, "semantic_dtype": ty}));
        }
    }
    expected_ops.push(json!(
        {"op": "store", "args": [ids[ids.len()-1]], "buffer": 1,
         "dtype": ty, "semantic_dtype": ty, "index": {"op": "linear"},
         "iteration_scope": "outer"}
    ));
    if d["ops"] != json!(expected_ops) {
        return Err(
            "qualified final requires direct FP32 sum and unmodified outer owner store".into(),
        );
    }
    Ok(())
}
