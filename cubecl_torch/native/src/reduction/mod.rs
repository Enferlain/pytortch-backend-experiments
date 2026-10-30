//! Semantic reduction DAG + independently validated cooperative physical plan.
//! No handwritten target source: all loops, shared accesses and barriers are IR.
mod analysis;
mod combiner;
mod emit;
mod tile;
mod vector;

pub(crate) use analysis::validate_description;
pub(crate) use emit::{CodegenOptions, build, build_physical, build_physical_codegen};
pub(crate) use vector::reduction_coefficient;
