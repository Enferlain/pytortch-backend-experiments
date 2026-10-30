//! sk8 check: an Inductor loop body inside cubecl-tile loops, emitted by our own emitter.
//!
//! `fused_probe <out dir> <semantic.json> <rows> <cols> [reps] [width]`
#[allow(dead_code)]
#[path = "../../../../../cubecl_torch/native/src/ir.rs"]
mod ir;
mod probe;
#[allow(dead_code)]
#[path = "../../../../../cubecl_torch/native/src/semantic.rs"]
mod semantic;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    probe::run(&std::env::args().collect::<Vec<_>>())
}
