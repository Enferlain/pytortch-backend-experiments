mod packed;
mod wqc;
fn main() -> Result<(), Box<dyn std::error::Error>> { match std::env::args().nth(3).as_deref() { Some("packed") => packed::generate(), Some("wqc") => wqc::generate(), _ => Err("observer mode".into()) } }
