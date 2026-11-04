//! Burn's row sum on inputs written by `probe.py`: for each case, the first
//! call's wall time (compile and autotune included) and the output bytes.
//! CubeCL's autotune log (`CUBECL_DEBUG_LOG=stderr`) names the pick.
//!
//! `burn-probe DIR`: DIR holds `cases.txt` (`name dtype rows cols` lines,
//! dtype `float32` or `bfloat16`) and `name.in`; writes `name.out` and prints
//! `FIRST name ms` and `WARM name ms` (median of 20 calls, wall time with sync).

use burn::prelude::{Device, Tensor};
use burn::tensor::{DType, DeviceIndex, TensorData};
use std::time::Instant;

fn main() {
    let dir = std::path::PathBuf::from(std::env::args().nth(1).expect("DIR"));
    let device = Device::cuda(DeviceIndex::Default);
    // CUDA, the allocator and the runtime warmed on another op, so the first
    // sum below pays for its own compile and tuning only.
    let warm = Tensor::<2>::ones([8, 8], &device) * 2.0;
    warm.into_data();
    let cases = std::fs::read_to_string(dir.join("cases.txt")).expect("cases.txt");
    for line in cases.lines().filter(|l| !l.trim().is_empty()) {
        let fields: Vec<&str> = line.split_whitespace().collect();
        let (name, dtype, rows, cols) = (
            fields[0],
            match fields[1] {
                "float32" => DType::F32,
                "bfloat16" => DType::BF16,
                other => panic!("dtype {other}"),
            },
            fields[2].parse::<usize>().unwrap(),
            fields[3].parse::<usize>().unwrap(),
        );
        let bytes = std::fs::read(dir.join(format!("{name}.in"))).expect("input");
        // The dtype given, or the device's default float type (float32) is taken.
        let x = Tensor::<2>::from_data(
            TensorData::from_bytes_vec(bytes, [rows, cols], dtype),
            (&device, dtype),
        );
        assert_eq!(x.dtype(), dtype);
        x.to_data();
        let start = Instant::now();
        let y = x.clone().sum_dim(1);
        let data = y.into_data();
        println!("FIRST {name} {:.3}", start.elapsed().as_secs_f64() * 1e3);
        std::fs::write(dir.join(format!("{name}.out")), data.as_bytes()).expect("output");
        let mut warm = Vec::new();
        for _ in 0..20 {
            let start = Instant::now();
            x.clone().sum_dim(1).into_data();
            warm.push(start.elapsed().as_secs_f64() * 1e3);
        }
        warm.sort_by(f64::total_cmp);
        println!("WARM {name} {:.3}", warm[10]);
    }
}
