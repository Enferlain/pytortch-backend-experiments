//! RMSNorm weight gradient, grad_w[c] = sum_r g[r,c] * x[r,c] * rstd[r], through Burn's
//! CUDA backend with fusion. Args: rows cols dtype(f32|bf16) mode(check|time) [cuda index].
use burn::prelude::*;
use burn::tensor::{DeviceType, FloatDType, TensorData};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let rows: usize = a[1].parse().unwrap();
    let cols: usize = a[2].parse().unwrap();
    let dtype = if a[3] == "bf16" { FloatDType::BF16 } else { FloatDType::F32 };
    let mode = a[4].as_str();
    let index: usize = a.get(5).map(|s| s.parse().unwrap()).unwrap_or(0);
    let device = Device::enumerate(DeviceType::Cuda).into_vec()[index].clone();

    let make = |shape: [usize; 2], seed: u64| -> Tensor<2> {
        let n = shape[0] * shape[1];
        if mode == "check" {
            // Small integers: every sum is exact in any order.
            let v: Vec<f32> = (0..n as u64)
                .map(|i| ((i.wrapping_mul(2654435761).wrapping_add(seed * 97) >> 7) % 7) as f32 - 3.0)
                .collect();
            Tensor::<2>::from_data(TensorData::new(v, shape), &device).cast(dtype)
        } else {
            // Host-side normal values (Box-Muller over a 64-bit LCG): no generator kernel.
            let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let mut uniform = || {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
            };
            let v: Vec<f32> = (0..n)
                .map(|_| ((-2.0 * uniform().ln()).sqrt() * (std::f64::consts::TAU * uniform()).cos()) as f32)
                .collect();
            Tensor::<2>::from_data(TensorData::new(v, shape), &device).cast(dtype)
        }
    };
    let g = make([rows, cols], 1);
    let x = make([rows, cols], 2);
    let rstd = make([rows, 1], 3);
    let variant = std::env::var("VARIANT").unwrap_or_default();
    let f = || {
        let p = match variant.as_str() {
            "gx" => g.clone().cast(FloatDType::F32) * x.clone().cast(FloatDType::F32),
            "exp" => g.clone().exp(),
            "gg" => g.clone().cast(FloatDType::F32) * g.clone().cast(FloatDType::F32),
            _ => g.clone().cast(FloatDType::F32) * x.clone().cast(FloatDType::F32) * rstd.clone().cast(FloatDType::F32),
        };
        let dim = if variant.ends_with("row") || std::env::var("ROW").is_ok() { 1 } else { 0 };
        p.sum_dim(dim).cast(dtype)
    };
    if mode == "check" {
        let out: Vec<f32> = f().cast(FloatDType::F32).into_data().to_vec().unwrap();
        let host = |t: &Tensor<2>| -> Vec<f32> { t.clone().cast(FloatDType::F32).into_data().to_vec().unwrap() };
        let (gv, xv, rv) = (host(&g), host(&x), host(&rstd));
        let mut bad = 0;
        for c in 0..cols {
            let s: f64 = (0..rows).map(|r| gv[r * cols + c] as f64 * xv[r * cols + c] as f64 * rv[r] as f64).sum();
            let expect = if dtype == FloatDType::BF16 { half::bf16::from_f64(s).to_f64() } else { s };
            if out[c] as f64 != expect {
                bad += 1;
            }
        }
        println!("check {rows}x{cols} {dtype:?}: {bad} mismatches of {cols}");
    } else {
        for _ in 0..20 {
            let _ = f().into_data();
        }
        println!("timed {rows}x{cols} {dtype:?}");
    }
}
