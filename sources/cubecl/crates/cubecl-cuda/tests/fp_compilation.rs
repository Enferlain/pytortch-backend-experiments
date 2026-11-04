//! Floating-point compilation settings reach NVRTC per kernel, and kernels
//! compiled under different settings are different kernels: a division
//! compiled with precise division is correctly rounded, one compiled without
//! is approximate, and neither is served from the other's compilation.
//!
//! Needs the C++ backend (`--features cpp`) and NVRTC's headers (`CUDA_PATH`).

use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_core::{CompilationObserver, FpCompilation};
use cubecl_cuda::CudaRuntime;
use cubecl_server::runtime::Runtime;
use std::sync::{Arc, Mutex};

#[cube(launch)]
fn divide(a: &[f32], b: &[f32], out: &mut [f32]) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = a[ABSOLUTE_POS] / b[ABSOLUTE_POS];
    }
}

#[derive(Default)]
struct Compiled(Mutex<Vec<(FpCompilation, String, String)>>);

impl CompilationObserver for Compiled {
    fn compiled(&self, kernel: &KernelId, source: Option<&str>, binary: &[u8]) {
        self.0.lock().unwrap().push((
            kernel.fp,
            source.unwrap_or_default().to_string(),
            String::from_utf8_lossy(binary).into_owned(),
        ));
    }
}

// The settings are NVRTC options: the LLVM backend refuses them.
#[cfg(feature = "cpp")]
#[test]
fn division_precision_follows_the_setting() {
    let client = CudaRuntime::client(&Default::default());
    let compiled = Arc::new(Compiled::default());
    client
        .set_compilation_observer(Some(compiled.clone()))
        .unwrap();

    let precise = FpCompilation {
        contraction: Some(true),
        precise_division: Some(true),
    };
    let fast = FpCompilation {
        contraction: Some(true),
        precise_division: Some(false),
    };
    let n = 256;
    let a = client.create_from_slice(bytemuck::cast_slice(&vec![1.0f32; n]));
    let b = client.create_from_slice(bytemuck::cast_slice(&vec![3.0f32; n]));
    for fp in [precise, fast, precise] {
        let out = client.empty(n * 4);
        with_fp_compilation(fp, || {
            divide::launch(
                &client,
                CubeCount::new_single(),
                CubeDim::new_1d(n as u32),
                unsafe { BufferArg::from_raw_parts(a.clone(), n) },
                unsafe { BufferArg::from_raw_parts(b.clone(), n) },
                unsafe { BufferArg::from_raw_parts(out.clone(), n) },
            );
        });
        client.read_one(out).unwrap();
    }

    let compiled = compiled.0.lock().unwrap();
    // The second precise launch reused the first's kernel; the fast one did not.
    assert_eq!(compiled.len(), 2, "one compilation per setting");
    let ptx = |fp| {
        let (_, source, ptx) = compiled.iter().find(|(f, ..)| *f == fp).unwrap();
        (source.clone(), ptx.clone())
    };
    let ((precise_source, precise_ptx), (fast_source, fast_ptx)) = (ptx(precise), ptx(fast));
    assert!(precise_source.starts_with("// --fmad=true --prec-div=true\n"));
    assert!(fast_source.starts_with("// --fmad=true --prec-div=false\n"));
    assert!(precise_ptx.contains("div.rn.f32"), "{precise_ptx}");
    assert!(!precise_ptx.contains("div.approx") && !precise_ptx.contains("div.full"));
    assert!(
        fast_ptx.contains("div.approx") || fast_ptx.contains("div.full"),
        "{fast_ptx}"
    );
    assert!(!fast_ptx.contains("div.rn.f32"));
}
