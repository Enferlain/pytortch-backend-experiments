//! Bridge prototype (bead zah.2): a C interface that launches CubeCL and CubeK kernels
//! through CubeCL's CUDA client on memory and a stream PyTorch owns.
//!
//! Each call takes PyTorch's raw stream; the kernel is on that stream when the call
//! returns (the worktree's device handle runs server calls on the calling thread).

use cubecl::cuda::external;
use cubecl::prelude::*;
use cubecl::server::Handle;
use cubek_reduce::{
    ReduceDtypes, ReduceStrategy,
    components::instructions::ReduceOperationConfig,
    launch::{RoutineStrategy, VectorizationStrategy},
    routines::{BlueprintStrategy, plane::PlaneStrategy},
};
use std::sync::OnceLock;

static CLIENT: OnceLock<Client> = OnceLock::new();

fn client() -> &'static Client {
    CLIENT.get_or_init(|| cubecl::Device::Cuda(Default::default()).client())
}

/// Borrow `size` bytes at `ptr` (a PyTorch allocation) for one call.
fn borrow(ptr: u64, size: u64) -> Handle {
    unsafe { client().register_external(ptr, size) }.expect("register external memory")
}

#[cube(launch_unchecked)]
fn touch(output: &mut [f32]) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = output[ABSOLUTE_POS] + 1.0;
    }
}

/// Initialize the client (compiles nothing). Returns 0.
#[unsafe(no_mangle)]
pub extern "C" fn bridge_init() -> i32 {
    client();
    0
}

/// `output[i] += 1` for `n` f32 values at `ptr`, on `stream`.
#[unsafe(no_mangle)]
pub extern "C" fn bridge_touch(ptr: u64, n: u64, stream: u64) -> i32 {
    external::on_stream(stream, || {
        let out = borrow(ptr, n * 4);
        unsafe {
            touch::launch_unchecked(
                client(),
                CubeCount::Static(n.div_ceil(256) as u32, 1, 1),
                CubeDim::new_1d(256),
                BufferArg::from_raw_parts(out, n as usize),
            )
        };
    });
    0
}

/// Row sums: `output[r] = sum(input[r, :])` for a contiguous f32 `rows x cols` input, with
/// CubeK's plane routine and its inferred blueprint.
#[unsafe(no_mangle)]
pub extern "C" fn bridge_sum_rows(input: u64, output: u64, rows: u64, cols: u64, stream: u64) -> i32 {
    external::on_stream(stream, || {
        let input = borrow(input, rows * cols * 4);
        let output = borrow(output, rows * 4);
        let (rows, cols) = (rows as usize, cols as usize);
        let input = unsafe { TensorBinding::from_raw_parts(input, [cols, 1].into(), [rows, cols].into()) };
        let output = unsafe { TensorBinding::from_raw_parts(output, [1, 1].into(), [rows, 1].into()) };
        let strategy = ReduceStrategy {
            routine: RoutineStrategy::Plane(BlueprintStrategy::Inferred(PlaneStrategy { independent: true })),
            vectorization: VectorizationStrategy { parallel_output_vectorization: false },
            autotune_level: Default::default(),
        };
        let f32_ = cubecl::ir::ElemType::Float(cubecl::ir::FloatKind::F32);
        let dtypes = ReduceDtypes { input: f32_, output: f32_, accumulation: f32_ };
        match cubek_reduce::reduce(client(), input, output, 1, strategy, ReduceOperationConfig::Sum, dtypes) {
            Ok(()) => 0,
            Err(e) => {
                eprintln!("reduce: {e:?}");
                1
            }
        }
    })
}

static HANDLES: std::sync::Mutex<Vec<Handle>> = std::sync::Mutex::new(Vec::new());

/// Register `size` bytes at `ptr` once; returns an index for [`bridge_touch_handle`].
#[unsafe(no_mangle)]
pub extern "C" fn bridge_register(ptr: u64, size: u64) -> u64 {
    let mut handles = HANDLES.lock().unwrap();
    handles.push(borrow(ptr, size));
    (handles.len() - 1) as u64
}

/// [`bridge_touch`] on a handle registered once, to separate registration from launch.
#[unsafe(no_mangle)]
pub extern "C" fn bridge_touch_handle(index: u64, n: u64, stream: u64) -> i32 {
    let out = HANDLES.lock().unwrap()[index as usize].clone();
    external::on_stream(stream, || unsafe {
        touch::launch_unchecked(
            client(),
            CubeCount::Static(n.div_ceil(256) as u32, 1, 1),
            CubeDim::new_1d(256),
            BufferArg::from_raw_parts(out, n as usize),
        )
    });
    0
}
