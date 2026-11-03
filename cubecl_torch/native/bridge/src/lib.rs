//! The runtime bridge: a C interface over CubeCL's CUDA client and CubeK, for
//! `cubecl_torch/runtime/bridge.py`.
//!
//! Every call runs on memory the caller owns and on a stream the caller owns,
//! and has reached that stream when it returns: each device's client uses the
//! calling-thread transport, chosen before the client exists. Memory is
//! borrowed for the call: the handle registered for an address stays as a
//! mapping for later calls, but nothing refers to the memory between calls,
//! and keeping it alive on the GPU is the caller's (PyTorch's
//! `record_stream`). CubeCL's own device memory for a stream comes from the
//! allocator the caller installs.
//!
//! Every function returns 0 on success and nonzero on failure, with the
//! reason in [`bridge_last_error`]; none panics across the boundary, and none
//! changes the calling thread's current CUDA context (PyTorch's current
//! device), as a PyTorch operator does not.

use cubecl::cuda::CudaDevice;
use cubecl::device_handle::{Transport, set_transport};
use cubecl::ir::{ElemType, FloatKind};
use cubecl::prelude::*;
use cubecl::server::{Handle, StreamAllocator};
use cubecl::{CompilationObserver, FpCompilation, with_fp_compilation};
use cubecl_common::device::Device as _;
use cubecl_environment::stream::StreamId;
use cubek_reduce::{
    ReduceDtypes, ReduceStrategy,
    components::instructions::ReduceOperationConfig,
    launch::{RoutineStrategy, VectorizationStrategy},
    routines::{
        BlueprintStrategy, CubeBlueprint, cube::CubeStrategy, plane::PlaneStrategy,
        unit::UnitStrategy,
    },
    {BoundChecks, IdleMode},
};
use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::{CString, c_char};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Mutex, OnceLock, RwLock};

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

/// Restores the calling thread's CUDA context when dropped, as PyTorch's
/// `CUDAGuard` restores its device: CubeCL makes its device's context current
/// on the thread that runs its server, which with the calling-thread transport
/// is the caller's.
struct ContextGuard(Option<cudarc::driver::sys::CUcontext>);

impl ContextGuard {
    fn new() -> Self {
        Self(cudarc::driver::result::ctx::get_current().ok().flatten())
    }
}

impl Drop for ContextGuard {
    fn drop(&mut self) {
        let current = cudarc::driver::result::ctx::get_current().ok().flatten();
        if current != self.0 {
            // SAFETY: the context was current on this thread when the call
            // began, and primary contexts PyTorch uses outlive the process's
            // calls; null (none was current) unbinds.
            let _ = unsafe {
                cudarc::driver::result::ctx::set_current(self.0.unwrap_or(std::ptr::null_mut()))
            };
        }
    }
}

/// Runs `f`, turning an error or a panic into a nonzero return and the
/// thread's last error, with the thread's CUDA context as it found it.
fn guarded(f: impl FnOnce() -> Result<(), String>) -> i32 {
    let _context = ContextGuard::new();
    let message = match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(())) => return 0,
        Ok(Err(message)) => message,
        Err(panic) => panic
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_else(|| "panic".into()),
    };
    LAST_ERROR
        .with(|e| *e.borrow_mut() = CString::new(message.replace('\0', " ")).unwrap_or_default());
    1
}

/// The reason the last failing call on this thread failed. Valid until the
/// thread's next failing call.
#[unsafe(no_mangle)]
pub extern "C" fn bridge_last_error() -> *const c_char {
    LAST_ERROR.with(|e| e.borrow().as_ptr())
}

/// Device memory from the caller: `allocate(size, stream, device)` returns a
/// pointer or 0 when out of memory, `free(ptr, stream, device)` returns it.
/// Called on the thread issuing the call that needs the memory.
pub type AllocateFn = extern "C" fn(size: u64, stream: u64, device: u32) -> u64;
pub type FreeFn = extern "C" fn(ptr: u64, stream: u64, device: u32);

#[derive(Clone, Copy)]
struct Callbacks {
    allocate: AllocateFn,
    free: FreeFn,
}

/// The caller's allocator, for one device's streams.
struct CallerAllocator {
    callbacks: Callbacks,
    device: u32,
}

impl StreamAllocator for CallerAllocator {
    fn allocate(&self, size: u64, stream: u64) -> Option<u64> {
        match (self.callbacks.allocate)(size, stream, self.device) {
            0 => None,
            ptr => Some(ptr),
        }
    }

    fn free(&self, ptr: u64, stream: u64) {
        (self.callbacks.free)(ptr, stream, self.device)
    }
}

/// A kernel was compiled: its id (`Debug`), source and PTX.
pub type CompiledFn = extern "C" fn(
    kernel: *const c_char,
    source: *const u8,
    source_len: usize,
    binary: *const u8,
    binary_len: usize,
);

struct CallerObserver(CompiledFn);

impl CompilationObserver for CallerObserver {
    fn compiled(&self, kernel: &KernelId, source: Option<&str>, binary: &[u8]) {
        let kernel = CString::new(format!("{kernel:?}").replace('\0', " ")).unwrap_or_default();
        let source = source.unwrap_or_default();
        (self.0)(
            kernel.as_ptr(),
            source.as_ptr(),
            source.len(),
            binary.as_ptr(),
            binary.len(),
        );
    }
}

/// The most registered handles a device keeps between calls; past it a new
/// address is registered for its call only.
const HANDLE_CACHE: usize = 4096;

/// A device's client, the ids its bound streams were given, and the handles
/// registered for memory it was given, by stream, address and size.
///
/// A handle is only an address mapping: the memory behind an address may be
/// freed and reallocated by its owner, and the handle then names the new
/// allocation, which is what a call with that address means. A handle whose
/// call failed is released, so no failure claim outlives the call.
struct Bridged {
    index: u32,
    client: Client,
    streams: RwLock<HashMap<u64, StreamId>>,
    handles: Mutex<HashMap<(u64, u64, u64), Handle>>,
}

static CALLBACKS: OnceLock<Callbacks> = OnceLock::new();
const MAX_DEVICES: usize = 64;
static DEVICES: [OnceLock<Bridged>; MAX_DEVICES] = [const { OnceLock::new() }; MAX_DEVICES];
static DEVICE_INIT: Mutex<()> = Mutex::new(());

fn device(index: u32) -> Result<&'static Bridged, String> {
    let slot = DEVICES
        .get(index as usize)
        .ok_or_else(|| format!("cuda:{index} is past the bridge's {MAX_DEVICES} devices"))?;
    if let Some(bridged) = slot.get() {
        return Ok(bridged);
    }
    let _init = DEVICE_INIT.lock().map_err(|e| e.to_string())?;
    if let Some(bridged) = slot.get() {
        return Ok(bridged);
    }
    let cuda = CudaDevice::new(index as usize);
    set_transport(cuda.to_id(), Transport::CallingThread).map_err(|e| {
        format!("{e}: CubeCL already serves cuda:{index} in this process on another transport")
    })?;
    let _ = slot.set(Bridged {
        index,
        client: cubecl::Device::Cuda(cuda).client(),
        streams: RwLock::new(HashMap::new()),
        handles: Mutex::new(HashMap::new()),
    });
    Ok(slot.get().expect("set above"))
}

impl Bridged {
    /// The id of `stream`, bound on first use with the caller's allocator.
    fn stream(&self, stream: u64) -> Result<StreamId, String> {
        if let Some(id) = self.streams.read().map_err(|e| e.to_string())?.get(&stream) {
            return Ok(*id);
        }
        let mut streams = self.streams.write().map_err(|e| e.to_string())?;
        if let Some(id) = streams.get(&stream) {
            return Ok(*id);
        }
        let callbacks = *CALLBACKS
            .get()
            .ok_or("no allocator installed: call bridge_set_allocator first")?;
        let allocator: Arc<dyn StreamAllocator> = Arc::new(CallerAllocator {
            callbacks,
            device: self.index,
        });
        // SAFETY: the caller hands only streams that live as long as the
        // process (PyTorch's default and pooled streams).
        let id = unsafe { self.client.bind_stream_with(stream, Some(allocator)) }
            .map_err(|e| e.to_string())?;
        streams.insert(stream, id);
        Ok(id)
    }

    /// The handles for `memory` on `id`, registered as needed, and those
    /// registered for this call only (the cache is full), to release after it.
    fn handles(
        &self,
        id: StreamId,
        memory: &[(u64, u64)],
    ) -> Result<(Vec<Handle>, Vec<Handle>), String> {
        let mut cache = self.handles.lock().map_err(|e| e.to_string())?;
        let (mut handles, mut temporary) = (Vec::with_capacity(memory.len()), Vec::new());
        for &(ptr, bytes) in memory {
            if let Some(handle) = cache.get(&(id.value, ptr, bytes)) {
                handles.push(handle.clone());
                continue;
            }
            // SAFETY: the caller keeps the memory alive until the stream has
            // run the work issued with it.
            let handle = match unsafe { self.client.register_external(ptr, bytes) } {
                Ok(handle) => handle,
                Err(error) => {
                    for handle in temporary {
                        let _ = self.client.unregister_external(handle);
                    }
                    return Err(error.to_string());
                }
            };
            if cache.len() < HANDLE_CACHE {
                cache.insert((id.value, ptr, bytes), handle.clone());
            } else {
                temporary.push(handle.clone());
            }
            handles.push(handle);
        }
        Ok((handles, temporary))
    }

    /// Releases the handles of `memory` on `id`, after a failed call, so no
    /// failure claim outlives it. Another thread's call on the same stream
    /// with the same memory, at the same moment, then fails too.
    fn forget(&self, id: StreamId, memory: &[(u64, u64)]) {
        if let Ok(mut cache) = self.handles.lock() {
            for &(ptr, bytes) in memory {
                if let Some(handle) = cache.remove(&(id.value, ptr, bytes)) {
                    let _ = self.client.unregister_external(handle);
                }
            }
        }
    }

    /// Runs `f` on `stream` with `fp`, borrowing each `(ptr, bytes)` in
    /// `memory` as a handle for the call.
    fn call(
        &self,
        stream: u64,
        fp: FpCompilation,
        memory: &[(u64, u64)],
        f: impl FnOnce(&Client, &[Handle]) -> Result<(), String>,
    ) -> Result<(), String> {
        let id = self.stream(stream)?;
        id.executes(|| {
            let client = &self.client;
            let (handles, temporary) = self.handles(id, memory)?;
            let result = with_fp_compilation(fp, || f(client, &handles))
                // A launch that failed to enqueue claims what it was given
                // instead of returning an error: ask, so it is raised here.
                .and_then(|()| client.check(&handles).map_err(|e| e.to_string()));
            for handle in temporary {
                let _ = client.unregister_external(handle);
            }
            if result.is_err() {
                self.forget(id, memory);
            }
            result
        })
    }
}

/// Installs the allocator every bound stream's own device memory comes from.
/// Once per process, before the first launch.
#[unsafe(no_mangle)]
pub extern "C" fn bridge_set_allocator(allocate: AllocateFn, free: FreeFn) -> i32 {
    guarded(|| {
        CALLBACKS
            .set(Callbacks { allocate, free })
            .map_err(|_| "an allocator is already installed".to_string())
    })
}

/// Creates the client for `cuda:device` on the calling-thread transport.
/// Compiles nothing; later calls create it on first use too.
#[unsafe(no_mangle)]
pub extern "C" fn bridge_init(device_index: u32) -> i32 {
    guarded(|| device(device_index).map(|_| ()))
}

/// Reports every kernel compiled for `cuda:device` from now on to `compiled`.
#[unsafe(no_mangle)]
pub extern "C" fn bridge_set_observer(device_index: u32, compiled: CompiledFn) -> i32 {
    guarded(|| {
        device(device_index)?
            .client
            .set_compilation_observer(Some(Arc::new(CallerObserver(compiled))))
            .map_err(|e| e.to_string())
    })
}

/// The floating-point settings a launch is compiled with: each of
/// `contraction` and `precise_division` is 0 (off), 1 (on) or -1 (NVRTC's
/// default).
fn fp(contraction: i32, precise_division: i32) -> FpCompilation {
    let setting = |value: i32| match value {
        0 => Some(false),
        1 => Some(true),
        _ => None,
    };
    FpCompilation {
        contraction: setting(contraction),
        precise_division: setting(precise_division),
    }
}

#[cube(launch_unchecked)]
fn add_one(input: &[f32], output: &mut [f32]) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = input[ABSOLUTE_POS] + 1.0;
    }
}

/// `output[i] = input[i] + 1` for `n` float32 values, on `stream`: the
/// bridge's own test kernel, for ordering and lifetime checks.
#[unsafe(no_mangle)]
pub extern "C" fn bridge_test_add_one(
    device_index: u32,
    stream: u64,
    input: u64,
    output: u64,
    n: u64,
) -> i32 {
    guarded(|| {
        let bytes = n * 4;
        device(device_index)?.call(
            stream,
            FpCompilation::default(),
            &[(input, bytes), (output, bytes)],
            |client, handles| {
                // SAFETY: both handles cover `n` values, and the kernel
                // bounds its index by the output's length.
                unsafe {
                    add_one::launch_unchecked(
                        client,
                        CubeCount::new_1d(n.div_ceil(256).max(1) as u32),
                        CubeDim::new_1d(256),
                        BufferArg::from_raw_parts(handles[0].clone(), n as usize),
                        BufferArg::from_raw_parts(handles[1].clone(), n as usize),
                    )
                };
                Ok(())
            },
        )
    })
}

/// CubeK's reduce routines, as the caller names them. `forced_planes` > 0
/// forces a cube routine's blueprint to that many planes of the device's plane
/// width per cube, instead of CubeK's inference (at most 8 planes): with planes
/// (code 3) one accumulator per plane, merged by a plane reduction in plane 0;
/// without (code 4) one per unit, merged as a shared-memory tree. Each unit
/// keeps `unit_elements` elements as separate partials (at least one vector).
/// Bounds and idle cubes are masked, which is correct for every shape.
fn routine(
    code: u32,
    forced_planes: u32,
    unit_elements: u32,
    plane_size: u32,
) -> Result<RoutineStrategy, String> {
    if forced_planes == 0 && unit_elements > 1 {
        return Err("unit_elements takes a forced blueprint (forced_planes > 0)".into());
    }
    if forced_planes > 0 {
        let use_planes = match code {
            3 => true,
            4 => false,
            other => return Err(format!("routine {other} takes no forced blueprint")),
        };
        let cube_dim = CubeDim::new_2d(plane_size, forced_planes);
        return Ok(RoutineStrategy::Cube(BlueprintStrategy::Forced((
            CubeBlueprint {
                cube_idle: IdleMode::Mask,
                bound_checks: BoundChecks::Mask,
                num_shared_accumulators: if use_planes {
                    forced_planes as usize
                } else {
                    cube_dim.num_elems() as usize
                },
                use_planes,
                unit_elements: unit_elements.max(1) as usize,
            },
            cube_dim,
        ))));
    }
    Ok(match code {
        0 => RoutineStrategy::Unit(BlueprintStrategy::Inferred(UnitStrategy)),
        1 => RoutineStrategy::Plane(BlueprintStrategy::Inferred(PlaneStrategy {
            independent: true,
        })),
        2 => RoutineStrategy::Plane(BlueprintStrategy::Inferred(PlaneStrategy {
            independent: false,
        })),
        3 => RoutineStrategy::Cube(BlueprintStrategy::Inferred(CubeStrategy {
            use_planes: true,
        })),
        4 => RoutineStrategy::Cube(BlueprintStrategy::Inferred(CubeStrategy {
            use_planes: false,
        })),
        other => return Err(format!("unknown reduce routine {other}")),
    })
}

fn float(code: u32) -> Result<(ElemType, u64), String> {
    Ok(match code {
        0 => (ElemType::Float(FloatKind::F32), 4),
        1 => (ElemType::Float(FloatKind::F16), 2),
        2 => (ElemType::Float(FloatKind::BF16), 2),
        other => return Err(format!("unknown float type {other}")),
    })
}

/// Row sums with CubeK's reduce: `output[r] = sum(input[r, :])` for a
/// contiguous `rows x cols` input, accumulated in float32. `routine` names
/// the CubeK routine (0 unit; 1, 2 plane with independent accumulators or
/// not; 3, 4 cube with planes or not), each with CubeK's inferred blueprint.
/// `forced_planes` (0: CubeK's inference) forces a cube routine's planes per
/// cube, each unit keeping `unit_elements` elements (0 or 1: one vector) as
/// separate partials (see [`routine`]).
/// `parallel_output` (0 or 1) is CubeK's `parallel_output_vectorization`:
/// each unit reduces as many rows as its output vector holds. `input_type`
/// and `output_type` are 0 float32, 1 float16, 2 bfloat16. Both addresses
/// must be aligned to CubeK's widest vector (16 bytes): CubeCL vectorizes by
/// shape and strides alone, as for the memory it allocates itself.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn bridge_sum_rows(
    device_index: u32,
    stream: u64,
    input: u64,
    output: u64,
    rows: u64,
    cols: u64,
    input_type: u32,
    output_type: u32,
    routine_code: u32,
    forced_planes: u32,
    unit_elements: u32,
    parallel_output: i32,
    contraction: i32,
    precise_division: i32,
) -> i32 {
    guarded(|| {
        let (input_elem, input_size) = float(input_type)?;
        let (output_elem, output_size) = float(output_type)?;
        let bridged = device(device_index)?;
        let plane_size = bridged.client.properties().hardware.plane_size_max;
        let strategy = ReduceStrategy {
            routine: routine(routine_code, forced_planes, unit_elements, plane_size)?,
            vectorization: VectorizationStrategy {
                parallel_output_vectorization: parallel_output != 0,
            },
            autotune_level: Default::default(),
        };
        let dtypes = ReduceDtypes {
            input: input_elem,
            output: output_elem,
            accumulation: ElemType::Float(FloatKind::F32),
        };
        let (r, c) = (rows as usize, cols as usize);
        bridged.call(
            stream,
            fp(contraction, precise_division),
            &[
                (input, rows * cols * input_size),
                (output, rows * output_size),
            ],
            |client, handles| {
                // SAFETY: the shapes and strides describe exactly the bytes
                // each handle covers.
                let (input, output) = unsafe {
                    (
                        TensorBinding::from_raw_parts(
                            handles[0].clone(),
                            [c, 1].into(),
                            [r, c].into(),
                        ),
                        TensorBinding::from_raw_parts(
                            handles[1].clone(),
                            [1, 1].into(),
                            [r, 1].into(),
                        ),
                    )
                };
                cubek_reduce::reduce(
                    client,
                    input,
                    output,
                    1,
                    strategy,
                    ReduceOperationConfig::Sum,
                    dtypes,
                )
                .map_err(|e| format!("{e:?}"))
            },
        )
    })
}
