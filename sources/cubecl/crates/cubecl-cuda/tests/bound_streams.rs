//! Running on streams and memory a caller owns: with the calling-thread
//! transport, a launch issued under a bound stream's id has reached that stream
//! when it returns, so the caller's own work on the stream, before and after
//! it, is ordered around it with no synchronization through the client.
//!
//! The caller's side uses the driver directly: its own streams, allocations,
//! copies and synchronizations, none of which go through `CubeCL`.

use cubecl_common::device::Device;
use cubecl_common::device_handle::{Transport, set_transport};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_cuda::{CudaDevice, CudaRuntime};
use cubecl_server::runtime::Runtime;
use cudarc::driver::result;
use cudarc::driver::sys::CUstream;
use std::sync::OnceLock;

#[cube(launch)]
fn add_one(input: &[u32], out: &mut [u32]) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = input[ABSOLUTE_POS] + 1u32;
    }
}

const N: usize = 1 << 20;

/// The client, on the calling-thread transport chosen before it exists, and
/// the driver context the caller's own work runs in.
fn setup() -> (Client, std::sync::Arc<cudarc::driver::CudaContext>) {
    static CONTEXT: OnceLock<std::sync::Arc<cudarc::driver::CudaContext>> = OnceLock::new();
    let context = CONTEXT
        .get_or_init(|| {
            set_transport(CudaDevice::default().to_id(), Transport::CallingThread).unwrap();
            cudarc::driver::CudaContext::new(0).unwrap()
        })
        .clone();
    context.bind_to_thread().unwrap();
    (CudaRuntime::client(&CudaDevice::default()), context)
}

struct Owned {
    stream: CUstream,
    input: u64,
    output: u64,
}

impl Owned {
    fn new() -> Self {
        let stream = result::stream::create(result::stream::StreamKind::NonBlocking).unwrap();
        // SAFETY: a context is current (`setup`); sizes are nonzero.
        unsafe {
            Self {
                stream,
                input: result::malloc_sync(N * 4).unwrap(),
                output: result::malloc_sync(N * 4).unwrap(),
            }
        }
    }

    fn write(&self, ptr: u64, values: &[u32]) {
        // SAFETY: `values` outlives the copy: `read`/`synchronize` below.
        unsafe { result::memcpy_htod_async(ptr, values, self.stream).unwrap() }
    }

    /// The stream's result, by the caller's own synchronization.
    fn read(&self, ptr: u64) -> Vec<u32> {
        let mut out = vec![0u32; N];
        // SAFETY: `out` lives until the stream is synchronized.
        unsafe {
            result::memcpy_dtoh_async(&mut out, ptr, self.stream).unwrap();
            result::stream::synchronize(self.stream).unwrap();
        }
        out
    }

    /// Destroys the stream and frees the memory: errors if `CubeCL` had
    /// destroyed or freed any of them.
    fn release(self) {
        // SAFETY: nothing is queued on the stream any more (`read` synchronized).
        unsafe {
            result::stream::destroy(self.stream).unwrap();
            result::free_sync(self.input).unwrap();
            result::free_sync(self.output).unwrap();
        }
    }
}

#[test]
fn launches_reach_the_bound_stream_before_returning() {
    let (client, _context) = setup();
    let owned = [Owned::new(), Owned::new()];

    // SAFETY: both streams outlive the client: they are destroyed at the end of
    // this test binary's only test, after its last call.
    let ids: Vec<_> = owned
        .iter()
        .map(|o| unsafe { client.bind_stream(o.stream as u64) }.unwrap())
        .collect();
    assert_ne!(ids[0], ids[1], "two caller streams share a slot");
    assert!(ids.iter().all(|id| id.is_bound()));
    assert_eq!(
        unsafe { client.bind_stream(owned[0].stream as u64) }.unwrap(),
        ids[0],
        "binding a stream again gives the same id"
    );

    for round in 0..4u32 {
        for (k, (o, id)) in owned.iter().zip(&ids).enumerate() {
            let value = round * 10 + k as u32;
            let input: Vec<u32> = (0..N as u32).map(|i| i ^ value).collect();
            // The caller writes the input on its stream, then the kernel runs on
            // the same stream: no synchronization in between.
            o.write(o.input, &input);
            let (input_h, output_h) = id.executes(|| {
                // SAFETY: the memory lives until `release`, after the handles
                // are unregistered and the stream synchronized.
                let input_h = unsafe { client.register_external(o.input, (N * 4) as u64) }.unwrap();
                let output_h =
                    unsafe { client.register_external(o.output, (N * 4) as u64) }.unwrap();
                add_one::launch(
                    &client,
                    CubeCount::new_1d((N / 256) as u32),
                    CubeDim::new_1d(256),
                    unsafe { BufferArg::from_raw_parts(input_h.clone(), N) },
                    unsafe { BufferArg::from_raw_parts(output_h.clone(), N) },
                );
                (input_h, output_h)
            });
            // The caller reads on its stream right away, synchronizing only it.
            // Nothing between the launch and this read waits on the server:
            // only the transport can have put the launch on the stream first.
            let out = o.read(o.output);
            let expected: Vec<u32> = input.iter().map(|v| v + 1).collect();
            assert!(
                out == expected,
                "round {round}, stream {k}: output not the kernel's"
            );
            id.executes(|| {
                client.unregister_external(input_h).unwrap();
                client.unregister_external(output_h).unwrap();
            });
            // Unregistering freed nothing: the caller's memory is still its own.
            let again = o.read(o.output);
            assert!(
                again == expected,
                "round {round}, stream {k}: memory changed after unregistering"
            );
        }
    }

    for o in owned {
        o.release();
    }
}

/// Counts what the server takes from and returns to the caller's allocator.
#[derive(Default)]
struct Counting {
    live: std::sync::Mutex<std::collections::HashMap<u64, u64>>,
    allocations: std::sync::atomic::AtomicUsize,
    frees: std::sync::atomic::AtomicUsize,
}

impl cubecl_core::server::StreamAllocator for Counting {
    fn allocate(&self, size: u64, _stream: u64) -> Option<u64> {
        // SAFETY: the server calls this with the device's context current.
        let ptr = unsafe { result::malloc_sync(size as usize) }.ok()?;
        self.live.lock().unwrap().insert(ptr, size);
        self.allocations
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Some(ptr)
    }

    fn free(&self, ptr: u64, _stream: u64) {
        assert!(
            self.live.lock().unwrap().remove(&ptr).is_some(),
            "freed memory the allocator did not hand out, or freed it twice"
        );
        // SAFETY: `ptr` came from `allocate`; `cuMemFree` waits for work using it.
        unsafe { result::free_sync(ptr) }.unwrap();
        self.frees
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

#[test]
fn a_bound_streams_memory_comes_from_its_allocator() {
    use std::sync::atomic::Ordering::Relaxed;
    let (client, _context) = setup();
    let owned = Owned::new();
    let counting = std::sync::Arc::new(Counting::default());
    // SAFETY: the stream outlives the client's use of it (destroyed at the end).
    let id =
        unsafe { client.bind_stream_with(owned.stream as u64, Some(counting.clone())) }.unwrap();

    for round in 0..4u32 {
        let input: Vec<u32> = (0..N as u32).map(|i| i ^ round).collect();
        owned.write(owned.input, &input);
        let out = id.executes(|| {
            let input_h = unsafe { client.register_external(owned.input, (N * 4) as u64) }.unwrap();
            // The output is the server's own: it comes from the allocator.
            let output_h = client.empty(N * 4);
            add_one::launch(
                &client,
                CubeCount::new_1d((N / 256) as u32),
                CubeDim::new_1d(256),
                unsafe { BufferArg::from_raw_parts(input_h.clone(), N) },
                unsafe { BufferArg::from_raw_parts(output_h.clone(), N) },
            );
            let bytes = client.read_one(output_h).unwrap();
            // An external handle reads like any other: it counts as allocated,
            // and no pool claims it.
            let read_back = client.read_one(input_h.clone()).unwrap();
            assert!(read_back[..] == *bytemuck::cast_slice::<u32, u8>(&input));
            client.unregister_external(input_h).unwrap();
            bytes
        });
        let out: &[u32] = bytemuck::cast_slice(&out);
        assert!(
            out.iter().zip(&input).all(|(o, i)| *o == i + 1),
            "round {round}"
        );
    }

    let held = |client: &Client| {
        id.executes(|| client.memory_report(cubecl_core::MemoryScope::CurrentStream))
    };
    let report = held(&client);
    let pools = &report.streams[0].pools;
    assert!(
        pools.dynamic.iter().all(|pool| pool.pages == 0) && pools.persistent.pages == 0,
        "a bound stream with an allocator pools nothing: {pools:?}"
    );
    assert!(
        counting.allocations.load(Relaxed) >= 4,
        "the outputs came from the allocator"
    );

    id.executes(|| client.memory_cleanup()).unwrap();
    let live = counting.live.lock().unwrap().len();
    let dedicated = held(&client).streams[0].pools.dedicated.pages;
    assert_eq!(
        live as u64, dedicated,
        "every allocation the server still holds is one the allocator handed out, and nothing else is live"
    );
    // Nothing is kept once freed: every allocation is back with the allocator.
    assert_eq!(live, 0, "dedicated allocations are returned once freed");
    assert_eq!(
        counting.frees.load(Relaxed),
        counting.allocations.load(Relaxed)
    );
    owned.release();
}

/// Device memory reserved before the capture and handed out in pieces, as
/// a framework's graph pool is: an allocation inside the capture calls no API.
struct Reserved {
    base: u64,
    next: std::sync::atomic::AtomicU64,
}

impl Reserved {
    const SIZE: u64 = 1 << 20;

    fn new() -> Self {
        // SAFETY: a context is current (`setup`).
        let base = unsafe { result::malloc_sync(Self::SIZE as usize) }.unwrap();
        Self {
            base,
            next: std::sync::atomic::AtomicU64::new(0),
        }
    }
}

impl cubecl_core::server::StreamAllocator for Reserved {
    fn allocate(&self, size: u64, _stream: u64) -> Option<u64> {
        let size = size.next_multiple_of(256);
        let offset = self
            .next
            .fetch_add(size, std::sync::atomic::Ordering::Relaxed);
        (offset + size <= Self::SIZE).then_some(self.base + offset)
    }

    fn free(&self, _ptr: u64, _stream: u64) {}
}

#[cube(launch_unchecked)]
fn shapes(input: &Tensor<f32>, out: &mut [u32]) {
    if ABSOLUTE_POS == 0 {
        out[0] = input.shape(0) as u32;
        out[1] = input.shape(1) as u32;
        out[2] = input.stride(0) as u32;
    }
}

#[test]
fn a_launch_first_seen_inside_the_callers_capture_replays() {
    // The caller captures its own stream; the server never learns of it. A
    // shape the server has not seen needs its metadata written inside the
    // capture, with no host memory: every replay must see that shape.
    use cudarc::driver::sys;
    let (client, _context) = setup();
    let owned = Owned::new();
    // A stream its owner captures needs an allocator that allocates nothing
    // through the driver inside the capture.
    let reserved = std::sync::Arc::new(Reserved::new());
    // SAFETY: the stream outlives the client's use of it (destroyed at the end).
    let id = unsafe { client.bind_stream_with(owned.stream as u64, Some(reserved)) }.unwrap();
    let (rows, cols) = (37usize, 113usize);
    let launch = || {
        id.executes(|| {
            let input =
                unsafe { client.register_external(owned.input, (rows * cols * 4) as u64) }.unwrap();
            let out = unsafe { client.register_external(owned.output, 16) }.unwrap();
            unsafe {
                shapes::launch_unchecked(
                    &client,
                    CubeCount::new_single(),
                    CubeDim::new_1d(1),
                    TensorArg::from_raw_parts(input.clone(), [cols, 1].into(), [rows, cols].into()),
                    BufferArg::from_raw_parts(out.clone(), 4),
                )
            };
            client.unregister_external(input).unwrap();
            client.unregister_external(out).unwrap();
        })
    };
    let capture = || {
        use cudarc::driver::sys;
        let (mut graph, mut exec) = (core::ptr::null_mut(), core::ptr::null_mut());
        // SAFETY: the graph and its executable are destroyed before the memory.
        unsafe {
            sys::cuStreamBeginCapture_v2(
                owned.stream,
                sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            )
            .result()
            .unwrap();
            launch();
            sys::cuStreamEndCapture(owned.stream, &mut graph)
                .result()
                .expect("the capture holds: nothing in the launch broke it");
            sys::cuGraphInstantiateWithFlags(&mut exec, graph, 0)
                .result()
                .unwrap();
        }
        (graph, exec)
    };
    let expected = [rows as u32, cols as u32, cols as u32];
    let check = |what: &str| {
        let out = owned.read(owned.output);
        assert_eq!(&out[..3], &expected, "{what}");
    };
    let replay = |exec| {
        owned.write(owned.output, &[0u32; 4]);
        unsafe { sys::cuGraphLaunch(exec, owned.stream).result().unwrap() };
    };

    let (graph, exec) = capture();
    // The shape was first seen inside the capture, whose writes have not run:
    // an eager launch of it must not find their buffer.
    owned.write(owned.output, &[0u32; 4]);
    launch();
    check("eager launch after the capture, before any replay");
    for _ in 0..2 {
        replay(exec);
        check("replay");
    }
    // A second capture of the same shape must write its own metadata, not rely
    // on the first graph's replay or on an eager buffer.
    let (graph_2, exec_2) = capture();
    replay(exec_2);
    check("replay of a second graph of the same shape");
    unsafe {
        for (graph, exec) in [(graph, exec), (graph_2, exec_2)] {
            sys::cuGraphExecDestroy(exec).result().unwrap();
            sys::cuGraphDestroy(graph).result().unwrap();
        }
    }
    owned.release();
}
