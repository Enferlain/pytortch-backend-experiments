//! Writing launch metadata to device memory without host memory, for streams a
//! caller owns.
//!
//! A bound stream may be captured into a graph by its owner, which this server
//! never learns of. Staging through pinned host memory cannot be captured: the
//! pinned allocation is refused while a stream captures, and the recorded copy
//! would read host memory whose lifetime only the owner's graph knows. A
//! kernel carrying the words in its parameters captures as a node with the
//! words baked in, and needs nothing else.

use crate::compute::events::driver_error;
use cubecl_core::server::ServerError;
use cudarc::driver::sys::{CUfunction, CUstream};
use std::ffi::{CString, c_void};

/// Words one launch carries in its parameter block.
const CHUNK: usize = 256;

const SOURCE: &str = r#"
struct Words { unsigned long long values[256]; };
extern "C" __global__ void cubecl_write_words(
    unsigned long long* dst, const __grid_constant__ Words words, unsigned int count) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) dst[i] = words.values[i];
}
"#;

#[repr(C)]
struct Words {
    values: [u64; CHUNK],
}

/// The word-writing kernel of one context, compiled on first use.
#[derive(Debug, Default)]
pub(crate) struct WordWriter {
    function: Option<usize>,
}

impl WordWriter {
    fn function(&mut self) -> Result<CUfunction, ServerError> {
        if let Some(function) = self.function {
            return Ok(function as CUfunction);
        }
        // For the current context's device, whose compute capability NVRTC
        // accepts whatever its release supports.
        let arch = unsafe {
            use cudarc::driver::sys::{self, CUdevice_attribute as A};
            let mut device = 0;
            sys::cuCtxGetDevice(&mut device)
                .result()
                .map_err(|err| driver_error("cuCtxGetDevice", err))?;
            let major = cudarc::driver::result::device::get_attribute(
                device,
                A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            )
            .map_err(|err| driver_error("cuDeviceGetAttribute", err))?;
            let minor = cudarc::driver::result::device::get_attribute(
                device,
                A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            )
            .map_err(|err| driver_error("cuDeviceGetAttribute", err))?;
            format!("compute_{major}{minor}")
        };
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(
            SOURCE,
            cudarc::nvrtc::CompileOptions {
                arch: Some(Box::leak(arch.into_boxed_str())),
                ..Default::default()
            },
        )
        .map_err(|err| ServerError::Generic {
            reason: format!("compiling the metadata writer: {err}"),
            backtrace: cubecl_environment::backtrace::BackTrace::capture(),
        })?;
        let image = CString::new(ptx.to_src()).expect("PTX has no NUL");
        // SAFETY: the context is current (the caller holds a command), and the
        // image is NUL-terminated PTX.
        let function = unsafe {
            let module = cudarc::driver::result::module::load_data(image.as_ptr() as *const c_void)
                .map_err(|err| driver_error("cuModuleLoadData", err))?;
            cudarc::driver::result::module::get_function(
                module,
                CString::from(c"cubecl_write_words"),
            )
            .map_err(|err| driver_error("cuModuleGetFunction", err))?
        };
        self.function = Some(function as usize);
        Ok(function)
    }

    /// Enqueues on `stream` the writes of `words` to `dst`.
    pub(crate) fn write(
        &mut self,
        stream: CUstream,
        dst: u64,
        words: &[u64],
    ) -> Result<(), ServerError> {
        let function = self.function()?;
        for (index, chunk) in words.chunks(CHUNK).enumerate() {
            let mut values = Words { values: [0; CHUNK] };
            values.values[..chunk.len()].copy_from_slice(chunk);
            let mut target = dst + (index * CHUNK * 8) as u64;
            let mut count = chunk.len() as u32;
            let mut params = [
                &mut target as *mut u64 as *mut c_void,
                &mut values as *mut Words as *mut c_void,
                &mut count as *mut u32 as *mut c_void,
            ];
            // SAFETY: the parameters match the kernel's signature, and the
            // driver copies them before the call returns.
            unsafe {
                cudarc::driver::result::launch_kernel(
                    function,
                    (1, 1, 1),
                    (CHUNK as u32, 1, 1),
                    0,
                    stream,
                    &mut params,
                )
            }
            .map_err(|err| driver_error("cuLaunchKernel", err))?;
        }
        Ok(())
    }
}
