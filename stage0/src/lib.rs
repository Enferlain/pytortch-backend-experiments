use cubecl_core::InfoBuilder;
use cubecl_core::ir::AddressType;

pub fn metadata(n: u64) -> Vec<u64> {
    let mut info = InfoBuilder::default();
    info.metadata.register_buffer(n, AddressType::U64);
    info.metadata.register_buffer(n, AddressType::U64);
    info.finish(AddressType::U64).data
}

/// Fixed Stage 0 metadata, copied into caller-owned host storage.
/// Returns the required number of u64 words. No pointers are retained.
/// # Safety
/// `out` must point to `capacity` writable u64 words when non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stage0_metadata(n: u64, out: *mut u64, capacity: usize) -> usize {
    // A Rust panic must never unwind through the C/ctypes boundary.
    let Ok(data) = std::panic::catch_unwind(|| metadata(n)) else {
        return 0;
    };
    if !out.is_null() && capacity >= data.len() {
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), out, data.len()) };
    }
    data.len()
}

/// Stage 1: identical static-length metadata for a bounded number of buffers.
/// # Safety
/// `out` must point to `capacity` writable u64 words when non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stage1_metadata(
    n: u64,
    buffers: usize,
    out: *mut u64,
    capacity: usize,
) -> usize {
    if !(2..=3).contains(&buffers) {
        return 0;
    }
    let Ok(data) = std::panic::catch_unwind(|| {
        let mut info = InfoBuilder::default();
        for _ in 0..buffers {
            info.metadata.register_buffer(n, AddressType::U64);
        }
        info.finish(AddressType::U64).data
    }) else {
        return 0;
    };
    if !out.is_null() && capacity >= data.len() {
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), out, data.len()) };
    }
    data.len()
}
