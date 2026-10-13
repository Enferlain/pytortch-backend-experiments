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

/// Stage 2 independent buffer spans and signed-i64 runtime metadata.
/// # Safety
/// All input pointers reference the declared counts; `out` has `capacity` words.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stage2_metadata(
    lengths: *const u64,
    buffers: usize,
    scalars: *const i64,
    count: usize,
    out: *mut u64,
    capacity: usize,
) -> usize {
    unsafe { serialize_metadata(lengths, buffers, scalars, count, out, capacity, 5) }
}
/// Stage 4 permits five external reads plus one destination.
/// # Safety
/// Input pointers cover their declared counts; output covers capacity words.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stage4_metadata(
    lengths: *const u64,
    buffers: usize,
    scalars: *const i64,
    count: usize,
    out: *mut u64,
    capacity: usize,
) -> usize {
    unsafe { serialize_metadata(lengths, buffers, scalars, count, out, capacity, 6) }
}
unsafe fn serialize_metadata(
    lengths: *const u64,
    buffers: usize,
    scalars: *const i64,
    count: usize,
    out: *mut u64,
    capacity: usize,
    maximum: usize,
) -> usize {
    if !(1..=maximum).contains(&buffers)
        || count == 0
        || count > 33
        || lengths.is_null()
        || scalars.is_null()
    {
        return 0;
    }
    let Ok(data) = std::panic::catch_unwind(|| {
        let lengths = unsafe { std::slice::from_raw_parts(lengths, buffers) };
        let scalars = unsafe { std::slice::from_raw_parts(scalars, count) };
        let mut info = InfoBuilder::default();
        for v in scalars {
            info.scalars.push(*v);
        }
        for n in lengths {
            info.metadata.register_buffer(*n, AddressType::U64);
        }
        info.finish(AddressType::U64).data
    }) else {
        return 0;
    };
    if !out.is_null() && capacity >= data.len() {
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), out, data.len());
        }
    }
    data.len()
}
