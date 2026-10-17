//! Host-only calls into the actual pinned policy helpers. No runtime/client/GPU.
use cubecl::ir::HardwareProperties;
use cubecl::tensor_vector_size_parallel;
use cubek_reduce::launch::calculate_plane_count_per_cube;
use serde_json::json;

fn main() {
    // Relevant values: recorded CUDA32/1024-thread device and CUDA source's
    // 128-bit load width. Other fields are unused by the called plane helper.
    let hardware = HardwareProperties {
        load_width: 128,
        vector_register_count: None,
        plane_size_min: 32,
        plane_size_max: 32,
        max_bindings: 0,
        max_shared_memory_size: 0,
        max_cube_count: (65535, 65535, 65535),
        max_units_per_cube: 1024,
        max_cube_dim: (1024, 1024, 64),
        num_streaming_multiprocessors: Some(82),
        num_cpu_cores: None,
        last_level_cache_size: None,
        num_tensor_cores: None,
        min_tensor_cores_dim: None,
        max_vector_size: usize::MAX,
        cube_mma_reserved_shared_memory: 0,
    };
    let mut rows = Vec::new();
    for storage in ["float32", "float16", "bfloat16"] {
        for (b, n) in [(64usize, 1024usize), (64, 255), (16, 4099)] {
            // All three trial virtual operands are FP32 squares, not raw storage.
            // CUDA 128-bit IO gives [4,2,1] at FP32, with no IO override assumed.
            let shape = vec![b, n].into();
            let strides = vec![n, 1].into();
            let input_width =
                tensor_vector_size_parallel([4, 2, 1].into_iter(), &shape, &strides, 1);
            let planes = calculate_plane_count_per_cube(b * 32, 32, &hardware);
            rows.push(json!({
                "storage":storage,"shape":[b,n],"virtual_operand":"float32",
                "conditional_routine":"Plane(independent=true), parallel_output_vectorization=false",
                "input_vector_size":input_width,"output_vector_size":1,
                "plane_count":planes,"grid":b.div_ceil(planes as usize),
                "stock_plane_iterations":n.div_ceil(32 * input_width),
                "trial_plane_iterations":n.div_ceil(32),
                "automatic_routine_winner":null,
            }));
        }
    }
    let zero_work_planes = calculate_plane_count_per_cube(0, 32, &hardware);
    let one_row_planes = calculate_plane_count_per_cube(32, 32, &hardware);
    let odd_shape = vec![64, 1025].into();
    let odd_strides = vec![1025, 1].into();
    let odd_width = tensor_vector_size_parallel([4, 2, 1].into_iter(), &odd_shape, &odd_strides, 1);
    let aligned_shape = vec![64, 1024].into();
    let bad_strides = vec![1025, 1].into();
    let bad_alignment =
        tensor_vector_size_parallel([4, 2, 1].into_iter(), &aligned_shape, &bad_strides, 1);
    assert_eq!(
        (zero_work_planes, one_row_planes, odd_width, bad_alignment),
        (1, 1, 1, 1)
    );
    println!("{}", serde_json::to_string_pretty(&json!({
        "scope":"Actual host policy helper calls, with explicit CUDA property model; no CubeK client, GPU compile, launch or autotuning",
        "cuda_io_width_bits":128,"io_override":null,
        "rows":rows,
        "controls": {"zero_work_planes":zero_work_planes,"one_row_planes":one_row_planes,
            "odd_width_vector":odd_width,"unaligned_row_stride_vector":bad_alignment}
    })).unwrap());
}
