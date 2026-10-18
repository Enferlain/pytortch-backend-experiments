#[cube]
fn copy_perpendicular<T: Numeric, N: Size>(
    input: &VirtualTensor<T, N>,
    output: &mut VirtualTensor<T, N, ReadWrite>,
    axis_vectorized: usize,
    #[define(T)] _elem: ElemType,
) {
    let vector_size = input.vector_size();
    let last_axis = input.rank() - 1;

    // Calculate how many vectorized vectors fit into the last dimension's shape.
    let num_batch = output.shape(last_axis) / vector_size;

    // Local registers to perform a small in-register transpose.
    let mut accumulators = Sequence::<Vector<T, N>>::new();

    #[unroll]
    for _ in 0..vector_size {
        accumulators.push(Vector::empty());
    }

    let channel_input_stride_elem = input.stride(last_axis);
    let channel_output_stride_elem = output.stride(axis_vectorized);

    // Strides adjusted for vectorization (vector_size).
    let channel_input_stride = channel_input_stride_elem / vector_size;
    let channel_output_stride = channel_output_stride_elem / vector_size;

    // Total parallel units needed to cover the output space.
    let num_runs = output.len() / (num_batch * vector_size);

    if ABSOLUTE_POS >= num_runs {
        terminate!()
    }

    // Mapping the global worker ID to the specific tensor coordinates.
    let batch_index = ABSOLUTE_POS * num_batch;
    let skip_interval = batch_index / channel_output_stride;
    let skip_index = batch_index % channel_output_stride;
    let skip_size = channel_output_stride_elem;
    let global_index = (skip_interval * skip_size) + skip_index;

    for b in 0..num_batch {
        let offset_output = global_index + b;

        // Calculate the physical offset in the input tensor for the current output coordinate.
        let mut batch_offset = 0;
        for axis in 0..input.rank() {
            let coordinate = ((offset_output * vector_size) / output.stride(axis)) % output.shape(axis);
            batch_offset += coordinate * input.stride(axis);
        }
        let batch_offset = batch_offset / vector_size;

        // --- STEP 1: GATHER ---
        // Load data from the input tensor. Since the data is "perpendicular",
        // we read across the stride-1 axis to fill the accumulators.
        #[unroll]
        for i in 0..vector_size {
            let index = batch_offset + i * channel_input_stride;
            let batched = input.read(index);

            // --- STEP 2: TRANSPOSE ---
            // Rearrange the loaded vector components into the accumulators.
            #[unroll]
            for o in 0..vector_size {
                let vector = accumulators.index_mut(o);
                vector.insert(i, batched.extract(o));
            }
        }

        // --- STEP 3: STORE ---
        // Write the transposed vectors to the output in a contiguous fashion.
        #[unroll]
        for o in 0..vector_size {
            let index_out = offset_output + o * channel_output_stride;
            let batched = accumulators[o];

            output.write(index_out, batched);
        }
    }
}

