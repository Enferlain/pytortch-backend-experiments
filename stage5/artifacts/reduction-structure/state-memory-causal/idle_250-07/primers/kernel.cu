
extern "C" __global__ void primer_read(const unsigned *p, unsigned *sink,
                                      unsigned count, unsigned stride) {
    unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned v = 0;
    const volatile unsigned *q = p;
    for (unsigned i = t; i < count; i += gridDim.x * blockDim.x)
        v += q[(unsigned long long)i * stride];
    sink[t] = v;
}
extern "C" __global__ void primer_compute(float *sink, unsigned iterations) {
    unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
    float a = (float)t, b = 0.5f;
    for (unsigned i = 0; i < iterations; ++i) {
        a = __fadd_rn(__fmul_rn(a, 1.00000011920928955078125f), b);
        b = __fadd_rn(__fmul_rn(b, 0.999999940395355224609375f), 0.000001f);
    }
    sink[t] = a + b;
}
extern "C" __global__ void primer_write(unsigned *p, unsigned *sink,
                                       unsigned count, unsigned stride) {
    unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
    volatile unsigned *q = p;
    for (unsigned i = t; i < count; i += gridDim.x * blockDim.x)
        q[(unsigned long long)i * stride] = 1u;
    sink[t] = t;
}
