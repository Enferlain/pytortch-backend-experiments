typedef unsigned int uint32_t;
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned long long int uint64_t;
typedef signed char int8_t;
typedef signed short int16_t;
typedef signed int int32_t;
typedef signed long long int int64_t;

template <typename T, size_t N>
struct array {
    T data[N];
    __device__ T& operator[](size_t i) { return data[i]; }
    __device__ const T& operator[](size_t i) const { return data[i]; }
};


typedef struct CUtensorMap_st {
alignas(128) unsigned long long int opaque[16];
} CUtensorMap;
#include <cuda_bf16.h>

struct info_st {
    uint64_t static_meta[3];
};
extern "C" __global__ void __launch_bounds__(256)  cubecl_fp_mul_negative_zero_left_bfloat16(__nv_bfloat16 const* const __restrict__ v1, __nv_bfloat16 const* const __restrict__ v2, __nv_bfloat16 * const __restrict__ v3, info_st const* const __restrict__ v17) {
uint64_t const* const dynamic_meta_v19 = reinterpret_cast<uint64_t const*>(v17 + 1);
const info_st& info_v18 = *v17;uint32_t const v22 = blockIdx.z;
uint64_t const v23 = uint64_t(v22);
uint32_t const v24 = gridDim.y;
uint64_t const v25 = uint64_t(v24);
uint64_t const v26 = v23 * v25;
uint32_t const v27 = gridDim.x;
uint64_t const v28 = uint64_t(v27);
uint64_t const v29 = v26 * v28;
uint32_t const v30 = blockIdx.y;
uint64_t const v31 = uint64_t(v30);
uint64_t const v34 = v31 * v28;
uint64_t const v35 = v29 + v34;
uint32_t const v36 = blockIdx.x;
uint64_t const v37 = uint64_t(v36);
uint64_t const v38 = v35 + v37;
uint64_t const v65 = 256;
uint64_t const v41 = v38 * v65;
uint32_t const v46 = threadIdx.x;
uint64_t const v43 = uint64_t(v46);
uint64_t const v44 = v41 + v43;
uint64_t const v20 = info_v18.static_meta[0];
bool const v6 = v44 < v20;
if(v6) {
__nv_bfloat16 const* const v7 = &v1[v44];
__nv_bfloat16 const v8 = *v7;
float const v9 = float(v8);
float const v13 = -0.0;
float const v14 = v13 * v9;
__nv_bfloat16 const v15 = __nv_bfloat16(v14);
__nv_bfloat16 * const v16 = &v3[v44];
*v16 = v15;
}
return;
}
