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
    int64_t scalars_i64[2];uint64_t static_meta[2];
};
extern "C" __global__ void __launch_bounds__(256)  cubecl4_ba438e5c4f95f5da(__nv_bfloat16 const* const __restrict__ v1, __nv_bfloat16 * const __restrict__ v2, info_st const* const __restrict__ v23) {
uint64_t const* const dynamic_meta_v25 = reinterpret_cast<uint64_t const*>(v23 + 1);
const info_st& info_v24 = *v23;int64_t const v26 = info_v24.scalars_i64[0];
uint64_t const v5 = blockIdx.x;
int64_t const v6 = int64_t(v5);
uint64_t const v7 = threadIdx.x;
int64_t const v8 = int64_t(v7);
int64_t const v9 = 256;
int64_t const v10 = v6 * v9;
int64_t const v11 = v10 + v8;
bool const v12 = v11 < v26;
if(v12) {
uint64_t const v13 = uint64_t(v11);
__nv_bfloat16 const* const v14 = &v1[v13];
__nv_bfloat16 const v15 = *v14;
float const v16 = float(v15);
float const v29 = 0.0;
float const v19 = v16 * v29;
__nv_bfloat16 const v20 = __nv_bfloat16(v19);
__nv_bfloat16 * const v22 = &v2[v13];
*v22 = v20;
}
return;
}
