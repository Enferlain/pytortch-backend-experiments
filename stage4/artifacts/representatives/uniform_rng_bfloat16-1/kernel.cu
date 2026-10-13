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
    int64_t scalars_i64[1];uint64_t static_meta[2];
};
extern "C" __global__ void __launch_bounds__(256)  cubecl4_00c8e200d4f96f29(float const* const __restrict__ v1, __nv_bfloat16 * const __restrict__ v2, info_st const* const __restrict__ v22) {
uint64_t const* const dynamic_meta_v24 = reinterpret_cast<uint64_t const*>(v22 + 1);
const info_st& info_v23 = *v22;int64_t const v25 = info_v23.scalars_i64[0];
uint64_t const v4 = blockIdx.x;
int64_t const v5 = int64_t(v4);
uint64_t const v6 = threadIdx.x;
int64_t const v7 = int64_t(v6);
int64_t const v8 = 256;
int64_t const v9 = v5 * v8;
int64_t const v10 = v9 + v7;
bool const v11 = v10 < v25;
if(v11) {
uint64_t const v28 = 0;
float const* const v14 = &v1[v28];
float const v15 = *v14;
__nv_bfloat16 const v16 = __nv_bfloat16(v15);
float const v17 = float(v16);
__nv_bfloat16 const v18 = __nv_bfloat16(v17);
__nv_bfloat16 * const v21 = &v2[v28];
*v21 = v18;
}
return;
}
