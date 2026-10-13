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
    int64_t scalars_i64[2];uint64_t static_meta[3];
};
extern "C" __global__ void __launch_bounds__(256)  cubecl4_d5310cf7b3c494e3(bool const* const __restrict__ v1, __nv_bfloat16 const* const __restrict__ v2, __nv_bfloat16 * const __restrict__ v3, info_st const* const __restrict__ v33) {
uint64_t const* const dynamic_meta_v35 = reinterpret_cast<uint64_t const*>(v33 + 1);
const info_st& info_v34 = *v33;int64_t const v36 = info_v34.scalars_i64[0];
uint64_t const v6 = blockIdx.x;
int64_t const v7 = int64_t(v6);
uint64_t const v8 = threadIdx.x;
int64_t const v9 = int64_t(v8);
int64_t const v10 = 256;
int64_t const v11 = v7 * v10;
int64_t const v12 = v11 + v9;
bool const v13 = v12 < v36;
if(v13) {
uint64_t const v14 = uint64_t(v12);
bool const* const v15 = &v1[v14];
bool const v16 = *v15;
float const v17 = float(v16);
__nv_bfloat16 const* const v19 = &v2[v14];
__nv_bfloat16 const v20 = *v19;
float const v21 = float(v20);
float const v22 = v17 * v21;
__nv_bfloat16 const v23 = __nv_bfloat16(v22);
float const v24 = float(v23);
float const v39 = 1.3359375;
float const v27 = v24 * v39;
__nv_bfloat16 const v28 = __nv_bfloat16(v27);
float const v29 = float(v28);
__nv_bfloat16 const v30 = __nv_bfloat16(v29);
__nv_bfloat16 * const v32 = &v3[v14];
*v32 = v30;
}
return;
}
