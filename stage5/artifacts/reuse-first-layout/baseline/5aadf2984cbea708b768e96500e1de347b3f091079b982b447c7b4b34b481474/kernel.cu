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
    int64_t scalars_i64[3];uint64_t static_meta[2];
};
extern "C" __global__ void __launch_bounds__(256)  cubecl5_7d7439611f1d7677(__nv_bfloat16 const* const __restrict__ v1, __nv_bfloat16 * const __restrict__ v2, info_st const* const __restrict__ v45) {
uint64_t const* const dynamic_meta_v47 = reinterpret_cast<uint64_t const*>(v45 + 1);
const info_st& info_v46 = *v45;int64_t const v48 = info_v46.scalars_i64[0];
int64_t const v49 = info_v46.scalars_i64[1];
uint64_t const v6 = blockIdx.x;
int64_t const v7 = int64_t(v6);
uint64_t const v8 = threadIdx.x;
int64_t const v9 = int64_t(v8);
int64_t const v10 = 256;
int64_t const v11 = v7 * v10;
int64_t const v12 = 1;
int64_t const v14 = 0;
int64_t const v18 = v11 + v9;
bool const v19 = v18 < v48;
if(v19) {
int64_t const v20 = v18 / v49;
int64_t const v21 = v18 % v49;
bool const v24 = v21 < v14;
int64_t const v25 = v20 - v12;
int64_t const v26 = v24 ? v25 : v20;
uint64_t const v27 = uint64_t(v26);
__nv_bfloat16 const* const v28 = &v1[v27];
__nv_bfloat16 const v29 = *v28;
float const v30 = float(v29);
__nv_bfloat16 const v31 = __nv_bfloat16(v30);
float const v32 = float(v31);
float const v33 = sin(v32);__nv_bfloat16 const v34 = __nv_bfloat16(v33);
float const v35 = float(v34);
__nv_bfloat16 const v36 = __nv_bfloat16(v35);
float const v37 = float(v36);
float const v38 = 1.0;
float const v39 = v37 + v38;
__nv_bfloat16 const v40 = __nv_bfloat16(v39);
float const v41 = float(v40);
__nv_bfloat16 const v42 = __nv_bfloat16(v41);
uint64_t const v43 = uint64_t(v18);
__nv_bfloat16 * const v44 = &v2[v43];
*v44 = v42;
}
return;
}
