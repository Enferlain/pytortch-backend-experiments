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
extern "C" __global__ void __launch_bounds__(256)  cubecl5_9214b7425b78538c(__nv_bfloat16 const* const __restrict__ v1, __nv_bfloat16 * const __restrict__ v2, info_st const* const __restrict__ v37) {
uint64_t const* const dynamic_meta_v39 = reinterpret_cast<uint64_t const*>(v37 + 1);
const info_st& info_v38 = *v37;int64_t const v40 = info_v38.scalars_i64[0];
int64_t const v41 = info_v38.scalars_i64[1];
int64_t const v42 = info_v38.scalars_i64[2];
uint64_t const v6 = blockIdx.x;
int64_t const v7 = int64_t(v6);
uint64_t const v8 = threadIdx.x;
int64_t const v9 = int64_t(v8);
int64_t const v10 = 256;
int64_t const v11 = v7 * v10;
int64_t const v12 = 1;
int64_t const v14 = 0;
int64_t const v18 = v11 + v9;
bool const v19 = v18 < v40;
if(v19) {
int64_t const v44 = v18 % v42;
int64_t const v46 = v44 ^ v42;
bool const v47 = v46 < v14;
bool const v48 = v44 != v14;
bool const v49 = v48 && v47;
int64_t const v50 = v49 ? v42 : v14;
int64_t const v51 = v44 + v50;
int64_t const v21 = v41 * v51;
int64_t const v22 = v18 / v42;
bool const v26 = v44 < v14;
int64_t const v27 = v22 - v12;
int64_t const v28 = v26 ? v27 : v22;
int64_t const v29 = v21 + v28;
uint64_t const v30 = uint64_t(v29);
__nv_bfloat16 const* const v31 = &v1[v30];
__nv_bfloat16 const v32 = *v31;
float const v33 = float(v32);
__nv_bfloat16 const v34 = __nv_bfloat16(v33);
uint64_t const v35 = uint64_t(v18);
__nv_bfloat16 * const v36 = &v2[v35];
*v36 = v34;
}
return;
}
