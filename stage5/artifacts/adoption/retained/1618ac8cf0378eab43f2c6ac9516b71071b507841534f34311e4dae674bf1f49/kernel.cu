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
    int64_t scalars_i64[7];uint64_t static_meta[3];
};
extern "C" __global__ void __launch_bounds__(256)  cubecl5_5eb45c1177ed210a(__nv_bfloat16 const* const __restrict__ v1, __nv_bfloat16 const* const __restrict__ v2, __nv_bfloat16 * const __restrict__ v3, info_st const* const __restrict__ v52) {
uint64_t const* const dynamic_meta_v54 = reinterpret_cast<uint64_t const*>(v52 + 1);
const info_st& info_v53 = *v52;int64_t const v55 = info_v53.scalars_i64[0];
int64_t const v60 = info_v53.scalars_i64[5];
int64_t const v61 = info_v53.scalars_i64[6];
uint32_t const v16 = uint32_t(v61);
uint64_t const v18 = uint64_t(v60);
uint64_t const v22 = blockIdx.x;
int64_t const v23 = int64_t(v22);
uint64_t const v24 = threadIdx.x;
int64_t const v25 = int64_t(v24);
int64_t const v26 = 256;
int64_t const v27 = v23 * v26;
int64_t const v28 = v27 + v25;
bool const v29 = v28 < v55;
if(v29) {
uint64_t const v30 = uint64_t(v28);
__nv_bfloat16 const* const v31 = &v1[v30];
__nv_bfloat16 const v32 = *v31;
float const v33 = float(v32);
__nv_bfloat16 const v34 = __nv_bfloat16(v33);
float const v35 = float(v34);
uint64_t const v37 = __umul64hi(v30, v18);
uint64_t const v38 = v37 + v30;
uint64_t const v39 = uint64_t(v16);
uint64_t const v40 = v38 >> v39;
__nv_bfloat16 const* const v41 = &v2[v40];
__nv_bfloat16 const v42 = *v41;
float const v43 = float(v42);
__nv_bfloat16 const v44 = __nv_bfloat16(v43);
float const v45 = float(v44);
float const v46 = v35 + v45;
__nv_bfloat16 const v47 = __nv_bfloat16(v46);
float const v48 = float(v47);
__nv_bfloat16 const v49 = __nv_bfloat16(v48);
__nv_bfloat16 * const v51 = &v3[v30];
*v51 = v49;
}
return;
}
