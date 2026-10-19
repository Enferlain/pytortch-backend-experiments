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
    int64_t scalars_i64[3];uint64_t static_meta[3];
};
extern "C" __global__ void __launch_bounds__(256)  cubecl5_93245d2f226524eb(__nv_bfloat16 const* const __restrict__ v1, __nv_bfloat16 const* const __restrict__ v2, __nv_bfloat16 * const __restrict__ v3, info_st const* const __restrict__ v46) {
uint64_t const* const dynamic_meta_v48 = reinterpret_cast<uint64_t const*>(v46 + 1);
const info_st& info_v47 = *v46;int64_t const v49 = info_v47.scalars_i64[0];
int64_t const v50 = info_v47.scalars_i64[1];
uint64_t const v7 = blockIdx.x;
int64_t const v8 = int64_t(v7);
uint64_t const v9 = threadIdx.x;
int64_t const v10 = int64_t(v9);
int64_t const v11 = 256;
int64_t const v12 = v8 * v11;
int64_t const v13 = 1;
int64_t const v15 = 0;
int64_t const v19 = v12 + v10;
bool const v20 = v19 < v49;
if(v20) {
uint64_t const v21 = uint64_t(v19);
__nv_bfloat16 const* const v22 = &v1[v21];
__nv_bfloat16 const v23 = *v22;
float const v24 = float(v23);
__nv_bfloat16 const v25 = __nv_bfloat16(v24);
float const v26 = float(v25);
int64_t const v27 = v19 / v50;
int64_t const v28 = v19 % v50;
bool const v31 = v28 < v15;
int64_t const v32 = v27 - v13;
int64_t const v33 = v31 ? v32 : v27;
uint64_t const v34 = uint64_t(v33);
__nv_bfloat16 const* const v35 = &v2[v34];
__nv_bfloat16 const v36 = *v35;
float const v37 = float(v36);
__nv_bfloat16 const v38 = __nv_bfloat16(v37);
float const v39 = float(v38);
float const v40 = v26 + v39;
__nv_bfloat16 const v41 = __nv_bfloat16(v40);
float const v42 = float(v41);
__nv_bfloat16 const v43 = __nv_bfloat16(v42);
__nv_bfloat16 * const v45 = &v3[v21];
*v45 = v43;
}
return;
}
