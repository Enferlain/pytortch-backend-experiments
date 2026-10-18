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
extern "C" __global__ void __launch_bounds__(256)  cubecl5_6ca8fa1fd160633d(__nv_bfloat16 const* const __restrict__ v1, __nv_bfloat16 const* const __restrict__ v2, __nv_bfloat16 * const __restrict__ v3, info_st const* const __restrict__ v40) {
uint64_t const* const dynamic_meta_v42 = reinterpret_cast<uint64_t const*>(v40 + 1);
const info_st& info_v41 = *v40;int64_t const v43 = info_v41.scalars_i64[0];
int64_t const v44 = info_v41.scalars_i64[1];
uint64_t const v7 = blockIdx.x;
int64_t const v8 = int64_t(v7);
uint64_t const v9 = threadIdx.x;
int64_t const v10 = int64_t(v9);
int64_t const v11 = 256;
int64_t const v12 = v8 * v11;
int64_t const v15 = 0;
int64_t const v19 = v12 + v10;
bool const v20 = v19 < v43;
if(v20) {
uint64_t const v21 = uint64_t(v19);
__nv_bfloat16 const* const v22 = &v1[v21];
__nv_bfloat16 const v23 = *v22;
float const v24 = float(v23);
__nv_bfloat16 const v25 = __nv_bfloat16(v24);
float const v26 = float(v25);
int64_t const v47 = v19 % v44;
int64_t const v49 = v47 ^ v44;
bool const v50 = v49 < v15;
bool const v51 = v47 != v15;
bool const v52 = v51 && v50;
int64_t const v53 = v52 ? v44 : v15;
int64_t const v54 = v47 + v53;
uint64_t const v28 = uint64_t(v54);
__nv_bfloat16 const* const v29 = &v2[v28];
__nv_bfloat16 const v30 = *v29;
float const v31 = float(v30);
__nv_bfloat16 const v32 = __nv_bfloat16(v31);
float const v33 = float(v32);
float const v34 = v26 + v33;
__nv_bfloat16 const v35 = __nv_bfloat16(v34);
float const v36 = float(v35);
__nv_bfloat16 const v37 = __nv_bfloat16(v36);
__nv_bfloat16 * const v39 = &v3[v21];
*v39 = v37;
}
return;
}
