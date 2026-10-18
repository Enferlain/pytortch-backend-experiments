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
extern "C" __global__ void __launch_bounds__(256)  cubecl5_dbf34f87fb508074(__nv_bfloat16 const* const __restrict__ v1, __nv_bfloat16 * const __restrict__ v2, info_st const* const __restrict__ v39) {
uint64_t const* const dynamic_meta_v41 = reinterpret_cast<uint64_t const*>(v39 + 1);
const info_st& info_v40 = *v39;int64_t const v42 = info_v40.scalars_i64[0];
int64_t const v43 = info_v40.scalars_i64[1];
uint64_t const v6 = blockIdx.x;
int64_t const v7 = int64_t(v6);
uint64_t const v8 = threadIdx.x;
int64_t const v9 = int64_t(v8);
int64_t const v10 = 256;
int64_t const v11 = v7 * v10;
int64_t const v18 = v11 + v9;
bool const v19 = v18 < v42;
if(v19) {
int64_t const v20 = v18 * v43;
uint64_t const v21 = uint64_t(v20);
__nv_bfloat16 const* const v22 = &v1[v21];
__nv_bfloat16 const v23 = *v22;
float const v24 = float(v23);
__nv_bfloat16 const v25 = __nv_bfloat16(v24);
float const v26 = float(v25);
float const v27 = sin(v26);__nv_bfloat16 const v28 = __nv_bfloat16(v27);
float const v29 = float(v28);
__nv_bfloat16 const v30 = __nv_bfloat16(v29);
float const v31 = float(v30);
float const v32 = 1.0;
float const v33 = v31 + v32;
__nv_bfloat16 const v34 = __nv_bfloat16(v33);
float const v35 = float(v34);
__nv_bfloat16 const v36 = __nv_bfloat16(v35);
uint64_t const v37 = uint64_t(v18);
__nv_bfloat16 * const v38 = &v2[v37];
*v38 = v36;
}
return;
}
