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
extern "C" __global__ void __launch_bounds__(256)  cubecl5_958eea99ca131a20(__nv_bfloat16 const* const __restrict__ v1, __nv_bfloat16 * const __restrict__ v2, info_st const* const __restrict__ v48) {
uint64_t const* const dynamic_meta_v50 = reinterpret_cast<uint64_t const*>(v48 + 1);
const info_st& info_v49 = *v48;int64_t const v51 = info_v49.scalars_i64[0];
int64_t const v52 = info_v49.scalars_i64[1];
int64_t const v53 = info_v49.scalars_i64[2];
uint64_t const v6 = blockIdx.x;
int64_t const v7 = int64_t(v6);
uint64_t const v8 = threadIdx.x;
int64_t const v9 = int64_t(v8);
int64_t const v10 = 256;
int64_t const v11 = v7 * v10;
int64_t const v12 = 1;
int64_t const v14 = 0;
int64_t const v18 = v11 + v9;
bool const v19 = v18 < v51;
if(v19) {
int64_t const v55 = v18 % v53;
int64_t const v57 = v55 ^ v53;
bool const v58 = v57 < v14;
bool const v59 = v55 != v14;
bool const v60 = v59 && v58;
int64_t const v61 = v60 ? v53 : v14;
int64_t const v62 = v55 + v61;
int64_t const v21 = v52 * v62;
int64_t const v22 = v18 / v53;
bool const v26 = v55 < v14;
int64_t const v27 = v22 - v12;
int64_t const v28 = v26 ? v27 : v22;
int64_t const v29 = v21 + v28;
uint64_t const v30 = uint64_t(v29);
__nv_bfloat16 const* const v31 = &v1[v30];
__nv_bfloat16 const v32 = *v31;
float const v33 = float(v32);
__nv_bfloat16 const v34 = __nv_bfloat16(v33);
float const v35 = float(v34);
float const v36 = sin(v35);__nv_bfloat16 const v37 = __nv_bfloat16(v36);
float const v38 = float(v37);
__nv_bfloat16 const v39 = __nv_bfloat16(v38);
float const v40 = float(v39);
float const v41 = 1.0;
float const v42 = v40 + v41;
__nv_bfloat16 const v43 = __nv_bfloat16(v42);
float const v44 = float(v43);
__nv_bfloat16 const v45 = __nv_bfloat16(v44);
uint64_t const v46 = uint64_t(v18);
__nv_bfloat16 * const v47 = &v2[v46];
*v47 = v45;
}
return;
}
