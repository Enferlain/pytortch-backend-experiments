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
extern "C" __global__ void __launch_bounds__(256)  cubecl5_layout_f43fd452db53d6fd(__nv_bfloat16 const* const __restrict__ v1, __nv_bfloat16 const* const __restrict__ v2, __nv_bfloat16 * const __restrict__ v3, info_st const* const __restrict__ v87) {
uint64_t const* const dynamic_meta_v89 = reinterpret_cast<uint64_t const*>(v87 + 1);
const info_st& info_v88 = *v87;int64_t const v90 = info_v88.scalars_i64[0];
int64_t const v92 = info_v88.scalars_i64[2];
int64_t const v93 = info_v88.scalars_i64[3];
int64_t const v94 = info_v88.scalars_i64[4];
int64_t const v95 = info_v88.scalars_i64[5];
int64_t const v96 = info_v88.scalars_i64[6];
uint32_t const v12 = uint32_t(v94);
uint64_t const v13 = uint64_t(v92);
uint64_t const v14 = uint64_t(v93);
uint32_t const v16 = uint32_t(v96);
uint64_t const v18 = uint64_t(v95);
uint64_t const v26 = blockIdx.x;
int64_t const v27 = int64_t(v26);
uint64_t const v28 = threadIdx.x;
int64_t const v29 = int64_t(v28);
int64_t const v30 = 256;
int64_t const v31 = v27 * v30;
int64_t const v32 = v31 + v29;
bool const v33 = v32 < v90;
if(v33) {
uint64_t const v34 = uint64_t(v32);
__nv_bfloat16 const* const v35 = &v1[v34];
__nv_bfloat16 const v36 = *v35;
float const v37 = float(v36);
__nv_bfloat16 const v38 = __nv_bfloat16(v37);
float const v39 = float(v38);
uint64_t const v50 = __umul64hi(v34, v18);
uint64_t const v52 = v50 + v34;
uint64_t const v53 = uint64_t(v16);
uint64_t const v54 = v52 >> v53;
uint64_t const v62 = __umul64hi(v54, v14);
uint64_t const v64 = v62 + v54;
uint64_t const v65 = uint64_t(v12);
uint64_t const v66 = v64 >> v65;
uint64_t const v67 = v66 * v13;
uint64_t const v69 = v54 - v67;
__nv_bfloat16 const* const v76 = &v2[v69];
__nv_bfloat16 const v77 = *v76;
float const v78 = float(v77);
__nv_bfloat16 const v79 = __nv_bfloat16(v78);
float const v80 = float(v79);
float const v81 = v39 + v80;
__nv_bfloat16 const v82 = __nv_bfloat16(v81);
float const v83 = float(v82);
__nv_bfloat16 const v84 = __nv_bfloat16(v83);
__nv_bfloat16 * const v86 = &v3[v34];
*v86 = v84;
}
return;
}
