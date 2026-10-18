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
    int64_t scalars_i64[7];uint64_t static_meta[2];
};
extern "C" __global__ void __launch_bounds__(256)  cubecl5_layout_caf48303184cc55e(__nv_bfloat16 const* const __restrict__ v1, __nv_bfloat16 * const __restrict__ v2, info_st const* const __restrict__ v86) {
uint64_t const* const dynamic_meta_v88 = reinterpret_cast<uint64_t const*>(v86 + 1);
const info_st& info_v87 = *v86;int64_t const v89 = info_v87.scalars_i64[0];
int64_t const v91 = info_v87.scalars_i64[2];
int64_t const v92 = info_v87.scalars_i64[3];
int64_t const v93 = info_v87.scalars_i64[4];
int64_t const v94 = info_v87.scalars_i64[5];
int64_t const v95 = info_v87.scalars_i64[6];
uint32_t const v11 = uint32_t(v93);
uint64_t const v12 = uint64_t(v91);
uint64_t const v13 = uint64_t(v92);
uint32_t const v15 = uint32_t(v95);
uint64_t const v17 = uint64_t(v94);
uint64_t const v25 = blockIdx.x;
int64_t const v26 = int64_t(v25);
uint64_t const v27 = threadIdx.x;
int64_t const v28 = int64_t(v27);
int64_t const v29 = 256;
int64_t const v30 = v26 * v29;
int64_t const v31 = v30 + v28;
bool const v32 = v31 < v89;
if(v32) {
uint64_t const v33 = uint64_t(v31);
uint64_t const v43 = __umul64hi(v33, v17);
uint64_t const v45 = v43 + v33;
uint64_t const v46 = uint64_t(v15);
uint64_t const v47 = v45 >> v46;
uint64_t const v55 = __umul64hi(v47, v13);
uint64_t const v57 = v55 + v47;
uint64_t const v58 = uint64_t(v11);
uint64_t const v59 = v57 >> v58;
uint64_t const v60 = v59 * v12;
uint64_t const v62 = v47 - v60;
__nv_bfloat16 const* const v69 = &v1[v62];
__nv_bfloat16 const v70 = *v69;
float const v71 = float(v70);
__nv_bfloat16 const v72 = __nv_bfloat16(v71);
float const v73 = float(v72);
float const v74 = sin(v73);__nv_bfloat16 const v75 = __nv_bfloat16(v74);
float const v76 = float(v75);
__nv_bfloat16 const v77 = __nv_bfloat16(v76);
float const v78 = float(v77);
float const v79 = 1.0;
float const v80 = v78 + v79;
__nv_bfloat16 const v81 = __nv_bfloat16(v80);
float const v82 = float(v81);
__nv_bfloat16 const v83 = __nv_bfloat16(v82);
__nv_bfloat16 * const v85 = &v2[v33];
*v85 = v83;
}
return;
}
