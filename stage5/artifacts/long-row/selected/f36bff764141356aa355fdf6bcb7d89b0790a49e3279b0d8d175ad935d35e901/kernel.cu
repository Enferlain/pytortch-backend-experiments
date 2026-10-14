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
    int64_t scalars_i64[2];uint64_t static_meta[2];
};
extern "C" __global__ void __launch_bounds__(256)  cubecl5_256f413b1f1dd337(float const* const __restrict__ v1, __nv_bfloat16 * const __restrict__ v2, info_st const* const __restrict__ v41) {
uint64_t const* const dynamic_meta_v43 = reinterpret_cast<uint64_t const*>(v41 + 1);
const info_st& info_v42 = *v41;int64_t const v45 = info_v42.scalars_i64[1];
uint64_t const v5 = blockIdx.x;
int64_t const v6 = int64_t(v5);
uint64_t const v7 = threadIdx.x;
int64_t const v8 = int64_t(v7);
uint64_t const v9 = threadIdx.y;
int64_t const v10 = int64_t(v9);
int64_t const v11 = 8;
int64_t const v12 = v6 * v11;
int64_t const v13 = v12 + v10;
int64_t const v14 = 0;
int64_t const v15 = 32;
bool const v17 = v13 < v45;
if(v17) {
float v47_store;
float * v47 = &v47_store;float const v19 = 0.0;
*v47 = v19;
for(int64_t v23 = v8; v23 < v15; v23 += v15) {
int64_t const v25 = v15 * v13;
int64_t const v26 = v23 + v25;
uint64_t const v27 = uint64_t(v26);
float const* const v28 = &v1[v27];
float const v29 = *v28;
float const v31 = *v47;
float const v32 = v31 + v29;
*v47 = v32;
}
float const v33 = *v47;
uint32_t const v48 = warpSize;
uint32_t const v71 = 256;
uint32_t const v50 = min(v48, v71);
float v68_store;
float * v68 = &v68_store;*v68 = v33;
uint32_t const v52 = 1;
uint32_t v69_store;
uint32_t * v69 = &v69_store;*v69 = v52;
while([&]{
uint32_t const v55 = *v69;
bool const v56 = v55 < v50;
return v56;
}()) {
float const v59 = *v68;
uint32_t const v60 = *v69;
float const v61 = __shfl_xor_sync(__activemask(), v59, v60);
float const v63 = v59 + v61;
*v68 = v63;
uint32_t const v65 = 2;
uint32_t const v66 = v60 * v65;
*v69 = v66;
}
float const v67 = *v68;
bool const v35 = v8 == v14;
if(v35) {
__nv_bfloat16 const v36 = __nv_bfloat16(v67);
float const v37 = float(v36);
__nv_bfloat16 const v38 = __nv_bfloat16(v37);
uint64_t const v39 = uint64_t(v13);
__nv_bfloat16 * const v40 = &v2[v39];
*v40 = v38;
}
}
return;
}
