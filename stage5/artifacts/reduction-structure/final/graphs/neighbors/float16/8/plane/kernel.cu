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
#include <cuda_fp16.h>

struct info_st {
    int64_t scalars_i64[2];uint64_t static_meta[2];
};
extern "C" __global__ void __launch_bounds__(128)  cubecl5_9208721374227411(float const* const __restrict__ v1, __half * const __restrict__ v2, info_st const* const __restrict__ v55) {
uint64_t const* const dynamic_meta_v57 = reinterpret_cast<uint64_t const*>(v55 + 1);
const info_st& info_v56 = *v55;int64_t const v59 = info_v56.scalars_i64[1];
uint64_t const v5 = blockIdx.x;
int64_t const v6 = int64_t(v5);
uint64_t const v7 = threadIdx.x;
int64_t const v8 = int64_t(v7);
uint64_t const v9 = threadIdx.y;
int64_t const v10 = int64_t(v9);
int64_t const v11 = 4;
int64_t const v12 = v6 * v11;
int64_t const v13 = v12 + v10;
int64_t const v14 = 0;
int64_t const v15 = 32;
int64_t const v16 = 8;
bool const v17 = v13 < v59;
if(v17) {
float v61_store;
float * v61 = &v61_store;float const v19 = 0.0;
*v61 = v19;
for(int64_t v23 = v8; v23 < v16; v23 += v15) {
int64_t const v25 = v16 * v13;
int64_t const v26 = v23 + v25;
uint64_t const v27 = uint64_t(v26);
float const* const v28 = &v1[v27];
float const v29 = *v28;
float const v31 = *v61;
float const v32 = v31 + v29;
*v61 = v32;
}
float const v33 = *v61;
int32_t const v34 = 16;
float const v35 = __shfl_xor_sync(__activemask(), v33, v34);
float const v36 = v33 + v35;
int32_t const v37 = 8;
float const v38 = __shfl_xor_sync(__activemask(), v36, v37);
float const v39 = v36 + v38;
int32_t const v40 = 4;
float const v41 = __shfl_xor_sync(__activemask(), v39, v40);
float const v42 = v39 + v41;
int32_t const v43 = 2;
float const v44 = __shfl_xor_sync(__activemask(), v42, v43);
float const v45 = v42 + v44;
int32_t const v46 = 1;
float const v47 = __shfl_xor_sync(__activemask(), v45, v46);
float const v48 = v45 + v47;
bool const v49 = v8 == v14;
if(v49) {
__half const v50 = __half(v48);
float const v51 = float(v50);
__half const v52 = __half(v51);
uint64_t const v53 = uint64_t(v13);
__half * const v54 = &v2[v53];
*v54 = v52;
}
}
return;
}
