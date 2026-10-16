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
extern "C" __global__ void __launch_bounds__(128)  cubecl5_final_plane_float16(float const* const __restrict__ v1, __half * const __restrict__ v2, info_st const* const __restrict__ v59) {
uint64_t const* const dynamic_meta_v61 = reinterpret_cast<uint64_t const*>(v59 + 1);
const info_st& info_v60 = *v59;int64_t const v63 = info_v60.scalars_i64[1];
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
int64_t const v16 = 9;
bool const v17 = v13 < v63;
if(v17) {
float v65_store;
float * v65 = &v65_store;float const v19 = 0.0;
*v65 = v19;
for(int64_t v23 = v14; v23 < v16; v23 += v15) {
int64_t const v24 = v23 + v8;
float v66_store;
float * v66 = &v66_store;*v66 = v19;
bool const v26 = v24 < v16;
if(v26) {
int64_t const v28 = v16 * v13;
int64_t const v29 = v24 + v28;
uint64_t const v30 = uint64_t(v29);
float const* const v31 = &v1[v30];
float const v32 = *v31;
*v66 = v32;
}
float const v34 = *v66;
float const v35 = *v65;
float const v36 = v35 + v34;
*v65 = v36;
}
float const v37 = *v65;
int32_t const v38 = 16;
float const v39 = __shfl_xor_sync(__activemask(), v37, v38);
float const v40 = v37 + v39;
int32_t const v41 = 8;
float const v42 = __shfl_xor_sync(__activemask(), v40, v41);
float const v43 = v40 + v42;
int32_t const v44 = 4;
float const v45 = __shfl_xor_sync(__activemask(), v43, v44);
float const v46 = v43 + v45;
int32_t const v47 = 2;
float const v48 = __shfl_xor_sync(__activemask(), v46, v47);
float const v49 = v46 + v48;
int32_t const v50 = 1;
float const v51 = __shfl_xor_sync(__activemask(), v49, v50);
float const v52 = v49 + v51;
bool const v53 = v8 == v14;
if(v53) {
__half const v54 = __half(v52);
float const v55 = float(v54);
__half const v56 = __half(v55);
uint64_t const v57 = uint64_t(v13);
__half * const v58 = &v2[v57];
*v58 = v56;
}
}
return;
}
