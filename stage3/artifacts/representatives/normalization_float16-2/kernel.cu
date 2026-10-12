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
    int64_t scalars_i64[3];uint64_t static_meta[3];
};
extern "C" __global__ void __launch_bounds__(256)  cubecl2_3565bf76d76bb5d7(__half const* const __restrict__ v1, float const* const __restrict__ v2, __half * const __restrict__ v3, info_st const* const __restrict__ v40) {
uint64_t const* const dynamic_meta_v42 = reinterpret_cast<uint64_t const*>(v40 + 1);
const info_st& info_v41 = *v40;int64_t const v43 = info_v41.scalars_i64[0];
int64_t const v44 = info_v41.scalars_i64[1];
uint64_t const v7 = blockIdx.x;
int64_t const v8 = int64_t(v7);
uint64_t const v9 = threadIdx.x;
int64_t const v10 = int64_t(v9);
int64_t const v11 = 256;
int64_t const v12 = v8 * v11;
int64_t const v13 = v12 + v10;
bool const v14 = v13 < v43;
if(v14) {
uint64_t const v15 = uint64_t(v13);
__half const* const v16 = &v1[v15];
__half const v17 = *v16;
float const v18 = float(v17);
__half const v19 = __half(v18);
float const v20 = float(v19);
int64_t const v22 = v13 / v44;
int64_t const v23 = v13 % v44;
int64_t const v24 = 0;
int64_t const v25 = 1;
bool const v26 = v23 < v24;
int64_t const v27 = v22 - v25;
int64_t const v28 = v26 ? v27 : v22;
uint64_t const v29 = uint64_t(v28);
float const* const v30 = &v2[v29];
float const v31 = *v30;
float const v32 = v20 * v31;
__half const v33 = __half(v32);
float const v34 = float(v33);
__half const v35 = __half(v34);
float const v36 = float(v35);
__half const v37 = __half(v36);
__half * const v39 = &v3[v15];
*v39 = v37;
}
return;
}
