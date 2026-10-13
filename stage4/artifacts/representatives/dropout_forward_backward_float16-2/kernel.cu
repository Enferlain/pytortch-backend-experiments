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
    int64_t scalars_i64[1];uint64_t static_meta[3];
};
extern "C" __global__ void __launch_bounds__(256)  cubecl4_f55627606b47b29f(bool const* const __restrict__ v1, __half const* const __restrict__ v2, __half * const __restrict__ v3, info_st const* const __restrict__ v35) {
uint64_t const* const dynamic_meta_v37 = reinterpret_cast<uint64_t const*>(v35 + 1);
const info_st& info_v36 = *v35;int64_t const v38 = info_v36.scalars_i64[0];
uint64_t const v5 = blockIdx.x;
int64_t const v6 = int64_t(v5);
uint64_t const v7 = threadIdx.x;
int64_t const v8 = int64_t(v7);
int64_t const v9 = 256;
int64_t const v10 = v6 * v9;
int64_t const v11 = v10 + v8;
bool const v12 = v11 < v38;
if(v12) {
uint64_t const v43 = 0;
bool const* const v15 = &v1[v43];
bool const v16 = *v15;
float const v17 = float(v16);
__half const* const v20 = &v2[v43];
__half const v21 = *v20;
float const v22 = float(v21);
float const v23 = v17 * v22;
__half const v24 = __half(v23);
float const v25 = float(v24);
float const v41 = 1.33300781;
float const v28 = v25 * v41;
__half const v29 = __half(v28);
float const v30 = float(v29);
__half const v31 = __half(v30);
__half * const v34 = &v3[v43];
*v34 = v31;
}
return;
}
