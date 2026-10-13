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
extern "C" __global__ void __launch_bounds__(256)  cubecl4_9e12d88b7af97fb0(__half const* const __restrict__ v1, bool const* const __restrict__ v2, __half * const __restrict__ v3, info_st const* const __restrict__ v32) {
uint64_t const* const dynamic_meta_v34 = reinterpret_cast<uint64_t const*>(v32 + 1);
const info_st& info_v33 = *v32;int64_t const v35 = info_v33.scalars_i64[0];
uint64_t const v5 = blockIdx.x;
int64_t const v6 = int64_t(v5);
uint64_t const v7 = threadIdx.x;
int64_t const v8 = int64_t(v7);
int64_t const v9 = 256;
int64_t const v10 = v6 * v9;
int64_t const v11 = v10 + v8;
bool const v12 = v11 < v35;
if(v12) {
uint64_t const v40 = 0;
__half const* const v15 = &v1[v40];
__half const v16 = *v15;
float const v17 = float(v16);
bool const* const v20 = &v2[v40];
bool const v21 = *v20;
__half const v22 = __half(v21);
float const v23 = float(v22);
float const v38 = 1.33300781;
float const v26 = v23 * v38;
float const v27 = v17 * v26;
__half const v28 = __half(v27);
__half * const v31 = &v3[v40];
*v31 = v28;
}
return;
}
