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
    int64_t scalars_i64[2];uint64_t static_meta[1];
};
extern "C" __global__ void __launch_bounds__(256)  cubecl4_dc81d64514a942a8(__half * const __restrict__ v1, info_st const* const __restrict__ v17) {
uint64_t const* const dynamic_meta_v19 = reinterpret_cast<uint64_t const*>(v17 + 1);
const info_st& info_v18 = *v17;int64_t const v20 = info_v18.scalars_i64[0];
uint64_t const v4 = blockIdx.x;
int64_t const v5 = int64_t(v4);
uint64_t const v6 = threadIdx.x;
int64_t const v7 = int64_t(v6);
int64_t const v8 = 256;
int64_t const v9 = v5 * v8;
int64_t const v10 = v9 + v7;
bool const v11 = v10 < v20;
if(v11) {
__half const v12 = 0.0;
uint64_t const v15 = uint64_t(v10);
__half * const v16 = &v1[v15];
*v16 = v12;
}
return;
}
