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
extern "C" __global__ void __launch_bounds__(256)  cubecl5_3d0811ba23ef24a2(__half const* const __restrict__ v1, __half * const __restrict__ v2, info_st const* const __restrict__ v26) {
uint64_t const* const dynamic_meta_v28 = reinterpret_cast<uint64_t const*>(v26 + 1);
const info_st& info_v27 = *v26;int64_t const v29 = info_v27.scalars_i64[0];
uint64_t const v5 = blockIdx.x;
int64_t const v6 = int64_t(v5);
uint64_t const v7 = threadIdx.x;
int64_t const v8 = int64_t(v7);
int64_t const v9 = 256;
int64_t const v10 = v6 * v9;
int64_t const v17 = v10 + v8;
bool const v18 = v17 < v29;
if(v18) {
uint64_t const v19 = uint64_t(v17);
__half const* const v20 = &v1[v19];
__half const v21 = *v20;
float const v22 = float(v21);
__half const v23 = __half(v22);
__half * const v25 = &v2[v19];
*v25 = v23;
}
return;
}
