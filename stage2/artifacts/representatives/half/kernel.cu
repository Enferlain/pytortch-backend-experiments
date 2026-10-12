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
extern "C" __global__ void __launch_bounds__(256)  cubecl2_0fb1d1fe81bec527(__half const* const __restrict__ v1, __half * const __restrict__ v2, info_st const* const __restrict__ v32) {
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
uint64_t const v13 = uint64_t(v11);
__half const* const v14 = &v1[v13];
__half const v15 = *v14;
float const v16 = float(v15);
__half const v17 = __half(v16);
float const v18 = float(v17);
float const v19 = sin(v18);__half const v20 = __half(v19);
float const v21 = float(v20);
__half const v22 = __half(v21);
float const v23 = float(v22);
float const v38 = 1.0;
float const v26 = v23 + v38;
__half const v27 = __half(v26);
float const v28 = float(v27);
__half const v29 = __half(v28);
__half * const v31 = &v2[v13];
*v31 = v29;
}
return;
}
