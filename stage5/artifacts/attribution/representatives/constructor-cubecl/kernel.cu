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
    int64_t scalars_i64[1];uint64_t static_meta[2];
};
extern "C" __global__ void __launch_bounds__(256)  cubecl5_0dba5ab15c720310(__half const* const __restrict__ v1, __half * const __restrict__ v2, info_st const* const __restrict__ v31) {
uint64_t const* const dynamic_meta_v33 = reinterpret_cast<uint64_t const*>(v31 + 1);
const info_st& info_v32 = *v31;int64_t const v34 = info_v32.scalars_i64[0];
uint64_t const v4 = blockIdx.x;
int64_t const v5 = int64_t(v4);
uint64_t const v6 = threadIdx.x;
int64_t const v7 = int64_t(v6);
int64_t const v8 = 256;
int64_t const v9 = v5 * v8;
int64_t const v16 = v9 + v7;
bool const v17 = v16 < v34;
if(v17) {
uint64_t const v18 = uint64_t(v16);
__half const* const v19 = &v1[v18];
__half const v20 = *v19;
float const v21 = float(v20);
__half const v22 = __half(v21);
float const v23 = float(v22);
float const v24 = 0.333333343;
float const v25 = v24 * v23;
__half const v26 = __half(v25);
float const v27 = float(v26);
__half const v28 = __half(v27);
__half * const v30 = &v2[v18];
*v30 = v28;
}
return;
}
