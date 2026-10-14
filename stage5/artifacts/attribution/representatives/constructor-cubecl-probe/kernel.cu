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
extern "C" __global__ void __launch_bounds__(256)  cubecl5_062b34e07584ddb5(__half const* const __restrict__ v1, __half * const __restrict__ v2, info_st const* const __restrict__ v33) {
uint64_t const* const dynamic_meta_v35 = reinterpret_cast<uint64_t const*>(v33 + 1);
const info_st& info_v34 = *v33;int64_t const v36 = info_v34.scalars_i64[0];
uint64_t const v4 = blockIdx.x;
int64_t const v5 = int64_t(v4);
uint64_t const v6 = threadIdx.x;
int64_t const v7 = int64_t(v6);
int64_t const v8 = 256;
int64_t const v9 = v5 * v8;
int64_t const v16 = v9 + v7;
bool const v17 = v16 < v36;
if(v17) {
float const v38 = 0.333251953;
uint64_t const v21 = uint64_t(v16);
__half const* const v22 = &v1[v21];
__half const v23 = *v22;
float const v24 = float(v23);
__half const v25 = __half(v24);
float const v26 = float(v25);
float const v27 = v38 * v26;
__half const v28 = __half(v27);
float const v29 = float(v28);
__half const v30 = __half(v29);
__half * const v32 = &v2[v21];
*v32 = v30;
}
return;
}
