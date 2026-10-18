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
    int64_t scalars_i64[3];uint64_t static_meta[2];
};
extern "C" __global__ void __launch_bounds__(256)  cubecl5_b60977cd54c04aea(__half const* const __restrict__ v1, __half * const __restrict__ v2, info_st const* const __restrict__ v38) {
uint64_t const* const dynamic_meta_v40 = reinterpret_cast<uint64_t const*>(v38 + 1);
const info_st& info_v39 = *v38;int64_t const v41 = info_v39.scalars_i64[0];
uint64_t const v6 = blockIdx.x;
int64_t const v7 = int64_t(v6);
uint64_t const v8 = threadIdx.x;
int64_t const v9 = int64_t(v8);
int64_t const v10 = 256;
int64_t const v11 = v7 * v10;
int64_t const v18 = v11 + v9;
bool const v19 = v18 < v41;
if(v19) {
uint64_t const v20 = uint64_t(v18);
__half const* const v21 = &v1[v20];
__half const v22 = *v21;
float const v23 = float(v22);
__half const v24 = __half(v23);
float const v25 = float(v24);
float const v26 = sin(v25);__half const v27 = __half(v26);
float const v28 = float(v27);
__half const v29 = __half(v28);
float const v30 = float(v29);
float const v31 = 1.0;
float const v32 = v30 + v31;
__half const v33 = __half(v32);
float const v34 = float(v33);
__half const v35 = __half(v34);
__half * const v37 = &v2[v20];
*v37 = v35;
}
return;
}
