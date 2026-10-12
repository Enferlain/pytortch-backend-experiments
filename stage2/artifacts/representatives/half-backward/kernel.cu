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
    int64_t scalars_i64[2];uint64_t static_meta[3];
};
extern "C" __global__ void __launch_bounds__(256)  cubecl2_0cdd46aaf3bc4827(__half const* const __restrict__ v1, __half const* const __restrict__ v2, __half * const __restrict__ v3, info_st const* const __restrict__ v37) {
uint64_t const* const dynamic_meta_v39 = reinterpret_cast<uint64_t const*>(v37 + 1);
const info_st& info_v38 = *v37;int64_t const v40 = info_v38.scalars_i64[0];
uint64_t const v6 = blockIdx.x;
int64_t const v7 = int64_t(v6);
uint64_t const v8 = threadIdx.x;
int64_t const v9 = int64_t(v8);
int64_t const v10 = 256;
int64_t const v11 = v7 * v10;
int64_t const v12 = v11 + v9;
bool const v13 = v12 < v40;
if(v13) {
uint64_t const v14 = uint64_t(v12);
__half const* const v15 = &v1[v14];
__half const v16 = *v15;
float const v17 = float(v16);
__half const v18 = __half(v17);
float const v19 = float(v18);
__half const* const v21 = &v2[v14];
__half const v22 = *v21;
float const v23 = float(v22);
__half const v24 = __half(v23);
float const v25 = float(v24);
float const v26 = cos(v25);__half const v27 = __half(v26);
float const v28 = float(v27);
__half const v29 = __half(v28);
float const v30 = float(v29);
float const v31 = v19 * v30;
__half const v32 = __half(v31);
float const v33 = float(v32);
__half const v34 = __half(v33);
__half * const v36 = &v3[v14];
*v36 = v34;
}
return;
}
