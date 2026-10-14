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
extern "C" __global__ void __launch_bounds__(256)  cubecl5_528835ad936ed85f(__half const* const __restrict__ v1, __half * const __restrict__ v2, info_st const* const __restrict__ v34) {
uint64_t const* const dynamic_meta_v36 = reinterpret_cast<uint64_t const*>(v34 + 1);
const info_st& info_v35 = *v34;int64_t const v37 = info_v35.scalars_i64[0];
uint64_t const v5 = blockIdx.x;
int64_t const v6 = int64_t(v5);
uint64_t const v7 = threadIdx.x;
int64_t const v8 = int64_t(v7);
int64_t const v9 = 256;
int64_t const v10 = v6 * v9;
int64_t const v17 = v10 + v8;
bool const v18 = v17 < v37;
if(v18) {
float const v40 = 0.333251953;
uint64_t const v22 = uint64_t(v17);
__half const* const v23 = &v1[v22];
__half const v24 = *v23;
float const v25 = float(v24);
__half const v26 = __half(v25);
float const v27 = float(v26);
float const v28 = v40 * v27;
__half const v29 = __half(v28);
float const v30 = float(v29);
__half const v31 = __half(v30);
__half * const v33 = &v2[v22];
*v33 = v31;
}
return;
}
