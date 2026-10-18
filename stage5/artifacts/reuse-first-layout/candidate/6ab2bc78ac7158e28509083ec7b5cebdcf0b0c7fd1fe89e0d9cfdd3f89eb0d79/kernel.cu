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
    int64_t scalars_i64[5];uint64_t static_meta[2];
};
extern "C" __global__ void __launch_bounds__(256)  cubecl5_layout_6ab2bc78ac7158e2(__half const* const __restrict__ v1, __half * const __restrict__ v2, info_st const* const __restrict__ v64) {
uint64_t const* const dynamic_meta_v66 = reinterpret_cast<uint64_t const*>(v64 + 1);
const info_st& info_v65 = *v64;int64_t const v67 = info_v65.scalars_i64[0];
int64_t const v68 = info_v65.scalars_i64[1];
int64_t const v69 = info_v65.scalars_i64[2];
int64_t const v70 = info_v65.scalars_i64[3];
int64_t const v71 = info_v65.scalars_i64[4];
uint32_t const v9 = uint32_t(v71);
uint64_t const v10 = uint64_t(v69);
uint64_t const v11 = uint64_t(v70);
uint64_t const v13 = uint64_t(v68);
uint64_t const v15 = blockIdx.x;
int64_t const v16 = int64_t(v15);
uint64_t const v17 = threadIdx.x;
int64_t const v18 = int64_t(v17);
int64_t const v19 = 256;
int64_t const v20 = v16 * v19;
int64_t const v21 = v20 + v18;
bool const v22 = v21 < v67;
if(v22) {
uint64_t const v23 = uint64_t(v21);
uint64_t const v33 = __umul64hi(v23, v11);
uint64_t const v35 = v33 + v23;
uint64_t const v36 = uint64_t(v9);
uint64_t const v37 = v35 >> v36;
uint64_t const v38 = v37 * v10;
uint64_t const v40 = v23 - v38;
uint64_t const v41 = v40 * v13;
__half const* const v47 = &v1[v41];
__half const v48 = *v47;
float const v49 = float(v48);
__half const v50 = __half(v49);
float const v51 = float(v50);
float const v52 = sin(v51);__half const v53 = __half(v52);
float const v54 = float(v53);
__half const v55 = __half(v54);
float const v56 = float(v55);
float const v57 = 1.0;
float const v58 = v56 + v57;
__half const v59 = __half(v58);
float const v60 = float(v59);
__half const v61 = __half(v60);
__half * const v63 = &v2[v23];
*v63 = v61;
}
return;
}
