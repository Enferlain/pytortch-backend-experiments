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

struct info_st {
    uint64_t static_meta[3];
};
extern "C" __global__ void __launch_bounds__(256)  cubecl_fp_loaded_rem_float32(float const* const __restrict__ v1, float const* const __restrict__ v2, float * const __restrict__ v3, info_st const* const __restrict__ v13) {
uint64_t const* const dynamic_meta_v15 = reinterpret_cast<uint64_t const*>(v13 + 1);
const info_st& info_v14 = *v13;uint32_t const v18 = blockIdx.z;
uint64_t const v19 = uint64_t(v18);
uint32_t const v20 = gridDim.y;
uint64_t const v21 = uint64_t(v20);
uint64_t const v22 = v19 * v21;
uint32_t const v23 = gridDim.x;
uint64_t const v24 = uint64_t(v23);
uint64_t const v25 = v22 * v24;
uint32_t const v26 = blockIdx.y;
uint64_t const v27 = uint64_t(v26);
uint64_t const v30 = v27 * v24;
uint64_t const v31 = v25 + v30;
uint32_t const v32 = blockIdx.x;
uint64_t const v33 = uint64_t(v32);
uint64_t const v34 = v31 + v33;
uint64_t const v61 = 256;
uint64_t const v37 = v34 * v61;
uint32_t const v42 = threadIdx.x;
uint64_t const v39 = uint64_t(v42);
uint64_t const v40 = v37 + v39;
uint64_t const v16 = info_v14.static_meta[0];
bool const v6 = v40 < v16;
if(v6) {
float const* const v7 = &v1[v40];
float const v8 = *v7;
float const* const v9 = &v2[v40];
float const v10 = *v9;
float const v11 = fmod(v8, v10);
float * const v12 = &v3[v40];
*v12 = v11;
}
return;
}
