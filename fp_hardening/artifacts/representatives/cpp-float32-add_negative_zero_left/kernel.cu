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
extern "C" __global__ void __launch_bounds__(256)  cubecl_fp_add_negative_zero_left_float32(float const* const __restrict__ v1, float const* const __restrict__ v2, float * const __restrict__ v3, info_st const* const __restrict__ v14) {
uint64_t const* const dynamic_meta_v16 = reinterpret_cast<uint64_t const*>(v14 + 1);
const info_st& info_v15 = *v14;uint32_t const v19 = blockIdx.z;
uint64_t const v20 = uint64_t(v19);
uint32_t const v21 = gridDim.y;
uint64_t const v22 = uint64_t(v21);
uint64_t const v23 = v20 * v22;
uint32_t const v24 = gridDim.x;
uint64_t const v25 = uint64_t(v24);
uint64_t const v26 = v23 * v25;
uint32_t const v27 = blockIdx.y;
uint64_t const v28 = uint64_t(v27);
uint64_t const v31 = v28 * v25;
uint64_t const v32 = v26 + v31;
uint32_t const v33 = blockIdx.x;
uint64_t const v34 = uint64_t(v33);
uint64_t const v35 = v32 + v34;
uint64_t const v62 = 256;
uint64_t const v38 = v35 * v62;
uint32_t const v43 = threadIdx.x;
uint64_t const v40 = uint64_t(v43);
uint64_t const v41 = v38 + v40;
uint64_t const v17 = info_v15.static_meta[0];
bool const v6 = v41 < v17;
if(v6) {
float const* const v7 = &v1[v41];
float const v8 = *v7;
float const v11 = -0.0;
float const v12 = v11 + v8;
float * const v13 = &v3[v41];
*v13 = v12;
}
return;
}
