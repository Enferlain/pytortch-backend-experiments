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
    uint64_t static_meta[2];
};
extern "C" __global__ void __launch_bounds__(256)  cubecl_66627c35b250b257(float const* const __restrict__ v1, float * const __restrict__ v2, info_st const* const __restrict__ v12) {
uint64_t const* const dynamic_meta_v14 = reinterpret_cast<uint64_t const*>(v12 + 1);
const info_st& info_v13 = *v12;uint32_t const v17 = blockIdx.z;
uint64_t const v18 = uint64_t(v17);
uint32_t const v19 = gridDim.y;
uint64_t const v20 = uint64_t(v19);
uint64_t const v21 = v18 * v20;
uint32_t const v22 = gridDim.x;
uint64_t const v23 = uint64_t(v22);
uint64_t const v24 = v21 * v23;
uint32_t const v25 = blockIdx.y;
uint64_t const v26 = uint64_t(v25);
uint64_t const v29 = v26 * v23;
uint64_t const v30 = v24 + v29;
uint32_t const v31 = blockIdx.x;
uint64_t const v32 = uint64_t(v31);
uint64_t const v33 = v30 + v32;
uint64_t const v60 = 256;
uint64_t const v36 = v33 * v60;
uint32_t const v41 = threadIdx.x;
uint64_t const v38 = uint64_t(v41);
uint64_t const v39 = v36 + v38;
uint64_t const v15 = info_v13.static_meta[0];
bool const v5 = v39 < v15;
if(v5) {
float const* const v6 = &v1[v39];
float const v7 = *v6;
float const v8 = sin(v7);float const v9 = 1;
float const v10 = v8 + v9;
float * const v11 = &v2[v39];
*v11 = v10;
}
return;
}
