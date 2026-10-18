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
    int64_t scalars_i64[3];uint64_t static_meta[2];
};
extern "C" __global__ void __launch_bounds__(256)  cubecl5_154628e287e9e3de(float const* const __restrict__ v1, float * const __restrict__ v2, info_st const* const __restrict__ v28) {
uint64_t const* const dynamic_meta_v30 = reinterpret_cast<uint64_t const*>(v28 + 1);
const info_st& info_v29 = *v28;int64_t const v31 = info_v29.scalars_i64[0];
uint64_t const v6 = blockIdx.x;
int64_t const v7 = int64_t(v6);
uint64_t const v8 = threadIdx.x;
int64_t const v9 = int64_t(v8);
int64_t const v10 = 256;
int64_t const v11 = v7 * v10;
int64_t const v18 = v11 + v9;
bool const v19 = v18 < v31;
if(v19) {
uint64_t const v20 = uint64_t(v18);
float const* const v21 = &v1[v20];
float const v22 = *v21;
float const v23 = sin(v22);float const v24 = 1.0;
float const v25 = v23 + v24;
float * const v27 = &v2[v20];
*v27 = v25;
}
return;
}
