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
extern "C" __global__ void __launch_bounds__(256)  cubecl2_eb9de07e161502d8(float const* const __restrict__ v1, float * const __restrict__ v2, info_st const* const __restrict__ v22) {
uint64_t const* const dynamic_meta_v24 = reinterpret_cast<uint64_t const*>(v22 + 1);
const info_st& info_v23 = *v22;int64_t const v25 = info_v23.scalars_i64[0];
uint64_t const v6 = blockIdx.x;
int64_t const v7 = int64_t(v6);
uint64_t const v8 = threadIdx.x;
int64_t const v9 = int64_t(v8);
int64_t const v10 = 256;
int64_t const v11 = v7 * v10;
int64_t const v12 = v11 + v9;
bool const v13 = v12 < v25;
if(v13) {
uint64_t const v14 = uint64_t(v12);
float const* const v15 = &v1[v14];
float const v16 = *v15;
float const v17 = sin(v16);float const v18 = 1.0;
float const v19 = v17 + v18;
float * const v21 = &v2[v14];
*v21 = v19;
}
return;
}
