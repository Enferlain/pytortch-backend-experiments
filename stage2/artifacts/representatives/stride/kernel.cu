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
extern "C" __global__ void __launch_bounds__(256)  cubecl2_5a9e7f8baf65b146(float const* const __restrict__ v1, float * const __restrict__ v2, info_st const* const __restrict__ v23) {
uint64_t const* const dynamic_meta_v25 = reinterpret_cast<uint64_t const*>(v23 + 1);
const info_st& info_v24 = *v23;int64_t const v26 = info_v24.scalars_i64[0];
int64_t const v27 = info_v24.scalars_i64[1];
uint64_t const v6 = blockIdx.x;
int64_t const v7 = int64_t(v6);
uint64_t const v8 = threadIdx.x;
int64_t const v9 = int64_t(v8);
int64_t const v10 = 256;
int64_t const v11 = v7 * v10;
int64_t const v12 = v11 + v9;
bool const v13 = v12 < v26;
if(v13) {
int64_t const v14 = v12 * v27;
uint64_t const v15 = uint64_t(v14);
float const* const v16 = &v1[v15];
float const v17 = *v16;
float const v18 = sin(v17);float const v19 = 1.0;
float const v20 = v18 + v19;
uint64_t const v21 = uint64_t(v12);
float * const v22 = &v2[v21];
*v22 = v20;
}
return;
}
