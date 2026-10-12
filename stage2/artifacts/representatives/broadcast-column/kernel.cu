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
    int64_t scalars_i64[3];uint64_t static_meta[3];
};
extern "C" __global__ void __launch_bounds__(256)  cubecl2_5c48ec828d4c7c5b(float const* const __restrict__ v1, float const* const __restrict__ v2, float * const __restrict__ v3, info_st const* const __restrict__ v32) {
uint64_t const* const dynamic_meta_v34 = reinterpret_cast<uint64_t const*>(v32 + 1);
const info_st& info_v33 = *v32;int64_t const v35 = info_v33.scalars_i64[0];
int64_t const v36 = info_v33.scalars_i64[1];
uint64_t const v7 = blockIdx.x;
int64_t const v8 = int64_t(v7);
uint64_t const v9 = threadIdx.x;
int64_t const v10 = int64_t(v9);
int64_t const v11 = 256;
int64_t const v12 = v8 * v11;
int64_t const v13 = v12 + v10;
bool const v14 = v13 < v35;
if(v14) {
uint64_t const v15 = uint64_t(v13);
float const* const v16 = &v1[v15];
float const v17 = *v16;
float const v18 = sin(v17);int64_t const v19 = v13 / v36;
int64_t const v20 = v13 % v36;
int64_t const v21 = 0;
int64_t const v22 = 1;
bool const v23 = v20 < v21;
int64_t const v24 = v19 - v22;
int64_t const v25 = v23 ? v24 : v19;
uint64_t const v26 = uint64_t(v25);
float const* const v27 = &v2[v26];
float const v28 = *v27;
float const v29 = v18 + v28;
float * const v31 = &v3[v15];
*v31 = v29;
}
return;
}
