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
extern "C" __global__ void __launch_bounds__(256)  cubecl2_019f60b7092bd043(float const* const __restrict__ v1, float const* const __restrict__ v2, float * const __restrict__ v3, info_st const* const __restrict__ v31) {
uint64_t const* const dynamic_meta_v33 = reinterpret_cast<uint64_t const*>(v31 + 1);
const info_st& info_v32 = *v31;int64_t const v34 = info_v32.scalars_i64[0];
int64_t const v35 = info_v32.scalars_i64[1];
uint64_t const v7 = blockIdx.x;
int64_t const v8 = int64_t(v7);
uint64_t const v9 = threadIdx.x;
int64_t const v10 = int64_t(v9);
int64_t const v11 = 256;
int64_t const v12 = v8 * v11;
int64_t const v13 = v12 + v10;
bool const v14 = v13 < v34;
if(v14) {
uint64_t const v15 = uint64_t(v13);
float const* const v16 = &v1[v15];
float const v17 = *v16;
int64_t const v18 = v13 / v35;
int64_t const v19 = v13 % v35;
int64_t const v20 = 0;
int64_t const v21 = 1;
bool const v22 = v19 < v20;
int64_t const v23 = v18 - v21;
int64_t const v24 = v22 ? v23 : v18;
uint64_t const v25 = uint64_t(v24);
float const* const v26 = &v2[v25];
float const v27 = *v26;
float const v28 = v17 * v27;
float * const v30 = &v3[v15];
*v30 = v28;
}
return;
}
