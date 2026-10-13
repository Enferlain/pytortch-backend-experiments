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
    int64_t scalars_i64[1];uint64_t static_meta[3];
};
extern "C" __global__ void __launch_bounds__(256)  cubecl4_e30fe16d7bbdd2a1(bool const* const __restrict__ v1, float const* const __restrict__ v2, float * const __restrict__ v3, info_st const* const __restrict__ v28) {
uint64_t const* const dynamic_meta_v30 = reinterpret_cast<uint64_t const*>(v28 + 1);
const info_st& info_v29 = *v28;int64_t const v31 = info_v29.scalars_i64[0];
uint64_t const v5 = blockIdx.x;
int64_t const v6 = int64_t(v5);
uint64_t const v7 = threadIdx.x;
int64_t const v8 = int64_t(v7);
int64_t const v9 = 256;
int64_t const v10 = v6 * v9;
int64_t const v11 = v10 + v8;
bool const v12 = v11 < v31;
if(v12) {
uint64_t const v35 = 0;
bool const* const v15 = &v1[v35];
bool const v16 = *v15;
float const v17 = float(v16);
float const* const v20 = &v2[v35];
float const v21 = *v20;
float const v22 = v17 * v21;
float const v23 = 1.33333337;
float const v24 = v22 * v23;
float * const v27 = &v3[v35];
*v27 = v24;
}
return;
}
