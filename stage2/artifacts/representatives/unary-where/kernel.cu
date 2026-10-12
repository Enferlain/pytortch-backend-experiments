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
    int64_t scalars_i64[2];uint64_t static_meta[2];
};
extern "C" __global__ void __launch_bounds__(256)  cubecl2_13aaaba68e4fc032(float const* const __restrict__ v1, float * const __restrict__ v2, info_st const* const __restrict__ v26) {
uint64_t const* const dynamic_meta_v28 = reinterpret_cast<uint64_t const*>(v26 + 1);
const info_st& info_v27 = *v26;int64_t const v29 = info_v27.scalars_i64[0];
uint64_t const v5 = blockIdx.x;
int64_t const v6 = int64_t(v5);
uint64_t const v7 = threadIdx.x;
int64_t const v8 = int64_t(v7);
int64_t const v9 = 256;
int64_t const v10 = v6 * v9;
int64_t const v11 = v10 + v8;
bool const v12 = v11 < v29;
if(v12) {
uint64_t const v13 = uint64_t(v11);
float const* const v14 = &v1[v13];
float const v15 = *v14;
float const v16 = 0.0;
bool const v17 = v15 <= v16;
float const v18 = -v15;
float const v19 = exp(v15);float const v20 = 1.0;
float const v21 = v19 - v20;
float const v22 = sqrt(v21);float const v23 = v17 ? v18 : v22;
float * const v25 = &v2[v13];
*v25 = v23;
}
return;
}
