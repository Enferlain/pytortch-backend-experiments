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
    int64_t scalars_i64[2];uint64_t static_meta[3];
};
extern "C" __global__ void __launch_bounds__(256)  cubecl2_5a34aebe35a7d520(bool const* const __restrict__ v1, float const* const __restrict__ v2, float * const __restrict__ v3, info_st const* const __restrict__ v24) {
uint64_t const* const dynamic_meta_v26 = reinterpret_cast<uint64_t const*>(v24 + 1);
const info_st& info_v25 = *v24;int64_t const v27 = info_v25.scalars_i64[0];
uint64_t const v6 = blockIdx.x;
int64_t const v7 = int64_t(v6);
uint64_t const v8 = threadIdx.x;
int64_t const v9 = int64_t(v8);
int64_t const v10 = 256;
int64_t const v11 = v7 * v10;
int64_t const v12 = v11 + v9;
bool const v13 = v12 < v27;
if(v13) {
uint64_t const v14 = uint64_t(v12);
bool const* const v15 = &v1[v14];
bool const v16 = *v15;
float const* const v18 = &v2[v14];
float const v19 = *v18;
float const v20 = -v19;
float const v21 = v16 ? v19 : v20;
float * const v23 = &v3[v14];
*v23 = v21;
}
return;
}
