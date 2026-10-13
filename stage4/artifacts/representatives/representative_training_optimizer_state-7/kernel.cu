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
    int64_t scalars_i64[1];uint64_t static_meta[1];
};
extern "C" __global__ void __launch_bounds__(256)  cubecl4_0cc2284e3e64bf91(float * const __restrict__ v1, info_st const* const __restrict__ v14) {
uint64_t const* const dynamic_meta_v16 = reinterpret_cast<uint64_t const*>(v14 + 1);
const info_st& info_v15 = *v14;int64_t const v17 = info_v15.scalars_i64[0];
uint64_t const v3 = blockIdx.x;
int64_t const v4 = int64_t(v3);
uint64_t const v5 = threadIdx.x;
int64_t const v6 = int64_t(v5);
int64_t const v7 = 256;
int64_t const v8 = v4 * v7;
int64_t const v9 = v8 + v6;
bool const v10 = v9 < v17;
if(v10) {
float const v11 = 0.0;
uint64_t const v12 = uint64_t(v9);
float * const v13 = &v1[v12];
*v13 = v11;
}
return;
}
