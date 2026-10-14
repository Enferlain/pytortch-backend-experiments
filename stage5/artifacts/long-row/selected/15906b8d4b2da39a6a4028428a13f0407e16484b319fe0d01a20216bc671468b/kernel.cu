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
    int64_t scalars_i64[2];uint64_t static_meta[1];
};
extern "C" __global__ void __launch_bounds__(256)  cubecl5_3a38b13cfe72a595(float * const __restrict__ v1, info_st const* const __restrict__ v21) {
uint64_t const* const dynamic_meta_v23 = reinterpret_cast<uint64_t const*>(v21 + 1);
const info_st& info_v22 = *v21;int64_t const v24 = info_v22.scalars_i64[0];
uint64_t const v4 = blockIdx.x;
int64_t const v5 = int64_t(v4);
uint64_t const v6 = threadIdx.x;
int64_t const v7 = int64_t(v6);
int64_t const v8 = 256;
int64_t const v9 = v5 * v8;
int64_t const v16 = v9 + v7;
bool const v17 = v16 < v24;
if(v17) {
float const v18 = 0.0;
uint64_t const v19 = uint64_t(v16);
float * const v20 = &v1[v19];
*v20 = v18;
}
return;
}
