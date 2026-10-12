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
extern "C" __global__ void __launch_bounds__(256)  cubecl2_1539f3de9281af77(float * const __restrict__ v1, info_st const* const __restrict__ v15) {
uint64_t const* const dynamic_meta_v17 = reinterpret_cast<uint64_t const*>(v15 + 1);
const info_st& info_v16 = *v15;int64_t const v18 = info_v16.scalars_i64[0];
uint64_t const v4 = blockIdx.x;
int64_t const v5 = int64_t(v4);
uint64_t const v6 = threadIdx.x;
int64_t const v7 = int64_t(v6);
int64_t const v8 = 256;
int64_t const v9 = v5 * v8;
int64_t const v10 = v9 + v7;
bool const v11 = v10 < v18;
if(v11) {
float const v12 = 0.0;
uint64_t const v13 = uint64_t(v10);
float * const v14 = &v1[v13];
*v14 = v12;
}
return;
}
