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
extern "C" __global__ void __launch_bounds__(256)  cubecl2_d8dfa68056872427(float const* const __restrict__ v1, float * const __restrict__ v2, info_st const* const __restrict__ v34) {
uint64_t const* const dynamic_meta_v36 = reinterpret_cast<uint64_t const*>(v34 + 1);
const info_st& info_v35 = *v34;int64_t const v37 = info_v35.scalars_i64[0];
int64_t const v39 = info_v35.scalars_i64[2];
uint64_t const v6 = blockIdx.x;
int64_t const v7 = int64_t(v6);
uint64_t const v8 = threadIdx.x;
int64_t const v9 = int64_t(v8);
int64_t const v10 = 256;
int64_t const v11 = v7 * v10;
int64_t const v12 = v11 + v9;
bool const v13 = v12 < v37;
if(v13) {
int64_t const v14 = v39 + v12;
int64_t const v15 = -9;
int64_t const v16 = v14 + v15;
int64_t const v20 = 7;
int64_t const v21 = v16 / v20;
int64_t const v22 = v16 % v20;
int64_t const v23 = 0;
int64_t const v24 = 1;
bool const v25 = v22 < v23;
int64_t const v26 = v21 - v24;
int64_t const v27 = v25 ? v26 : v21;
int64_t const v28 = -7;
int64_t const v29 = v27 * v28;
int64_t const v30 = v16 + v29;
float const v31 = float(v30);
uint64_t const v32 = uint64_t(v12);
float * const v33 = &v2[v32];
*v33 = v31;
}
return;
}
