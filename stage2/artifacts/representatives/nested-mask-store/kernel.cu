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
extern "C" __global__ void __launch_bounds__(256)  cubecl2_7e8bd35e0ec2e803(float const* const __restrict__ v1, float * const __restrict__ v2, info_st const* const __restrict__ v33) {
uint64_t const* const dynamic_meta_v35 = reinterpret_cast<uint64_t const*>(v33 + 1);
const info_st& info_v34 = *v33;int64_t const v36 = info_v34.scalars_i64[0];
uint64_t const v5 = blockIdx.x;
int64_t const v6 = int64_t(v5);
uint64_t const v7 = threadIdx.x;
int64_t const v8 = int64_t(v7);
int64_t const v9 = 256;
int64_t const v10 = v6 * v9;
int64_t const v11 = v10 + v8;
bool const v12 = v11 < v36;
if(v12) {
int64_t const v13 = 128;
int64_t const v14 = 64;
bool const v15 = v11 < v13;
bool const v16 = v11 < v14;
float const v17 = -7.0;
uint64_t const v18 = uint64_t(v11);
float * const v19 = &v2[v18];
*v19 = v17;
float v39_store;
float * v39 = &v39_store;if(v15) {
float v40_store;
float * v40 = &v40_store;if(v16) {
int64_t const v22 = 2;
int64_t const v23 = v11 * v22;
uint64_t const v24 = uint64_t(v23);
float const* const v25 = &v1[v24];
float const v26 = *v25;
*v40 = v26;
}
 else {
float const v27 = 9.0;
*v40 = v27;
}
float const v28 = *v40;
*v19 = v28;
*v39 = v28;
}
 else {
float const v31 = -11.0;
*v39 = v31;
}
}
return;
}
