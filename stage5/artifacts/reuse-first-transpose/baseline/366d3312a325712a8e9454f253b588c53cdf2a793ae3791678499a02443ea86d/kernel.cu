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
extern "C" __global__ void __launch_bounds__(256)  cubecl5_350b06e7d7838d0f(float const* const __restrict__ v1, float * const __restrict__ v2, info_st const* const __restrict__ v35) {
uint64_t const* const dynamic_meta_v37 = reinterpret_cast<uint64_t const*>(v35 + 1);
const info_st& info_v36 = *v35;int64_t const v38 = info_v36.scalars_i64[0];
int64_t const v39 = info_v36.scalars_i64[1];
int64_t const v40 = info_v36.scalars_i64[2];
uint64_t const v6 = blockIdx.x;
int64_t const v7 = int64_t(v6);
uint64_t const v8 = threadIdx.x;
int64_t const v9 = int64_t(v8);
int64_t const v10 = 256;
int64_t const v11 = v7 * v10;
int64_t const v12 = 1;
int64_t const v14 = 0;
int64_t const v18 = v11 + v9;
bool const v19 = v18 < v38;
if(v19) {
int64_t const v42 = v18 % v40;
int64_t const v44 = v42 ^ v40;
bool const v45 = v44 < v14;
bool const v46 = v42 != v14;
bool const v47 = v46 && v45;
int64_t const v48 = v47 ? v40 : v14;
int64_t const v49 = v42 + v48;
int64_t const v21 = v39 * v49;
int64_t const v22 = v18 / v40;
bool const v26 = v42 < v14;
int64_t const v27 = v22 - v12;
int64_t const v28 = v26 ? v27 : v22;
int64_t const v29 = v21 + v28;
uint64_t const v30 = uint64_t(v29);
float const* const v31 = &v1[v30];
float const v32 = *v31;
uint64_t const v33 = uint64_t(v18);
float * const v34 = &v2[v33];
*v34 = v32;
}
return;
}
