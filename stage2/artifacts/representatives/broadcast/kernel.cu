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
extern "C" __global__ void __launch_bounds__(256)  cubecl2_007dea30664a414a(float const* const __restrict__ v1, float const* const __restrict__ v2, float * const __restrict__ v3, info_st const* const __restrict__ v26) {
uint64_t const* const dynamic_meta_v28 = reinterpret_cast<uint64_t const*>(v26 + 1);
const info_st& info_v27 = *v26;int64_t const v29 = info_v27.scalars_i64[0];
int64_t const v30 = info_v27.scalars_i64[1];
uint64_t const v7 = blockIdx.x;
int64_t const v8 = int64_t(v7);
uint64_t const v9 = threadIdx.x;
int64_t const v10 = int64_t(v9);
int64_t const v11 = 256;
int64_t const v12 = v8 * v11;
int64_t const v13 = v12 + v10;
bool const v14 = v13 < v29;
if(v14) {
uint64_t const v15 = uint64_t(v13);
float const* const v16 = &v1[v15];
float const v17 = *v16;
float const v18 = sin(v17);int64_t const v33 = v13 % v30;
int64_t const v34 = 0;
int64_t const v35 = v33 ^ v30;
bool const v36 = v35 < v34;
bool const v37 = v33 != v34;
bool const v38 = v37 && v36;
int64_t const v39 = v38 ? v30 : v34;
int64_t const v40 = v33 + v39;
uint64_t const v20 = uint64_t(v40);
float const* const v21 = &v2[v20];
float const v22 = *v21;
float const v23 = v18 + v22;
float * const v25 = &v3[v15];
*v25 = v23;
}
return;
}
