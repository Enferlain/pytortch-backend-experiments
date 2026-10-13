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
extern "C" __global__ void __launch_bounds__(256)  cubecl4_f2036cd26b23a744(int64_t const* const __restrict__ v1, float const* const __restrict__ v2, float * const __restrict__ v3, info_st const* const __restrict__ v34) {
uint64_t const* const dynamic_meta_v36 = reinterpret_cast<uint64_t const*>(v34 + 1);
const info_st& info_v35 = *v34;int64_t const v37 = info_v35.scalars_i64[0];
int64_t const v38 = info_v35.scalars_i64[1];
int64_t const v39 = info_v35.scalars_i64[2];
uint64_t const v7 = blockIdx.x;
int64_t const v8 = int64_t(v7);
uint64_t const v9 = threadIdx.x;
int64_t const v10 = int64_t(v9);
int64_t const v11 = 256;
int64_t const v12 = v8 * v11;
int64_t const v13 = v12 + v10;
bool const v14 = v13 < v37;
if(v14) {
int64_t const v15 = 0;
uint64_t const v42 = 0;
int64_t const* const v17 = &v1[v42];
int64_t const v18 = *v17;
int64_t const v19 = v18 + v38;
bool const v21 = v18 < v15;
int64_t const v22 = v21 ? v19 : v18;
bool const v24 = v22 >= v15;
bool const v25 = v22 < v38;
bool const v26 = v24 & v25;
if (!v26) { __trap(); }
int64_t const v27 = v22 * v39;
uint64_t const v28 = uint64_t(v27);
float const* const v29 = &v2[v28];
float const v30 = *v29;
float * const v33 = &v3[v42];
*v33 = v30;
}
return;
}
