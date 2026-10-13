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
extern "C" __global__ void __launch_bounds__(256)  cubecl4_bddc0bf737f78b05(float const* const __restrict__ v1, int64_t const* const __restrict__ v2, float * const __restrict__ v3, info_st const* const __restrict__ v32) {
uint64_t const* const dynamic_meta_v34 = reinterpret_cast<uint64_t const*>(v32 + 1);
const info_st& info_v33 = *v32;int64_t const v35 = info_v33.scalars_i64[0];
uint64_t const v6 = blockIdx.x;
int64_t const v7 = int64_t(v6);
uint64_t const v8 = threadIdx.x;
int64_t const v9 = int64_t(v8);
int64_t const v10 = 256;
int64_t const v11 = v7 * v10;
int64_t const v12 = v11 + v9;
bool const v13 = v12 < v35;
if(v13) {
int64_t const v14 = 128;
bool const v15 = v12 < v14;
float v38_store;
float * v38 = &v38_store;if(v15) {
uint64_t const v17 = uint64_t(v12);
int64_t const* const v18 = &v2[v17];
int64_t const v19 = *v18;
int64_t const v20 = 7;
int64_t const v21 = 0;
bool const v22 = v19 >= v21;
bool const v23 = v19 < v20;
bool const v24 = v22 & v23;
if (!v24) { __trap(); }
uint64_t const v25 = uint64_t(v19);
float const* const v26 = &v1[v25];
float const v27 = *v26;
*v38 = v27;
}
 else {
float const v28 = 7.0;
*v38 = v28;
}
float const v29 = *v38;
uint64_t const v30 = uint64_t(v12);
float * const v31 = &v3[v30];
*v31 = v29;
}
return;
}
