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
extern "C" __global__ void __launch_bounds__(256)  cubecl4_f0134a174f6b741f(float const* const __restrict__ v1, int64_t const* const __restrict__ v2, float * const __restrict__ v3, info_st const* const __restrict__ v37) {
uint64_t const* const dynamic_meta_v39 = reinterpret_cast<uint64_t const*>(v37 + 1);
const info_st& info_v38 = *v37;int64_t const v40 = info_v38.scalars_i64[0];
uint64_t const v6 = blockIdx.x;
int64_t const v7 = int64_t(v6);
uint64_t const v8 = threadIdx.x;
int64_t const v9 = int64_t(v8);
int64_t const v10 = 256;
int64_t const v11 = v7 * v10;
int64_t const v12 = v11 + v9;
bool const v13 = v12 < v40;
if(v13) {
int64_t const v14 = 128;
bool const v15 = v12 < v14;
float v43_store;
float * v43 = &v43_store;if(v15) {
int64_t const v17 = 64;
bool const v18 = v12 < v17;
float v44_store;
float * v44 = &v44_store;if(v18) {
uint64_t const v20 = uint64_t(v12);
int64_t const* const v21 = &v2[v20];
int64_t const v22 = *v21;
int64_t const v23 = 7;
int64_t const v24 = 0;
bool const v25 = v22 >= v24;
bool const v26 = v22 < v23;
bool const v27 = v25 & v26;
if (!v27) { __trap(); }
uint64_t const v28 = uint64_t(v22);
float const* const v29 = &v1[v28];
float const v30 = *v29;
*v44 = v30;
}
 else {
float const v31 = 7.0;
*v44 = v31;
}
float const v32 = *v44;
*v43 = v32;
}
 else {
float const v33 = 7.0;
*v43 = v33;
}
float const v34 = *v43;
uint64_t const v35 = uint64_t(v12);
float * const v36 = &v3[v35];
*v36 = v34;
}
return;
}
