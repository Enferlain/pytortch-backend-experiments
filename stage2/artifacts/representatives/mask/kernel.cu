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
extern "C" __global__ void __launch_bounds__(256)  cubecl2_77c8a9b1209e694e(float const* const __restrict__ v1, float * const __restrict__ v2, info_st const* const __restrict__ v38) {
uint64_t const* const dynamic_meta_v40 = reinterpret_cast<uint64_t const*>(v38 + 1);
const info_st& info_v39 = *v38;int64_t const v41 = info_v39.scalars_i64[0];
int64_t const v42 = info_v39.scalars_i64[1];
uint64_t const v5 = blockIdx.x;
int64_t const v6 = int64_t(v5);
uint64_t const v7 = threadIdx.x;
int64_t const v8 = int64_t(v7);
int64_t const v9 = 256;
int64_t const v10 = v6 * v9;
int64_t const v11 = v10 + v8;
bool const v12 = v11 < v41;
if(v12) {
bool const v15 = v11 < v42;
float v44_store;
float * v44 = &v44_store;if(v15) {
uint64_t const v17 = uint64_t(v11);
float const* const v18 = &v1[v17];
float const v19 = *v18;
*v44 = v19;
}
 else {
float const v20 = 0;
*v44 = v20;
}
float const v21 = *v44;
bool const v22 = v11 >= v42;
float v45_store;
float * v45 = &v45_store;if(v22) {
uint64_t const v46 = 0;
float const* const v29 = &v1[v46];
float const v30 = *v29;
*v45 = v30;
}
 else {
float const v31 = 0;
*v45 = v31;
}
float const v32 = *v45;
float const v33 = v15 ? v21 : v32;
float const v34 = 1;
float const v35 = v33 + v34;
uint64_t const v36 = uint64_t(v11);
float * const v37 = &v2[v36];
*v37 = v35;
}
return;
}
