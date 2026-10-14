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
extern "C" __global__ void __launch_bounds__(128)  cubecl5_a269a53acef989a6(float const* const __restrict__ v1, float * const __restrict__ v2, info_st const* const __restrict__ v37) {
uint64_t const* const dynamic_meta_v39 = reinterpret_cast<uint64_t const*>(v37 + 1);
const info_st& info_v38 = *v37;int64_t const v41 = info_v38.scalars_i64[1];
uint64_t const v5 = blockIdx.x;
int64_t const v6 = int64_t(v5);
uint64_t const v7 = threadIdx.x;
int64_t const v8 = int64_t(v7);
uint64_t const v9 = threadIdx.y;
int64_t const v10 = int64_t(v9);
int64_t const v11 = 4;
int64_t const v12 = v6 * v11;
int64_t const v13 = v12 + v10;
int64_t const v14 = 0;
int64_t const v15 = 32;
bool const v17 = v13 < v41;
if(v17) {
float v43_store;
float * v43 = &v43_store;float const v19 = 0.0;
*v43 = v19;
for(int64_t v23 = v8; v23 < v15; v23 += v15) {
int64_t const v25 = v15 * v13;
int64_t const v26 = v23 + v25;
uint64_t const v27 = uint64_t(v26);
float const* const v28 = &v1[v27];
float const v29 = *v28;
float const v30 = *v43;
float const v31 = v30 + v29;
*v43 = v31;
}
float const v32 = *v43;
uint32_t const v44 = warpSize;
uint32_t const v67 = 128;
uint32_t const v46 = min(v44, v67);
float v64_store;
float * v64 = &v64_store;*v64 = v32;
uint32_t const v48 = 1;
uint32_t v65_store;
uint32_t * v65 = &v65_store;*v65 = v48;
while([&]{
uint32_t const v51 = *v65;
bool const v52 = v51 < v46;
return v52;
}()) {
float const v55 = *v64;
uint32_t const v56 = *v65;
float const v57 = __shfl_xor_sync(__activemask(), v55, v56);
float const v59 = v55 + v57;
*v64 = v59;
uint32_t const v61 = 2;
uint32_t const v62 = v56 * v61;
*v65 = v62;
}
float const v63 = *v64;
bool const v34 = v8 == v14;
if(v34) {
uint64_t const v35 = uint64_t(v13);
float * const v36 = &v2[v35];
*v36 = v63;
}
}
return;
}
