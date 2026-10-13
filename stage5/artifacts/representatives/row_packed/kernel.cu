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
extern "C" __global__ void __launch_bounds__(128)  cubecl5_a1f23b555f1720ef(float const* const __restrict__ v1, float * const __restrict__ v2, info_st const* const __restrict__ v36) {
uint64_t const* const dynamic_meta_v38 = reinterpret_cast<uint64_t const*>(v36 + 1);
const info_st& info_v37 = *v36;int64_t const v40 = info_v37.scalars_i64[1];
int64_t const v41 = info_v37.scalars_i64[2];
uint64_t const v6 = blockIdx.x;
int64_t const v7 = int64_t(v6);
uint64_t const v8 = threadIdx.x;
int64_t const v9 = int64_t(v8);
uint64_t const v10 = threadIdx.y;
int64_t const v11 = int64_t(v10);
int64_t const v12 = 4;
int64_t const v13 = v7 * v12;
int64_t const v14 = v13 + v11;
int64_t const v15 = 0;
int64_t const v16 = 32;
bool const v17 = v14 < v41;
if(v17) {
float v43_store;
float * v43 = &v43_store;float const v19 = 0.0;
*v43 = v19;
for(int64_t v23 = v9; v23 < v40; v23 += v16) {
int64_t const v24 = v14 * v40;
int64_t const v25 = v23 + v24;
uint64_t const v26 = uint64_t(v25);
float const* const v27 = &v1[v26];
float const v28 = *v27;
float const v29 = *v43;
float const v30 = v29 + v28;
*v43 = v30;
}
float const v31 = *v43;
uint32_t const v44 = warpSize;
uint32_t const v67 = 128;
uint32_t const v46 = min(v44, v67);
float v64_store;
float * v64 = &v64_store;*v64 = v31;
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
bool const v33 = v9 == v15;
if(v33) {
uint64_t const v34 = uint64_t(v14);
float * const v35 = &v2[v34];
*v35 = v63;
}
}
return;
}
