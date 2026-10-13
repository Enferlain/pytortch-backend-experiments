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
extern "C" __global__ void __launch_bounds__(128)  cubecl5_1fdf55173e1412d4(float const* const __restrict__ v1, float * const __restrict__ v2, info_st const* const __restrict__ v44) {
uint64_t const* const dynamic_meta_v46 = reinterpret_cast<uint64_t const*>(v44 + 1);
const info_st& info_v45 = *v44;int64_t const v48 = info_v45.scalars_i64[1];
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
int64_t const v16 = 9;
bool const v18 = v13 < v48;
if(v18) {
float v52_store;
float * v52 = &v52_store;float const v20 = 0.0;
*v52 = v20;
for(int64_t v24 = v8; v24 < v16; v24 += v15) {
int64_t const v26 = v16 * v13;
int64_t const v27 = v24 + v26;
uint64_t const v28 = uint64_t(v27);
float const* const v29 = &v1[v28];
float const v30 = *v29;
float const v31 = v30 * v30;
float const v32 = *v52;
float const v33 = v32 + v31;
*v52 = v33;
}
float const v34 = *v52;
uint32_t const v53 = warpSize;
uint32_t const v76 = 128;
uint32_t const v55 = min(v53, v76);
float v73_store;
float * v73 = &v73_store;*v73 = v34;
uint32_t const v57 = 1;
uint32_t v74_store;
uint32_t * v74 = &v74_store;*v74 = v57;
while([&]{
uint32_t const v60 = *v74;
bool const v61 = v60 < v55;
return v61;
}()) {
float const v64 = *v73;
uint32_t const v65 = *v74;
float const v66 = __shfl_xor_sync(__activemask(), v64, v65);
float const v68 = v64 + v66;
*v73 = v68;
uint32_t const v70 = 2;
uint32_t const v71 = v65 * v70;
*v74 = v71;
}
float const v72 = *v73;
bool const v36 = v8 == v14;
if(v36) {
float const v37 = 9.0;
float const v38 = v72 / v37;
float const v39 = 9.999999740e-06;
float const v40 = v38 + v39;
float const v41 = rsqrt(v40);uint64_t const v42 = uint64_t(v13);
float * const v43 = &v2[v42];
*v43 = v41;
}
}
return;
}
