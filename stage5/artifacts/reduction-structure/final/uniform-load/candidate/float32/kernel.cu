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
extern "C" __global__ void __launch_bounds__(128)  cubecl5_final_plane_float32(float const* const __restrict__ v1, float * const __restrict__ v2, info_st const* const __restrict__ v55) {
uint64_t const* const dynamic_meta_v57 = reinterpret_cast<uint64_t const*>(v55 + 1);
const info_st& info_v56 = *v55;int64_t const v59 = info_v56.scalars_i64[1];
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
bool const v17 = v13 < v59;
if(v17) {
float v61_store;
float * v61 = &v61_store;float const v19 = 0.0;
*v61 = v19;
for(int64_t v23 = v14; v23 < v16; v23 += v15) {
int64_t const v24 = v23 + v8;
float v62_store;
float * v62 = &v62_store;*v62 = v19;
bool const v26 = v24 < v16;
if(v26) {
int64_t const v28 = v16 * v13;
int64_t const v29 = v24 + v28;
uint64_t const v30 = uint64_t(v29);
float const* const v31 = &v1[v30];
float const v32 = *v31;
*v62 = v32;
}
float const v33 = *v62;
float const v34 = *v61;
float const v35 = v34 + v33;
*v61 = v35;
}
float const v36 = *v61;
int32_t const v37 = 16;
float const v38 = __shfl_xor_sync(__activemask(), v36, v37);
float const v39 = v36 + v38;
int32_t const v40 = 8;
float const v41 = __shfl_xor_sync(__activemask(), v39, v40);
float const v42 = v39 + v41;
int32_t const v43 = 4;
float const v44 = __shfl_xor_sync(__activemask(), v42, v43);
float const v45 = v42 + v44;
int32_t const v46 = 2;
float const v47 = __shfl_xor_sync(__activemask(), v45, v46);
float const v48 = v45 + v47;
int32_t const v49 = 1;
float const v50 = __shfl_xor_sync(__activemask(), v48, v49);
float const v51 = v48 + v50;
bool const v52 = v8 == v14;
if(v52) {
uint64_t const v53 = uint64_t(v13);
float * const v54 = &v2[v53];
*v54 = v51;
}
}
return;
}
