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
extern "C" __global__ void __launch_bounds__(128)  cubecl5_9ace0512031aa9a3(float const* const __restrict__ v1, float * const __restrict__ v2, info_st const* const __restrict__ v51) {
uint64_t const* const dynamic_meta_v53 = reinterpret_cast<uint64_t const*>(v51 + 1);
const info_st& info_v52 = *v51;int64_t const v55 = info_v52.scalars_i64[1];
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
int64_t const v16 = 8;
bool const v17 = v13 < v55;
if(v17) {
float v57_store;
float * v57 = &v57_store;float const v19 = 0.0;
*v57 = v19;
for(int64_t v23 = v8; v23 < v16; v23 += v15) {
int64_t const v25 = v16 * v13;
int64_t const v26 = v23 + v25;
uint64_t const v27 = uint64_t(v26);
float const* const v28 = &v1[v27];
float const v29 = *v28;
float const v30 = *v57;
float const v31 = v30 + v29;
*v57 = v31;
}
float const v32 = *v57;
int32_t const v33 = 16;
float const v34 = __shfl_xor_sync(__activemask(), v32, v33);
float const v35 = v32 + v34;
int32_t const v36 = 8;
float const v37 = __shfl_xor_sync(__activemask(), v35, v36);
float const v38 = v35 + v37;
int32_t const v39 = 4;
float const v40 = __shfl_xor_sync(__activemask(), v38, v39);
float const v41 = v38 + v40;
int32_t const v42 = 2;
float const v43 = __shfl_xor_sync(__activemask(), v41, v42);
float const v44 = v41 + v43;
int32_t const v45 = 1;
float const v46 = __shfl_xor_sync(__activemask(), v44, v45);
float const v47 = v44 + v46;
bool const v48 = v8 == v14;
if(v48) {
uint64_t const v49 = uint64_t(v13);
float * const v50 = &v2[v49];
*v50 = v47;
}
}
return;
}
