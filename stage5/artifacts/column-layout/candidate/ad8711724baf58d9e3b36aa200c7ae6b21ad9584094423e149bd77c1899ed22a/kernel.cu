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
    int64_t scalars_i64[7];uint64_t static_meta[3];
};
extern "C" __global__ void __launch_bounds__(256)  cubecl5_column_ad8711724baf58d9(float const* const __restrict__ v1, float const* const __restrict__ v2, float * const __restrict__ v3, info_st const* const __restrict__ v43) {
uint64_t const* const dynamic_meta_v45 = reinterpret_cast<uint64_t const*>(v43 + 1);
const info_st& info_v44 = *v43;int64_t const v46 = info_v44.scalars_i64[0];
int64_t const v51 = info_v44.scalars_i64[5];
int64_t const v52 = info_v44.scalars_i64[6];
uint32_t const v16 = uint32_t(v52);
uint64_t const v18 = uint64_t(v51);
uint64_t const v22 = blockIdx.x;
int64_t const v23 = int64_t(v22);
uint64_t const v24 = threadIdx.x;
int64_t const v25 = int64_t(v24);
int64_t const v26 = 256;
int64_t const v27 = v23 * v26;
int64_t const v28 = v27 + v25;
bool const v29 = v28 < v46;
if(v29) {
uint64_t const v30 = uint64_t(v28);
float const* const v31 = &v1[v30];
float const v32 = *v31;
uint64_t const v34 = __umul64hi(v30, v18);
uint64_t const v35 = v34 + v30;
uint64_t const v36 = uint64_t(v16);
uint64_t const v37 = v35 >> v36;
float const* const v38 = &v2[v37];
float const v39 = *v38;
float const v40 = v32 + v39;
float * const v42 = &v3[v30];
*v42 = v40;
}
return;
}
