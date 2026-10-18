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
extern "C" __global__ void __launch_bounds__(128)  cubecl5_6d0c0bb351246543(float const* const __restrict__ v1, float const* const __restrict__ v2, float * const __restrict__ v3, info_st const* const __restrict__ v31) {
uint64_t const* const dynamic_meta_v33 = reinterpret_cast<uint64_t const*>(v31 + 1);
const info_st& info_v32 = *v31;int64_t const v34 = info_v32.scalars_i64[0];
int64_t const v35 = info_v32.scalars_i64[1];
uint64_t const v7 = blockIdx.x;
int64_t const v8 = int64_t(v7);
uint64_t const v9 = threadIdx.x;
int64_t const v10 = int64_t(v9);
int64_t const v11 = 128;
int64_t const v12 = v8 * v11;
int64_t const v15 = 0;
int64_t const v19 = v12 + v10;
bool const v20 = v19 < v34;
if(v20) {
uint64_t const v21 = uint64_t(v19);
float const* const v22 = &v1[v21];
float const v23 = *v22;
int64_t const v38 = v19 % v35;
int64_t const v40 = v38 ^ v35;
bool const v41 = v40 < v15;
bool const v42 = v38 != v15;
bool const v43 = v42 && v41;
int64_t const v44 = v43 ? v35 : v15;
int64_t const v45 = v38 + v44;
uint64_t const v25 = uint64_t(v45);
float const* const v26 = &v2[v25];
float const v27 = *v26;
float const v28 = v23 + v27;
float * const v30 = &v3[v21];
*v30 = v28;
}
return;
}
