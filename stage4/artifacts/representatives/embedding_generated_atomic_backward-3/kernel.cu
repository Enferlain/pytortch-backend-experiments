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
    int64_t scalars_i64[4];uint64_t static_meta[3];
};
extern "C" __global__ void __launch_bounds__(256)  cubecl4_6b4e98d129f24a92(int64_t const* const __restrict__ v1, float const* const __restrict__ v2, float * const __restrict__ v3, info_st const* const __restrict__ v38) {
uint64_t const* const dynamic_meta_v40 = reinterpret_cast<uint64_t const*>(v38 + 1);
const info_st& info_v39 = *v38;int64_t const v41 = info_v39.scalars_i64[0];
int64_t const v42 = info_v39.scalars_i64[1];
int64_t const v43 = info_v39.scalars_i64[2];
uint64_t const v8 = blockIdx.x;
int64_t const v9 = int64_t(v8);
uint64_t const v10 = threadIdx.x;
int64_t const v11 = int64_t(v10);
int64_t const v12 = 256;
int64_t const v13 = v9 * v12;
int64_t const v14 = v13 + v11;
bool const v15 = v14 < v41;
if(v15) {
int64_t const v16 = v14 / v42;
int64_t const v17 = v14 % v42;
int64_t const v18 = 0;
int64_t const v19 = 1;
bool const v20 = v17 < v18;
int64_t const v21 = v16 - v19;
int64_t const v22 = v20 ? v21 : v16;
uint64_t const v23 = uint64_t(v22);
int64_t const* const v24 = &v1[v23];
int64_t const v25 = *v24;
bool const v27 = v25 >= v18;
bool const v28 = v25 < v43;
bool const v29 = v27 & v28;
if (!v29) { __trap(); }
int64_t const v30 = v25 * v42;
int64_t const v48 = v17 ^ v42;
bool const v49 = v48 < v18;
bool const v50 = v17 != v18;
bool const v51 = v50 && v49;
int64_t const v52 = v51 ? v42 : v18;
int64_t const v53 = v17 + v52;
int64_t const v32 = v30 + v53;
uint64_t const v33 = uint64_t(v32);
float const* const v34 = &v2[v33];
float const v35 = *v34;
uint64_t const v36 = uint64_t(v14);
float * const v37 = &v3[v36];
*v37 = v35;
}
return;
}
