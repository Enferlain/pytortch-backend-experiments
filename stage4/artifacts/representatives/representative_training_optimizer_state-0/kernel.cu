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
extern "C" __global__ void __launch_bounds__(256)  cubecl4_cb3d90ed69c26973(int64_t const* const __restrict__ v1, float const* const __restrict__ v2, float * const __restrict__ v3, info_st const* const __restrict__ v40) {
uint64_t const* const dynamic_meta_v42 = reinterpret_cast<uint64_t const*>(v40 + 1);
const info_st& info_v41 = *v40;int64_t const v43 = info_v41.scalars_i64[0];
uint64_t const v6 = blockIdx.x;
int64_t const v7 = int64_t(v6);
uint64_t const v8 = threadIdx.x;
int64_t const v9 = int64_t(v8);
int64_t const v10 = 256;
int64_t const v11 = v7 * v10;
int64_t const v12 = v11 + v9;
bool const v13 = v12 < v43;
if(v13) {
int64_t const v14 = 7;
int64_t const v15 = v12 / v14;
int64_t const v16 = v12 % v14;
int64_t const v17 = 0;
int64_t const v18 = 1;
bool const v19 = v16 < v17;
int64_t const v20 = v15 - v18;
int64_t const v21 = v19 ? v20 : v15;
uint64_t const v22 = uint64_t(v21);
int64_t const* const v23 = &v1[v22];
int64_t const v24 = *v23;
int64_t const v25 = 17;
bool const v27 = v24 >= v17;
bool const v28 = v24 < v25;
bool const v29 = v27 & v28;
if (!v29) { __trap(); }
int64_t const v31 = v14 * v24;
int64_t const v48 = v16 ^ v14;
bool const v49 = v48 < v17;
bool const v50 = v16 != v17;
bool const v51 = v50 && v49;
int64_t const v52 = v51 ? v14 : v17;
int64_t const v53 = v16 + v52;
int64_t const v34 = v31 + v53;
uint64_t const v35 = uint64_t(v34);
float const* const v36 = &v2[v35];
float const v37 = *v36;
uint64_t const v38 = uint64_t(v12);
float * const v39 = &v3[v38];
*v39 = v37;
}
return;
}
