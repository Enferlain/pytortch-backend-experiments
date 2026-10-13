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
extern "C" __global__ void __launch_bounds__(256)  cubecl5_8fe48112363dde63(int64_t const* const __restrict__ v1, float const* const __restrict__ v2, float * const __restrict__ v3, info_st const* const __restrict__ v46) {
uint64_t const* const dynamic_meta_v48 = reinterpret_cast<uint64_t const*>(v46 + 1);
const info_st& info_v47 = *v46;int64_t const v49 = info_v47.scalars_i64[0];
uint64_t const v6 = blockIdx.x;
int64_t const v7 = int64_t(v6);
uint64_t const v8 = threadIdx.x;
int64_t const v9 = int64_t(v8);
int64_t const v10 = 256;
int64_t const v11 = v7 * v10;
int64_t const v12 = 1;
int64_t const v14 = 0;
int64_t const v18 = v11 + v9;
bool const v19 = v18 < v49;
if(v19) {
int64_t const v20 = 7;
int64_t const v21 = v18 / v20;
int64_t const v22 = v18 % v20;
bool const v25 = v22 < v14;
int64_t const v26 = v21 - v12;
int64_t const v27 = v25 ? v26 : v21;
uint64_t const v28 = uint64_t(v27);
int64_t const* const v29 = &v1[v28];
int64_t const v30 = *v29;
int64_t const v31 = 17;
bool const v33 = v30 >= v14;
bool const v34 = v30 < v31;
bool const v35 = v33 & v34;
if (!v35) { __trap(); }
int64_t const v37 = v20 * v30;
int64_t const v54 = v22 ^ v20;
bool const v55 = v54 < v14;
bool const v56 = v22 != v14;
bool const v57 = v56 && v55;
int64_t const v58 = v57 ? v20 : v14;
int64_t const v59 = v22 + v58;
int64_t const v40 = v37 + v59;
uint64_t const v41 = uint64_t(v40);
float const* const v42 = &v2[v41];
float const v43 = *v42;
uint64_t const v44 = uint64_t(v18);
float * const v45 = &v3[v44];
*v45 = v43;
}
return;
}
