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
extern "C" __global__ void __launch_bounds__(256)  cubecl2_853f1cfa9d5700ec(float const* const __restrict__ v1, float * const __restrict__ v2, info_st const* const __restrict__ v30) {
uint64_t const* const dynamic_meta_v32 = reinterpret_cast<uint64_t const*>(v30 + 1);
const info_st& info_v31 = *v30;int64_t const v33 = info_v31.scalars_i64[0];
int64_t const v35 = info_v31.scalars_i64[2];
uint64_t const v6 = blockIdx.x;
int64_t const v7 = int64_t(v6);
uint64_t const v8 = threadIdx.x;
int64_t const v9 = int64_t(v8);
int64_t const v10 = 256;
int64_t const v11 = v7 * v10;
int64_t const v12 = v11 + v9;
bool const v13 = v12 < v33;
if(v13) {
int64_t const v14 = v35 + v12;
int64_t const v15 = -9;
int64_t const v16 = v14 + v15;
int64_t const v17 = 13;
int64_t const v18 = v16 / v17;
int64_t const v19 = v16 % v17;
int64_t const v20 = 0;
int64_t const v21 = 1;
bool const v22 = v19 < v20;
int64_t const v23 = v18 - v21;
int64_t const v24 = v22 ? v23 : v18;
int64_t const v25 = 257;
int64_t const v37 = v24 % v25;
int64_t const v39 = v37 ^ v25;
bool const v40 = v39 < v20;
bool const v41 = v37 != v20;
bool const v42 = v41 && v40;
int64_t const v43 = v42 ? v25 : v20;
int64_t const v44 = v37 + v43;
float const v27 = float(v44);
uint64_t const v28 = uint64_t(v12);
float * const v29 = &v2[v28];
*v29 = v27;
}
return;
}
