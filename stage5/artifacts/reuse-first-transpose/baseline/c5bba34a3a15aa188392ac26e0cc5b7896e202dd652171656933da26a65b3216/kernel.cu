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
extern "C" __global__ void __launch_bounds__(256)  cubecl5_85c64fe1360672a1(float const* const __restrict__ v1, float * const __restrict__ v2, info_st const* const __restrict__ v38) {
uint64_t const* const dynamic_meta_v40 = reinterpret_cast<uint64_t const*>(v38 + 1);
const info_st& info_v39 = *v38;int64_t const v41 = info_v39.scalars_i64[0];
int64_t const v42 = info_v39.scalars_i64[1];
int64_t const v43 = info_v39.scalars_i64[2];
uint64_t const v6 = blockIdx.x;
int64_t const v7 = int64_t(v6);
uint64_t const v8 = threadIdx.x;
int64_t const v9 = int64_t(v8);
int64_t const v10 = 256;
int64_t const v11 = v7 * v10;
int64_t const v12 = 1;
int64_t const v14 = 0;
int64_t const v18 = v11 + v9;
bool const v19 = v18 < v41;
if(v19) {
int64_t const v45 = v18 % v43;
int64_t const v47 = v45 ^ v43;
bool const v48 = v47 < v14;
bool const v49 = v45 != v14;
bool const v50 = v49 && v48;
int64_t const v51 = v50 ? v43 : v14;
int64_t const v52 = v45 + v51;
int64_t const v21 = v42 * v52;
int64_t const v22 = v18 / v43;
bool const v26 = v45 < v14;
int64_t const v27 = v22 - v12;
int64_t const v28 = v26 ? v27 : v22;
int64_t const v29 = v21 + v28;
uint64_t const v30 = uint64_t(v29);
float const* const v31 = &v1[v30];
float const v32 = *v31;
float const v33 = sin(v32);float const v34 = 1.0;
float const v35 = v33 + v34;
uint64_t const v36 = uint64_t(v18);
float * const v37 = &v2[v36];
*v37 = v35;
}
return;
}
