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
    int64_t scalars_i64[5];uint64_t static_meta[2];
};
extern "C" __global__ void __launch_bounds__(256)  cubecl5_layout_f2619300f7ce2547(float const* const __restrict__ v1, float * const __restrict__ v2, info_st const* const __restrict__ v54) {
uint64_t const* const dynamic_meta_v56 = reinterpret_cast<uint64_t const*>(v54 + 1);
const info_st& info_v55 = *v54;int64_t const v57 = info_v55.scalars_i64[0];
int64_t const v58 = info_v55.scalars_i64[1];
int64_t const v59 = info_v55.scalars_i64[2];
int64_t const v60 = info_v55.scalars_i64[3];
int64_t const v61 = info_v55.scalars_i64[4];
uint32_t const v9 = uint32_t(v61);
uint64_t const v10 = uint64_t(v59);
uint64_t const v11 = uint64_t(v60);
uint64_t const v13 = uint64_t(v58);
uint64_t const v15 = blockIdx.x;
int64_t const v16 = int64_t(v15);
uint64_t const v17 = threadIdx.x;
int64_t const v18 = int64_t(v17);
int64_t const v19 = 256;
int64_t const v20 = v16 * v19;
int64_t const v21 = v20 + v18;
bool const v22 = v21 < v57;
if(v22) {
uint64_t const v23 = uint64_t(v21);
uint64_t const v33 = __umul64hi(v23, v11);
uint64_t const v35 = v33 + v23;
uint64_t const v36 = uint64_t(v9);
uint64_t const v37 = v35 >> v36;
uint64_t const v38 = v37 * v10;
uint64_t const v40 = v23 - v38;
uint64_t const v41 = v40 * v13;
float const* const v47 = &v1[v41];
float const v48 = *v47;
float const v49 = sin(v48);float const v50 = 1.0;
float const v51 = v49 + v50;
float * const v53 = &v2[v23];
*v53 = v51;
}
return;
}
