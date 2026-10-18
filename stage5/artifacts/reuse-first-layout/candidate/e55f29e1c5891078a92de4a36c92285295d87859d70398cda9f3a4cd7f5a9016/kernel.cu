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
extern "C" __global__ void __launch_bounds__(256)  cubecl5_layout_e55f29e1c5891078(float const* const __restrict__ v1, float const* const __restrict__ v2, float * const __restrict__ v3, info_st const* const __restrict__ v78) {
uint64_t const* const dynamic_meta_v80 = reinterpret_cast<uint64_t const*>(v78 + 1);
const info_st& info_v79 = *v78;int64_t const v81 = info_v79.scalars_i64[0];
int64_t const v83 = info_v79.scalars_i64[2];
int64_t const v84 = info_v79.scalars_i64[3];
int64_t const v85 = info_v79.scalars_i64[4];
int64_t const v86 = info_v79.scalars_i64[5];
int64_t const v87 = info_v79.scalars_i64[6];
uint32_t const v12 = uint32_t(v85);
uint64_t const v13 = uint64_t(v83);
uint64_t const v14 = uint64_t(v84);
uint32_t const v16 = uint32_t(v87);
uint64_t const v18 = uint64_t(v86);
uint64_t const v26 = blockIdx.x;
int64_t const v27 = int64_t(v26);
uint64_t const v28 = threadIdx.x;
int64_t const v29 = int64_t(v28);
int64_t const v30 = 256;
int64_t const v31 = v27 * v30;
int64_t const v32 = v31 + v29;
bool const v33 = v32 < v81;
if(v33) {
uint64_t const v34 = uint64_t(v32);
float const* const v35 = &v1[v34];
float const v36 = *v35;
uint64_t const v47 = __umul64hi(v34, v18);
uint64_t const v49 = v47 + v34;
uint64_t const v50 = uint64_t(v16);
uint64_t const v51 = v49 >> v50;
uint64_t const v59 = __umul64hi(v51, v14);
uint64_t const v61 = v59 + v51;
uint64_t const v62 = uint64_t(v12);
uint64_t const v63 = v61 >> v62;
uint64_t const v64 = v63 * v13;
uint64_t const v66 = v51 - v64;
float const* const v73 = &v2[v66];
float const v74 = *v73;
float const v75 = v36 + v74;
float * const v77 = &v3[v34];
*v77 = v75;
}
return;
}
