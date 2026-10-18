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
    int64_t scalars_i64[7];uint64_t static_meta[2];
};
extern "C" __global__ void __launch_bounds__(256)  cubecl5_layout_0939e43f7a680dad(float const* const __restrict__ v1, float * const __restrict__ v2, info_st const* const __restrict__ v76) {
uint64_t const* const dynamic_meta_v78 = reinterpret_cast<uint64_t const*>(v76 + 1);
const info_st& info_v77 = *v76;int64_t const v79 = info_v77.scalars_i64[0];
int64_t const v81 = info_v77.scalars_i64[2];
int64_t const v82 = info_v77.scalars_i64[3];
int64_t const v83 = info_v77.scalars_i64[4];
int64_t const v84 = info_v77.scalars_i64[5];
int64_t const v85 = info_v77.scalars_i64[6];
uint32_t const v11 = uint32_t(v83);
uint64_t const v12 = uint64_t(v81);
uint64_t const v13 = uint64_t(v82);
uint32_t const v15 = uint32_t(v85);
uint64_t const v17 = uint64_t(v84);
uint64_t const v25 = blockIdx.x;
int64_t const v26 = int64_t(v25);
uint64_t const v27 = threadIdx.x;
int64_t const v28 = int64_t(v27);
int64_t const v29 = 256;
int64_t const v30 = v26 * v29;
int64_t const v31 = v30 + v28;
bool const v32 = v31 < v79;
if(v32) {
uint64_t const v33 = uint64_t(v31);
uint64_t const v43 = __umul64hi(v33, v17);
uint64_t const v45 = v43 + v33;
uint64_t const v46 = uint64_t(v15);
uint64_t const v47 = v45 >> v46;
uint64_t const v55 = __umul64hi(v47, v13);
uint64_t const v57 = v55 + v47;
uint64_t const v58 = uint64_t(v11);
uint64_t const v59 = v57 >> v58;
uint64_t const v60 = v59 * v12;
uint64_t const v62 = v47 - v60;
float const* const v69 = &v1[v62];
float const v70 = *v69;
float const v71 = sin(v70);float const v72 = 1.0;
float const v73 = v71 + v72;
float * const v75 = &v2[v33];
*v75 = v73;
}
return;
}
