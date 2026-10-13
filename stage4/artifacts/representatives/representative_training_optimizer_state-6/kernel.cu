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
    int64_t scalars_i64[2];uint64_t static_meta[6];
};
extern "C" __global__ void __launch_bounds__(256)  cubecl4_a7a817980b349606(float const* const __restrict__ v1, bool const* const __restrict__ v2, float const* const __restrict__ v3, float const* const __restrict__ v4, float const* const __restrict__ v5, float * const __restrict__ v6, info_st const* const __restrict__ v76) {
uint64_t const* const dynamic_meta_v78 = reinterpret_cast<uint64_t const*>(v76 + 1);
const info_st& info_v77 = *v76;int64_t const v79 = info_v77.scalars_i64[0];
int64_t const v80 = info_v77.scalars_i64[1];
uint64_t const v9 = blockIdx.x;
int64_t const v10 = int64_t(v9);
uint64_t const v11 = threadIdx.x;
int64_t const v12 = int64_t(v11);
int64_t const v13 = 256;
int64_t const v14 = v10 * v13;
int64_t const v15 = v14 + v12;
bool const v16 = v15 < v79;
if(v16) {
int64_t const v17 = 0;
uint64_t const v82 = 0;
float const* const v19 = &v1[v82];
float const v20 = *v19;
int64_t const v21 = 9;
int64_t const v22 = v21 * v80;
float const v23 = float(v22);
float const v24 = v20 / v23;
uint64_t const v25 = uint64_t(v15);
bool const* const v26 = &v2[v25];
bool const v27 = *v26;
float const v28 = float(v27);
float const* const v30 = &v3[v25];
float const v31 = *v30;
int64_t const v33 = v15 / v21;
int64_t const v34 = v15 % v21;
int64_t const v36 = 1;
bool const v37 = v34 < v17;
int64_t const v38 = v33 - v36;
int64_t const v39 = v37 ? v38 : v33;
uint64_t const v40 = uint64_t(v39);
float const* const v41 = &v4[v40];
float const v42 = *v41;
float const v43 = v31 * v42;
float const v44 = v28 * v43;
float const v45 = 1.33333337;
float const v46 = v44 * v45;
float const v47 = 2.0;
float const v48 = v46 * v47;
float const v49 = v24 * v48;
float const v50 = v28 * v45;
float const v51 = v49 * v50;
float const v52 = v51 * v42;
float const* const v62 = &v5[v40];
float const v63 = *v62;
float const v64 = -0.5;
float const v65 = v63 * v64;
float const v66 = v42 * v42;
float const v67 = v66 * v42;
float const v68 = v65 * v67;
float const v69 = 0.111111112;
float const v70 = v68 * v69;
float const v71 = v70 * v31;
float const v72 = v52 + v71;
float const v73 = v72 + v71;
float * const v75 = &v6[v25];
*v75 = v73;
}
return;
}
