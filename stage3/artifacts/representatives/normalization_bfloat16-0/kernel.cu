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
#include <cuda_bf16.h>

struct info_st {
    int64_t scalars_i64[3];uint64_t static_meta[2];
};
extern "C" __global__ void __launch_bounds__(256)  cubecl3_c84843fabafaecd4(__nv_bfloat16 const* const __restrict__ v1, __nv_bfloat16 * const __restrict__ v2, info_st const* const __restrict__ v140) {
extern __shared__ __align__(4) char v146[];uint64_t const* const dynamic_meta_v142 = reinterpret_cast<uint64_t const*>(v140 + 1);
const info_st& info_v141 = *v140;int64_t const v144 = info_v141.scalars_i64[1];
int64_t const v145 = info_v141.scalars_i64[2];
uint64_t const v6 = blockIdx.x;
int64_t const v7 = int64_t(v6);
uint64_t const v8 = threadIdx.x;
int64_t const v9 = int64_t(v8);
int64_t const v11 = 256;
array<float, 256> * const v147 = reinterpret_cast<array<float, 256> *>(&v146[0]);
bool const v13 = v7 < v145;
if(v13) {
float v149_store;
float * v149 = &v149_store;float const v15 = 0.0;
*v149 = v15;
for(int64_t v16 = v9; v16 < v144; v16 += v11) {
int64_t const v17 = v7 * v144;
int64_t const v18 = v16 + v17;
uint64_t const v19 = uint64_t(v18);
__nv_bfloat16 const* const v20 = &v1[v19];
__nv_bfloat16 const v21 = *v20;
float const v22 = float(v21);
__nv_bfloat16 const v23 = __nv_bfloat16(v22);
float const v24 = float(v23);
float const v26 = v24 * v24;
float const v27 = *v149;
float const v28 = v27 + v26;
*v149 = v28;
}
float const v29 = *v149;
uint64_t const v30 = uint64_t(v9);
float * const v31 = &(*v147)[v30];
*v31 = v29;
__syncthreads();
int64_t const v32 = 128;
bool const v33 = v9 < v32;
if(v33) {
int64_t const v34 = v9 + v32;
uint64_t const v37 = uint64_t(v34);
float * const v38 = &(*v147)[v37];
float const v39 = *v31;
float const v40 = *v38;
float const v41 = v39 + v40;
*v31 = v41;
}
__syncthreads();
int64_t const v42 = 64;
bool const v43 = v9 < v42;
if(v43) {
int64_t const v44 = v9 + v42;
uint64_t const v47 = uint64_t(v44);
float * const v48 = &(*v147)[v47];
float const v49 = *v31;
float const v50 = *v48;
float const v51 = v49 + v50;
*v31 = v51;
}
__syncthreads();
int64_t const v52 = 32;
bool const v53 = v9 < v52;
if(v53) {
int64_t const v54 = v9 + v52;
uint64_t const v57 = uint64_t(v54);
float * const v58 = &(*v147)[v57];
float const v59 = *v31;
float const v60 = *v58;
float const v61 = v59 + v60;
*v31 = v61;
}
__syncthreads();
int64_t const v62 = 16;
bool const v63 = v9 < v62;
if(v63) {
int64_t const v64 = v9 + v62;
uint64_t const v67 = uint64_t(v64);
float * const v68 = &(*v147)[v67];
float const v69 = *v31;
float const v70 = *v68;
float const v71 = v69 + v70;
*v31 = v71;
}
__syncthreads();
int64_t const v72 = 8;
bool const v73 = v9 < v72;
if(v73) {
int64_t const v74 = v9 + v72;
uint64_t const v77 = uint64_t(v74);
float * const v78 = &(*v147)[v77];
float const v79 = *v31;
float const v80 = *v78;
float const v81 = v79 + v80;
*v31 = v81;
}
__syncthreads();
int64_t const v82 = 4;
bool const v83 = v9 < v82;
if(v83) {
int64_t const v84 = v9 + v82;
uint64_t const v87 = uint64_t(v84);
float * const v88 = &(*v147)[v87];
float const v89 = *v31;
float const v90 = *v88;
float const v91 = v89 + v90;
*v31 = v91;
}
__syncthreads();
int64_t const v92 = 2;
bool const v93 = v9 < v92;
if(v93) {
int64_t const v94 = v9 + v92;
uint64_t const v97 = uint64_t(v94);
float * const v98 = &(*v147)[v97];
float const v99 = *v31;
float const v100 = *v98;
float const v101 = v99 + v100;
*v31 = v101;
}
__syncthreads();
int64_t const v102 = 1;
bool const v103 = v9 < v102;
if(v103) {
int64_t const v104 = v9 + v102;
uint64_t const v107 = uint64_t(v104);
float * const v108 = &(*v147)[v107];
float const v109 = *v31;
float const v110 = *v108;
float const v111 = v109 + v110;
*v31 = v111;
}
__syncthreads();
uint64_t const v150 = 0;
float * const v113 = &(*v147)[v150];
float const v114 = *v113;
__syncthreads();
for(int64_t v115 = v9; v115 < v144; v115 += v11) {
int64_t const v116 = v7 * v144;
int64_t const v117 = v115 + v116;
uint64_t const v118 = uint64_t(v117);
__nv_bfloat16 const* const v119 = &v1[v118];
__nv_bfloat16 const v120 = *v119;
float const v121 = float(v120);
__nv_bfloat16 const v122 = __nv_bfloat16(v121);
float const v123 = float(v122);
float const v125 = float(v144);
float const v126 = v114 / v125;
float const v127 = 9.999999740e-06;
float const v128 = v126 + v127;
float const v129 = rsqrt(v128);float const v130 = v123 * v129;
__nv_bfloat16 const v131 = __nv_bfloat16(v130);
float const v132 = float(v131);
__nv_bfloat16 const v133 = __nv_bfloat16(v132);
float const v134 = float(v133);
__nv_bfloat16 const v135 = __nv_bfloat16(v134);
__nv_bfloat16 * const v139 = &v2[v118];
*v139 = v135;
}
}
return;
}
