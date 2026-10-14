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
extern "C" __global__ void __launch_bounds__(256)  cubecl5_580379bcbcde6f6c(__nv_bfloat16 const* const __restrict__ v1, __nv_bfloat16 * const __restrict__ v2, info_st const* const __restrict__ v108) {
extern __shared__ __align__(4) char v114[];uint64_t const* const dynamic_meta_v110 = reinterpret_cast<uint64_t const*>(v108 + 1);
const info_st& info_v109 = *v108;int64_t const v112 = info_v109.scalars_i64[1];
int64_t const v113 = info_v109.scalars_i64[2];
uint64_t const v6 = blockIdx.x;
int64_t const v7 = int64_t(v6);
uint64_t const v8 = threadIdx.x;
int64_t const v9 = int64_t(v8);
int64_t const v10 = 0;
int64_t const v15 = 256;
array<float, 8> * const v115 = reinterpret_cast<array<float, 8> *>(&v114[0]);
bool const v17 = v7 < v113;
if(v17) {
float v117_store;
float * v117 = &v117_store;float const v19 = 0.0;
*v117 = v19;
float v118_store;
float * v118 = &v118_store;*v118 = v19;
float v119_store;
float * v119 = &v119_store;*v119 = v19;
float v120_store;
float * v120 = &v120_store;*v120 = v19;
int64_t const v171 = 1024;
for(int64_t v28 = v9; v28 < v112; v28 += v171) {
bool const v31 = v28 < v112;
if(v31) {
int64_t const v32 = v7 * v112;
int64_t const v33 = v28 + v32;
uint64_t const v34 = uint64_t(v33);
__nv_bfloat16 const* const v35 = &v1[v34];
__nv_bfloat16 const v36 = *v35;
float const v37 = float(v36);
float const v38 = *v117;
float const v39 = v38 + v37;
*v117 = v39;
}
int64_t const v41 = v28 + v15;
bool const v42 = v41 < v112;
if(v42) {
int64_t const v43 = v7 * v112;
int64_t const v44 = v41 + v43;
uint64_t const v45 = uint64_t(v44);
__nv_bfloat16 const* const v46 = &v1[v45];
__nv_bfloat16 const v47 = *v46;
float const v48 = float(v47);
float const v49 = *v118;
float const v50 = v49 + v48;
*v118 = v50;
}
int64_t const v51 = 512;
int64_t const v52 = v28 + v51;
bool const v53 = v52 < v112;
if(v53) {
int64_t const v54 = v7 * v112;
int64_t const v55 = v52 + v54;
uint64_t const v56 = uint64_t(v55);
__nv_bfloat16 const* const v57 = &v1[v56];
__nv_bfloat16 const v58 = *v57;
float const v59 = float(v58);
float const v60 = *v119;
float const v61 = v60 + v59;
*v119 = v61;
}
int64_t const v62 = 768;
int64_t const v63 = v28 + v62;
bool const v64 = v63 < v112;
if(v64) {
int64_t const v65 = v7 * v112;
int64_t const v66 = v63 + v65;
uint64_t const v67 = uint64_t(v66);
__nv_bfloat16 const* const v68 = &v1[v67];
__nv_bfloat16 const v69 = *v68;
float const v70 = float(v69);
float const v71 = *v120;
float const v72 = v71 + v70;
*v120 = v72;
}
}
float const v73 = *v117;
float const v74 = *v118;
float const v75 = *v119;
float const v76 = *v120;
float const v77 = v73 + v75;
float const v78 = v74 + v76;
float const v79 = v77 + v78;
uint32_t const v121 = warpSize;
uint32_t const v168 = 256;
uint32_t const v123 = min(v121, v168);
float v162_store;
float * v162 = &v162_store;*v162 = v79;
uint32_t const v125 = 1;
uint32_t v163_store;
uint32_t * v163 = &v163_store;*v163 = v125;
while([&]{
uint32_t const v128 = *v163;
bool const v129 = v128 < v123;
return v129;
}()) {
float const v132 = *v162;
uint32_t const v133 = *v163;
float const v134 = __shfl_xor_sync(__activemask(), v132, v133);
float const v136 = v132 + v134;
*v162 = v136;
uint32_t const v138 = 2;
uint32_t const v139 = v133 * v138;
*v163 = v139;
}
float const v140 = *v162;
int64_t const v81 = 32;
int64_t const v82 = v9 / v81;
int64_t const v83 = v9 % v81;
bool const v84 = v83 == v10;
if(v84) {
uint64_t const v85 = uint64_t(v82);
float * const v86 = &(*v115)[v85];
*v86 = v140;
}
__syncthreads();
bool const v87 = v82 == v10;
if(v87) {
float v141_store;
float * v141 = &v141_store;*v141 = v19;
int64_t const v90 = 8;
bool const v91 = v83 < v90;
if(v91) {
uint64_t const v92 = uint64_t(v83);
float * const v93 = &(*v115)[v92];
float const v94 = *v93;
*v141 = v94;
}
float const v95 = *v141;
float v165_store;
float * v165 = &v165_store;*v165 = v95;
uint32_t v166_store;
uint32_t * v166 = &v166_store;*v166 = v125;
while([&]{
uint32_t const v149 = *v166;
bool const v150 = v149 < v123;
return v150;
}()) {
float const v153 = *v165;
uint32_t const v154 = *v166;
float const v155 = __shfl_xor_sync(__activemask(), v153, v154);
float const v157 = v153 + v155;
*v165 = v157;
uint32_t const v159 = 2;
uint32_t const v160 = v154 * v159;
*v166 = v160;
}
float const v161 = *v165;
if(v84) {
uint64_t const v172 = 0;
float * const v98 = &(*v115)[v172];
*v98 = v161;
}
}
__syncthreads();
uint64_t const v170 = 0;
float * const v100 = &(*v115)[v170];
float const v101 = *v100;
__syncthreads();
bool const v102 = v9 == v10;
if(v102) {
__nv_bfloat16 const v103 = __nv_bfloat16(v101);
float const v104 = float(v103);
__nv_bfloat16 const v105 = __nv_bfloat16(v104);
uint64_t const v106 = uint64_t(v7);
__nv_bfloat16 * const v107 = &v2[v106];
*v107 = v105;
}
}
return;
}
