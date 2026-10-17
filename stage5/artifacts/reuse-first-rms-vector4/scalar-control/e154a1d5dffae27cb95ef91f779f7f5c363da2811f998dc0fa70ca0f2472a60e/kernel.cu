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
extern "C" __global__ void __launch_bounds__(256)  cubecl5_rms_cubek_e154a1d5dffae27c(__nv_bfloat16 const* const __restrict__ v1, __nv_bfloat16 * const __restrict__ v2, info_st const* const __restrict__ v150) {
uint64_t const* const dynamic_meta_v152 = reinterpret_cast<uint64_t const*>(v150 + 1);
const info_st& info_v151 = *v150;int64_t const v154 = info_v151.scalars_i64[1];
int64_t const v155 = info_v151.scalars_i64[2];
uint64_t const v6 = blockIdx.x;
int64_t const v7 = int64_t(v6);
uint64_t const v8 = threadIdx.y;
int64_t const v9 = int64_t(v8);
int64_t const v10 = 8;
int64_t const v11 = v7 * v10;
int64_t const v12 = v11 + v9;
uint64_t const v13 = uint64_t(v12);
uint64_t const v14 = uint64_t(v154);
uint64_t const v15 = uint64_t(v155);
bool const v16 = v13 < v15;
if(v16) {
uint64_t const v18 = 0;
uint64_t v157_store;
uint64_t * v157 = &v157_store;*v157 = v18;
uint64_t const v22 = 2;
uint64_t const v23 = 1;
for(uint64_t v24 = v18; v24 < v22; v24 += v23) {
int64_t const v27 = int64_t(v24);
int64_t const v28 = 0;
bool const v29 = v27 == v28;
int64_t const v30 = 1;
int64_t const v39 = v29 ? v155 : v30;
uint64_t const v40 = uint64_t(v39);
uint64_t const v41 = v13 % v40;
int64_t const v46 = v29 ? v154 : v30;
uint64_t const v47 = uint64_t(v46);
uint64_t const v48 = v41 * v47;
uint64_t const v49 = *v157;
uint64_t const v50 = v49 + v48;
*v157 = v50;
}
uint64_t const v51 = *v157;
*v157 = v51;
float const v62 = 0.0;
float v162_store;
float * v162 = &v162_store;*v162 = v62;
uint64_t const v193 = 32;
uint64_t const v72 = v14 + v193;
uint64_t const v74 = v72 - v23;
uint64_t const v75 = v74 / v193;
for(uint64_t v78 = v18; v78 < v75; v78 += v23) {
uint64_t const v82 = v78 * v193;
uint32_t const v83 = threadIdx.x;
uint64_t const v84 = uint64_t(v83);
uint64_t const v85 = v82 + v84;
uint64_t const v86 = *v157;
uint64_t const v87 = v85 + v86;
bool const v88 = v85 < v14;
float v165_store;
float * v165 = &v165_store;if(v88) {
int64_t const v89 = int64_t(v87);
int64_t const v90 = v12 * v154;
int64_t const v91 = v89 - v90;
int64_t const v93 = v91 + v90;
uint64_t const v94 = uint64_t(v93);
__nv_bfloat16 const* const v95 = &v1[v94];
__nv_bfloat16 const v96 = *v95;
float const v97 = float(v96);
__nv_bfloat16 const v98 = __nv_bfloat16(v97);
float const v99 = float(v98);
float const v101 = v99 * v99;
*v165 = v101;
}
 else {
*v165 = v62;
}
float const v108 = *v165;
float const v109 = *v162;
float const v110 = v109 + v108;
*v162 = v110;
}
float const v111 = *v162;
uint32_t const v166 = warpSize;
uint32_t const v192 = 256;
uint32_t const v168 = min(v166, v192);
float v188_store;
float * v188 = &v188_store;*v188 = v111;
uint32_t const v170 = 1;
uint32_t v189_store;
uint32_t * v189 = &v189_store;*v189 = v170;
while([&]{
uint32_t const v173 = *v189;
bool const v174 = v173 < v168;
return v174;
}()) {
float const v177 = *v188;
uint32_t const v178 = *v189;
float const v179 = __shfl_xor_sync(__activemask(), v177, v178);
float const v181 = v177 + v179;
*v188 = v181;
uint32_t const v183 = 2;
uint32_t const v184 = v178 * v183;
*v189 = v184;
}
float const v185 = *v188;
*v162 = v185;
uint32_t const v113 = threadIdx.x;
uint64_t const v114 = uint64_t(v113);
for(uint64_t v116 = v114; v116 < v14; v116 += v193) {
uint64_t const v119 = v13 * v14;
uint64_t const v120 = v119 + v116;
float const v121 = *v162;
int64_t const v122 = int64_t(v120);
int64_t const v123 = v12 * v154;
int64_t const v124 = v122 - v123;
int64_t const v126 = v124 + v123;
uint64_t const v127 = uint64_t(v126);
__nv_bfloat16 const* const v128 = &v1[v127];
__nv_bfloat16 const v129 = *v128;
float const v130 = float(v129);
__nv_bfloat16 const v131 = __nv_bfloat16(v130);
float const v132 = float(v131);
float const v135 = float(v154);
float const v136 = v121 / v135;
float const v137 = 9.999999740e-06;
float const v138 = v136 + v137;
float const v139 = rsqrt(v138);float const v140 = v132 * v139;
__nv_bfloat16 const v141 = __nv_bfloat16(v140);
float const v142 = float(v141);
__nv_bfloat16 const v143 = __nv_bfloat16(v142);
float const v144 = float(v143);
__nv_bfloat16 const v145 = __nv_bfloat16(v144);
__nv_bfloat16 * const v149 = &v2[v127];
*v149 = v145;
}
}
return;
}
