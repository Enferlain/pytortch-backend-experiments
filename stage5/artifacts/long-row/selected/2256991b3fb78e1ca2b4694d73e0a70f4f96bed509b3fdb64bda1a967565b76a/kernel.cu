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
extern "C" __global__ void __launch_bounds__(256)  cubecl5_e96ad8796bebbfb7(__nv_bfloat16 const* const __restrict__ v1, float * const __restrict__ v2, info_st const* const __restrict__ v224) {
extern __shared__ __align__(4) char v230[];uint64_t const* const dynamic_meta_v226 = reinterpret_cast<uint64_t const*>(v224 + 1);
const info_st& info_v225 = *v224;int64_t const v228 = info_v225.scalars_i64[1];
int64_t const v229 = info_v225.scalars_i64[2];
uint64_t const v6 = blockIdx.x;
int64_t const v7 = int64_t(v6);
uint64_t const v8 = threadIdx.x;
int64_t const v9 = int64_t(v8);
int64_t const v10 = 0;
int64_t const v11 = 1;
int64_t const v15 = 256;
int64_t const v16 = 9;
int64_t const v17 = v16 * v229;
int64_t const v18 = 8;
int64_t const v19 = v18 + v228;
int64_t const v21 = v19 / v16;
int64_t const v22 = v19 % v16;
bool const v25 = v22 < v10;
int64_t const v26 = v21 - v11;
int64_t const v27 = v25 ? v26 : v21;
array<float, 256> * const v231 = reinterpret_cast<array<float, 256> *>(&v230[0]);
bool const v29 = v7 < v17;
if(v29) {
float v233_store;
float * v233 = &v233_store;float const v31 = 0.0;
*v233 = v31;
for(int64_t v35 = v9; v35 < v27; v35 += v15) {
int64_t const v234 = v7 % v16;
int64_t const v236 = v234 ^ v16;
bool const v237 = v236 < v10;
bool const v238 = v234 != v10;
bool const v239 = v238 && v237;
int64_t const v240 = v239 ? v16 : v10;
int64_t const v241 = v234 + v240;
int64_t const v48 = v27 * v241;
int64_t const v49 = v35 + v48;
bool const v50 = v49 < v228;
float v242_store;
float * v242 = &v242_store;if(v50) {
int64_t const v53 = v7 / v16;
bool const v57 = v234 < v10;
int64_t const v58 = v53 - v11;
int64_t const v59 = v57 ? v58 : v53;
int64_t const v60 = v228 * v59;
int64_t const v61 = v35 + v60;
int64_t const v75 = v61 + v48;
uint64_t const v76 = uint64_t(v75);
__nv_bfloat16 const* const v77 = &v1[v76];
__nv_bfloat16 const v78 = *v77;
float const v79 = float(v78);
*v242 = v79;
}
 else {
*v242 = v31;
}
float const v81 = *v242;
float const v82 = *v233;
float const v83 = v82 + v81;
*v233 = v83;
}
float const v84 = *v233;
uint64_t const v88 = uint64_t(v9);
float * const v89 = &(*v231)[v88];
*v89 = v84;
__syncthreads();
int64_t const v90 = 128;
bool const v91 = v9 < v90;
if(v91) {
int64_t const v92 = v9 + v90;
uint64_t const v101 = uint64_t(v92);
float * const v102 = &(*v231)[v101];
float const v103 = *v89;
float const v104 = *v102;
float const v105 = v103 + v104;
*v89 = v105;
}
__syncthreads();
int64_t const v106 = 64;
bool const v107 = v9 < v106;
if(v107) {
int64_t const v108 = v9 + v106;
uint64_t const v117 = uint64_t(v108);
float * const v118 = &(*v231)[v117];
float const v119 = *v89;
float const v120 = *v118;
float const v121 = v119 + v120;
*v89 = v121;
}
__syncthreads();
int64_t const v122 = 32;
bool const v123 = v9 < v122;
if(v123) {
int64_t const v124 = v9 + v122;
uint64_t const v133 = uint64_t(v124);
float * const v134 = &(*v231)[v133];
float const v135 = *v89;
float const v136 = *v134;
float const v137 = v135 + v136;
*v89 = v137;
}
__syncthreads();
int64_t const v138 = 16;
bool const v139 = v9 < v138;
if(v139) {
int64_t const v140 = v9 + v138;
uint64_t const v149 = uint64_t(v140);
float * const v150 = &(*v231)[v149];
float const v151 = *v89;
float const v152 = *v150;
float const v153 = v151 + v152;
*v89 = v153;
}
__syncthreads();
bool const v155 = v9 < v18;
if(v155) {
int64_t const v156 = v9 + v18;
uint64_t const v165 = uint64_t(v156);
float * const v166 = &(*v231)[v165];
float const v167 = *v89;
float const v168 = *v166;
float const v169 = v167 + v168;
*v89 = v169;
}
__syncthreads();
int64_t const v170 = 4;
bool const v171 = v9 < v170;
if(v171) {
int64_t const v172 = v9 + v170;
uint64_t const v181 = uint64_t(v172);
float * const v182 = &(*v231)[v181];
float const v183 = *v89;
float const v184 = *v182;
float const v185 = v183 + v184;
*v89 = v185;
}
__syncthreads();
int64_t const v186 = 2;
bool const v187 = v9 < v186;
if(v187) {
int64_t const v188 = v9 + v186;
uint64_t const v197 = uint64_t(v188);
float * const v198 = &(*v231)[v197];
float const v199 = *v89;
float const v200 = *v198;
float const v201 = v199 + v200;
*v89 = v201;
}
__syncthreads();
bool const v203 = v9 < v11;
if(v203) {
int64_t const v204 = v9 + v11;
uint64_t const v213 = uint64_t(v204);
float * const v214 = &(*v231)[v213];
float const v215 = *v89;
float const v216 = *v214;
float const v217 = v215 + v216;
*v89 = v217;
}
__syncthreads();
uint64_t const v251 = 0;
float * const v219 = &(*v231)[v251];
float const v220 = *v219;
__syncthreads();
bool const v221 = v9 == v10;
if(v221) {
uint64_t const v222 = uint64_t(v7);
float * const v223 = &v2[v222];
*v223 = v220;
}
}
return;
}
