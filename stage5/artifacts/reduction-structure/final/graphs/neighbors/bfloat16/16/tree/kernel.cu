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
    int64_t scalars_i64[2];uint64_t static_meta[2];
};
extern "C" __global__ void __launch_bounds__(256)  cubecl5_daddfbb5a958b83f(float const* const __restrict__ v1, __nv_bfloat16 * const __restrict__ v2, info_st const* const __restrict__ v176) {
extern __shared__ __align__(4) char v181[];uint64_t const* const dynamic_meta_v178 = reinterpret_cast<uint64_t const*>(v176 + 1);
const info_st& info_v177 = *v176;int64_t const v180 = info_v177.scalars_i64[1];
uint64_t const v5 = blockIdx.x;
int64_t const v6 = int64_t(v5);
uint64_t const v7 = threadIdx.x;
int64_t const v8 = int64_t(v7);
int64_t const v9 = 0;
int64_t const v10 = 1;
int64_t const v14 = 256;
int64_t const v15 = 16;
array<float, 256> * const v182 = reinterpret_cast<array<float, 256> *>(&v181[0]);
bool const v17 = v6 < v180;
if(v17) {
float v184_store;
float * v184 = &v184_store;float const v19 = 0.0;
*v184 = v19;
for(int64_t v23 = v8; v23 < v15; v23 += v14) {
int64_t const v25 = v15 * v6;
int64_t const v26 = v23 + v25;
uint64_t const v27 = uint64_t(v26);
float const* const v28 = &v1[v27];
float const v29 = *v28;
float const v31 = *v184;
float const v32 = v31 + v29;
*v184 = v32;
}
float const v33 = *v184;
uint64_t const v37 = uint64_t(v8);
float * const v38 = &(*v182)[v37];
*v38 = v33;
__syncthreads();
int64_t const v39 = 128;
bool const v40 = v8 < v39;
if(v40) {
int64_t const v41 = v8 + v39;
uint64_t const v50 = uint64_t(v41);
float * const v51 = &(*v182)[v50];
float const v52 = *v38;
float const v53 = *v51;
float const v54 = v52 + v53;
*v38 = v54;
}
__syncthreads();
int64_t const v55 = 64;
bool const v56 = v8 < v55;
if(v56) {
int64_t const v57 = v8 + v55;
uint64_t const v66 = uint64_t(v57);
float * const v67 = &(*v182)[v66];
float const v68 = *v38;
float const v69 = *v67;
float const v70 = v68 + v69;
*v38 = v70;
}
__syncthreads();
int64_t const v71 = 32;
bool const v72 = v8 < v71;
if(v72) {
int64_t const v73 = v8 + v71;
uint64_t const v82 = uint64_t(v73);
float * const v83 = &(*v182)[v82];
float const v84 = *v38;
float const v85 = *v83;
float const v86 = v84 + v85;
*v38 = v86;
}
__syncthreads();
bool const v88 = v8 < v15;
if(v88) {
int64_t const v89 = v8 + v15;
uint64_t const v98 = uint64_t(v89);
float * const v99 = &(*v182)[v98];
float const v100 = *v38;
float const v101 = *v99;
float const v102 = v100 + v101;
*v38 = v102;
}
__syncthreads();
int64_t const v103 = 8;
bool const v104 = v8 < v103;
if(v104) {
int64_t const v105 = v8 + v103;
uint64_t const v114 = uint64_t(v105);
float * const v115 = &(*v182)[v114];
float const v116 = *v38;
float const v117 = *v115;
float const v118 = v116 + v117;
*v38 = v118;
}
__syncthreads();
int64_t const v119 = 4;
bool const v120 = v8 < v119;
if(v120) {
int64_t const v121 = v8 + v119;
uint64_t const v130 = uint64_t(v121);
float * const v131 = &(*v182)[v130];
float const v132 = *v38;
float const v133 = *v131;
float const v134 = v132 + v133;
*v38 = v134;
}
__syncthreads();
int64_t const v135 = 2;
bool const v136 = v8 < v135;
if(v136) {
int64_t const v137 = v8 + v135;
uint64_t const v146 = uint64_t(v137);
float * const v147 = &(*v182)[v146];
float const v148 = *v38;
float const v149 = *v147;
float const v150 = v148 + v149;
*v38 = v150;
}
__syncthreads();
bool const v152 = v8 < v10;
if(v152) {
int64_t const v153 = v8 + v10;
uint64_t const v162 = uint64_t(v153);
float * const v163 = &(*v182)[v162];
float const v164 = *v38;
float const v165 = *v163;
float const v166 = v164 + v165;
*v38 = v166;
}
__syncthreads();
uint64_t const v185 = 0;
float * const v168 = &(*v182)[v185];
float const v169 = *v168;
bool const v170 = v8 == v9;
if(v170) {
__nv_bfloat16 const v171 = __nv_bfloat16(v169);
float const v172 = float(v171);
__nv_bfloat16 const v173 = __nv_bfloat16(v172);
uint64_t const v174 = uint64_t(v6);
__nv_bfloat16 * const v175 = &v2[v174];
*v175 = v173;
}
}
return;
}
