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
    int64_t scalars_i64[3];uint64_t static_meta[4];
};
extern "C" __global__ void __launch_bounds__(256)  cubecl3_8fc01e3a0823ce87(__nv_bfloat16 const* const __restrict__ v1, __nv_bfloat16 const* const __restrict__ v2, float const* const __restrict__ v3, __nv_bfloat16 * const __restrict__ v4, info_st const* const __restrict__ v168) {
extern __shared__ __align__(4) char v174[];uint64_t const* const dynamic_meta_v170 = reinterpret_cast<uint64_t const*>(v168 + 1);
const info_st& info_v169 = *v168;int64_t const v172 = info_v169.scalars_i64[1];
int64_t const v173 = info_v169.scalars_i64[2];
uint64_t const v8 = blockIdx.x;
int64_t const v9 = int64_t(v8);
uint64_t const v10 = threadIdx.x;
int64_t const v11 = int64_t(v10);
int64_t const v13 = 256;
array<float, 256> * const v175 = reinterpret_cast<array<float, 256> *>(&v174[0]);
bool const v15 = v9 < v173;
if(v15) {
float v177_store;
float * v177 = &v177_store;float const v17 = 0.0;
*v177 = v17;
for(int64_t v18 = v11; v18 < v172; v18 += v13) {
int64_t const v19 = v9 * v172;
int64_t const v20 = v18 + v19;
uint64_t const v21 = uint64_t(v20);
__nv_bfloat16 const* const v22 = &v1[v21];
__nv_bfloat16 const v23 = *v22;
float const v24 = float(v23);
__nv_bfloat16 const v25 = __nv_bfloat16(v24);
float const v26 = float(v25);
__nv_bfloat16 const* const v31 = &v2[v21];
__nv_bfloat16 const v32 = *v31;
float const v33 = float(v32);
__nv_bfloat16 const v34 = __nv_bfloat16(v33);
float const v35 = float(v34);
float const v37 = v26 * v35;
float const v38 = *v177;
float const v39 = v38 + v37;
*v177 = v39;
}
float const v40 = *v177;
uint64_t const v41 = uint64_t(v11);
float * const v42 = &(*v175)[v41];
*v42 = v40;
__syncthreads();
int64_t const v43 = 128;
bool const v44 = v11 < v43;
if(v44) {
int64_t const v45 = v11 + v43;
uint64_t const v48 = uint64_t(v45);
float * const v49 = &(*v175)[v48];
float const v50 = *v42;
float const v51 = *v49;
float const v52 = v50 + v51;
*v42 = v52;
}
__syncthreads();
int64_t const v53 = 64;
bool const v54 = v11 < v53;
if(v54) {
int64_t const v55 = v11 + v53;
uint64_t const v58 = uint64_t(v55);
float * const v59 = &(*v175)[v58];
float const v60 = *v42;
float const v61 = *v59;
float const v62 = v60 + v61;
*v42 = v62;
}
__syncthreads();
int64_t const v63 = 32;
bool const v64 = v11 < v63;
if(v64) {
int64_t const v65 = v11 + v63;
uint64_t const v68 = uint64_t(v65);
float * const v69 = &(*v175)[v68];
float const v70 = *v42;
float const v71 = *v69;
float const v72 = v70 + v71;
*v42 = v72;
}
__syncthreads();
int64_t const v73 = 16;
bool const v74 = v11 < v73;
if(v74) {
int64_t const v75 = v11 + v73;
uint64_t const v78 = uint64_t(v75);
float * const v79 = &(*v175)[v78];
float const v80 = *v42;
float const v81 = *v79;
float const v82 = v80 + v81;
*v42 = v82;
}
__syncthreads();
int64_t const v83 = 8;
bool const v84 = v11 < v83;
if(v84) {
int64_t const v85 = v11 + v83;
uint64_t const v88 = uint64_t(v85);
float * const v89 = &(*v175)[v88];
float const v90 = *v42;
float const v91 = *v89;
float const v92 = v90 + v91;
*v42 = v92;
}
__syncthreads();
int64_t const v93 = 4;
bool const v94 = v11 < v93;
if(v94) {
int64_t const v95 = v11 + v93;
uint64_t const v98 = uint64_t(v95);
float * const v99 = &(*v175)[v98];
float const v100 = *v42;
float const v101 = *v99;
float const v102 = v100 + v101;
*v42 = v102;
}
__syncthreads();
int64_t const v103 = 2;
bool const v104 = v11 < v103;
if(v104) {
int64_t const v105 = v11 + v103;
uint64_t const v108 = uint64_t(v105);
float * const v109 = &(*v175)[v108];
float const v110 = *v42;
float const v111 = *v109;
float const v112 = v110 + v111;
*v42 = v112;
}
__syncthreads();
int64_t const v113 = 1;
bool const v114 = v11 < v113;
if(v114) {
int64_t const v115 = v11 + v113;
uint64_t const v118 = uint64_t(v115);
float * const v119 = &(*v175)[v118];
float const v120 = *v42;
float const v121 = *v119;
float const v122 = v120 + v121;
*v42 = v122;
}
__syncthreads();
uint64_t const v178 = 0;
float * const v124 = &(*v175)[v178];
float const v125 = *v124;
__syncthreads();
for(int64_t v126 = v11; v126 < v172; v126 += v13) {
int64_t const v127 = v9 * v172;
int64_t const v128 = v126 + v127;
uint64_t const v129 = uint64_t(v128);
__nv_bfloat16 const* const v130 = &v1[v129];
__nv_bfloat16 const v131 = *v130;
float const v132 = float(v131);
__nv_bfloat16 const v133 = __nv_bfloat16(v132);
float const v134 = float(v133);
__nv_bfloat16 const* const v139 = &v2[v129];
__nv_bfloat16 const v140 = *v139;
float const v141 = float(v140);
__nv_bfloat16 const v142 = __nv_bfloat16(v141);
float const v143 = float(v142);
uint64_t const v145 = uint64_t(v9);
float const* const v146 = &v3[v145];
float const v147 = *v146;
float const v148 = v134 * v147;
float const v149 = -0.5;
float const v150 = v125 * v149;
float const v151 = v147 * v147;
float const v152 = v151 * v147;
float const v153 = v150 * v152;
float const v154 = float(v172);
float const v155 = v153 / v154;
float const v156 = v155 * v143;
float const v157 = v148 + v156;
float const v158 = v157 + v156;
__nv_bfloat16 const v159 = __nv_bfloat16(v158);
float const v160 = float(v159);
__nv_bfloat16 const v161 = __nv_bfloat16(v160);
float const v162 = float(v161);
__nv_bfloat16 const v163 = __nv_bfloat16(v162);
__nv_bfloat16 * const v167 = &v4[v129];
*v167 = v163;
}
}
return;
}
