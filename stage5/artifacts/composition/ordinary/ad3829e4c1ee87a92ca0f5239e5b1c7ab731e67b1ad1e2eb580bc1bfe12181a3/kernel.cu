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
extern "C" __global__ void __launch_bounds__(128)  cubecl5_c3acfd580bd37061(__nv_bfloat16 const* const __restrict__ v1, __nv_bfloat16 * const __restrict__ v2, info_st const* const __restrict__ v182) {
extern __shared__ __align__(4) char v188[];uint64_t const* const dynamic_meta_v184 = reinterpret_cast<uint64_t const*>(v182 + 1);
const info_st& info_v183 = *v182;int64_t const v186 = info_v183.scalars_i64[1];
int64_t const v187 = info_v183.scalars_i64[2];
uint64_t const v6 = blockIdx.x;
int64_t const v7 = int64_t(v6);
uint64_t const v8 = threadIdx.x;
int64_t const v9 = int64_t(v8);
int64_t const v11 = 1;
int64_t const v15 = 128;
array<float, 128> * const v189 = reinterpret_cast<array<float, 128> *>(&v188[0]);
bool const v17 = v7 < v187;
if(v17) {
float v191_store;
float * v191 = &v191_store;float const v19 = 0.0;
*v191 = v19;
for(int64_t v23 = v9; v23 < v186; v23 += v15) {
int64_t const v24 = v7 * v186;
int64_t const v25 = v23 + v24;
uint64_t const v26 = uint64_t(v25);
__nv_bfloat16 const* const v27 = &v1[v26];
__nv_bfloat16 const v28 = *v27;
float const v29 = float(v28);
__nv_bfloat16 const v30 = __nv_bfloat16(v29);
float const v31 = float(v30);
float const v33 = v31 * v31;
float const v34 = *v191;
float const v35 = v34 + v33;
*v191 = v35;
}
float const v36 = *v191;
uint64_t const v40 = uint64_t(v9);
float * const v41 = &(*v189)[v40];
*v41 = v36;
__syncthreads();
int64_t const v42 = 64;
bool const v43 = v9 < v42;
if(v43) {
int64_t const v44 = v9 + v42;
uint64_t const v53 = uint64_t(v44);
float * const v54 = &(*v189)[v53];
float const v55 = *v41;
float const v56 = *v54;
float const v57 = v55 + v56;
*v41 = v57;
}
__syncthreads();
int64_t const v58 = 32;
bool const v59 = v9 < v58;
if(v59) {
int64_t const v60 = v9 + v58;
uint64_t const v69 = uint64_t(v60);
float * const v70 = &(*v189)[v69];
float const v71 = *v41;
float const v72 = *v70;
float const v73 = v71 + v72;
*v41 = v73;
}
__syncthreads();
int64_t const v74 = 16;
bool const v75 = v9 < v74;
if(v75) {
int64_t const v76 = v9 + v74;
uint64_t const v85 = uint64_t(v76);
float * const v86 = &(*v189)[v85];
float const v87 = *v41;
float const v88 = *v86;
float const v89 = v87 + v88;
*v41 = v89;
}
__syncthreads();
int64_t const v90 = 8;
bool const v91 = v9 < v90;
if(v91) {
int64_t const v92 = v9 + v90;
uint64_t const v101 = uint64_t(v92);
float * const v102 = &(*v189)[v101];
float const v103 = *v41;
float const v104 = *v102;
float const v105 = v103 + v104;
*v41 = v105;
}
__syncthreads();
int64_t const v106 = 4;
bool const v107 = v9 < v106;
if(v107) {
int64_t const v108 = v9 + v106;
uint64_t const v117 = uint64_t(v108);
float * const v118 = &(*v189)[v117];
float const v119 = *v41;
float const v120 = *v118;
float const v121 = v119 + v120;
*v41 = v121;
}
__syncthreads();
int64_t const v122 = 2;
bool const v123 = v9 < v122;
if(v123) {
int64_t const v124 = v9 + v122;
uint64_t const v133 = uint64_t(v124);
float * const v134 = &(*v189)[v133];
float const v135 = *v41;
float const v136 = *v134;
float const v137 = v135 + v136;
*v41 = v137;
}
__syncthreads();
bool const v139 = v9 < v11;
if(v139) {
int64_t const v140 = v9 + v11;
uint64_t const v149 = uint64_t(v140);
float * const v150 = &(*v189)[v149];
float const v151 = *v41;
float const v152 = *v150;
float const v153 = v151 + v152;
*v41 = v153;
}
__syncthreads();
uint64_t const v192 = 0;
float * const v155 = &(*v189)[v192];
float const v156 = *v155;
for(int64_t v157 = v9; v157 < v186; v157 += v15) {
int64_t const v158 = v7 * v186;
int64_t const v159 = v157 + v158;
uint64_t const v160 = uint64_t(v159);
__nv_bfloat16 const* const v161 = &v1[v160];
__nv_bfloat16 const v162 = *v161;
float const v163 = float(v162);
__nv_bfloat16 const v164 = __nv_bfloat16(v163);
float const v165 = float(v164);
float const v167 = float(v186);
float const v168 = v156 / v167;
float const v169 = 9.999999740e-06;
float const v170 = v168 + v169;
float const v171 = rsqrt(v170);float const v172 = v165 * v171;
__nv_bfloat16 const v173 = __nv_bfloat16(v172);
float const v174 = float(v173);
__nv_bfloat16 const v175 = __nv_bfloat16(v174);
float const v176 = float(v175);
__nv_bfloat16 const v177 = __nv_bfloat16(v176);
__nv_bfloat16 * const v181 = &v2[v160];
*v181 = v177;
}
}
return;
}
