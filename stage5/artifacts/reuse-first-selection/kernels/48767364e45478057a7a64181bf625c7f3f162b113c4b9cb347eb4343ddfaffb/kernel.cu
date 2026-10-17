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
extern "C" __global__ void __launch_bounds__(256)  cubecl5_3e57ddb773700701(__nv_bfloat16 const* const __restrict__ v1, __nv_bfloat16 * const __restrict__ v2, info_st const* const __restrict__ v198) {
extern __shared__ __align__(4) char v204[];uint64_t const* const dynamic_meta_v200 = reinterpret_cast<uint64_t const*>(v198 + 1);
const info_st& info_v199 = *v198;int64_t const v202 = info_v199.scalars_i64[1];
int64_t const v203 = info_v199.scalars_i64[2];
uint64_t const v6 = blockIdx.x;
int64_t const v7 = int64_t(v6);
uint64_t const v8 = threadIdx.x;
int64_t const v9 = int64_t(v8);
int64_t const v11 = 1;
int64_t const v15 = 256;
array<float, 256> * const v205 = reinterpret_cast<array<float, 256> *>(&v204[0]);
bool const v17 = v7 < v203;
if(v17) {
float v207_store;
float * v207 = &v207_store;float const v19 = 0.0;
*v207 = v19;
for(int64_t v23 = v9; v23 < v202; v23 += v15) {
int64_t const v24 = v7 * v202;
int64_t const v25 = v23 + v24;
uint64_t const v26 = uint64_t(v25);
__nv_bfloat16 const* const v27 = &v1[v26];
__nv_bfloat16 const v28 = *v27;
float const v29 = float(v28);
__nv_bfloat16 const v30 = __nv_bfloat16(v29);
float const v31 = float(v30);
float const v33 = v31 * v31;
float const v34 = *v207;
float const v35 = v34 + v33;
*v207 = v35;
}
float const v36 = *v207;
uint64_t const v40 = uint64_t(v9);
float * const v41 = &(*v205)[v40];
*v41 = v36;
__syncthreads();
int64_t const v42 = 128;
bool const v43 = v9 < v42;
if(v43) {
int64_t const v44 = v9 + v42;
uint64_t const v53 = uint64_t(v44);
float * const v54 = &(*v205)[v53];
float const v55 = *v41;
float const v56 = *v54;
float const v57 = v55 + v56;
*v41 = v57;
}
__syncthreads();
int64_t const v58 = 64;
bool const v59 = v9 < v58;
if(v59) {
int64_t const v60 = v9 + v58;
uint64_t const v69 = uint64_t(v60);
float * const v70 = &(*v205)[v69];
float const v71 = *v41;
float const v72 = *v70;
float const v73 = v71 + v72;
*v41 = v73;
}
__syncthreads();
int64_t const v74 = 32;
bool const v75 = v9 < v74;
if(v75) {
int64_t const v76 = v9 + v74;
uint64_t const v85 = uint64_t(v76);
float * const v86 = &(*v205)[v85];
float const v87 = *v41;
float const v88 = *v86;
float const v89 = v87 + v88;
*v41 = v89;
}
__syncthreads();
int64_t const v90 = 16;
bool const v91 = v9 < v90;
if(v91) {
int64_t const v92 = v9 + v90;
uint64_t const v101 = uint64_t(v92);
float * const v102 = &(*v205)[v101];
float const v103 = *v41;
float const v104 = *v102;
float const v105 = v103 + v104;
*v41 = v105;
}
__syncthreads();
int64_t const v106 = 8;
bool const v107 = v9 < v106;
if(v107) {
int64_t const v108 = v9 + v106;
uint64_t const v117 = uint64_t(v108);
float * const v118 = &(*v205)[v117];
float const v119 = *v41;
float const v120 = *v118;
float const v121 = v119 + v120;
*v41 = v121;
}
__syncthreads();
int64_t const v122 = 4;
bool const v123 = v9 < v122;
if(v123) {
int64_t const v124 = v9 + v122;
uint64_t const v133 = uint64_t(v124);
float * const v134 = &(*v205)[v133];
float const v135 = *v41;
float const v136 = *v134;
float const v137 = v135 + v136;
*v41 = v137;
}
__syncthreads();
int64_t const v138 = 2;
bool const v139 = v9 < v138;
if(v139) {
int64_t const v140 = v9 + v138;
uint64_t const v149 = uint64_t(v140);
float * const v150 = &(*v205)[v149];
float const v151 = *v41;
float const v152 = *v150;
float const v153 = v151 + v152;
*v41 = v153;
}
__syncthreads();
bool const v155 = v9 < v11;
if(v155) {
int64_t const v156 = v9 + v11;
uint64_t const v165 = uint64_t(v156);
float * const v166 = &(*v205)[v165];
float const v167 = *v41;
float const v168 = *v166;
float const v169 = v167 + v168;
*v41 = v169;
}
__syncthreads();
uint64_t const v208 = 0;
float * const v171 = &(*v205)[v208];
float const v172 = *v171;
for(int64_t v173 = v9; v173 < v202; v173 += v15) {
int64_t const v174 = v7 * v202;
int64_t const v175 = v173 + v174;
uint64_t const v176 = uint64_t(v175);
__nv_bfloat16 const* const v177 = &v1[v176];
__nv_bfloat16 const v178 = *v177;
float const v179 = float(v178);
__nv_bfloat16 const v180 = __nv_bfloat16(v179);
float const v181 = float(v180);
float const v183 = float(v202);
float const v184 = v172 / v183;
float const v185 = 9.999999740e-06;
float const v186 = v184 + v185;
float const v187 = rsqrt(v186);float const v188 = v181 * v187;
__nv_bfloat16 const v189 = __nv_bfloat16(v188);
float const v190 = float(v189);
__nv_bfloat16 const v191 = __nv_bfloat16(v190);
float const v192 = float(v191);
__nv_bfloat16 const v193 = __nv_bfloat16(v192);
__nv_bfloat16 * const v197 = &v2[v176];
*v197 = v193;
}
}
return;
}
