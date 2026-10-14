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
    int64_t scalars_i64[3];uint64_t static_meta[2];
};
extern "C" __global__ void __launch_bounds__(256)  cubecl5_9e1cc7cdb65e5621(float const* const __restrict__ v1, float * const __restrict__ v2, info_st const* const __restrict__ v187) {
extern __shared__ __align__(4) char v193[];uint64_t const* const dynamic_meta_v189 = reinterpret_cast<uint64_t const*>(v187 + 1);
const info_st& info_v188 = *v187;int64_t const v191 = info_v188.scalars_i64[1];
int64_t const v192 = info_v188.scalars_i64[2];
uint64_t const v6 = blockIdx.x;
int64_t const v7 = int64_t(v6);
uint64_t const v8 = threadIdx.x;
int64_t const v9 = int64_t(v8);
int64_t const v10 = 0;
int64_t const v11 = 1;
int64_t const v15 = 256;
array<float, 256> * const v194 = reinterpret_cast<array<float, 256> *>(&v193[0]);
bool const v17 = v7 < v192;
if(v17) {
float v196_store;
float * v196 = &v196_store;float const v19 = 0.0;
*v196 = v19;
float v197_store;
float * v197 = &v197_store;*v197 = v19;
int64_t const v22 = 2;
int64_t const v199 = 512;
for(int64_t v24 = v9; v24 < v191; v24 += v199) {
bool const v27 = v24 < v191;
if(v27) {
int64_t const v28 = v7 * v191;
int64_t const v29 = v24 + v28;
uint64_t const v30 = uint64_t(v29);
float const* const v31 = &v1[v30];
float const v32 = *v31;
float const v33 = *v196;
float const v34 = v33 + v32;
*v196 = v34;
}
int64_t const v36 = v24 + v15;
bool const v37 = v36 < v191;
if(v37) {
int64_t const v38 = v7 * v191;
int64_t const v39 = v36 + v38;
uint64_t const v40 = uint64_t(v39);
float const* const v41 = &v1[v40];
float const v42 = *v41;
float const v43 = *v197;
float const v44 = v43 + v42;
*v197 = v44;
}
}
float const v45 = *v196;
float const v46 = *v197;
float const v47 = v45 + v46;
uint64_t const v51 = uint64_t(v9);
float * const v52 = &(*v194)[v51];
*v52 = v47;
__syncthreads();
int64_t const v53 = 128;
bool const v54 = v9 < v53;
if(v54) {
int64_t const v55 = v9 + v53;
uint64_t const v64 = uint64_t(v55);
float * const v65 = &(*v194)[v64];
float const v66 = *v52;
float const v67 = *v65;
float const v68 = v66 + v67;
*v52 = v68;
}
__syncthreads();
int64_t const v69 = 64;
bool const v70 = v9 < v69;
if(v70) {
int64_t const v71 = v9 + v69;
uint64_t const v80 = uint64_t(v71);
float * const v81 = &(*v194)[v80];
float const v82 = *v52;
float const v83 = *v81;
float const v84 = v82 + v83;
*v52 = v84;
}
__syncthreads();
int64_t const v85 = 32;
bool const v86 = v9 < v85;
if(v86) {
int64_t const v87 = v9 + v85;
uint64_t const v96 = uint64_t(v87);
float * const v97 = &(*v194)[v96];
float const v98 = *v52;
float const v99 = *v97;
float const v100 = v98 + v99;
*v52 = v100;
}
__syncthreads();
int64_t const v101 = 16;
bool const v102 = v9 < v101;
if(v102) {
int64_t const v103 = v9 + v101;
uint64_t const v112 = uint64_t(v103);
float * const v113 = &(*v194)[v112];
float const v114 = *v52;
float const v115 = *v113;
float const v116 = v114 + v115;
*v52 = v116;
}
__syncthreads();
int64_t const v117 = 8;
bool const v118 = v9 < v117;
if(v118) {
int64_t const v119 = v9 + v117;
uint64_t const v128 = uint64_t(v119);
float * const v129 = &(*v194)[v128];
float const v130 = *v52;
float const v131 = *v129;
float const v132 = v130 + v131;
*v52 = v132;
}
__syncthreads();
int64_t const v133 = 4;
bool const v134 = v9 < v133;
if(v134) {
int64_t const v135 = v9 + v133;
uint64_t const v144 = uint64_t(v135);
float * const v145 = &(*v194)[v144];
float const v146 = *v52;
float const v147 = *v145;
float const v148 = v146 + v147;
*v52 = v148;
}
__syncthreads();
bool const v150 = v9 < v22;
if(v150) {
int64_t const v151 = v9 + v22;
uint64_t const v160 = uint64_t(v151);
float * const v161 = &(*v194)[v160];
float const v162 = *v52;
float const v163 = *v161;
float const v164 = v162 + v163;
*v52 = v164;
}
__syncthreads();
bool const v166 = v9 < v11;
if(v166) {
int64_t const v167 = v9 + v11;
uint64_t const v176 = uint64_t(v167);
float * const v177 = &(*v194)[v176];
float const v178 = *v52;
float const v179 = *v177;
float const v180 = v178 + v179;
*v52 = v180;
}
__syncthreads();
uint64_t const v198 = 0;
float * const v182 = &(*v194)[v198];
float const v183 = *v182;
__syncthreads();
bool const v184 = v9 == v10;
if(v184) {
uint64_t const v185 = uint64_t(v7);
float * const v186 = &v2[v185];
*v186 = v183;
}
}
return;
}
