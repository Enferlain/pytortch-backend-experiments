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
extern "C" __global__ void __launch_bounds__(256)  cubecl5_b7c9af3ee0cd4bb5(float const* const __restrict__ v1, float * const __restrict__ v2, info_st const* const __restrict__ v171) {
extern __shared__ __align__(4) char v177[];uint64_t const* const dynamic_meta_v173 = reinterpret_cast<uint64_t const*>(v171 + 1);
const info_st& info_v172 = *v171;int64_t const v175 = info_v172.scalars_i64[1];
int64_t const v176 = info_v172.scalars_i64[2];
uint64_t const v6 = blockIdx.x;
int64_t const v7 = int64_t(v6);
uint64_t const v8 = threadIdx.x;
int64_t const v9 = int64_t(v8);
int64_t const v10 = 0;
int64_t const v11 = 1;
int64_t const v15 = 256;
array<float, 256> * const v178 = reinterpret_cast<array<float, 256> *>(&v177[0]);
bool const v17 = v7 < v176;
if(v17) {
float v180_store;
float * v180 = &v180_store;float const v19 = 0.0;
*v180 = v19;
for(int64_t v23 = v9; v23 < v175; v23 += v15) {
int64_t const v24 = v7 * v175;
int64_t const v25 = v23 + v24;
uint64_t const v26 = uint64_t(v25);
float const* const v27 = &v1[v26];
float const v28 = *v27;
float const v29 = *v180;
float const v30 = v29 + v28;
*v180 = v30;
}
float const v31 = *v180;
uint64_t const v35 = uint64_t(v9);
float * const v36 = &(*v178)[v35];
*v36 = v31;
__syncthreads();
int64_t const v37 = 128;
bool const v38 = v9 < v37;
if(v38) {
int64_t const v39 = v9 + v37;
uint64_t const v48 = uint64_t(v39);
float * const v49 = &(*v178)[v48];
float const v50 = *v36;
float const v51 = *v49;
float const v52 = v50 + v51;
*v36 = v52;
}
__syncthreads();
int64_t const v53 = 64;
bool const v54 = v9 < v53;
if(v54) {
int64_t const v55 = v9 + v53;
uint64_t const v64 = uint64_t(v55);
float * const v65 = &(*v178)[v64];
float const v66 = *v36;
float const v67 = *v65;
float const v68 = v66 + v67;
*v36 = v68;
}
__syncthreads();
int64_t const v69 = 32;
bool const v70 = v9 < v69;
if(v70) {
int64_t const v71 = v9 + v69;
uint64_t const v80 = uint64_t(v71);
float * const v81 = &(*v178)[v80];
float const v82 = *v36;
float const v83 = *v81;
float const v84 = v82 + v83;
*v36 = v84;
}
__syncthreads();
int64_t const v85 = 16;
bool const v86 = v9 < v85;
if(v86) {
int64_t const v87 = v9 + v85;
uint64_t const v96 = uint64_t(v87);
float * const v97 = &(*v178)[v96];
float const v98 = *v36;
float const v99 = *v97;
float const v100 = v98 + v99;
*v36 = v100;
}
__syncthreads();
int64_t const v101 = 8;
bool const v102 = v9 < v101;
if(v102) {
int64_t const v103 = v9 + v101;
uint64_t const v112 = uint64_t(v103);
float * const v113 = &(*v178)[v112];
float const v114 = *v36;
float const v115 = *v113;
float const v116 = v114 + v115;
*v36 = v116;
}
__syncthreads();
int64_t const v117 = 4;
bool const v118 = v9 < v117;
if(v118) {
int64_t const v119 = v9 + v117;
uint64_t const v128 = uint64_t(v119);
float * const v129 = &(*v178)[v128];
float const v130 = *v36;
float const v131 = *v129;
float const v132 = v130 + v131;
*v36 = v132;
}
__syncthreads();
int64_t const v133 = 2;
bool const v134 = v9 < v133;
if(v134) {
int64_t const v135 = v9 + v133;
uint64_t const v144 = uint64_t(v135);
float * const v145 = &(*v178)[v144];
float const v146 = *v36;
float const v147 = *v145;
float const v148 = v146 + v147;
*v36 = v148;
}
__syncthreads();
bool const v150 = v9 < v11;
if(v150) {
int64_t const v151 = v9 + v11;
uint64_t const v160 = uint64_t(v151);
float * const v161 = &(*v178)[v160];
float const v162 = *v36;
float const v163 = *v161;
float const v164 = v162 + v163;
*v36 = v164;
}
__syncthreads();
uint64_t const v181 = 0;
float * const v166 = &(*v178)[v181];
float const v167 = *v166;
__syncthreads();
bool const v168 = v9 == v10;
if(v168) {
uint64_t const v169 = uint64_t(v7);
float * const v170 = &v2[v169];
*v170 = v167;
}
}
return;
}
