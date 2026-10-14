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
extern "C" __global__ void __launch_bounds__(256)  cubecl5_9be05f7964f82c03(float const* const __restrict__ v1, float * const __restrict__ v2, info_st const* const __restrict__ v215) {
extern __shared__ __align__(4) char v221[];uint64_t const* const dynamic_meta_v217 = reinterpret_cast<uint64_t const*>(v215 + 1);
const info_st& info_v216 = *v215;int64_t const v219 = info_v216.scalars_i64[1];
int64_t const v220 = info_v216.scalars_i64[2];
uint64_t const v6 = blockIdx.x;
int64_t const v7 = int64_t(v6);
uint64_t const v8 = threadIdx.x;
int64_t const v9 = int64_t(v8);
int64_t const v10 = 0;
int64_t const v11 = 1;
int64_t const v15 = 256;
array<float, 256> * const v222 = reinterpret_cast<array<float, 256> *>(&v221[0]);
bool const v17 = v7 < v220;
if(v17) {
float v224_store;
float * v224 = &v224_store;float const v19 = 0.0;
*v224 = v19;
float v225_store;
float * v225 = &v225_store;*v225 = v19;
float v226_store;
float * v226 = &v226_store;*v226 = v19;
float v227_store;
float * v227 = &v227_store;*v227 = v19;
int64_t const v26 = 4;
int64_t const v229 = 1024;
for(int64_t v28 = v9; v28 < v219; v28 += v229) {
bool const v31 = v28 < v219;
if(v31) {
int64_t const v32 = v7 * v219;
int64_t const v33 = v28 + v32;
uint64_t const v34 = uint64_t(v33);
float const* const v35 = &v1[v34];
float const v36 = *v35;
float const v37 = *v224;
float const v38 = v37 + v36;
*v224 = v38;
}
int64_t const v40 = v28 + v15;
bool const v41 = v40 < v219;
if(v41) {
int64_t const v42 = v7 * v219;
int64_t const v43 = v40 + v42;
uint64_t const v44 = uint64_t(v43);
float const* const v45 = &v1[v44];
float const v46 = *v45;
float const v47 = *v225;
float const v48 = v47 + v46;
*v225 = v48;
}
int64_t const v49 = 512;
int64_t const v50 = v28 + v49;
bool const v51 = v50 < v219;
if(v51) {
int64_t const v52 = v7 * v219;
int64_t const v53 = v50 + v52;
uint64_t const v54 = uint64_t(v53);
float const* const v55 = &v1[v54];
float const v56 = *v55;
float const v57 = *v226;
float const v58 = v57 + v56;
*v226 = v58;
}
int64_t const v59 = 768;
int64_t const v60 = v28 + v59;
bool const v61 = v60 < v219;
if(v61) {
int64_t const v62 = v7 * v219;
int64_t const v63 = v60 + v62;
uint64_t const v64 = uint64_t(v63);
float const* const v65 = &v1[v64];
float const v66 = *v65;
float const v67 = *v227;
float const v68 = v67 + v66;
*v227 = v68;
}
}
float const v69 = *v224;
float const v70 = *v225;
float const v71 = *v226;
float const v72 = *v227;
float const v73 = v69 + v71;
float const v74 = v70 + v72;
float const v75 = v73 + v74;
uint64_t const v79 = uint64_t(v9);
float * const v80 = &(*v222)[v79];
*v80 = v75;
__syncthreads();
int64_t const v81 = 128;
bool const v82 = v9 < v81;
if(v82) {
int64_t const v83 = v9 + v81;
uint64_t const v92 = uint64_t(v83);
float * const v93 = &(*v222)[v92];
float const v94 = *v80;
float const v95 = *v93;
float const v96 = v94 + v95;
*v80 = v96;
}
__syncthreads();
int64_t const v97 = 64;
bool const v98 = v9 < v97;
if(v98) {
int64_t const v99 = v9 + v97;
uint64_t const v108 = uint64_t(v99);
float * const v109 = &(*v222)[v108];
float const v110 = *v80;
float const v111 = *v109;
float const v112 = v110 + v111;
*v80 = v112;
}
__syncthreads();
int64_t const v113 = 32;
bool const v114 = v9 < v113;
if(v114) {
int64_t const v115 = v9 + v113;
uint64_t const v124 = uint64_t(v115);
float * const v125 = &(*v222)[v124];
float const v126 = *v80;
float const v127 = *v125;
float const v128 = v126 + v127;
*v80 = v128;
}
__syncthreads();
int64_t const v129 = 16;
bool const v130 = v9 < v129;
if(v130) {
int64_t const v131 = v9 + v129;
uint64_t const v140 = uint64_t(v131);
float * const v141 = &(*v222)[v140];
float const v142 = *v80;
float const v143 = *v141;
float const v144 = v142 + v143;
*v80 = v144;
}
__syncthreads();
int64_t const v145 = 8;
bool const v146 = v9 < v145;
if(v146) {
int64_t const v147 = v9 + v145;
uint64_t const v156 = uint64_t(v147);
float * const v157 = &(*v222)[v156];
float const v158 = *v80;
float const v159 = *v157;
float const v160 = v158 + v159;
*v80 = v160;
}
__syncthreads();
bool const v162 = v9 < v26;
if(v162) {
int64_t const v163 = v9 + v26;
uint64_t const v172 = uint64_t(v163);
float * const v173 = &(*v222)[v172];
float const v174 = *v80;
float const v175 = *v173;
float const v176 = v174 + v175;
*v80 = v176;
}
__syncthreads();
int64_t const v177 = 2;
bool const v178 = v9 < v177;
if(v178) {
int64_t const v179 = v9 + v177;
uint64_t const v188 = uint64_t(v179);
float * const v189 = &(*v222)[v188];
float const v190 = *v80;
float const v191 = *v189;
float const v192 = v190 + v191;
*v80 = v192;
}
__syncthreads();
bool const v194 = v9 < v11;
if(v194) {
int64_t const v195 = v9 + v11;
uint64_t const v204 = uint64_t(v195);
float * const v205 = &(*v222)[v204];
float const v206 = *v80;
float const v207 = *v205;
float const v208 = v206 + v207;
*v80 = v208;
}
__syncthreads();
uint64_t const v228 = 0;
float * const v210 = &(*v222)[v228];
float const v211 = *v210;
__syncthreads();
bool const v212 = v9 == v10;
if(v212) {
uint64_t const v213 = uint64_t(v7);
float * const v214 = &v2[v213];
*v214 = v211;
}
}
return;
}
