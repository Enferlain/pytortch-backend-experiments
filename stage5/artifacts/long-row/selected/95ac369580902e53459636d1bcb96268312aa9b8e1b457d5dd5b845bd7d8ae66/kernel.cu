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
    int64_t scalars_i64[2];uint64_t static_meta[2];
};
extern "C" __global__ void __launch_bounds__(256)  cubecl5_89434481d08018c2(float const* const __restrict__ v1, float * const __restrict__ v2, info_st const* const __restrict__ v172) {
extern __shared__ __align__(4) char v177[];uint64_t const* const dynamic_meta_v174 = reinterpret_cast<uint64_t const*>(v172 + 1);
const info_st& info_v173 = *v172;int64_t const v176 = info_v173.scalars_i64[1];
uint64_t const v5 = blockIdx.x;
int64_t const v6 = int64_t(v5);
uint64_t const v7 = threadIdx.x;
int64_t const v8 = int64_t(v7);
int64_t const v9 = 0;
int64_t const v10 = 1;
int64_t const v14 = 256;
int64_t const v15 = 4;
array<float, 256> * const v178 = reinterpret_cast<array<float, 256> *>(&v177[0]);
bool const v17 = v6 < v176;
if(v17) {
float v180_store;
float * v180 = &v180_store;float const v19 = 0.0;
*v180 = v19;
for(int64_t v23 = v8; v23 < v15; v23 += v14) {
int64_t const v25 = v15 * v6;
int64_t const v26 = v23 + v25;
uint64_t const v27 = uint64_t(v26);
float const* const v28 = &v1[v27];
float const v29 = *v28;
float const v30 = *v180;
float const v31 = v30 + v29;
*v180 = v31;
}
float const v32 = *v180;
uint64_t const v36 = uint64_t(v8);
float * const v37 = &(*v178)[v36];
*v37 = v32;
__syncthreads();
int64_t const v38 = 128;
bool const v39 = v8 < v38;
if(v39) {
int64_t const v40 = v8 + v38;
uint64_t const v49 = uint64_t(v40);
float * const v50 = &(*v178)[v49];
float const v51 = *v37;
float const v52 = *v50;
float const v53 = v51 + v52;
*v37 = v53;
}
__syncthreads();
int64_t const v54 = 64;
bool const v55 = v8 < v54;
if(v55) {
int64_t const v56 = v8 + v54;
uint64_t const v65 = uint64_t(v56);
float * const v66 = &(*v178)[v65];
float const v67 = *v37;
float const v68 = *v66;
float const v69 = v67 + v68;
*v37 = v69;
}
__syncthreads();
int64_t const v70 = 32;
bool const v71 = v8 < v70;
if(v71) {
int64_t const v72 = v8 + v70;
uint64_t const v81 = uint64_t(v72);
float * const v82 = &(*v178)[v81];
float const v83 = *v37;
float const v84 = *v82;
float const v85 = v83 + v84;
*v37 = v85;
}
__syncthreads();
int64_t const v86 = 16;
bool const v87 = v8 < v86;
if(v87) {
int64_t const v88 = v8 + v86;
uint64_t const v97 = uint64_t(v88);
float * const v98 = &(*v178)[v97];
float const v99 = *v37;
float const v100 = *v98;
float const v101 = v99 + v100;
*v37 = v101;
}
__syncthreads();
int64_t const v102 = 8;
bool const v103 = v8 < v102;
if(v103) {
int64_t const v104 = v8 + v102;
uint64_t const v113 = uint64_t(v104);
float * const v114 = &(*v178)[v113];
float const v115 = *v37;
float const v116 = *v114;
float const v117 = v115 + v116;
*v37 = v117;
}
__syncthreads();
bool const v119 = v8 < v15;
if(v119) {
int64_t const v120 = v8 + v15;
uint64_t const v129 = uint64_t(v120);
float * const v130 = &(*v178)[v129];
float const v131 = *v37;
float const v132 = *v130;
float const v133 = v131 + v132;
*v37 = v133;
}
__syncthreads();
int64_t const v134 = 2;
bool const v135 = v8 < v134;
if(v135) {
int64_t const v136 = v8 + v134;
uint64_t const v145 = uint64_t(v136);
float * const v146 = &(*v178)[v145];
float const v147 = *v37;
float const v148 = *v146;
float const v149 = v147 + v148;
*v37 = v149;
}
__syncthreads();
bool const v151 = v8 < v10;
if(v151) {
int64_t const v152 = v8 + v10;
uint64_t const v161 = uint64_t(v152);
float * const v162 = &(*v178)[v161];
float const v163 = *v37;
float const v164 = *v162;
float const v165 = v163 + v164;
*v37 = v165;
}
__syncthreads();
uint64_t const v181 = 0;
float * const v167 = &(*v178)[v181];
float const v168 = *v167;
__syncthreads();
bool const v169 = v8 == v9;
if(v169) {
uint64_t const v170 = uint64_t(v6);
float * const v171 = &v2[v170];
*v171 = v168;
}
}
return;
}
