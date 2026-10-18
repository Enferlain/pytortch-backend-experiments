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
extern "C" __global__ void __launch_bounds__(256)  cubecl5_2c68f193a70eb284(float const* const __restrict__ v1, float * const __restrict__ v2, info_st const* const __restrict__ v185) {
extern __shared__ __align__(4) char v191[];uint64_t const* const dynamic_meta_v187 = reinterpret_cast<uint64_t const*>(v185 + 1);
const info_st& info_v186 = *v185;int64_t const v189 = info_v186.scalars_i64[1];
int64_t const v190 = info_v186.scalars_i64[2];
uint64_t const v6 = blockIdx.x;
int64_t const v7 = int64_t(v6);
uint64_t const v8 = threadIdx.x;
int64_t const v9 = int64_t(v8);
int64_t const v11 = 1;
int64_t const v15 = 256;
array<float, 256> * const v192 = reinterpret_cast<array<float, 256> *>(&v191[0]);
bool const v17 = v7 < v190;
if(v17) {
float v194_store;
float * v194 = &v194_store;float const v19 = 0.0;
*v194 = v19;
for(int64_t v23 = v9; v23 < v189; v23 += v15) {
int64_t const v24 = v7 * v189;
int64_t const v25 = v23 + v24;
uint64_t const v26 = uint64_t(v25);
float const* const v27 = &v1[v26];
float const v28 = *v27;
float const v29 = v28 * v28;
float const v30 = *v194;
float const v31 = v30 + v29;
*v194 = v31;
}
float const v32 = *v194;
uint64_t const v36 = uint64_t(v9);
float * const v37 = &(*v192)[v36];
*v37 = v32;
__syncthreads();
int64_t const v38 = 128;
bool const v39 = v9 < v38;
if(v39) {
int64_t const v40 = v9 + v38;
uint64_t const v49 = uint64_t(v40);
float * const v50 = &(*v192)[v49];
float const v51 = *v37;
float const v52 = *v50;
float const v53 = v51 + v52;
*v37 = v53;
}
__syncthreads();
int64_t const v54 = 64;
bool const v55 = v9 < v54;
if(v55) {
int64_t const v56 = v9 + v54;
uint64_t const v65 = uint64_t(v56);
float * const v66 = &(*v192)[v65];
float const v67 = *v37;
float const v68 = *v66;
float const v69 = v67 + v68;
*v37 = v69;
}
__syncthreads();
int64_t const v70 = 32;
bool const v71 = v9 < v70;
if(v71) {
int64_t const v72 = v9 + v70;
uint64_t const v81 = uint64_t(v72);
float * const v82 = &(*v192)[v81];
float const v83 = *v37;
float const v84 = *v82;
float const v85 = v83 + v84;
*v37 = v85;
}
__syncthreads();
int64_t const v86 = 16;
bool const v87 = v9 < v86;
if(v87) {
int64_t const v88 = v9 + v86;
uint64_t const v97 = uint64_t(v88);
float * const v98 = &(*v192)[v97];
float const v99 = *v37;
float const v100 = *v98;
float const v101 = v99 + v100;
*v37 = v101;
}
__syncthreads();
int64_t const v102 = 8;
bool const v103 = v9 < v102;
if(v103) {
int64_t const v104 = v9 + v102;
uint64_t const v113 = uint64_t(v104);
float * const v114 = &(*v192)[v113];
float const v115 = *v37;
float const v116 = *v114;
float const v117 = v115 + v116;
*v37 = v117;
}
__syncthreads();
int64_t const v118 = 4;
bool const v119 = v9 < v118;
if(v119) {
int64_t const v120 = v9 + v118;
uint64_t const v129 = uint64_t(v120);
float * const v130 = &(*v192)[v129];
float const v131 = *v37;
float const v132 = *v130;
float const v133 = v131 + v132;
*v37 = v133;
}
__syncthreads();
int64_t const v134 = 2;
bool const v135 = v9 < v134;
if(v135) {
int64_t const v136 = v9 + v134;
uint64_t const v145 = uint64_t(v136);
float * const v146 = &(*v192)[v145];
float const v147 = *v37;
float const v148 = *v146;
float const v149 = v147 + v148;
*v37 = v149;
}
__syncthreads();
bool const v151 = v9 < v11;
if(v151) {
int64_t const v152 = v9 + v11;
uint64_t const v161 = uint64_t(v152);
float * const v162 = &(*v192)[v161];
float const v163 = *v37;
float const v164 = *v162;
float const v165 = v163 + v164;
*v37 = v165;
}
__syncthreads();
uint64_t const v195 = 0;
float * const v167 = &(*v192)[v195];
float const v168 = *v167;
for(int64_t v169 = v9; v169 < v189; v169 += v15) {
int64_t const v170 = v7 * v189;
int64_t const v171 = v169 + v170;
uint64_t const v172 = uint64_t(v171);
float const* const v173 = &v1[v172];
float const v174 = *v173;
float const v175 = float(v189);
float const v176 = v168 / v175;
float const v177 = 9.999999740e-06;
float const v178 = v176 + v177;
float const v179 = rsqrt(v178);float const v180 = v174 * v179;
float * const v184 = &v2[v172];
*v184 = v180;
}
}
return;
}
