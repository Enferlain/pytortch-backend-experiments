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
extern "C" __global__ void __launch_bounds__(64)  cubecl5_b9862d901ad45222(float const* const __restrict__ v1, float * const __restrict__ v2, info_st const* const __restrict__ v153) {
extern __shared__ __align__(4) char v159[];uint64_t const* const dynamic_meta_v155 = reinterpret_cast<uint64_t const*>(v153 + 1);
const info_st& info_v154 = *v153;int64_t const v157 = info_v154.scalars_i64[1];
int64_t const v158 = info_v154.scalars_i64[2];
uint64_t const v6 = blockIdx.x;
int64_t const v7 = int64_t(v6);
uint64_t const v8 = threadIdx.x;
int64_t const v9 = int64_t(v8);
int64_t const v11 = 1;
int64_t const v15 = 64;
array<float, 64> * const v160 = reinterpret_cast<array<float, 64> *>(&v159[0]);
bool const v17 = v7 < v158;
if(v17) {
float v162_store;
float * v162 = &v162_store;float const v19 = 0.0;
*v162 = v19;
for(int64_t v23 = v9; v23 < v157; v23 += v15) {
int64_t const v24 = v7 * v157;
int64_t const v25 = v23 + v24;
uint64_t const v26 = uint64_t(v25);
float const* const v27 = &v1[v26];
float const v28 = *v27;
float const v29 = v28 * v28;
float const v30 = *v162;
float const v31 = v30 + v29;
*v162 = v31;
}
float const v32 = *v162;
uint64_t const v36 = uint64_t(v9);
float * const v37 = &(*v160)[v36];
*v37 = v32;
__syncthreads();
int64_t const v38 = 32;
bool const v39 = v9 < v38;
if(v39) {
int64_t const v40 = v9 + v38;
uint64_t const v49 = uint64_t(v40);
float * const v50 = &(*v160)[v49];
float const v51 = *v37;
float const v52 = *v50;
float const v53 = v51 + v52;
*v37 = v53;
}
__syncthreads();
int64_t const v54 = 16;
bool const v55 = v9 < v54;
if(v55) {
int64_t const v56 = v9 + v54;
uint64_t const v65 = uint64_t(v56);
float * const v66 = &(*v160)[v65];
float const v67 = *v37;
float const v68 = *v66;
float const v69 = v67 + v68;
*v37 = v69;
}
__syncthreads();
int64_t const v70 = 8;
bool const v71 = v9 < v70;
if(v71) {
int64_t const v72 = v9 + v70;
uint64_t const v81 = uint64_t(v72);
float * const v82 = &(*v160)[v81];
float const v83 = *v37;
float const v84 = *v82;
float const v85 = v83 + v84;
*v37 = v85;
}
__syncthreads();
int64_t const v86 = 4;
bool const v87 = v9 < v86;
if(v87) {
int64_t const v88 = v9 + v86;
uint64_t const v97 = uint64_t(v88);
float * const v98 = &(*v160)[v97];
float const v99 = *v37;
float const v100 = *v98;
float const v101 = v99 + v100;
*v37 = v101;
}
__syncthreads();
int64_t const v102 = 2;
bool const v103 = v9 < v102;
if(v103) {
int64_t const v104 = v9 + v102;
uint64_t const v113 = uint64_t(v104);
float * const v114 = &(*v160)[v113];
float const v115 = *v37;
float const v116 = *v114;
float const v117 = v115 + v116;
*v37 = v117;
}
__syncthreads();
bool const v119 = v9 < v11;
if(v119) {
int64_t const v120 = v9 + v11;
uint64_t const v129 = uint64_t(v120);
float * const v130 = &(*v160)[v129];
float const v131 = *v37;
float const v132 = *v130;
float const v133 = v131 + v132;
*v37 = v133;
}
__syncthreads();
uint64_t const v163 = 0;
float * const v135 = &(*v160)[v163];
float const v136 = *v135;
for(int64_t v137 = v9; v137 < v157; v137 += v15) {
int64_t const v138 = v7 * v157;
int64_t const v139 = v137 + v138;
uint64_t const v140 = uint64_t(v139);
float const* const v141 = &v1[v140];
float const v142 = *v141;
float const v143 = float(v157);
float const v144 = v136 / v143;
float const v145 = 9.999999740e-06;
float const v146 = v144 + v145;
float const v147 = rsqrt(v146);float const v148 = v142 * v147;
float * const v152 = &v2[v140];
*v152 = v148;
}
}
return;
}
