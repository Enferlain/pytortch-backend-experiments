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
    int64_t scalars_i64[3];uint64_t static_meta[4];
};
extern "C" __global__ void __launch_bounds__(256)  cubecl3_bcb4deac8cf7c550(float const* const __restrict__ v1, float const* const __restrict__ v2, float const* const __restrict__ v3, float * const __restrict__ v4, info_st const* const __restrict__ v147) {
extern __shared__ __align__(4) char v153[];uint64_t const* const dynamic_meta_v149 = reinterpret_cast<uint64_t const*>(v147 + 1);
const info_st& info_v148 = *v147;int64_t const v151 = info_v148.scalars_i64[1];
int64_t const v152 = info_v148.scalars_i64[2];
uint64_t const v8 = blockIdx.x;
int64_t const v9 = int64_t(v8);
uint64_t const v10 = threadIdx.x;
int64_t const v11 = int64_t(v10);
int64_t const v13 = 256;
array<float, 256> * const v154 = reinterpret_cast<array<float, 256> *>(&v153[0]);
bool const v15 = v9 < v152;
if(v15) {
float v156_store;
float * v156 = &v156_store;float const v17 = 0.0;
*v156 = v17;
for(int64_t v18 = v11; v18 < v151; v18 += v13) {
int64_t const v19 = v9 * v151;
int64_t const v20 = v18 + v19;
uint64_t const v21 = uint64_t(v20);
float const* const v22 = &v1[v21];
float const v23 = *v22;
float const* const v27 = &v2[v21];
float const v28 = *v27;
float const v29 = v23 * v28;
float const v30 = *v156;
float const v31 = v30 + v29;
*v156 = v31;
}
float const v32 = *v156;
uint64_t const v33 = uint64_t(v11);
float * const v34 = &(*v154)[v33];
*v34 = v32;
__syncthreads();
int64_t const v35 = 128;
bool const v36 = v11 < v35;
if(v36) {
int64_t const v37 = v11 + v35;
uint64_t const v40 = uint64_t(v37);
float * const v41 = &(*v154)[v40];
float const v42 = *v34;
float const v43 = *v41;
float const v44 = v42 + v43;
*v34 = v44;
}
__syncthreads();
int64_t const v45 = 64;
bool const v46 = v11 < v45;
if(v46) {
int64_t const v47 = v11 + v45;
uint64_t const v50 = uint64_t(v47);
float * const v51 = &(*v154)[v50];
float const v52 = *v34;
float const v53 = *v51;
float const v54 = v52 + v53;
*v34 = v54;
}
__syncthreads();
int64_t const v55 = 32;
bool const v56 = v11 < v55;
if(v56) {
int64_t const v57 = v11 + v55;
uint64_t const v60 = uint64_t(v57);
float * const v61 = &(*v154)[v60];
float const v62 = *v34;
float const v63 = *v61;
float const v64 = v62 + v63;
*v34 = v64;
}
__syncthreads();
int64_t const v65 = 16;
bool const v66 = v11 < v65;
if(v66) {
int64_t const v67 = v11 + v65;
uint64_t const v70 = uint64_t(v67);
float * const v71 = &(*v154)[v70];
float const v72 = *v34;
float const v73 = *v71;
float const v74 = v72 + v73;
*v34 = v74;
}
__syncthreads();
int64_t const v75 = 8;
bool const v76 = v11 < v75;
if(v76) {
int64_t const v77 = v11 + v75;
uint64_t const v80 = uint64_t(v77);
float * const v81 = &(*v154)[v80];
float const v82 = *v34;
float const v83 = *v81;
float const v84 = v82 + v83;
*v34 = v84;
}
__syncthreads();
int64_t const v85 = 4;
bool const v86 = v11 < v85;
if(v86) {
int64_t const v87 = v11 + v85;
uint64_t const v90 = uint64_t(v87);
float * const v91 = &(*v154)[v90];
float const v92 = *v34;
float const v93 = *v91;
float const v94 = v92 + v93;
*v34 = v94;
}
__syncthreads();
int64_t const v95 = 2;
bool const v96 = v11 < v95;
if(v96) {
int64_t const v97 = v11 + v95;
uint64_t const v100 = uint64_t(v97);
float * const v101 = &(*v154)[v100];
float const v102 = *v34;
float const v103 = *v101;
float const v104 = v102 + v103;
*v34 = v104;
}
__syncthreads();
int64_t const v105 = 1;
bool const v106 = v11 < v105;
if(v106) {
int64_t const v107 = v11 + v105;
uint64_t const v110 = uint64_t(v107);
float * const v111 = &(*v154)[v110];
float const v112 = *v34;
float const v113 = *v111;
float const v114 = v112 + v113;
*v34 = v114;
}
__syncthreads();
uint64_t const v157 = 0;
float * const v116 = &(*v154)[v157];
float const v117 = *v116;
__syncthreads();
for(int64_t v118 = v11; v118 < v151; v118 += v13) {
int64_t const v119 = v9 * v151;
int64_t const v120 = v118 + v119;
uint64_t const v121 = uint64_t(v120);
float const* const v122 = &v1[v121];
float const v123 = *v122;
float const* const v127 = &v2[v121];
float const v128 = *v127;
uint64_t const v129 = uint64_t(v9);
float const* const v130 = &v3[v129];
float const v131 = *v130;
float const v132 = v123 * v131;
float const v133 = -0.5;
float const v134 = v117 * v133;
float const v135 = v131 * v131;
float const v136 = v135 * v131;
float const v137 = v134 * v136;
float const v138 = float(v151);
float const v139 = v137 / v138;
float const v140 = v139 * v128;
float const v141 = v132 + v140;
float const v142 = v141 + v140;
float * const v146 = &v4[v121];
*v146 = v142;
}
}
return;
}
