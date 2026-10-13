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
    int64_t scalars_i64[2];uint64_t static_meta[5];
};
extern "C" __global__ void __launch_bounds__(256)  cubecl3_73e7f92701858e6c(float const* const __restrict__ v1, bool const* const __restrict__ v2, float const* const __restrict__ v3, float const* const __restrict__ v4, float * const __restrict__ v5, info_st const* const __restrict__ v145) {
extern __shared__ __align__(4) char v150[];uint64_t const* const dynamic_meta_v147 = reinterpret_cast<uint64_t const*>(v145 + 1);
const info_st& info_v146 = *v145;int64_t const v149 = info_v146.scalars_i64[1];
uint64_t const v8 = blockIdx.x;
int64_t const v9 = int64_t(v8);
uint64_t const v10 = threadIdx.x;
int64_t const v11 = int64_t(v10);
int64_t const v12 = 0;
int64_t const v13 = 256;
int64_t const v14 = 9;
array<float, 256> * const v151 = reinterpret_cast<array<float, 256> *>(&v150[0]);
bool const v16 = v9 < v149;
if(v16) {
float v153_store;
float * v153 = &v153_store;float const v18 = 0.0;
*v153 = v18;
for(int64_t v19 = v11; v19 < v14; v19 += v13) {
uint64_t const v155 = 0;
float const* const v22 = &v1[v155];
float const v23 = *v22;
int64_t const v25 = v14 * v149;
float const v26 = float(v25);
float const v27 = v23 / v26;
int64_t const v29 = v14 * v9;
int64_t const v30 = v19 + v29;
uint64_t const v31 = uint64_t(v30);
bool const* const v32 = &v2[v31];
bool const v33 = *v32;
float const v34 = float(v33);
float const* const v39 = &v3[v31];
float const v40 = *v39;
uint64_t const v41 = uint64_t(v9);
float const* const v42 = &v4[v41];
float const v43 = *v42;
float const v44 = v40 * v43;
float const v45 = v34 * v44;
float const v46 = 1.33333337;
float const v47 = v45 * v46;
float const v48 = 2.0;
float const v49 = v47 * v48;
float const v50 = v27 * v49;
float const v51 = v34 * v46;
float const v52 = v50 * v51;
float const v53 = v52 * v40;
float const v54 = *v153;
float const v55 = v54 + v53;
*v153 = v55;
}
float const v56 = *v153;
uint64_t const v57 = uint64_t(v11);
float * const v58 = &(*v151)[v57];
*v58 = v56;
__syncthreads();
int64_t const v59 = 128;
bool const v60 = v11 < v59;
if(v60) {
int64_t const v61 = v11 + v59;
uint64_t const v64 = uint64_t(v61);
float * const v65 = &(*v151)[v64];
float const v66 = *v58;
float const v67 = *v65;
float const v68 = v66 + v67;
*v58 = v68;
}
__syncthreads();
int64_t const v69 = 64;
bool const v70 = v11 < v69;
if(v70) {
int64_t const v71 = v11 + v69;
uint64_t const v74 = uint64_t(v71);
float * const v75 = &(*v151)[v74];
float const v76 = *v58;
float const v77 = *v75;
float const v78 = v76 + v77;
*v58 = v78;
}
__syncthreads();
int64_t const v79 = 32;
bool const v80 = v11 < v79;
if(v80) {
int64_t const v81 = v11 + v79;
uint64_t const v84 = uint64_t(v81);
float * const v85 = &(*v151)[v84];
float const v86 = *v58;
float const v87 = *v85;
float const v88 = v86 + v87;
*v58 = v88;
}
__syncthreads();
int64_t const v89 = 16;
bool const v90 = v11 < v89;
if(v90) {
int64_t const v91 = v11 + v89;
uint64_t const v94 = uint64_t(v91);
float * const v95 = &(*v151)[v94];
float const v96 = *v58;
float const v97 = *v95;
float const v98 = v96 + v97;
*v58 = v98;
}
__syncthreads();
int64_t const v99 = 8;
bool const v100 = v11 < v99;
if(v100) {
int64_t const v101 = v11 + v99;
uint64_t const v104 = uint64_t(v101);
float * const v105 = &(*v151)[v104];
float const v106 = *v58;
float const v107 = *v105;
float const v108 = v106 + v107;
*v58 = v108;
}
__syncthreads();
int64_t const v109 = 4;
bool const v110 = v11 < v109;
if(v110) {
int64_t const v111 = v11 + v109;
uint64_t const v114 = uint64_t(v111);
float * const v115 = &(*v151)[v114];
float const v116 = *v58;
float const v117 = *v115;
float const v118 = v116 + v117;
*v58 = v118;
}
__syncthreads();
int64_t const v119 = 2;
bool const v120 = v11 < v119;
if(v120) {
int64_t const v121 = v11 + v119;
uint64_t const v124 = uint64_t(v121);
float * const v125 = &(*v151)[v124];
float const v126 = *v58;
float const v127 = *v125;
float const v128 = v126 + v127;
*v58 = v128;
}
__syncthreads();
int64_t const v129 = 1;
bool const v130 = v11 < v129;
if(v130) {
int64_t const v131 = v11 + v129;
uint64_t const v134 = uint64_t(v131);
float * const v135 = &(*v151)[v134];
float const v136 = *v58;
float const v137 = *v135;
float const v138 = v136 + v137;
*v58 = v138;
}
__syncthreads();
uint64_t const v154 = 0;
float * const v140 = &(*v151)[v154];
float const v141 = *v140;
__syncthreads();
bool const v142 = v11 == v12;
if(v142) {
uint64_t const v143 = uint64_t(v9);
float * const v144 = &v5[v143];
*v144 = v141;
}
}
return;
}
