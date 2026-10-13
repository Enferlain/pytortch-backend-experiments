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
    int64_t scalars_i64[2];uint64_t static_meta[4];
};
extern "C" __global__ void __launch_bounds__(256)  cubecl3_e7fb50b4e1c9d716(bool const* const __restrict__ v1, float const* const __restrict__ v2, float const* const __restrict__ v3, float * const __restrict__ v4, info_st const* const __restrict__ v140) {
extern __shared__ __align__(4) char v145[];uint64_t const* const dynamic_meta_v142 = reinterpret_cast<uint64_t const*>(v140 + 1);
const info_st& info_v141 = *v140;int64_t const v144 = info_v141.scalars_i64[1];
uint64_t const v7 = blockIdx.x;
int64_t const v8 = int64_t(v7);
uint64_t const v9 = threadIdx.x;
int64_t const v10 = int64_t(v9);
int64_t const v11 = 0;
int64_t const v12 = 256;
int64_t const v13 = 1;
int64_t const v14 = 9;
int64_t const v15 = v14 * v144;
array<float, 256> * const v146 = reinterpret_cast<array<float, 256> *>(&v145[0]);
bool const v17 = v8 < v13;
if(v17) {
float v148_store;
float * v148 = &v148_store;float const v19 = 0.0;
*v148 = v19;
for(int64_t v20 = v10; v20 < v15; v20 += v12) {
uint64_t const v21 = uint64_t(v20);
bool const* const v22 = &v1[v21];
bool const v23 = *v22;
float const v24 = float(v23);
float const* const v26 = &v2[v21];
float const v27 = *v26;
int64_t const v29 = v20 / v14;
int64_t const v30 = v20 % v14;
bool const v33 = v30 < v11;
int64_t const v34 = v29 - v13;
int64_t const v35 = v33 ? v34 : v29;
uint64_t const v36 = uint64_t(v35);
float const* const v37 = &v3[v36];
float const v38 = *v37;
float const v39 = v27 * v38;
float const v40 = v24 * v39;
float const v41 = 1.33333337;
float const v42 = v40 * v41;
float const v43 = v42 * v42;
float const v44 = *v148;
float const v45 = v44 + v43;
*v148 = v45;
}
float const v46 = *v148;
uint64_t const v47 = uint64_t(v10);
float * const v48 = &(*v146)[v47];
*v48 = v46;
__syncthreads();
int64_t const v49 = 128;
bool const v50 = v10 < v49;
if(v50) {
int64_t const v51 = v10 + v49;
uint64_t const v54 = uint64_t(v51);
float * const v55 = &(*v146)[v54];
float const v56 = *v48;
float const v57 = *v55;
float const v58 = v56 + v57;
*v48 = v58;
}
__syncthreads();
int64_t const v59 = 64;
bool const v60 = v10 < v59;
if(v60) {
int64_t const v61 = v10 + v59;
uint64_t const v64 = uint64_t(v61);
float * const v65 = &(*v146)[v64];
float const v66 = *v48;
float const v67 = *v65;
float const v68 = v66 + v67;
*v48 = v68;
}
__syncthreads();
int64_t const v69 = 32;
bool const v70 = v10 < v69;
if(v70) {
int64_t const v71 = v10 + v69;
uint64_t const v74 = uint64_t(v71);
float * const v75 = &(*v146)[v74];
float const v76 = *v48;
float const v77 = *v75;
float const v78 = v76 + v77;
*v48 = v78;
}
__syncthreads();
int64_t const v79 = 16;
bool const v80 = v10 < v79;
if(v80) {
int64_t const v81 = v10 + v79;
uint64_t const v84 = uint64_t(v81);
float * const v85 = &(*v146)[v84];
float const v86 = *v48;
float const v87 = *v85;
float const v88 = v86 + v87;
*v48 = v88;
}
__syncthreads();
int64_t const v89 = 8;
bool const v90 = v10 < v89;
if(v90) {
int64_t const v91 = v10 + v89;
uint64_t const v94 = uint64_t(v91);
float * const v95 = &(*v146)[v94];
float const v96 = *v48;
float const v97 = *v95;
float const v98 = v96 + v97;
*v48 = v98;
}
__syncthreads();
int64_t const v99 = 4;
bool const v100 = v10 < v99;
if(v100) {
int64_t const v101 = v10 + v99;
uint64_t const v104 = uint64_t(v101);
float * const v105 = &(*v146)[v104];
float const v106 = *v48;
float const v107 = *v105;
float const v108 = v106 + v107;
*v48 = v108;
}
__syncthreads();
int64_t const v109 = 2;
bool const v110 = v10 < v109;
if(v110) {
int64_t const v111 = v10 + v109;
uint64_t const v114 = uint64_t(v111);
float * const v115 = &(*v146)[v114];
float const v116 = *v48;
float const v117 = *v115;
float const v118 = v116 + v117;
*v48 = v118;
}
__syncthreads();
bool const v120 = v10 < v13;
if(v120) {
int64_t const v121 = v10 + v13;
uint64_t const v124 = uint64_t(v121);
float * const v125 = &(*v146)[v124];
float const v126 = *v48;
float const v127 = *v125;
float const v128 = v126 + v127;
*v48 = v128;
}
__syncthreads();
uint64_t const v149 = 0;
float * const v130 = &(*v146)[v149];
float const v131 = *v130;
__syncthreads();
bool const v132 = v10 == v11;
if(v132) {
float const v135 = float(v15);
float const v136 = v131 / v135;
float * const v139 = &v4[v149];
*v139 = v136;
}
}
return;
}
