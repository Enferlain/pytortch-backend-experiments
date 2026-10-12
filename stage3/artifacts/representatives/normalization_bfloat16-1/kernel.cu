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
extern "C" __global__ void __launch_bounds__(256)  cubecl3_264e59064206ba8d(__nv_bfloat16 const* const __restrict__ v1, float * const __restrict__ v2, info_st const* const __restrict__ v123) {
extern __shared__ __align__(4) char v129[];uint64_t const* const dynamic_meta_v125 = reinterpret_cast<uint64_t const*>(v123 + 1);
const info_st& info_v124 = *v123;int64_t const v127 = info_v124.scalars_i64[1];
int64_t const v128 = info_v124.scalars_i64[2];
uint64_t const v6 = blockIdx.x;
int64_t const v7 = int64_t(v6);
uint64_t const v8 = threadIdx.x;
int64_t const v9 = int64_t(v8);
int64_t const v10 = 0;
int64_t const v11 = 256;
array<float, 256> * const v130 = reinterpret_cast<array<float, 256> *>(&v129[0]);
bool const v13 = v7 < v128;
if(v13) {
float v132_store;
float * v132 = &v132_store;float const v15 = 0.0;
*v132 = v15;
for(int64_t v16 = v9; v16 < v127; v16 += v11) {
int64_t const v17 = v7 * v127;
int64_t const v18 = v16 + v17;
uint64_t const v19 = uint64_t(v18);
__nv_bfloat16 const* const v20 = &v1[v19];
__nv_bfloat16 const v21 = *v20;
float const v22 = float(v21);
__nv_bfloat16 const v23 = __nv_bfloat16(v22);
float const v24 = float(v23);
float const v26 = v24 * v24;
float const v27 = *v132;
float const v28 = v27 + v26;
*v132 = v28;
}
float const v29 = *v132;
uint64_t const v30 = uint64_t(v9);
float * const v31 = &(*v130)[v30];
*v31 = v29;
__syncthreads();
int64_t const v32 = 128;
bool const v33 = v9 < v32;
if(v33) {
int64_t const v34 = v9 + v32;
uint64_t const v37 = uint64_t(v34);
float * const v38 = &(*v130)[v37];
float const v39 = *v31;
float const v40 = *v38;
float const v41 = v39 + v40;
*v31 = v41;
}
__syncthreads();
int64_t const v42 = 64;
bool const v43 = v9 < v42;
if(v43) {
int64_t const v44 = v9 + v42;
uint64_t const v47 = uint64_t(v44);
float * const v48 = &(*v130)[v47];
float const v49 = *v31;
float const v50 = *v48;
float const v51 = v49 + v50;
*v31 = v51;
}
__syncthreads();
int64_t const v52 = 32;
bool const v53 = v9 < v52;
if(v53) {
int64_t const v54 = v9 + v52;
uint64_t const v57 = uint64_t(v54);
float * const v58 = &(*v130)[v57];
float const v59 = *v31;
float const v60 = *v58;
float const v61 = v59 + v60;
*v31 = v61;
}
__syncthreads();
int64_t const v62 = 16;
bool const v63 = v9 < v62;
if(v63) {
int64_t const v64 = v9 + v62;
uint64_t const v67 = uint64_t(v64);
float * const v68 = &(*v130)[v67];
float const v69 = *v31;
float const v70 = *v68;
float const v71 = v69 + v70;
*v31 = v71;
}
__syncthreads();
int64_t const v72 = 8;
bool const v73 = v9 < v72;
if(v73) {
int64_t const v74 = v9 + v72;
uint64_t const v77 = uint64_t(v74);
float * const v78 = &(*v130)[v77];
float const v79 = *v31;
float const v80 = *v78;
float const v81 = v79 + v80;
*v31 = v81;
}
__syncthreads();
int64_t const v82 = 4;
bool const v83 = v9 < v82;
if(v83) {
int64_t const v84 = v9 + v82;
uint64_t const v87 = uint64_t(v84);
float * const v88 = &(*v130)[v87];
float const v89 = *v31;
float const v90 = *v88;
float const v91 = v89 + v90;
*v31 = v91;
}
__syncthreads();
int64_t const v92 = 2;
bool const v93 = v9 < v92;
if(v93) {
int64_t const v94 = v9 + v92;
uint64_t const v97 = uint64_t(v94);
float * const v98 = &(*v130)[v97];
float const v99 = *v31;
float const v100 = *v98;
float const v101 = v99 + v100;
*v31 = v101;
}
__syncthreads();
int64_t const v102 = 1;
bool const v103 = v9 < v102;
if(v103) {
int64_t const v104 = v9 + v102;
uint64_t const v107 = uint64_t(v104);
float * const v108 = &(*v130)[v107];
float const v109 = *v31;
float const v110 = *v108;
float const v111 = v109 + v110;
*v31 = v111;
}
__syncthreads();
uint64_t const v133 = 0;
float * const v113 = &(*v130)[v133];
float const v114 = *v113;
__syncthreads();
bool const v115 = v9 == v10;
if(v115) {
float const v116 = float(v127);
float const v117 = v114 / v116;
float const v118 = 9.999999740e-06;
float const v119 = v117 + v118;
float const v120 = rsqrt(v119);uint64_t const v121 = uint64_t(v7);
float * const v122 = &v2[v121];
*v122 = v120;
}
}
return;
}
