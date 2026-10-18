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
    int64_t scalars_i64[3];uint64_t static_meta[3];
};
extern "C" __global__ void __launch_bounds__(128)  cubecl5_6623e7c33f6d3ee9(__nv_bfloat16 const* const __restrict__ v1, __nv_bfloat16 const* const __restrict__ v2, __nv_bfloat16 * const __restrict__ v3, info_st const* const __restrict__ v115) {
uint64_t const* const dynamic_meta_v117 = reinterpret_cast<uint64_t const*>(v115 + 1);
const info_st& info_v116 = *v115;int64_t const v118 = info_v116.scalars_i64[0];
int64_t const v119 = info_v116.scalars_i64[1];
uint64_t const v7 = blockIdx.x;
int64_t const v8 = int64_t(v7);
uint64_t const v9 = threadIdx.x;
int64_t const v10 = int64_t(v9);
int64_t const v11 = 128;
int64_t const v12 = v8 * v11;
int64_t const v13 = 4;
int64_t const v14 = v12 * v13;
int64_t const v15 = 0;
int64_t const v19 = v14 + v10;
bool const v20 = v19 < v118;
if(v20) {
uint64_t const v21 = uint64_t(v19);
__nv_bfloat16 const* const v22 = &v1[v21];
__nv_bfloat16 const v23 = *v22;
float const v24 = float(v23);
__nv_bfloat16 const v25 = __nv_bfloat16(v24);
float const v26 = float(v25);
int64_t const v122 = v19 % v119;
int64_t const v124 = v122 ^ v119;
bool const v125 = v124 < v15;
bool const v126 = v122 != v15;
bool const v127 = v126 && v125;
int64_t const v128 = v127 ? v119 : v15;
int64_t const v129 = v122 + v128;
uint64_t const v28 = uint64_t(v129);
__nv_bfloat16 const* const v29 = &v2[v28];
__nv_bfloat16 const v30 = *v29;
float const v31 = float(v30);
__nv_bfloat16 const v32 = __nv_bfloat16(v31);
float const v33 = float(v32);
float const v34 = v26 + v33;
__nv_bfloat16 const v35 = __nv_bfloat16(v34);
float const v36 = float(v35);
__nv_bfloat16 const v37 = __nv_bfloat16(v36);
__nv_bfloat16 * const v39 = &v3[v21];
*v39 = v37;
}
int64_t const v41 = v10 + v11;
int64_t const v44 = v14 + v41;
bool const v45 = v44 < v118;
if(v45) {
uint64_t const v46 = uint64_t(v44);
__nv_bfloat16 const* const v47 = &v1[v46];
__nv_bfloat16 const v48 = *v47;
float const v49 = float(v48);
__nv_bfloat16 const v50 = __nv_bfloat16(v49);
float const v51 = float(v50);
int64_t const v130 = v44 % v119;
int64_t const v132 = v130 ^ v119;
bool const v133 = v132 < v15;
bool const v134 = v130 != v15;
bool const v135 = v134 && v133;
int64_t const v136 = v135 ? v119 : v15;
int64_t const v137 = v130 + v136;
uint64_t const v53 = uint64_t(v137);
__nv_bfloat16 const* const v54 = &v2[v53];
__nv_bfloat16 const v55 = *v54;
float const v56 = float(v55);
__nv_bfloat16 const v57 = __nv_bfloat16(v56);
float const v58 = float(v57);
float const v59 = v51 + v58;
__nv_bfloat16 const v60 = __nv_bfloat16(v59);
float const v61 = float(v60);
__nv_bfloat16 const v62 = __nv_bfloat16(v61);
__nv_bfloat16 * const v64 = &v3[v46];
*v64 = v62;
}
int64_t const v65 = 256;
int64_t const v66 = v10 + v65;
int64_t const v69 = v14 + v66;
bool const v70 = v69 < v118;
if(v70) {
uint64_t const v71 = uint64_t(v69);
__nv_bfloat16 const* const v72 = &v1[v71];
__nv_bfloat16 const v73 = *v72;
float const v74 = float(v73);
__nv_bfloat16 const v75 = __nv_bfloat16(v74);
float const v76 = float(v75);
int64_t const v138 = v69 % v119;
int64_t const v140 = v138 ^ v119;
bool const v141 = v140 < v15;
bool const v142 = v138 != v15;
bool const v143 = v142 && v141;
int64_t const v144 = v143 ? v119 : v15;
int64_t const v145 = v138 + v144;
uint64_t const v78 = uint64_t(v145);
__nv_bfloat16 const* const v79 = &v2[v78];
__nv_bfloat16 const v80 = *v79;
float const v81 = float(v80);
__nv_bfloat16 const v82 = __nv_bfloat16(v81);
float const v83 = float(v82);
float const v84 = v76 + v83;
__nv_bfloat16 const v85 = __nv_bfloat16(v84);
float const v86 = float(v85);
__nv_bfloat16 const v87 = __nv_bfloat16(v86);
__nv_bfloat16 * const v89 = &v3[v71];
*v89 = v87;
}
int64_t const v90 = 384;
int64_t const v91 = v10 + v90;
int64_t const v94 = v14 + v91;
bool const v95 = v94 < v118;
if(v95) {
uint64_t const v96 = uint64_t(v94);
__nv_bfloat16 const* const v97 = &v1[v96];
__nv_bfloat16 const v98 = *v97;
float const v99 = float(v98);
__nv_bfloat16 const v100 = __nv_bfloat16(v99);
float const v101 = float(v100);
int64_t const v146 = v94 % v119;
int64_t const v148 = v146 ^ v119;
bool const v149 = v148 < v15;
bool const v150 = v146 != v15;
bool const v151 = v150 && v149;
int64_t const v152 = v151 ? v119 : v15;
int64_t const v153 = v146 + v152;
uint64_t const v103 = uint64_t(v153);
__nv_bfloat16 const* const v104 = &v2[v103];
__nv_bfloat16 const v105 = *v104;
float const v106 = float(v105);
__nv_bfloat16 const v107 = __nv_bfloat16(v106);
float const v108 = float(v107);
float const v109 = v101 + v108;
__nv_bfloat16 const v110 = __nv_bfloat16(v109);
float const v111 = float(v110);
__nv_bfloat16 const v112 = __nv_bfloat16(v111);
__nv_bfloat16 * const v114 = &v3[v96];
*v114 = v112;
}
return;
}
