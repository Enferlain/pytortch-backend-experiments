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
#include <cuda_fp16.h>

struct info_st {
    int64_t scalars_i64[3];uint64_t static_meta[2];
};
extern "C" __global__ void __launch_bounds__(128)  cubecl5_3e46f10102c50a5c(__half const* const __restrict__ v1, __half * const __restrict__ v2, info_st const* const __restrict__ v106) {
uint64_t const* const dynamic_meta_v108 = reinterpret_cast<uint64_t const*>(v106 + 1);
const info_st& info_v107 = *v106;int64_t const v109 = info_v107.scalars_i64[0];
int64_t const v110 = info_v107.scalars_i64[1];
int64_t const v111 = info_v107.scalars_i64[2];
uint64_t const v6 = blockIdx.x;
int64_t const v7 = int64_t(v6);
uint64_t const v8 = threadIdx.x;
int64_t const v9 = int64_t(v8);
int64_t const v10 = 128;
int64_t const v11 = v7 * v10;
int64_t const v12 = 4;
int64_t const v13 = v11 * v12;
int64_t const v14 = 0;
int64_t const v16 = 1;
int64_t const v18 = v13 + v9;
bool const v19 = v18 < v109;
if(v19) {
int64_t const v113 = v18 % v111;
int64_t const v115 = v113 ^ v111;
bool const v116 = v115 < v14;
bool const v117 = v113 != v14;
bool const v118 = v117 && v116;
int64_t const v119 = v118 ? v111 : v14;
int64_t const v120 = v113 + v119;
int64_t const v21 = v110 * v120;
int64_t const v22 = v18 / v111;
bool const v26 = v113 < v14;
int64_t const v27 = v22 - v16;
int64_t const v28 = v26 ? v27 : v22;
int64_t const v29 = v21 + v28;
uint64_t const v30 = uint64_t(v29);
__half const* const v31 = &v1[v30];
__half const v32 = *v31;
float const v33 = float(v32);
__half const v34 = __half(v33);
uint64_t const v35 = uint64_t(v18);
__half * const v36 = &v2[v35];
*v36 = v34;
}
int64_t const v38 = v9 + v10;
int64_t const v41 = v13 + v38;
bool const v42 = v41 < v109;
if(v42) {
int64_t const v121 = v41 % v111;
int64_t const v123 = v121 ^ v111;
bool const v124 = v123 < v14;
bool const v125 = v121 != v14;
bool const v126 = v125 && v124;
int64_t const v127 = v126 ? v111 : v14;
int64_t const v128 = v121 + v127;
int64_t const v44 = v110 * v128;
int64_t const v45 = v41 / v111;
bool const v49 = v121 < v14;
int64_t const v50 = v45 - v16;
int64_t const v51 = v49 ? v50 : v45;
int64_t const v52 = v44 + v51;
uint64_t const v53 = uint64_t(v52);
__half const* const v54 = &v1[v53];
__half const v55 = *v54;
float const v56 = float(v55);
__half const v57 = __half(v56);
uint64_t const v58 = uint64_t(v41);
__half * const v59 = &v2[v58];
*v59 = v57;
}
int64_t const v60 = 256;
int64_t const v61 = v9 + v60;
int64_t const v64 = v13 + v61;
bool const v65 = v64 < v109;
if(v65) {
int64_t const v129 = v64 % v111;
int64_t const v131 = v129 ^ v111;
bool const v132 = v131 < v14;
bool const v133 = v129 != v14;
bool const v134 = v133 && v132;
int64_t const v135 = v134 ? v111 : v14;
int64_t const v136 = v129 + v135;
int64_t const v67 = v110 * v136;
int64_t const v68 = v64 / v111;
bool const v72 = v129 < v14;
int64_t const v73 = v68 - v16;
int64_t const v74 = v72 ? v73 : v68;
int64_t const v75 = v67 + v74;
uint64_t const v76 = uint64_t(v75);
__half const* const v77 = &v1[v76];
__half const v78 = *v77;
float const v79 = float(v78);
__half const v80 = __half(v79);
uint64_t const v81 = uint64_t(v64);
__half * const v82 = &v2[v81];
*v82 = v80;
}
int64_t const v83 = 384;
int64_t const v84 = v9 + v83;
int64_t const v87 = v13 + v84;
bool const v88 = v87 < v109;
if(v88) {
int64_t const v137 = v87 % v111;
int64_t const v139 = v137 ^ v111;
bool const v140 = v139 < v14;
bool const v141 = v137 != v14;
bool const v142 = v141 && v140;
int64_t const v143 = v142 ? v111 : v14;
int64_t const v144 = v137 + v143;
int64_t const v90 = v110 * v144;
int64_t const v91 = v87 / v111;
bool const v95 = v137 < v14;
int64_t const v96 = v91 - v16;
int64_t const v97 = v95 ? v96 : v91;
int64_t const v98 = v90 + v97;
uint64_t const v99 = uint64_t(v98);
__half const* const v100 = &v1[v99];
__half const v101 = *v100;
float const v102 = float(v101);
__half const v103 = __half(v102);
uint64_t const v104 = uint64_t(v87);
__half * const v105 = &v2[v104];
*v105 = v103;
}
return;
}
