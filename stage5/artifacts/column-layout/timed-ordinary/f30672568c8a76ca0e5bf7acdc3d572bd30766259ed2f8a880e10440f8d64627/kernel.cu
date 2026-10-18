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
    int64_t scalars_i64[3];uint64_t static_meta[3];
};
extern "C" __global__ void __launch_bounds__(128)  cubecl5_89fd3f424d42b4e5(__half const* const __restrict__ v1, __half const* const __restrict__ v2, __half * const __restrict__ v3, info_st const* const __restrict__ v139) {
uint64_t const* const dynamic_meta_v141 = reinterpret_cast<uint64_t const*>(v139 + 1);
const info_st& info_v140 = *v139;int64_t const v142 = info_v140.scalars_i64[0];
int64_t const v143 = info_v140.scalars_i64[1];
uint64_t const v7 = blockIdx.x;
int64_t const v8 = int64_t(v7);
uint64_t const v9 = threadIdx.x;
int64_t const v10 = int64_t(v9);
int64_t const v11 = 128;
int64_t const v12 = v8 * v11;
int64_t const v13 = 4;
int64_t const v14 = v12 * v13;
int64_t const v15 = 0;
int64_t const v17 = 1;
int64_t const v19 = v14 + v10;
bool const v20 = v19 < v142;
if(v20) {
uint64_t const v21 = uint64_t(v19);
__half const* const v22 = &v1[v21];
__half const v23 = *v22;
float const v24 = float(v23);
__half const v25 = __half(v24);
float const v26 = float(v25);
int64_t const v27 = v19 / v143;
int64_t const v28 = v19 % v143;
bool const v31 = v28 < v15;
int64_t const v32 = v27 - v17;
int64_t const v33 = v31 ? v32 : v27;
uint64_t const v34 = uint64_t(v33);
__half const* const v35 = &v2[v34];
__half const v36 = *v35;
float const v37 = float(v36);
__half const v38 = __half(v37);
float const v39 = float(v38);
float const v40 = v26 + v39;
__half const v41 = __half(v40);
float const v42 = float(v41);
__half const v43 = __half(v42);
__half * const v45 = &v3[v21];
*v45 = v43;
}
int64_t const v47 = v10 + v11;
int64_t const v50 = v14 + v47;
bool const v51 = v50 < v142;
if(v51) {
uint64_t const v52 = uint64_t(v50);
__half const* const v53 = &v1[v52];
__half const v54 = *v53;
float const v55 = float(v54);
__half const v56 = __half(v55);
float const v57 = float(v56);
int64_t const v58 = v50 / v143;
int64_t const v59 = v50 % v143;
bool const v62 = v59 < v15;
int64_t const v63 = v58 - v17;
int64_t const v64 = v62 ? v63 : v58;
uint64_t const v65 = uint64_t(v64);
__half const* const v66 = &v2[v65];
__half const v67 = *v66;
float const v68 = float(v67);
__half const v69 = __half(v68);
float const v70 = float(v69);
float const v71 = v57 + v70;
__half const v72 = __half(v71);
float const v73 = float(v72);
__half const v74 = __half(v73);
__half * const v76 = &v3[v52];
*v76 = v74;
}
int64_t const v77 = 256;
int64_t const v78 = v10 + v77;
int64_t const v81 = v14 + v78;
bool const v82 = v81 < v142;
if(v82) {
uint64_t const v83 = uint64_t(v81);
__half const* const v84 = &v1[v83];
__half const v85 = *v84;
float const v86 = float(v85);
__half const v87 = __half(v86);
float const v88 = float(v87);
int64_t const v89 = v81 / v143;
int64_t const v90 = v81 % v143;
bool const v93 = v90 < v15;
int64_t const v94 = v89 - v17;
int64_t const v95 = v93 ? v94 : v89;
uint64_t const v96 = uint64_t(v95);
__half const* const v97 = &v2[v96];
__half const v98 = *v97;
float const v99 = float(v98);
__half const v100 = __half(v99);
float const v101 = float(v100);
float const v102 = v88 + v101;
__half const v103 = __half(v102);
float const v104 = float(v103);
__half const v105 = __half(v104);
__half * const v107 = &v3[v83];
*v107 = v105;
}
int64_t const v108 = 384;
int64_t const v109 = v10 + v108;
int64_t const v112 = v14 + v109;
bool const v113 = v112 < v142;
if(v113) {
uint64_t const v114 = uint64_t(v112);
__half const* const v115 = &v1[v114];
__half const v116 = *v115;
float const v117 = float(v116);
__half const v118 = __half(v117);
float const v119 = float(v118);
int64_t const v120 = v112 / v143;
int64_t const v121 = v112 % v143;
bool const v124 = v121 < v15;
int64_t const v125 = v120 - v17;
int64_t const v126 = v124 ? v125 : v120;
uint64_t const v127 = uint64_t(v126);
__half const* const v128 = &v2[v127];
__half const v129 = *v128;
float const v130 = float(v129);
__half const v131 = __half(v130);
float const v132 = float(v131);
float const v133 = v119 + v132;
__half const v134 = __half(v133);
float const v135 = float(v134);
__half const v136 = __half(v135);
__half * const v138 = &v3[v114];
*v138 = v136;
}
return;
}
