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
extern "C" __global__ void __launch_bounds__(256)  cubecl5_transpose_67047ac68fc8641d(__nv_bfloat16 const* const __restrict__ v1, __nv_bfloat16 * const __restrict__ v2, info_st const* const __restrict__ v127) {
uint64_t const* const dynamic_meta_v129 = reinterpret_cast<uint64_t const*>(v127 + 1);
const info_st& info_v128 = *v127;bool inv_terminate_flag_v133_store = true;
bool * inv_terminate_flag_v133 = &inv_terminate_flag_v133_store;int64_t const v131 = info_v128.scalars_i64[1];
int64_t const v132 = info_v128.scalars_i64[2];
uint64_t const v6 = 2;
uint64_t const v7 = 1;
int64_t const v237 = 1;
int64_t const v10 = 0;
bool const v236 = false;
uint64_t const v13 = uint64_t(v132);
__nv_bfloat16 const v18 = 0.0;
__nv_bfloat16 v136_store;
__nv_bfloat16 * v136 = &v136_store;*v136 = v18;
uint64_t const v26 = 0;
int64_t const v37 = v131 * v132;
uint64_t const v40 = uint64_t(v37);
uint64_t const v43 = v40 / v13;
uint32_t const v148 = blockIdx.z;
uint64_t const v149 = uint64_t(v148);
uint32_t const v150 = gridDim.y;
uint64_t const v151 = uint64_t(v150);
uint64_t const v152 = v149 * v151;
uint32_t const v153 = gridDim.x;
uint64_t const v154 = uint64_t(v153);
uint64_t const v155 = v152 * v154;
uint32_t const v156 = blockIdx.y;
uint64_t const v157 = uint64_t(v156);
uint64_t const v160 = v157 * v154;
uint64_t const v161 = v155 + v160;
uint32_t const v162 = blockIdx.x;
uint64_t const v163 = uint64_t(v162);
uint64_t const v164 = v161 + v163;
uint64_t const v231 = 256;
uint64_t const v167 = v164 * v231;
uint32_t const v195 = threadIdx.x;
uint64_t const v169 = uint64_t(v195);
uint64_t const v170 = v167 + v169;
bool const v45 = v170 >= v43;
if(v45) {
*inv_terminate_flag_v133 = v236;
}
bool const v47 = *inv_terminate_flag_v133;
if(v47) {
uint64_t const v49 = v170 * v13;
uint64_t const v50 = v49 / v13;
uint64_t const v51 = v49 % v13;
uint64_t const v52 = v50 * v13;
uint64_t const v53 = v52 + v51;
for(uint64_t v56 = v26; v56 < v13; v56 += v7) {
uint64_t const v59 = v53 + v56;
uint64_t v139_store;
uint64_t * v139 = &v139_store;*v139 = v26;
for(uint64_t v66 = v26; v66 < v6; v66 += v7) {
int64_t const v71 = int64_t(v66);
bool const v73 = v71 == v10;
int64_t const v75 = v73 ? v132 : v237;
uint64_t const v76 = uint64_t(v75);
uint64_t const v77 = v59 / v76;
int64_t const v81 = v73 ? v131 : v132;
uint64_t const v82 = uint64_t(v81);
uint64_t const v83 = v77 % v82;
int64_t const v88 = v73 ? v237 : v131;
uint64_t const v89 = uint64_t(v88);
uint64_t const v90 = v83 * v89;
uint64_t const v91 = *v139;
uint64_t const v92 = v91 + v90;
*v139 = v92;
}
uint64_t const v93 = *v139;
int64_t const v101 = int64_t(v93);
uint64_t const v104 = uint64_t(v101);
__nv_bfloat16 const* const v105 = &v1[v104];
__nv_bfloat16 const v107 = *v105;
*v136 = v107;
__nv_bfloat16 const v116 = *v136;
int64_t const v117 = int64_t(v59);
float const v122 = float(v116);
__nv_bfloat16 const v123 = __nv_bfloat16(v122);
uint64_t const v124 = uint64_t(v117);
__nv_bfloat16 * const v125 = &v2[v124];
*v125 = v123;
}
}
return;
}
