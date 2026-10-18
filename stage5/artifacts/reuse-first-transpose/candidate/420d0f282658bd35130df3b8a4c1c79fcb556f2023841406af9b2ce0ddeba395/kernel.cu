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
extern "C" __global__ void __launch_bounds__(256)  cubecl5_transpose_420d0f282658bd35(__nv_bfloat16 const* const __restrict__ v1, __nv_bfloat16 * const __restrict__ v2, info_st const* const __restrict__ v138) {
uint64_t const* const dynamic_meta_v140 = reinterpret_cast<uint64_t const*>(v138 + 1);
const info_st& info_v139 = *v138;bool inv_terminate_flag_v144_store = true;
bool * inv_terminate_flag_v144 = &inv_terminate_flag_v144_store;int64_t const v142 = info_v139.scalars_i64[1];
int64_t const v143 = info_v139.scalars_i64[2];
uint64_t const v6 = 2;
uint64_t const v7 = 1;
int64_t const v248 = 1;
int64_t const v10 = 0;
bool const v247 = false;
uint64_t const v13 = uint64_t(v143);
__nv_bfloat16 const v18 = 0.0;
__nv_bfloat16 v147_store;
__nv_bfloat16 * v147 = &v147_store;*v147 = v18;
uint64_t const v26 = 0;
int64_t const v37 = v142 * v143;
uint64_t const v40 = uint64_t(v37);
uint64_t const v43 = v40 / v13;
uint32_t const v159 = blockIdx.z;
uint64_t const v160 = uint64_t(v159);
uint32_t const v161 = gridDim.y;
uint64_t const v162 = uint64_t(v161);
uint64_t const v163 = v160 * v162;
uint32_t const v164 = gridDim.x;
uint64_t const v165 = uint64_t(v164);
uint64_t const v166 = v163 * v165;
uint32_t const v167 = blockIdx.y;
uint64_t const v168 = uint64_t(v167);
uint64_t const v171 = v168 * v165;
uint64_t const v172 = v166 + v171;
uint32_t const v173 = blockIdx.x;
uint64_t const v174 = uint64_t(v173);
uint64_t const v175 = v172 + v174;
uint64_t const v242 = 256;
uint64_t const v178 = v175 * v242;
uint32_t const v206 = threadIdx.x;
uint64_t const v180 = uint64_t(v206);
uint64_t const v181 = v178 + v180;
bool const v45 = v181 >= v43;
if(v45) {
*inv_terminate_flag_v144 = v247;
}
bool const v47 = *inv_terminate_flag_v144;
if(v47) {
uint64_t const v49 = v181 * v13;
uint64_t const v50 = v49 / v13;
uint64_t const v51 = v49 % v13;
uint64_t const v52 = v50 * v13;
uint64_t const v53 = v52 + v51;
for(uint64_t v56 = v26; v56 < v13; v56 += v7) {
uint64_t const v59 = v53 + v56;
uint64_t v150_store;
uint64_t * v150 = &v150_store;*v150 = v26;
for(uint64_t v66 = v26; v66 < v6; v66 += v7) {
int64_t const v71 = int64_t(v66);
bool const v73 = v71 == v10;
int64_t const v75 = v73 ? v143 : v248;
uint64_t const v76 = uint64_t(v75);
uint64_t const v77 = v59 / v76;
int64_t const v81 = v73 ? v142 : v143;
uint64_t const v82 = uint64_t(v81);
uint64_t const v83 = v77 % v82;
int64_t const v88 = v73 ? v248 : v142;
uint64_t const v89 = uint64_t(v88);
uint64_t const v90 = v83 * v89;
uint64_t const v91 = *v150;
uint64_t const v92 = v91 + v90;
*v150 = v92;
}
uint64_t const v93 = *v150;
int64_t const v101 = int64_t(v93);
uint64_t const v104 = uint64_t(v101);
__nv_bfloat16 const* const v105 = &v1[v104];
__nv_bfloat16 const v107 = *v105;
*v147 = v107;
__nv_bfloat16 const v116 = *v147;
int64_t const v117 = int64_t(v59);
float const v122 = float(v116);
__nv_bfloat16 const v123 = __nv_bfloat16(v122);
float const v124 = float(v123);
float const v125 = sin(v124);__nv_bfloat16 const v126 = __nv_bfloat16(v125);
float const v127 = float(v126);
__nv_bfloat16 const v128 = __nv_bfloat16(v127);
float const v129 = float(v128);
float const v130 = 1.0;
float const v131 = v129 + v130;
__nv_bfloat16 const v132 = __nv_bfloat16(v131);
float const v133 = float(v132);
__nv_bfloat16 const v134 = __nv_bfloat16(v133);
uint64_t const v135 = uint64_t(v117);
__nv_bfloat16 * const v136 = &v2[v135];
*v136 = v134;
}
}
return;
}
