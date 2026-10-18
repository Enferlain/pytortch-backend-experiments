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
extern "C" __global__ void __launch_bounds__(128)  cubecl5_f47b6c01d28226a6(__nv_bfloat16 const* const __restrict__ v1, __nv_bfloat16 * const __restrict__ v2, info_st const* const __restrict__ v150) {
uint64_t const* const dynamic_meta_v152 = reinterpret_cast<uint64_t const*>(v150 + 1);
const info_st& info_v151 = *v150;int64_t const v153 = info_v151.scalars_i64[0];
int64_t const v154 = info_v151.scalars_i64[1];
int64_t const v155 = info_v151.scalars_i64[2];
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
bool const v19 = v18 < v153;
if(v19) {
int64_t const v157 = v18 % v155;
int64_t const v159 = v157 ^ v155;
bool const v160 = v159 < v14;
bool const v161 = v157 != v14;
bool const v162 = v161 && v160;
int64_t const v163 = v162 ? v155 : v14;
int64_t const v164 = v157 + v163;
int64_t const v21 = v154 * v164;
int64_t const v22 = v18 / v155;
bool const v26 = v157 < v14;
int64_t const v27 = v22 - v16;
int64_t const v28 = v26 ? v27 : v22;
int64_t const v29 = v21 + v28;
uint64_t const v30 = uint64_t(v29);
__nv_bfloat16 const* const v31 = &v1[v30];
__nv_bfloat16 const v32 = *v31;
float const v33 = float(v32);
__nv_bfloat16 const v34 = __nv_bfloat16(v33);
float const v35 = float(v34);
float const v36 = sin(v35);__nv_bfloat16 const v37 = __nv_bfloat16(v36);
float const v38 = float(v37);
__nv_bfloat16 const v39 = __nv_bfloat16(v38);
float const v40 = float(v39);
float const v41 = 1.0;
float const v42 = v40 + v41;
__nv_bfloat16 const v43 = __nv_bfloat16(v42);
float const v44 = float(v43);
__nv_bfloat16 const v45 = __nv_bfloat16(v44);
uint64_t const v46 = uint64_t(v18);
__nv_bfloat16 * const v47 = &v2[v46];
*v47 = v45;
}
int64_t const v49 = v9 + v10;
int64_t const v52 = v13 + v49;
bool const v53 = v52 < v153;
if(v53) {
int64_t const v165 = v52 % v155;
int64_t const v167 = v165 ^ v155;
bool const v168 = v167 < v14;
bool const v169 = v165 != v14;
bool const v170 = v169 && v168;
int64_t const v171 = v170 ? v155 : v14;
int64_t const v172 = v165 + v171;
int64_t const v55 = v154 * v172;
int64_t const v56 = v52 / v155;
bool const v60 = v165 < v14;
int64_t const v61 = v56 - v16;
int64_t const v62 = v60 ? v61 : v56;
int64_t const v63 = v55 + v62;
uint64_t const v64 = uint64_t(v63);
__nv_bfloat16 const* const v65 = &v1[v64];
__nv_bfloat16 const v66 = *v65;
float const v67 = float(v66);
__nv_bfloat16 const v68 = __nv_bfloat16(v67);
float const v69 = float(v68);
float const v70 = sin(v69);__nv_bfloat16 const v71 = __nv_bfloat16(v70);
float const v72 = float(v71);
__nv_bfloat16 const v73 = __nv_bfloat16(v72);
float const v74 = float(v73);
float const v75 = 1.0;
float const v76 = v74 + v75;
__nv_bfloat16 const v77 = __nv_bfloat16(v76);
float const v78 = float(v77);
__nv_bfloat16 const v79 = __nv_bfloat16(v78);
uint64_t const v80 = uint64_t(v52);
__nv_bfloat16 * const v81 = &v2[v80];
*v81 = v79;
}
int64_t const v82 = 256;
int64_t const v83 = v9 + v82;
int64_t const v86 = v13 + v83;
bool const v87 = v86 < v153;
if(v87) {
int64_t const v173 = v86 % v155;
int64_t const v175 = v173 ^ v155;
bool const v176 = v175 < v14;
bool const v177 = v173 != v14;
bool const v178 = v177 && v176;
int64_t const v179 = v178 ? v155 : v14;
int64_t const v180 = v173 + v179;
int64_t const v89 = v154 * v180;
int64_t const v90 = v86 / v155;
bool const v94 = v173 < v14;
int64_t const v95 = v90 - v16;
int64_t const v96 = v94 ? v95 : v90;
int64_t const v97 = v89 + v96;
uint64_t const v98 = uint64_t(v97);
__nv_bfloat16 const* const v99 = &v1[v98];
__nv_bfloat16 const v100 = *v99;
float const v101 = float(v100);
__nv_bfloat16 const v102 = __nv_bfloat16(v101);
float const v103 = float(v102);
float const v104 = sin(v103);__nv_bfloat16 const v105 = __nv_bfloat16(v104);
float const v106 = float(v105);
__nv_bfloat16 const v107 = __nv_bfloat16(v106);
float const v108 = float(v107);
float const v109 = 1.0;
float const v110 = v108 + v109;
__nv_bfloat16 const v111 = __nv_bfloat16(v110);
float const v112 = float(v111);
__nv_bfloat16 const v113 = __nv_bfloat16(v112);
uint64_t const v114 = uint64_t(v86);
__nv_bfloat16 * const v115 = &v2[v114];
*v115 = v113;
}
int64_t const v116 = 384;
int64_t const v117 = v9 + v116;
int64_t const v120 = v13 + v117;
bool const v121 = v120 < v153;
if(v121) {
int64_t const v181 = v120 % v155;
int64_t const v183 = v181 ^ v155;
bool const v184 = v183 < v14;
bool const v185 = v181 != v14;
bool const v186 = v185 && v184;
int64_t const v187 = v186 ? v155 : v14;
int64_t const v188 = v181 + v187;
int64_t const v123 = v154 * v188;
int64_t const v124 = v120 / v155;
bool const v128 = v181 < v14;
int64_t const v129 = v124 - v16;
int64_t const v130 = v128 ? v129 : v124;
int64_t const v131 = v123 + v130;
uint64_t const v132 = uint64_t(v131);
__nv_bfloat16 const* const v133 = &v1[v132];
__nv_bfloat16 const v134 = *v133;
float const v135 = float(v134);
__nv_bfloat16 const v136 = __nv_bfloat16(v135);
float const v137 = float(v136);
float const v138 = sin(v137);__nv_bfloat16 const v139 = __nv_bfloat16(v138);
float const v140 = float(v139);
__nv_bfloat16 const v141 = __nv_bfloat16(v140);
float const v142 = float(v141);
float const v143 = 1.0;
float const v144 = v142 + v143;
__nv_bfloat16 const v145 = __nv_bfloat16(v144);
float const v146 = float(v145);
__nv_bfloat16 const v147 = __nv_bfloat16(v146);
uint64_t const v148 = uint64_t(v120);
__nv_bfloat16 * const v149 = &v2[v148];
*v149 = v147;
}
return;
}
