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
struct alignas(8) __nv_bfloat16_4 { __nv_bfloat16 i_0; __nv_bfloat16 i_1; __nv_bfloat16 i_2; __nv_bfloat16 i_3; };
struct alignas(16) float_4 { float i_0; float i_1; float i_2; float i_3; };

struct info_st {
    int64_t scalars_i64[3];uint64_t static_meta[2];
};
extern "C" __global__ void __launch_bounds__(256)  cubecl5_rms_cubek_pack4_2088018557b83813(__nv_bfloat16 const* const __restrict__ v1, __nv_bfloat16 * const __restrict__ v2, info_st const* const __restrict__ v206) {
uint64_t const* const dynamic_meta_v208 = reinterpret_cast<uint64_t const*>(v206 + 1);
const info_st& info_v207 = *v206;int64_t const v210 = info_v207.scalars_i64[1];
int64_t const v211 = info_v207.scalars_i64[2];
uint64_t const v6 = blockIdx.x;
int64_t const v7 = int64_t(v6);
uint64_t const v8 = threadIdx.y;
int64_t const v9 = int64_t(v8);
int64_t const v10 = 8;
int64_t const v11 = v7 * v10;
int64_t const v12 = v11 + v9;
uint64_t const v13 = uint64_t(v12);
uint64_t const v14 = uint64_t(v210);
uint64_t const v15 = uint64_t(v211);
bool const v16 = v13 < v15;
if(v16) {
uint64_t const v18 = 0;
uint64_t v213_store;
uint64_t * v213 = &v213_store;*v213 = v18;
uint64_t const v22 = 2;
uint64_t const v23 = 1;
for(uint64_t v24 = v18; v24 < v22; v24 += v23) {
int64_t const v27 = int64_t(v24);
int64_t const v28 = 0;
bool const v29 = v27 == v28;
int64_t const v30 = 1;
int64_t const v39 = v29 ? v211 : v30;
uint64_t const v40 = uint64_t(v39);
uint64_t const v41 = v13 % v40;
int64_t const v46 = v29 ? v210 : v30;
uint64_t const v47 = uint64_t(v46);
uint64_t const v48 = v41 * v47;
uint64_t const v49 = *v213;
uint64_t const v50 = v49 + v48;
*v213 = v50;
}
uint64_t const v51 = *v213;
uint64_t const v52 = 4;
uint64_t const v53 = v51 / v52;
*v213 = v53;
uint64_t const v61 = v14 / v52;
float const v62 = 0.0;
float_4 const v63 = float_4{v62, v62, v62, v62};
int64_t const v68 = 4;
float_4 v218_store;
float_4 * v218 = &v218_store;*v218 = v63;
uint64_t const v305 = 32;
uint64_t const v78 = v61 + v305;
uint64_t const v80 = v78 - v23;
uint64_t const v81 = v80 / v305;
for(uint64_t v84 = v18; v84 < v81; v84 += v23) {
uint64_t const v88 = v84 * v305;
uint32_t const v89 = threadIdx.x;
uint64_t const v90 = uint64_t(v89);
uint64_t const v91 = v88 + v90;
uint64_t const v92 = *v213;
uint64_t const v93 = v91 + v92;
bool const v94 = v91 < v61;
float_4 v221_store;
float_4 * v221 = &v221_store;if(v94) {
int64_t const v95 = int64_t(v93);
int64_t const v97 = v95 * v68;
uint64_t const v98 = uint64_t(v97);
int64_t const v99 = int64_t(v98);
int64_t const v100 = v12 * v210;
int64_t const v101 = v99 - v100;
int64_t const v103 = v101 + v100;
uint64_t const v104 = uint64_t(v103);
__nv_bfloat16 const* const v105 = &v1[v104];
__nv_bfloat16_4 const* const v106 = reinterpret_cast<__nv_bfloat16_4 const*>(v105);
__nv_bfloat16_4 const v107 = *v106;
__nv_bfloat16 const v108 = v107.i_0;
float const v115 = float(v108);
__nv_bfloat16 const v116 = __nv_bfloat16(v115);
float const v117 = float(v116);
float const v119 = v117 * v117;
__nv_bfloat16 const v120 = v107.i_1;
float const v127 = float(v120);
__nv_bfloat16 const v128 = __nv_bfloat16(v127);
float const v129 = float(v128);
float const v131 = v129 * v129;
__nv_bfloat16 const v132 = v107.i_2;
float const v139 = float(v132);
__nv_bfloat16 const v140 = __nv_bfloat16(v139);
float const v141 = float(v140);
float const v143 = v141 * v141;
__nv_bfloat16 const v144 = v107.i_3;
float const v151 = float(v144);
__nv_bfloat16 const v152 = __nv_bfloat16(v151);
float const v153 = float(v152);
float const v155 = v153 * v153;
float_4 const v156 = float_4{v119, v131, v143, v155};
*v221 = v156;
}
 else {
*v221 = v63;
}
float_4 const v163 = *v221;
float_4 const v164 = *v218;
float const v268 = v164.i_0;
float const v269 = v163.i_0;
float const v270 = v268 + v269;
float const v271 = v164.i_1;
float const v272 = v163.i_1;
float const v273 = v271 + v272;
float const v274 = v164.i_2;
float const v275 = v163.i_2;
float const v276 = v274 + v275;
float const v277 = v164.i_3;
float const v278 = v163.i_3;
float const v279 = v277 + v278;
float_4 const v280 = float_4{v270, v273, v276, v279};
*v218 = v280;
}
float_4 const v166 = *v218;
uint32_t const v222 = warpSize;
uint32_t const v304 = 256;
uint32_t const v224 = min(v222, v304);
float_4 v262_store;
float_4 * v262 = &v262_store;*v262 = v166;
uint32_t const v226 = 1;
uint32_t v263_store;
uint32_t * v263 = &v263_store;*v263 = v226;
while([&]{
uint32_t const v229 = *v263;
bool const v230 = v229 < v224;
return v230;
}()) {
float_4 const v233 = *v262;
uint32_t const v234 = *v263;
float const v281 = v233.i_0;
float const v282 = __shfl_xor_sync(__activemask(), v281, v234);
float const v283 = v233.i_1;
float const v284 = __shfl_xor_sync(__activemask(), v283, v234);
float const v285 = v233.i_2;
float const v286 = __shfl_xor_sync(__activemask(), v285, v234);
float const v287 = v233.i_3;
float const v288 = __shfl_xor_sync(__activemask(), v287, v234);
float const v292 = v281 + v282;
float const v295 = v283 + v284;
float const v298 = v285 + v286;
float const v301 = v287 + v288;
float_4 const v302 = float_4{v292, v295, v298, v301};
*v262 = v302;
uint32_t const v239 = 2;
uint32_t const v240 = v234 * v239;
*v263 = v240;
}
float_4 const v241 = *v262;
*v218 = v241;
float_4 const v168 = *v218;
float const v247 = v168.i_0;
float const v249 = v62 + v247;
float const v250 = v168.i_1;
float const v252 = v249 + v250;
float const v253 = v168.i_2;
float const v255 = v252 + v253;
float const v256 = v168.i_3;
float const v258 = v255 + v256;
uint32_t const v170 = threadIdx.x;
uint64_t const v171 = uint64_t(v170);
for(uint64_t v173 = v171; v173 < v14; v173 += v305) {
uint64_t const v176 = v13 * v14;
uint64_t const v177 = v176 + v173;
int64_t const v178 = int64_t(v177);
int64_t const v179 = v12 * v210;
int64_t const v180 = v178 - v179;
int64_t const v182 = v180 + v179;
uint64_t const v183 = uint64_t(v182);
__nv_bfloat16 const* const v184 = &v1[v183];
__nv_bfloat16 const v185 = *v184;
float const v186 = float(v185);
__nv_bfloat16 const v187 = __nv_bfloat16(v186);
float const v188 = float(v187);
float const v191 = float(v210);
float const v192 = v258 / v191;
float const v193 = 9.999999740e-06;
float const v194 = v192 + v193;
float const v195 = rsqrt(v194);float const v196 = v188 * v195;
__nv_bfloat16 const v197 = __nv_bfloat16(v196);
float const v198 = float(v197);
__nv_bfloat16 const v199 = __nv_bfloat16(v198);
float const v200 = float(v199);
__nv_bfloat16 const v201 = __nv_bfloat16(v200);
__nv_bfloat16 * const v205 = &v2[v183];
*v205 = v201;
}
}
return;
}
