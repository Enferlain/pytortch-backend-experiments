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
    int64_t scalars_i64[3];uint64_t static_meta[3];
};
extern "C" __global__ void __launch_bounds__(256)  cubecl5_rms_cubek_pack4_observe_bfloat16(__nv_bfloat16 const* const __restrict__ v1, float * const __restrict__ v2, __nv_bfloat16 * const __restrict__ v3, info_st const* const __restrict__ v293) {
uint64_t const* const dynamic_meta_v295 = reinterpret_cast<uint64_t const*>(v293 + 1);
const info_st& info_v294 = *v293;int64_t const v297 = info_v294.scalars_i64[1];
int64_t const v298 = info_v294.scalars_i64[2];
uint64_t const v7 = blockIdx.x;
int64_t const v8 = int64_t(v7);
uint64_t const v9 = threadIdx.y;
int64_t const v10 = int64_t(v9);
int64_t const v11 = 8;
int64_t const v12 = v8 * v11;
int64_t const v13 = v12 + v10;
uint64_t const v14 = uint64_t(v13);
uint64_t const v15 = uint64_t(v297);
uint64_t const v16 = uint64_t(v298);
bool const v17 = v14 < v16;
if(v17) {
uint64_t const v19 = 0;
uint64_t v300_store;
uint64_t * v300 = &v300_store;*v300 = v19;
uint64_t const v23 = 2;
uint64_t const v24 = 1;
for(uint64_t v25 = v19; v25 < v23; v25 += v24) {
int64_t const v28 = int64_t(v25);
int64_t const v29 = 0;
bool const v30 = v28 == v29;
int64_t const v31 = 1;
int64_t const v40 = v30 ? v298 : v31;
uint64_t const v41 = uint64_t(v40);
uint64_t const v42 = v14 % v41;
int64_t const v47 = v30 ? v297 : v31;
uint64_t const v48 = uint64_t(v47);
uint64_t const v49 = v42 * v48;
uint64_t const v50 = *v300;
uint64_t const v51 = v50 + v49;
*v300 = v51;
}
uint64_t const v52 = *v300;
uint64_t const v53 = 4;
uint64_t const v54 = v52 / v53;
*v300 = v54;
uint64_t const v62 = v15 / v53;
float const v63 = 0.0;
float_4 const v64 = float_4{v63, v63, v63, v63};
int64_t const v69 = 4;
float_4 v305_store;
float_4 * v305 = &v305_store;*v305 = v64;
uint64_t const v394 = 32;
uint64_t const v79 = v62 + v394;
uint64_t const v81 = v79 - v24;
uint64_t const v82 = v81 / v394;
for(uint64_t v85 = v19; v85 < v82; v85 += v24) {
uint64_t const v89 = v85 * v394;
uint32_t const v90 = threadIdx.x;
uint64_t const v91 = uint64_t(v90);
uint64_t const v92 = v89 + v91;
uint64_t const v93 = *v300;
uint64_t const v94 = v92 + v93;
bool const v95 = v92 < v62;
float_4 v308_store;
float_4 * v308 = &v308_store;if(v95) {
int64_t const v96 = int64_t(v94);
int64_t const v98 = v96 * v69;
uint64_t const v99 = uint64_t(v98);
int64_t const v100 = int64_t(v99);
int64_t const v101 = v13 * v297;
int64_t const v102 = v100 - v101;
int64_t const v104 = v102 + v101;
uint64_t const v105 = uint64_t(v104);
__nv_bfloat16 const* const v106 = &v1[v105];
__nv_bfloat16_4 const* const v107 = reinterpret_cast<__nv_bfloat16_4 const*>(v106);
__nv_bfloat16_4 const v108 = *v107;
__nv_bfloat16 const v109 = v108.i_0;
float const v116 = float(v109);
__nv_bfloat16 const v117 = __nv_bfloat16(v116);
float const v118 = float(v117);
float const v120 = v118 * v118;
__nv_bfloat16 const v121 = v108.i_1;
float const v128 = float(v121);
__nv_bfloat16 const v129 = __nv_bfloat16(v128);
float const v130 = float(v129);
float const v132 = v130 * v130;
__nv_bfloat16 const v133 = v108.i_2;
float const v140 = float(v133);
__nv_bfloat16 const v141 = __nv_bfloat16(v140);
float const v142 = float(v141);
float const v144 = v142 * v142;
__nv_bfloat16 const v145 = v108.i_3;
float const v152 = float(v145);
__nv_bfloat16 const v153 = __nv_bfloat16(v152);
float const v154 = float(v153);
float const v156 = v154 * v154;
float_4 const v157 = float_4{v120, v132, v144, v156};
*v308 = v157;
}
 else {
*v308 = v64;
}
uint64_t const v167 = v53 * v15;
uint64_t const v169 = v167 + v394;
uint64_t const v170 = v14 * v169;
uint64_t const v172 = v394 * v85;
uint64_t const v175 = v172 + v91;
uint64_t const v177 = v175 * v53;
uint64_t const v178 = v170 + v177;
float_4 const v181 = *v308;
float const v182 = v181.i_0;
float * const v183 = &v2[v178];
*v183 = v182;
uint64_t const v198 = v178 + v24;
float const v200 = v181.i_1;
float * const v201 = &v2[v198];
*v201 = v200;
uint64_t const v216 = v178 + v23;
float const v218 = v181.i_2;
float * const v219 = &v2[v216];
*v219 = v218;
uint64_t const v233 = 3;
uint64_t const v234 = v178 + v233;
float const v236 = v181.i_3;
float * const v237 = &v2[v234];
*v237 = v236;
float_4 const v239 = *v305;
float const v357 = v239.i_0;
float const v359 = v357 + v182;
float const v360 = v239.i_1;
float const v362 = v360 + v200;
float const v363 = v239.i_2;
float const v365 = v363 + v218;
float const v366 = v239.i_3;
float const v368 = v366 + v236;
float_4 const v369 = float_4{v359, v362, v365, v368};
*v305 = v369;
}
float_4 const v241 = *v305;
uint32_t const v311 = warpSize;
uint32_t const v393 = 256;
uint32_t const v313 = min(v311, v393);
float_4 v351_store;
float_4 * v351 = &v351_store;*v351 = v241;
uint32_t const v315 = 1;
uint32_t v352_store;
uint32_t * v352 = &v352_store;*v352 = v315;
while([&]{
uint32_t const v318 = *v352;
bool const v319 = v318 < v313;
return v319;
}()) {
float_4 const v322 = *v351;
uint32_t const v323 = *v352;
float const v370 = v322.i_0;
float const v371 = __shfl_xor_sync(__activemask(), v370, v323);
float const v372 = v322.i_1;
float const v373 = __shfl_xor_sync(__activemask(), v372, v323);
float const v374 = v322.i_2;
float const v375 = __shfl_xor_sync(__activemask(), v374, v323);
float const v376 = v322.i_3;
float const v377 = __shfl_xor_sync(__activemask(), v376, v323);
float const v381 = v370 + v371;
float const v384 = v372 + v373;
float const v387 = v374 + v375;
float const v390 = v376 + v377;
float_4 const v391 = float_4{v381, v384, v387, v390};
*v351 = v391;
uint32_t const v328 = 2;
uint32_t const v329 = v323 * v328;
*v352 = v329;
}
float_4 const v330 = *v351;
*v305 = v330;
float_4 const v243 = *v305;
float const v336 = v243.i_0;
float const v338 = v63 + v336;
float const v339 = v243.i_1;
float const v341 = v338 + v339;
float const v342 = v243.i_2;
float const v344 = v341 + v342;
float const v345 = v243.i_3;
float const v347 = v344 + v345;
uint64_t const v246 = v53 * v15;
uint64_t const v248 = v246 + v394;
uint64_t const v249 = v14 * v248;
uint64_t const v252 = v249 + v246;
uint32_t const v253 = threadIdx.x;
uint64_t const v254 = uint64_t(v253);
uint64_t const v255 = v252 + v254;
float * const v256 = &v2[v255];
*v256 = v347;
for(uint64_t v260 = v254; v260 < v15; v260 += v394) {
uint64_t const v263 = v14 * v15;
uint64_t const v264 = v263 + v260;
int64_t const v265 = int64_t(v264);
int64_t const v266 = v13 * v297;
int64_t const v267 = v265 - v266;
int64_t const v269 = v267 + v266;
uint64_t const v270 = uint64_t(v269);
__nv_bfloat16 const* const v271 = &v1[v270];
__nv_bfloat16 const v272 = *v271;
float const v273 = float(v272);
__nv_bfloat16 const v274 = __nv_bfloat16(v273);
float const v275 = float(v274);
float const v278 = float(v297);
float const v279 = v347 / v278;
float const v280 = 9.999999740e-06;
float const v281 = v279 + v280;
float const v282 = rsqrt(v281);float const v283 = v275 * v282;
__nv_bfloat16 const v284 = __nv_bfloat16(v283);
float const v285 = float(v284);
__nv_bfloat16 const v286 = __nv_bfloat16(v285);
float const v287 = float(v286);
__nv_bfloat16 const v288 = __nv_bfloat16(v287);
__nv_bfloat16 * const v292 = &v3[v270];
*v292 = v288;
}
}
return;
}
