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
extern "C" __global__ void __launch_bounds__(512)  cubecl5_b3b1334d946336ae(__nv_bfloat16 const* const __restrict__ v1, float * const __restrict__ v2, info_st const* const __restrict__ v320) {
extern __shared__ __align__(4) char v326[];uint64_t const* const dynamic_meta_v322 = reinterpret_cast<uint64_t const*>(v320 + 1);
const info_st& info_v321 = *v320;int64_t const v324 = info_v321.scalars_i64[1];
int64_t const v325 = info_v321.scalars_i64[2];
uint64_t const v6 = blockIdx.x;
int64_t const v7 = int64_t(v6);
uint64_t const v8 = threadIdx.x;
int64_t const v9 = int64_t(v8);
int64_t const v10 = 0;
int64_t const v11 = 1;
int64_t const v15 = 512;
int64_t const v16 = 9;
int64_t const v17 = v16 * v325;
int64_t const v18 = 8;
int64_t const v19 = v18 + v324;
int64_t const v21 = v19 / v16;
int64_t const v22 = v19 % v16;
bool const v25 = v22 < v10;
int64_t const v26 = v21 - v11;
int64_t const v27 = v25 ? v26 : v21;
array<float, 16> * const v327 = reinterpret_cast<array<float, 16> *>(&v326[0]);
bool const v29 = v7 < v17;
if(v29) {
float v329_store;
float * v329 = &v329_store;float const v31 = 0.0;
*v329 = v31;
float v330_store;
float * v330 = &v330_store;*v330 = v31;
float v331_store;
float * v331 = &v331_store;*v331 = v31;
float v332_store;
float * v332 = &v332_store;*v332 = v31;
uint32_t const v408 = 0;
uint32_t const v41 = uint32_t(v27);
uint32_t const v407 = 2048;
for(uint32_t v43 = v408; v43 < v41; v43 += v407) {
int64_t const v44 = int64_t(v43);
int64_t const v45 = v44 + v9;
bool const v48 = v45 < v27;
float v333_store;
float * v333 = &v333_store;*v333 = v31;
if(v48) {
int64_t const v334 = v7 % v16;
int64_t const v336 = v334 ^ v16;
bool const v337 = v336 < v10;
bool const v338 = v334 != v10;
bool const v339 = v338 && v337;
int64_t const v340 = v339 ? v16 : v10;
int64_t const v341 = v334 + v340;
int64_t const v63 = v27 * v341;
int64_t const v64 = v45 + v63;
bool const v65 = v64 < v324;
float v342_store;
float * v342 = &v342_store;if(v65) {
int64_t const v68 = v7 / v16;
bool const v72 = v334 < v10;
int64_t const v73 = v68 - v11;
int64_t const v74 = v72 ? v73 : v68;
int64_t const v75 = v324 * v74;
int64_t const v76 = v45 + v75;
int64_t const v90 = v76 + v63;
uint64_t const v91 = uint64_t(v90);
__nv_bfloat16 const* const v92 = &v1[v91];
__nv_bfloat16 const v93 = *v92;
float const v94 = float(v93);
*v342 = v94;
}
 else {
*v342 = v31;
}
float const v96 = *v342;
*v333 = v96;
}
int64_t const v98 = v45 + v15;
bool const v99 = v98 < v27;
float v351_store;
float * v351 = &v351_store;*v351 = v31;
if(v99) {
int64_t const v352 = v7 % v16;
int64_t const v354 = v352 ^ v16;
bool const v355 = v354 < v10;
bool const v356 = v352 != v10;
bool const v357 = v356 && v355;
int64_t const v358 = v357 ? v16 : v10;
int64_t const v359 = v352 + v358;
int64_t const v114 = v27 * v359;
int64_t const v115 = v98 + v114;
bool const v116 = v115 < v324;
float v360_store;
float * v360 = &v360_store;if(v116) {
int64_t const v119 = v7 / v16;
bool const v123 = v352 < v10;
int64_t const v124 = v119 - v11;
int64_t const v125 = v123 ? v124 : v119;
int64_t const v126 = v324 * v125;
int64_t const v127 = v98 + v126;
int64_t const v141 = v127 + v114;
uint64_t const v142 = uint64_t(v141);
__nv_bfloat16 const* const v143 = &v1[v142];
__nv_bfloat16 const v144 = *v143;
float const v145 = float(v144);
*v360 = v145;
}
 else {
*v360 = v31;
}
float const v147 = *v360;
*v351 = v147;
}
int64_t const v148 = 1024;
int64_t const v149 = v45 + v148;
bool const v150 = v149 < v27;
float v369_store;
float * v369 = &v369_store;*v369 = v31;
if(v150) {
int64_t const v370 = v7 % v16;
int64_t const v372 = v370 ^ v16;
bool const v373 = v372 < v10;
bool const v374 = v370 != v10;
bool const v375 = v374 && v373;
int64_t const v376 = v375 ? v16 : v10;
int64_t const v377 = v370 + v376;
int64_t const v165 = v27 * v377;
int64_t const v166 = v149 + v165;
bool const v167 = v166 < v324;
float v378_store;
float * v378 = &v378_store;if(v167) {
int64_t const v170 = v7 / v16;
bool const v174 = v370 < v10;
int64_t const v175 = v170 - v11;
int64_t const v176 = v174 ? v175 : v170;
int64_t const v177 = v324 * v176;
int64_t const v178 = v149 + v177;
int64_t const v192 = v178 + v165;
uint64_t const v193 = uint64_t(v192);
__nv_bfloat16 const* const v194 = &v1[v193];
__nv_bfloat16 const v195 = *v194;
float const v196 = float(v195);
*v378 = v196;
}
 else {
*v378 = v31;
}
float const v198 = *v378;
*v369 = v198;
}
int64_t const v199 = 1536;
int64_t const v200 = v45 + v199;
bool const v201 = v200 < v27;
float v387_store;
float * v387 = &v387_store;*v387 = v31;
if(v201) {
int64_t const v388 = v7 % v16;
int64_t const v390 = v388 ^ v16;
bool const v391 = v390 < v10;
bool const v392 = v388 != v10;
bool const v393 = v392 && v391;
int64_t const v394 = v393 ? v16 : v10;
int64_t const v395 = v388 + v394;
int64_t const v216 = v27 * v395;
int64_t const v217 = v200 + v216;
bool const v218 = v217 < v324;
float v396_store;
float * v396 = &v396_store;if(v218) {
int64_t const v221 = v7 / v16;
bool const v225 = v388 < v10;
int64_t const v226 = v221 - v11;
int64_t const v227 = v225 ? v226 : v221;
int64_t const v228 = v324 * v227;
int64_t const v229 = v200 + v228;
int64_t const v243 = v229 + v216;
uint64_t const v244 = uint64_t(v243);
__nv_bfloat16 const* const v245 = &v1[v244];
__nv_bfloat16 const v246 = *v245;
float const v247 = float(v246);
*v396 = v247;
}
 else {
*v396 = v31;
}
float const v249 = *v396;
*v387 = v249;
}
if(v48) {
float const v251 = *v333;
float const v252 = *v329;
float const v253 = v252 + v251;
*v329 = v253;
float const v254 = *v351;
float const v255 = *v330;
float const v256 = v255 + v254;
*v330 = v256;
float const v257 = *v369;
float const v258 = *v331;
float const v259 = v258 + v257;
*v331 = v259;
float const v260 = *v387;
float const v261 = *v332;
float const v262 = v261 + v260;
*v332 = v262;
}
}
float const v263 = *v329;
float const v264 = *v330;
float const v265 = *v331;
float const v266 = *v332;
float const v267 = v263 + v264;
float const v268 = v265 + v266;
float const v269 = v267 + v268;
int32_t const v270 = 16;
float const v271 = __shfl_xor_sync(__activemask(), v269, v270);
float const v272 = v269 + v271;
int32_t const v273 = 8;
float const v274 = __shfl_xor_sync(__activemask(), v272, v273);
float const v275 = v272 + v274;
int32_t const v276 = 4;
float const v277 = __shfl_xor_sync(__activemask(), v275, v276);
float const v278 = v275 + v277;
int32_t const v279 = 2;
float const v280 = __shfl_xor_sync(__activemask(), v278, v279);
float const v281 = v278 + v280;
int32_t const v282 = 1;
float const v283 = __shfl_xor_sync(__activemask(), v281, v282);
float const v284 = v281 + v283;
int64_t const v285 = 32;
int64_t const v286 = v9 / v285;
int64_t const v287 = v9 % v285;
bool const v288 = v287 == v10;
if(v288) {
uint64_t const v289 = uint64_t(v286);
float * const v290 = &(*v327)[v289];
*v290 = v284;
}
__syncthreads();
bool const v291 = v286 == v10;
if(v291) {
float v405_store;
float * v405 = &v405_store;*v405 = v31;
int64_t const v294 = 16;
bool const v295 = v287 < v294;
if(v295) {
uint64_t const v296 = uint64_t(v287);
float * const v297 = &(*v327)[v296];
float const v298 = *v297;
*v405 = v298;
}
float const v299 = *v405;
float const v301 = __shfl_xor_sync(__activemask(), v299, v273);
float const v302 = v299 + v301;
float const v304 = __shfl_xor_sync(__activemask(), v302, v276);
float const v305 = v302 + v304;
float const v307 = __shfl_xor_sync(__activemask(), v305, v279);
float const v308 = v305 + v307;
float const v310 = __shfl_xor_sync(__activemask(), v308, v282);
float const v311 = v308 + v310;
if(v288) {
uint64_t const v410 = 0;
float * const v313 = &(*v327)[v410];
*v313 = v311;
}
}
__syncthreads();
uint64_t const v406 = 0;
float * const v315 = &(*v327)[v406];
float const v316 = *v315;
bool const v317 = v9 == v10;
if(v317) {
uint64_t const v318 = uint64_t(v7);
float * const v319 = &v2[v318];
*v319 = v316;
}
}
return;
}
