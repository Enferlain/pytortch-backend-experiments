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
extern "C" __global__ void __launch_bounds__(512)  cubecl5_d7c40e98ba153ce6(__half const* const __restrict__ v1, float * const __restrict__ v2, info_st const* const __restrict__ v312) {
extern __shared__ __align__(4) char v318[];uint64_t const* const dynamic_meta_v314 = reinterpret_cast<uint64_t const*>(v312 + 1);
const info_st& info_v313 = *v312;int64_t const v316 = info_v313.scalars_i64[1];
int64_t const v317 = info_v313.scalars_i64[2];
uint64_t const v6 = blockIdx.x;
int64_t const v7 = int64_t(v6);
uint64_t const v8 = threadIdx.x;
int64_t const v9 = int64_t(v8);
int64_t const v10 = 0;
int64_t const v11 = 1;
int64_t const v15 = 512;
int64_t const v16 = 9;
int64_t const v17 = v16 * v317;
int64_t const v18 = 8;
int64_t const v19 = v18 + v316;
int64_t const v21 = v19 / v16;
int64_t const v22 = v19 % v16;
bool const v25 = v22 < v10;
int64_t const v26 = v21 - v11;
int64_t const v27 = v25 ? v26 : v21;
array<float, 16> * const v319 = reinterpret_cast<array<float, 16> *>(&v318[0]);
bool const v29 = v7 < v17;
if(v29) {
float v321_store;
float * v321 = &v321_store;float const v31 = 0.0;
*v321 = v31;
float v322_store;
float * v322 = &v322_store;*v322 = v31;
float v323_store;
float * v323 = &v323_store;*v323 = v31;
float v324_store;
float * v324 = &v324_store;*v324 = v31;
int64_t const v399 = 2048;
for(int64_t v40 = v9; v40 < v27; v40 += v399) {
bool const v43 = v40 < v27;
float v325_store;
float * v325 = &v325_store;*v325 = v31;
if(v43) {
int64_t const v326 = v7 % v16;
int64_t const v328 = v326 ^ v16;
bool const v329 = v328 < v10;
bool const v330 = v326 != v10;
bool const v331 = v330 && v329;
int64_t const v332 = v331 ? v16 : v10;
int64_t const v333 = v326 + v332;
int64_t const v58 = v27 * v333;
int64_t const v59 = v40 + v58;
bool const v60 = v59 < v316;
float v334_store;
float * v334 = &v334_store;if(v60) {
int64_t const v63 = v7 / v16;
bool const v67 = v326 < v10;
int64_t const v68 = v63 - v11;
int64_t const v69 = v67 ? v68 : v63;
int64_t const v70 = v316 * v69;
int64_t const v71 = v40 + v70;
int64_t const v85 = v71 + v58;
uint64_t const v86 = uint64_t(v85);
__half const* const v87 = &v1[v86];
__half const v88 = *v87;
float const v89 = float(v88);
*v334 = v89;
}
 else {
*v334 = v31;
}
float const v91 = *v334;
*v325 = v91;
}
int64_t const v93 = v40 + v15;
bool const v94 = v93 < v27;
float v343_store;
float * v343 = &v343_store;*v343 = v31;
if(v94) {
int64_t const v344 = v7 % v16;
int64_t const v346 = v344 ^ v16;
bool const v347 = v346 < v10;
bool const v348 = v344 != v10;
bool const v349 = v348 && v347;
int64_t const v350 = v349 ? v16 : v10;
int64_t const v351 = v344 + v350;
int64_t const v109 = v27 * v351;
int64_t const v110 = v93 + v109;
bool const v111 = v110 < v316;
float v352_store;
float * v352 = &v352_store;if(v111) {
int64_t const v114 = v7 / v16;
bool const v118 = v344 < v10;
int64_t const v119 = v114 - v11;
int64_t const v120 = v118 ? v119 : v114;
int64_t const v121 = v316 * v120;
int64_t const v122 = v93 + v121;
int64_t const v136 = v122 + v109;
uint64_t const v137 = uint64_t(v136);
__half const* const v138 = &v1[v137];
__half const v139 = *v138;
float const v140 = float(v139);
*v352 = v140;
}
 else {
*v352 = v31;
}
float const v142 = *v352;
*v343 = v142;
}
int64_t const v143 = 1024;
int64_t const v144 = v40 + v143;
bool const v145 = v144 < v27;
float v361_store;
float * v361 = &v361_store;*v361 = v31;
if(v145) {
int64_t const v362 = v7 % v16;
int64_t const v364 = v362 ^ v16;
bool const v365 = v364 < v10;
bool const v366 = v362 != v10;
bool const v367 = v366 && v365;
int64_t const v368 = v367 ? v16 : v10;
int64_t const v369 = v362 + v368;
int64_t const v160 = v27 * v369;
int64_t const v161 = v144 + v160;
bool const v162 = v161 < v316;
float v370_store;
float * v370 = &v370_store;if(v162) {
int64_t const v165 = v7 / v16;
bool const v169 = v362 < v10;
int64_t const v170 = v165 - v11;
int64_t const v171 = v169 ? v170 : v165;
int64_t const v172 = v316 * v171;
int64_t const v173 = v144 + v172;
int64_t const v187 = v173 + v160;
uint64_t const v188 = uint64_t(v187);
__half const* const v189 = &v1[v188];
__half const v190 = *v189;
float const v191 = float(v190);
*v370 = v191;
}
 else {
*v370 = v31;
}
float const v193 = *v370;
*v361 = v193;
}
int64_t const v194 = 1536;
int64_t const v195 = v40 + v194;
bool const v196 = v195 < v27;
float v379_store;
float * v379 = &v379_store;*v379 = v31;
if(v196) {
int64_t const v380 = v7 % v16;
int64_t const v382 = v380 ^ v16;
bool const v383 = v382 < v10;
bool const v384 = v380 != v10;
bool const v385 = v384 && v383;
int64_t const v386 = v385 ? v16 : v10;
int64_t const v387 = v380 + v386;
int64_t const v211 = v27 * v387;
int64_t const v212 = v195 + v211;
bool const v213 = v212 < v316;
float v388_store;
float * v388 = &v388_store;if(v213) {
int64_t const v216 = v7 / v16;
bool const v220 = v380 < v10;
int64_t const v221 = v216 - v11;
int64_t const v222 = v220 ? v221 : v216;
int64_t const v223 = v316 * v222;
int64_t const v224 = v195 + v223;
int64_t const v238 = v224 + v211;
uint64_t const v239 = uint64_t(v238);
__half const* const v240 = &v1[v239];
__half const v241 = *v240;
float const v242 = float(v241);
*v388 = v242;
}
 else {
*v388 = v31;
}
float const v244 = *v388;
*v379 = v244;
}
float const v245 = *v325;
float const v246 = *v321;
float const v247 = v246 + v245;
*v321 = v247;
float const v248 = *v343;
float const v249 = *v322;
float const v250 = v249 + v248;
*v322 = v250;
float const v251 = *v361;
float const v252 = *v323;
float const v253 = v252 + v251;
*v323 = v253;
float const v254 = *v379;
float const v255 = *v324;
float const v256 = v255 + v254;
*v324 = v256;
}
float const v257 = *v321;
float const v258 = *v322;
float const v259 = *v323;
float const v260 = *v324;
float const v261 = v257 + v258;
float const v262 = v259 + v260;
float const v263 = v261 + v262;
float v397_store;
float * v397 = &v397_store;*v397 = v31;
int32_t const v266 = 16;
float const v267 = __shfl_xor_sync(__activemask(), v263, v266);
float const v268 = v263 + v267;
int32_t const v269 = 8;
float const v270 = __shfl_xor_sync(__activemask(), v268, v269);
float const v271 = v268 + v270;
int32_t const v272 = 4;
float const v273 = __shfl_xor_sync(__activemask(), v271, v272);
float const v274 = v271 + v273;
int32_t const v275 = 2;
float const v276 = __shfl_xor_sync(__activemask(), v274, v275);
float const v277 = v274 + v276;
int32_t const v278 = 1;
float const v279 = __shfl_xor_sync(__activemask(), v277, v278);
float const v280 = v277 + v279;
int64_t const v281 = 32;
int64_t const v282 = v9 / v281;
int64_t const v283 = v9 % v281;
bool const v284 = v283 == v10;
if(v284) {
uint64_t const v285 = uint64_t(v282);
float * const v286 = &(*v319)[v285];
*v286 = v280;
}
__syncthreads();
bool const v287 = v282 == v10;
if(v287) {
float v398_store;
float * v398 = &v398_store;*v398 = v31;
int64_t const v290 = 16;
bool const v291 = v283 < v290;
if(v291) {
uint64_t const v292 = uint64_t(v283);
float * const v293 = &(*v319)[v292];
float const v294 = *v293;
*v398 = v294;
}
float const v295 = *v398;
float const v297 = __shfl_xor_sync(__activemask(), v295, v269);
float const v298 = v295 + v297;
float const v300 = __shfl_xor_sync(__activemask(), v298, v272);
float const v301 = v298 + v300;
float const v303 = __shfl_xor_sync(__activemask(), v301, v275);
float const v304 = v301 + v303;
float const v306 = __shfl_xor_sync(__activemask(), v304, v278);
float const v307 = v304 + v306;
*v397 = v307;
}
float const v308 = *v397;
bool const v309 = v9 == v10;
if(v309) {
uint64_t const v310 = uint64_t(v7);
float * const v311 = &v2[v310];
*v311 = v308;
}
}
return;
}
