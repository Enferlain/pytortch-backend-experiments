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
extern "C" __global__ void __launch_bounds__(512)  cubecl5_d7c40e98ba153ce6(__half const* const __restrict__ v1, float * const __restrict__ v2, info_st const* const __restrict__ v314) {
extern __shared__ __align__(4) char v320[];uint64_t const* const dynamic_meta_v316 = reinterpret_cast<uint64_t const*>(v314 + 1);
const info_st& info_v315 = *v314;int64_t const v318 = info_v315.scalars_i64[1];
int64_t const v319 = info_v315.scalars_i64[2];
uint64_t const v6 = blockIdx.x;
int64_t const v7 = int64_t(v6);
uint64_t const v8 = threadIdx.x;
int64_t const v9 = int64_t(v8);
int64_t const v10 = 0;
int64_t const v11 = 1;
int64_t const v15 = 512;
int64_t const v16 = 9;
int64_t const v17 = v16 * v319;
int64_t const v18 = 8;
int64_t const v19 = v18 + v318;
int64_t const v21 = v19 / v16;
int64_t const v22 = v19 % v16;
bool const v25 = v22 < v10;
int64_t const v26 = v21 - v11;
int64_t const v27 = v25 ? v26 : v21;
array<float, 16> * const v321 = reinterpret_cast<array<float, 16> *>(&v320[0]);
bool const v29 = v7 < v17;
if(v29) {
float v323_store;
float * v323 = &v323_store;float const v31 = 0.0;
*v323 = v31;
float v324_store;
float * v324 = &v324_store;*v324 = v31;
float v325_store;
float * v325 = &v325_store;*v325 = v31;
float v326_store;
float * v326 = &v326_store;*v326 = v31;
int64_t const v401 = 2048;
for(int64_t v40 = v9; v40 < v27; v40 += v401) {
bool const v43 = v40 < v27;
float v327_store;
float * v327 = &v327_store;*v327 = v31;
if(v43) {
int64_t const v328 = v7 % v16;
int64_t const v330 = v328 ^ v16;
bool const v331 = v330 < v10;
bool const v332 = v328 != v10;
bool const v333 = v332 && v331;
int64_t const v334 = v333 ? v16 : v10;
int64_t const v335 = v328 + v334;
int64_t const v58 = v27 * v335;
int64_t const v59 = v40 + v58;
bool const v60 = v59 < v318;
float v336_store;
float * v336 = &v336_store;if(v60) {
int64_t const v63 = v7 / v16;
bool const v67 = v328 < v10;
int64_t const v68 = v63 - v11;
int64_t const v69 = v67 ? v68 : v63;
int64_t const v70 = v318 * v69;
int64_t const v71 = v40 + v70;
int64_t const v85 = v71 + v58;
uint64_t const v86 = uint64_t(v85);
__half const* const v87 = &v1[v86];
__half const v88 = *v87;
float const v89 = float(v88);
*v336 = v89;
}
 else {
*v336 = v31;
}
float const v91 = *v336;
*v327 = v91;
}
int64_t const v93 = v40 + v15;
bool const v94 = v93 < v27;
float v345_store;
float * v345 = &v345_store;*v345 = v31;
if(v94) {
int64_t const v346 = v7 % v16;
int64_t const v348 = v346 ^ v16;
bool const v349 = v348 < v10;
bool const v350 = v346 != v10;
bool const v351 = v350 && v349;
int64_t const v352 = v351 ? v16 : v10;
int64_t const v353 = v346 + v352;
int64_t const v109 = v27 * v353;
int64_t const v110 = v93 + v109;
bool const v111 = v110 < v318;
float v354_store;
float * v354 = &v354_store;if(v111) {
int64_t const v114 = v7 / v16;
bool const v118 = v346 < v10;
int64_t const v119 = v114 - v11;
int64_t const v120 = v118 ? v119 : v114;
int64_t const v121 = v318 * v120;
int64_t const v122 = v93 + v121;
int64_t const v136 = v122 + v109;
uint64_t const v137 = uint64_t(v136);
__half const* const v138 = &v1[v137];
__half const v139 = *v138;
float const v140 = float(v139);
*v354 = v140;
}
 else {
*v354 = v31;
}
float const v142 = *v354;
*v345 = v142;
}
int64_t const v143 = 1024;
int64_t const v144 = v40 + v143;
bool const v145 = v144 < v27;
float v363_store;
float * v363 = &v363_store;*v363 = v31;
if(v145) {
int64_t const v364 = v7 % v16;
int64_t const v366 = v364 ^ v16;
bool const v367 = v366 < v10;
bool const v368 = v364 != v10;
bool const v369 = v368 && v367;
int64_t const v370 = v369 ? v16 : v10;
int64_t const v371 = v364 + v370;
int64_t const v160 = v27 * v371;
int64_t const v161 = v144 + v160;
bool const v162 = v161 < v318;
float v372_store;
float * v372 = &v372_store;if(v162) {
int64_t const v165 = v7 / v16;
bool const v169 = v364 < v10;
int64_t const v170 = v165 - v11;
int64_t const v171 = v169 ? v170 : v165;
int64_t const v172 = v318 * v171;
int64_t const v173 = v144 + v172;
int64_t const v187 = v173 + v160;
uint64_t const v188 = uint64_t(v187);
__half const* const v189 = &v1[v188];
__half const v190 = *v189;
float const v191 = float(v190);
*v372 = v191;
}
 else {
*v372 = v31;
}
float const v193 = *v372;
*v363 = v193;
}
int64_t const v194 = 1536;
int64_t const v195 = v40 + v194;
bool const v196 = v195 < v27;
float v381_store;
float * v381 = &v381_store;*v381 = v31;
if(v196) {
int64_t const v382 = v7 % v16;
int64_t const v384 = v382 ^ v16;
bool const v385 = v384 < v10;
bool const v386 = v382 != v10;
bool const v387 = v386 && v385;
int64_t const v388 = v387 ? v16 : v10;
int64_t const v389 = v382 + v388;
int64_t const v211 = v27 * v389;
int64_t const v212 = v195 + v211;
bool const v213 = v212 < v318;
float v390_store;
float * v390 = &v390_store;if(v213) {
int64_t const v216 = v7 / v16;
bool const v220 = v382 < v10;
int64_t const v221 = v216 - v11;
int64_t const v222 = v220 ? v221 : v216;
int64_t const v223 = v318 * v222;
int64_t const v224 = v195 + v223;
int64_t const v238 = v224 + v211;
uint64_t const v239 = uint64_t(v238);
__half const* const v240 = &v1[v239];
__half const v241 = *v240;
float const v242 = float(v241);
*v390 = v242;
}
 else {
*v390 = v31;
}
float const v244 = *v390;
*v381 = v244;
}
float const v245 = *v327;
float const v246 = *v323;
float const v247 = v246 + v245;
*v323 = v247;
float const v248 = *v345;
float const v249 = *v324;
float const v250 = v249 + v248;
*v324 = v250;
float const v251 = *v363;
float const v252 = *v325;
float const v253 = v252 + v251;
*v325 = v253;
float const v254 = *v381;
float const v255 = *v326;
float const v256 = v255 + v254;
*v326 = v256;
}
float const v257 = *v323;
float const v258 = *v324;
float const v259 = *v325;
float const v260 = *v326;
float const v261 = v257 + v258;
float const v262 = v259 + v260;
float const v263 = v261 + v262;
int32_t const v264 = 16;
float const v265 = __shfl_xor_sync(__activemask(), v263, v264);
float const v266 = v263 + v265;
int32_t const v267 = 8;
float const v268 = __shfl_xor_sync(__activemask(), v266, v267);
float const v269 = v266 + v268;
int32_t const v270 = 4;
float const v271 = __shfl_xor_sync(__activemask(), v269, v270);
float const v272 = v269 + v271;
int32_t const v273 = 2;
float const v274 = __shfl_xor_sync(__activemask(), v272, v273);
float const v275 = v272 + v274;
int32_t const v276 = 1;
float const v277 = __shfl_xor_sync(__activemask(), v275, v276);
float const v278 = v275 + v277;
int64_t const v279 = 32;
int64_t const v280 = v9 / v279;
int64_t const v281 = v9 % v279;
bool const v282 = v281 == v10;
if(v282) {
uint64_t const v283 = uint64_t(v280);
float * const v284 = &(*v321)[v283];
*v284 = v278;
}
__syncthreads();
bool const v285 = v280 == v10;
if(v285) {
float v399_store;
float * v399 = &v399_store;*v399 = v31;
int64_t const v288 = 16;
bool const v289 = v281 < v288;
if(v289) {
uint64_t const v290 = uint64_t(v281);
float * const v291 = &(*v321)[v290];
float const v292 = *v291;
*v399 = v292;
}
float const v293 = *v399;
float const v295 = __shfl_xor_sync(__activemask(), v293, v267);
float const v296 = v293 + v295;
float const v298 = __shfl_xor_sync(__activemask(), v296, v270);
float const v299 = v296 + v298;
float const v301 = __shfl_xor_sync(__activemask(), v299, v273);
float const v302 = v299 + v301;
float const v304 = __shfl_xor_sync(__activemask(), v302, v276);
float const v305 = v302 + v304;
if(v282) {
uint64_t const v402 = 0;
float * const v307 = &(*v321)[v402];
*v307 = v305;
}
}
__syncthreads();
uint64_t const v400 = 0;
float * const v309 = &(*v321)[v400];
float const v310 = *v309;
bool const v311 = v9 == v10;
if(v311) {
uint64_t const v312 = uint64_t(v7);
float * const v313 = &v2[v312];
*v313 = v310;
}
}
return;
}
