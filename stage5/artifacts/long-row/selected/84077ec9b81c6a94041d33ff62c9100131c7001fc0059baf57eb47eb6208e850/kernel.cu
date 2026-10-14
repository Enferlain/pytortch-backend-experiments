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
extern "C" __global__ void __launch_bounds__(256)  cubecl5_2a5964308db0e95f(__half const* const __restrict__ v1, float * const __restrict__ v2, info_st const* const __restrict__ v277) {
extern __shared__ __align__(4) char v283[];uint64_t const* const dynamic_meta_v279 = reinterpret_cast<uint64_t const*>(v277 + 1);
const info_st& info_v278 = *v277;int64_t const v281 = info_v278.scalars_i64[1];
int64_t const v282 = info_v278.scalars_i64[2];
uint64_t const v6 = blockIdx.x;
int64_t const v7 = int64_t(v6);
uint64_t const v8 = threadIdx.x;
int64_t const v9 = int64_t(v8);
int64_t const v10 = 0;
int64_t const v11 = 1;
int64_t const v15 = 256;
int64_t const v16 = 9;
int64_t const v17 = v16 * v282;
int64_t const v18 = 8;
int64_t const v19 = v18 + v281;
int64_t const v21 = v19 / v16;
int64_t const v22 = v19 % v16;
bool const v25 = v22 < v10;
int64_t const v26 = v21 - v11;
int64_t const v27 = v25 ? v26 : v21;
array<float, 8> * const v284 = reinterpret_cast<array<float, 8> *>(&v283[0]);
bool const v29 = v7 < v17;
if(v29) {
float v286_store;
float * v286 = &v286_store;float const v31 = 0.0;
*v286 = v31;
float v287_store;
float * v287 = &v287_store;*v287 = v31;
float v288_store;
float * v288 = &v288_store;*v288 = v31;
float v289_store;
float * v289 = &v289_store;*v289 = v31;
int64_t const v408 = 1024;
for(int64_t v40 = v9; v40 < v27; v40 += v408) {
bool const v43 = v40 < v27;
if(v43) {
int64_t const v290 = v7 % v16;
int64_t const v292 = v290 ^ v16;
bool const v293 = v292 < v10;
bool const v294 = v290 != v10;
bool const v295 = v294 && v293;
int64_t const v296 = v295 ? v16 : v10;
int64_t const v297 = v290 + v296;
int64_t const v56 = v27 * v297;
int64_t const v57 = v40 + v56;
bool const v58 = v57 < v281;
float v298_store;
float * v298 = &v298_store;if(v58) {
int64_t const v61 = v7 / v16;
bool const v65 = v290 < v10;
int64_t const v66 = v61 - v11;
int64_t const v67 = v65 ? v66 : v61;
int64_t const v68 = v281 * v67;
int64_t const v69 = v40 + v68;
int64_t const v83 = v69 + v56;
uint64_t const v84 = uint64_t(v83);
__half const* const v85 = &v1[v84];
__half const v86 = *v85;
float const v87 = float(v86);
*v298 = v87;
}
 else {
*v298 = v31;
}
float const v89 = *v298;
float const v90 = *v286;
float const v91 = v90 + v89;
*v286 = v91;
}
int64_t const v93 = v40 + v15;
bool const v94 = v93 < v27;
if(v94) {
int64_t const v307 = v7 % v16;
int64_t const v309 = v307 ^ v16;
bool const v310 = v309 < v10;
bool const v311 = v307 != v10;
bool const v312 = v311 && v310;
int64_t const v313 = v312 ? v16 : v10;
int64_t const v314 = v307 + v313;
int64_t const v107 = v27 * v314;
int64_t const v108 = v93 + v107;
bool const v109 = v108 < v281;
float v315_store;
float * v315 = &v315_store;if(v109) {
int64_t const v112 = v7 / v16;
bool const v116 = v307 < v10;
int64_t const v117 = v112 - v11;
int64_t const v118 = v116 ? v117 : v112;
int64_t const v119 = v281 * v118;
int64_t const v120 = v93 + v119;
int64_t const v134 = v120 + v107;
uint64_t const v135 = uint64_t(v134);
__half const* const v136 = &v1[v135];
__half const v137 = *v136;
float const v138 = float(v137);
*v315 = v138;
}
 else {
*v315 = v31;
}
float const v140 = *v315;
float const v141 = *v287;
float const v142 = v141 + v140;
*v287 = v142;
}
int64_t const v143 = 512;
int64_t const v144 = v40 + v143;
bool const v145 = v144 < v27;
if(v145) {
int64_t const v324 = v7 % v16;
int64_t const v326 = v324 ^ v16;
bool const v327 = v326 < v10;
bool const v328 = v324 != v10;
bool const v329 = v328 && v327;
int64_t const v330 = v329 ? v16 : v10;
int64_t const v331 = v324 + v330;
int64_t const v158 = v27 * v331;
int64_t const v159 = v144 + v158;
bool const v160 = v159 < v281;
float v332_store;
float * v332 = &v332_store;if(v160) {
int64_t const v163 = v7 / v16;
bool const v167 = v324 < v10;
int64_t const v168 = v163 - v11;
int64_t const v169 = v167 ? v168 : v163;
int64_t const v170 = v281 * v169;
int64_t const v171 = v144 + v170;
int64_t const v185 = v171 + v158;
uint64_t const v186 = uint64_t(v185);
__half const* const v187 = &v1[v186];
__half const v188 = *v187;
float const v189 = float(v188);
*v332 = v189;
}
 else {
*v332 = v31;
}
float const v191 = *v332;
float const v192 = *v288;
float const v193 = v192 + v191;
*v288 = v193;
}
int64_t const v194 = 768;
int64_t const v195 = v40 + v194;
bool const v196 = v195 < v27;
if(v196) {
int64_t const v341 = v7 % v16;
int64_t const v343 = v341 ^ v16;
bool const v344 = v343 < v10;
bool const v345 = v341 != v10;
bool const v346 = v345 && v344;
int64_t const v347 = v346 ? v16 : v10;
int64_t const v348 = v341 + v347;
int64_t const v209 = v27 * v348;
int64_t const v210 = v195 + v209;
bool const v211 = v210 < v281;
float v349_store;
float * v349 = &v349_store;if(v211) {
int64_t const v214 = v7 / v16;
bool const v218 = v341 < v10;
int64_t const v219 = v214 - v11;
int64_t const v220 = v218 ? v219 : v214;
int64_t const v221 = v281 * v220;
int64_t const v222 = v195 + v221;
int64_t const v236 = v222 + v209;
uint64_t const v237 = uint64_t(v236);
__half const* const v238 = &v1[v237];
__half const v239 = *v238;
float const v240 = float(v239);
*v349 = v240;
}
 else {
*v349 = v31;
}
float const v242 = *v349;
float const v243 = *v289;
float const v244 = v243 + v242;
*v289 = v244;
}
}
float const v245 = *v286;
float const v246 = *v287;
float const v247 = *v288;
float const v248 = *v289;
float const v249 = v245 + v247;
float const v250 = v246 + v248;
float const v251 = v249 + v250;
uint32_t const v358 = warpSize;
uint32_t const v405 = 256;
uint32_t const v360 = min(v358, v405);
float v399_store;
float * v399 = &v399_store;*v399 = v251;
uint32_t const v362 = 1;
uint32_t v400_store;
uint32_t * v400 = &v400_store;*v400 = v362;
while([&]{
uint32_t const v365 = *v400;
bool const v366 = v365 < v360;
return v366;
}()) {
float const v369 = *v399;
uint32_t const v370 = *v400;
float const v371 = __shfl_xor_sync(__activemask(), v369, v370);
float const v373 = v369 + v371;
*v399 = v373;
uint32_t const v375 = 2;
uint32_t const v376 = v370 * v375;
*v400 = v376;
}
float const v377 = *v399;
int64_t const v253 = 32;
int64_t const v254 = v9 / v253;
int64_t const v255 = v9 % v253;
bool const v256 = v255 == v10;
if(v256) {
uint64_t const v257 = uint64_t(v254);
float * const v258 = &(*v284)[v257];
*v258 = v377;
}
__syncthreads();
bool const v259 = v254 == v10;
if(v259) {
float v378_store;
float * v378 = &v378_store;*v378 = v31;
bool const v263 = v255 < v18;
if(v263) {
uint64_t const v264 = uint64_t(v255);
float * const v265 = &(*v284)[v264];
float const v266 = *v265;
*v378 = v266;
}
float const v267 = *v378;
float v402_store;
float * v402 = &v402_store;*v402 = v267;
uint32_t v403_store;
uint32_t * v403 = &v403_store;*v403 = v362;
while([&]{
uint32_t const v386 = *v403;
bool const v387 = v386 < v360;
return v387;
}()) {
float const v390 = *v402;
uint32_t const v391 = *v403;
float const v392 = __shfl_xor_sync(__activemask(), v390, v391);
float const v394 = v390 + v392;
*v402 = v394;
uint32_t const v396 = 2;
uint32_t const v397 = v391 * v396;
*v403 = v397;
}
float const v398 = *v402;
if(v256) {
uint64_t const v409 = 0;
float * const v270 = &(*v284)[v409];
*v270 = v398;
}
}
__syncthreads();
uint64_t const v407 = 0;
float * const v272 = &(*v284)[v407];
float const v273 = *v272;
__syncthreads();
bool const v274 = v9 == v10;
if(v274) {
uint64_t const v275 = uint64_t(v7);
float * const v276 = &v2[v275];
*v276 = v273;
}
}
return;
}
