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
extern "C" __global__ void __launch_bounds__(512)  cubecl5_b3b1334d946336ae(__nv_bfloat16 const* const __restrict__ v1, float * const __restrict__ v2, info_st const* const __restrict__ v318) {
extern __shared__ __align__(4) char v324[];uint64_t const* const dynamic_meta_v320 = reinterpret_cast<uint64_t const*>(v318 + 1);
const info_st& info_v319 = *v318;int64_t const v322 = info_v319.scalars_i64[1];
int64_t const v323 = info_v319.scalars_i64[2];
uint64_t const v6 = blockIdx.x;
int64_t const v7 = int64_t(v6);
uint64_t const v8 = threadIdx.x;
int64_t const v9 = int64_t(v8);
int64_t const v10 = 0;
int64_t const v11 = 1;
int64_t const v15 = 512;
int64_t const v16 = 9;
int64_t const v17 = v16 * v323;
int64_t const v18 = 8;
int64_t const v19 = v18 + v322;
int64_t const v21 = v19 / v16;
int64_t const v22 = v19 % v16;
bool const v25 = v22 < v10;
int64_t const v26 = v21 - v11;
int64_t const v27 = v25 ? v26 : v21;
array<float, 16> * const v325 = reinterpret_cast<array<float, 16> *>(&v324[0]);
bool const v29 = v7 < v17;
if(v29) {
float v327_store;
float * v327 = &v327_store;float const v31 = 0.0;
*v327 = v31;
float v328_store;
float * v328 = &v328_store;*v328 = v31;
float v329_store;
float * v329 = &v329_store;*v329 = v31;
float v330_store;
float * v330 = &v330_store;*v330 = v31;
uint32_t const v40 = uint32_t(v9);
uint32_t const v41 = uint32_t(v27);
uint32_t const v405 = 2048;
for(uint32_t v43 = v40; v43 < v41; v43 += v405) {
int64_t const v44 = int64_t(v43);
bool const v47 = v44 < v27;
float v331_store;
float * v331 = &v331_store;*v331 = v31;
if(v47) {
int64_t const v332 = v7 % v16;
int64_t const v334 = v332 ^ v16;
bool const v335 = v334 < v10;
bool const v336 = v332 != v10;
bool const v337 = v336 && v335;
int64_t const v338 = v337 ? v16 : v10;
int64_t const v339 = v332 + v338;
int64_t const v62 = v27 * v339;
int64_t const v63 = v44 + v62;
bool const v64 = v63 < v322;
float v340_store;
float * v340 = &v340_store;if(v64) {
int64_t const v67 = v7 / v16;
bool const v71 = v332 < v10;
int64_t const v72 = v67 - v11;
int64_t const v73 = v71 ? v72 : v67;
int64_t const v74 = v322 * v73;
int64_t const v75 = v44 + v74;
int64_t const v89 = v75 + v62;
uint64_t const v90 = uint64_t(v89);
__nv_bfloat16 const* const v91 = &v1[v90];
__nv_bfloat16 const v92 = *v91;
float const v93 = float(v92);
*v340 = v93;
}
 else {
*v340 = v31;
}
float const v95 = *v340;
*v331 = v95;
}
int64_t const v97 = v44 + v15;
bool const v98 = v97 < v27;
float v349_store;
float * v349 = &v349_store;*v349 = v31;
if(v98) {
int64_t const v350 = v7 % v16;
int64_t const v352 = v350 ^ v16;
bool const v353 = v352 < v10;
bool const v354 = v350 != v10;
bool const v355 = v354 && v353;
int64_t const v356 = v355 ? v16 : v10;
int64_t const v357 = v350 + v356;
int64_t const v113 = v27 * v357;
int64_t const v114 = v97 + v113;
bool const v115 = v114 < v322;
float v358_store;
float * v358 = &v358_store;if(v115) {
int64_t const v118 = v7 / v16;
bool const v122 = v350 < v10;
int64_t const v123 = v118 - v11;
int64_t const v124 = v122 ? v123 : v118;
int64_t const v125 = v322 * v124;
int64_t const v126 = v97 + v125;
int64_t const v140 = v126 + v113;
uint64_t const v141 = uint64_t(v140);
__nv_bfloat16 const* const v142 = &v1[v141];
__nv_bfloat16 const v143 = *v142;
float const v144 = float(v143);
*v358 = v144;
}
 else {
*v358 = v31;
}
float const v146 = *v358;
*v349 = v146;
}
int64_t const v147 = 1024;
int64_t const v148 = v44 + v147;
bool const v149 = v148 < v27;
float v367_store;
float * v367 = &v367_store;*v367 = v31;
if(v149) {
int64_t const v368 = v7 % v16;
int64_t const v370 = v368 ^ v16;
bool const v371 = v370 < v10;
bool const v372 = v368 != v10;
bool const v373 = v372 && v371;
int64_t const v374 = v373 ? v16 : v10;
int64_t const v375 = v368 + v374;
int64_t const v164 = v27 * v375;
int64_t const v165 = v148 + v164;
bool const v166 = v165 < v322;
float v376_store;
float * v376 = &v376_store;if(v166) {
int64_t const v169 = v7 / v16;
bool const v173 = v368 < v10;
int64_t const v174 = v169 - v11;
int64_t const v175 = v173 ? v174 : v169;
int64_t const v176 = v322 * v175;
int64_t const v177 = v148 + v176;
int64_t const v191 = v177 + v164;
uint64_t const v192 = uint64_t(v191);
__nv_bfloat16 const* const v193 = &v1[v192];
__nv_bfloat16 const v194 = *v193;
float const v195 = float(v194);
*v376 = v195;
}
 else {
*v376 = v31;
}
float const v197 = *v376;
*v367 = v197;
}
int64_t const v198 = 1536;
int64_t const v199 = v44 + v198;
bool const v200 = v199 < v27;
float v385_store;
float * v385 = &v385_store;*v385 = v31;
if(v200) {
int64_t const v386 = v7 % v16;
int64_t const v388 = v386 ^ v16;
bool const v389 = v388 < v10;
bool const v390 = v386 != v10;
bool const v391 = v390 && v389;
int64_t const v392 = v391 ? v16 : v10;
int64_t const v393 = v386 + v392;
int64_t const v215 = v27 * v393;
int64_t const v216 = v199 + v215;
bool const v217 = v216 < v322;
float v394_store;
float * v394 = &v394_store;if(v217) {
int64_t const v220 = v7 / v16;
bool const v224 = v386 < v10;
int64_t const v225 = v220 - v11;
int64_t const v226 = v224 ? v225 : v220;
int64_t const v227 = v322 * v226;
int64_t const v228 = v199 + v227;
int64_t const v242 = v228 + v215;
uint64_t const v243 = uint64_t(v242);
__nv_bfloat16 const* const v244 = &v1[v243];
__nv_bfloat16 const v245 = *v244;
float const v246 = float(v245);
*v394 = v246;
}
 else {
*v394 = v31;
}
float const v248 = *v394;
*v385 = v248;
}
float const v249 = *v331;
float const v250 = *v327;
float const v251 = v250 + v249;
*v327 = v251;
float const v252 = *v349;
float const v253 = *v328;
float const v254 = v253 + v252;
*v328 = v254;
float const v255 = *v367;
float const v256 = *v329;
float const v257 = v256 + v255;
*v329 = v257;
float const v258 = *v385;
float const v259 = *v330;
float const v260 = v259 + v258;
*v330 = v260;
}
float const v261 = *v327;
float const v262 = *v328;
float const v263 = *v329;
float const v264 = *v330;
float const v265 = v261 + v262;
float const v266 = v263 + v264;
float const v267 = v265 + v266;
int32_t const v268 = 16;
float const v269 = __shfl_xor_sync(__activemask(), v267, v268);
float const v270 = v267 + v269;
int32_t const v271 = 8;
float const v272 = __shfl_xor_sync(__activemask(), v270, v271);
float const v273 = v270 + v272;
int32_t const v274 = 4;
float const v275 = __shfl_xor_sync(__activemask(), v273, v274);
float const v276 = v273 + v275;
int32_t const v277 = 2;
float const v278 = __shfl_xor_sync(__activemask(), v276, v277);
float const v279 = v276 + v278;
int32_t const v280 = 1;
float const v281 = __shfl_xor_sync(__activemask(), v279, v280);
float const v282 = v279 + v281;
int64_t const v283 = 32;
int64_t const v284 = v9 / v283;
int64_t const v285 = v9 % v283;
bool const v286 = v285 == v10;
if(v286) {
uint64_t const v287 = uint64_t(v284);
float * const v288 = &(*v325)[v287];
*v288 = v282;
}
__syncthreads();
bool const v289 = v284 == v10;
if(v289) {
float v403_store;
float * v403 = &v403_store;*v403 = v31;
int64_t const v292 = 16;
bool const v293 = v285 < v292;
if(v293) {
uint64_t const v294 = uint64_t(v285);
float * const v295 = &(*v325)[v294];
float const v296 = *v295;
*v403 = v296;
}
float const v297 = *v403;
float const v299 = __shfl_xor_sync(__activemask(), v297, v271);
float const v300 = v297 + v299;
float const v302 = __shfl_xor_sync(__activemask(), v300, v274);
float const v303 = v300 + v302;
float const v305 = __shfl_xor_sync(__activemask(), v303, v277);
float const v306 = v303 + v305;
float const v308 = __shfl_xor_sync(__activemask(), v306, v280);
float const v309 = v306 + v308;
if(v286) {
uint64_t const v407 = 0;
float * const v311 = &(*v325)[v407];
*v311 = v309;
}
}
__syncthreads();
uint64_t const v404 = 0;
float * const v313 = &(*v325)[v404];
float const v314 = *v313;
bool const v315 = v9 == v10;
if(v315) {
uint64_t const v316 = uint64_t(v7);
float * const v317 = &v2[v316];
*v317 = v314;
}
}
return;
}
