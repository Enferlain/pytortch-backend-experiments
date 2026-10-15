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
extern "C" __global__ void __launch_bounds__(256)  cubecl5_22cea85731132b83(__half const* const __restrict__ v1, float * const __restrict__ v2, info_st const* const __restrict__ v289) {
extern __shared__ __align__(4) char v295[];uint64_t const* const dynamic_meta_v291 = reinterpret_cast<uint64_t const*>(v289 + 1);
const info_st& info_v290 = *v289;int64_t const v293 = info_v290.scalars_i64[1];
int64_t const v294 = info_v290.scalars_i64[2];
uint64_t const v6 = blockIdx.x;
int64_t const v7 = int64_t(v6);
uint64_t const v8 = threadIdx.x;
int64_t const v9 = int64_t(v8);
int64_t const v10 = 0;
int64_t const v11 = 1;
int64_t const v15 = 256;
int64_t const v16 = 9;
int64_t const v17 = v16 * v294;
int64_t const v18 = 8;
int64_t const v19 = v18 + v293;
int64_t const v21 = v19 / v16;
int64_t const v22 = v19 % v16;
bool const v25 = v22 < v10;
int64_t const v26 = v21 - v11;
int64_t const v27 = v25 ? v26 : v21;
array<float, 8> * const v296 = reinterpret_cast<array<float, 8> *>(&v295[0]);
bool const v29 = v7 < v17;
if(v29) {
float v298_store;
float * v298 = &v298_store;float const v31 = 0.0;
*v298 = v31;
float v299_store;
float * v299 = &v299_store;*v299 = v31;
float v300_store;
float * v300 = &v300_store;*v300 = v31;
float v301_store;
float * v301 = &v301_store;*v301 = v31;
int64_t const v424 = 1024;
for(int64_t v40 = v9; v40 < v27; v40 += v424) {
bool const v43 = v40 < v27;
float v302_store;
float * v302 = &v302_store;*v302 = v31;
if(v43) {
int64_t const v303 = v7 % v16;
int64_t const v305 = v303 ^ v16;
bool const v306 = v305 < v10;
bool const v307 = v303 != v10;
bool const v308 = v307 && v306;
int64_t const v309 = v308 ? v16 : v10;
int64_t const v310 = v303 + v309;
int64_t const v58 = v27 * v310;
int64_t const v59 = v40 + v58;
bool const v60 = v59 < v293;
float v311_store;
float * v311 = &v311_store;if(v60) {
int64_t const v63 = v7 / v16;
bool const v67 = v303 < v10;
int64_t const v68 = v63 - v11;
int64_t const v69 = v67 ? v68 : v63;
int64_t const v70 = v293 * v69;
int64_t const v71 = v40 + v70;
int64_t const v85 = v71 + v58;
uint64_t const v86 = uint64_t(v85);
__half const* const v87 = &v1[v86];
__half const v88 = *v87;
float const v89 = float(v88);
*v311 = v89;
}
 else {
*v311 = v31;
}
float const v91 = *v311;
*v302 = v91;
}
int64_t const v93 = v40 + v15;
bool const v94 = v93 < v27;
float v320_store;
float * v320 = &v320_store;*v320 = v31;
if(v94) {
int64_t const v321 = v7 % v16;
int64_t const v323 = v321 ^ v16;
bool const v324 = v323 < v10;
bool const v325 = v321 != v10;
bool const v326 = v325 && v324;
int64_t const v327 = v326 ? v16 : v10;
int64_t const v328 = v321 + v327;
int64_t const v109 = v27 * v328;
int64_t const v110 = v93 + v109;
bool const v111 = v110 < v293;
float v329_store;
float * v329 = &v329_store;if(v111) {
int64_t const v114 = v7 / v16;
bool const v118 = v321 < v10;
int64_t const v119 = v114 - v11;
int64_t const v120 = v118 ? v119 : v114;
int64_t const v121 = v293 * v120;
int64_t const v122 = v93 + v121;
int64_t const v136 = v122 + v109;
uint64_t const v137 = uint64_t(v136);
__half const* const v138 = &v1[v137];
__half const v139 = *v138;
float const v140 = float(v139);
*v329 = v140;
}
 else {
*v329 = v31;
}
float const v142 = *v329;
*v320 = v142;
}
int64_t const v143 = 512;
int64_t const v144 = v40 + v143;
bool const v145 = v144 < v27;
float v338_store;
float * v338 = &v338_store;*v338 = v31;
if(v145) {
int64_t const v339 = v7 % v16;
int64_t const v341 = v339 ^ v16;
bool const v342 = v341 < v10;
bool const v343 = v339 != v10;
bool const v344 = v343 && v342;
int64_t const v345 = v344 ? v16 : v10;
int64_t const v346 = v339 + v345;
int64_t const v160 = v27 * v346;
int64_t const v161 = v144 + v160;
bool const v162 = v161 < v293;
float v347_store;
float * v347 = &v347_store;if(v162) {
int64_t const v165 = v7 / v16;
bool const v169 = v339 < v10;
int64_t const v170 = v165 - v11;
int64_t const v171 = v169 ? v170 : v165;
int64_t const v172 = v293 * v171;
int64_t const v173 = v144 + v172;
int64_t const v187 = v173 + v160;
uint64_t const v188 = uint64_t(v187);
__half const* const v189 = &v1[v188];
__half const v190 = *v189;
float const v191 = float(v190);
*v347 = v191;
}
 else {
*v347 = v31;
}
float const v193 = *v347;
*v338 = v193;
}
int64_t const v194 = 768;
int64_t const v195 = v40 + v194;
bool const v196 = v195 < v27;
float v356_store;
float * v356 = &v356_store;*v356 = v31;
if(v196) {
int64_t const v357 = v7 % v16;
int64_t const v359 = v357 ^ v16;
bool const v360 = v359 < v10;
bool const v361 = v357 != v10;
bool const v362 = v361 && v360;
int64_t const v363 = v362 ? v16 : v10;
int64_t const v364 = v357 + v363;
int64_t const v211 = v27 * v364;
int64_t const v212 = v195 + v211;
bool const v213 = v212 < v293;
float v365_store;
float * v365 = &v365_store;if(v213) {
int64_t const v216 = v7 / v16;
bool const v220 = v357 < v10;
int64_t const v221 = v216 - v11;
int64_t const v222 = v220 ? v221 : v216;
int64_t const v223 = v293 * v222;
int64_t const v224 = v195 + v223;
int64_t const v238 = v224 + v211;
uint64_t const v239 = uint64_t(v238);
__half const* const v240 = &v1[v239];
__half const v241 = *v240;
float const v242 = float(v241);
*v365 = v242;
}
 else {
*v365 = v31;
}
float const v244 = *v365;
*v356 = v244;
}
float const v245 = *v302;
float const v246 = *v298;
float const v247 = v246 + v245;
*v298 = v247;
float const v248 = *v320;
float const v249 = *v299;
float const v250 = v249 + v248;
*v299 = v250;
float const v251 = *v338;
float const v252 = *v300;
float const v253 = v252 + v251;
*v300 = v253;
float const v254 = *v356;
float const v255 = *v301;
float const v256 = v255 + v254;
*v301 = v256;
}
float const v257 = *v298;
float const v258 = *v299;
float const v259 = *v300;
float const v260 = *v301;
float const v261 = v257 + v258;
float const v262 = v259 + v260;
float const v263 = v261 + v262;
uint32_t const v374 = warpSize;
uint32_t const v421 = 256;
uint32_t const v376 = min(v374, v421);
float v415_store;
float * v415 = &v415_store;*v415 = v263;
uint32_t const v378 = 1;
uint32_t v416_store;
uint32_t * v416 = &v416_store;*v416 = v378;
while([&]{
uint32_t const v381 = *v416;
bool const v382 = v381 < v376;
return v382;
}()) {
float const v385 = *v415;
uint32_t const v386 = *v416;
float const v387 = __shfl_xor_sync(__activemask(), v385, v386);
float const v389 = v385 + v387;
*v415 = v389;
uint32_t const v391 = 2;
uint32_t const v392 = v386 * v391;
*v416 = v392;
}
float const v393 = *v415;
int64_t const v265 = 32;
int64_t const v266 = v9 / v265;
int64_t const v267 = v9 % v265;
bool const v268 = v267 == v10;
if(v268) {
uint64_t const v269 = uint64_t(v266);
float * const v270 = &(*v296)[v269];
*v270 = v393;
}
__syncthreads();
bool const v271 = v266 == v10;
if(v271) {
float v394_store;
float * v394 = &v394_store;*v394 = v31;
bool const v275 = v267 < v18;
if(v275) {
uint64_t const v276 = uint64_t(v267);
float * const v277 = &(*v296)[v276];
float const v278 = *v277;
*v394 = v278;
}
float const v279 = *v394;
float v418_store;
float * v418 = &v418_store;*v418 = v279;
uint32_t v419_store;
uint32_t * v419 = &v419_store;*v419 = v378;
while([&]{
uint32_t const v402 = *v419;
bool const v403 = v402 < v376;
return v403;
}()) {
float const v406 = *v418;
uint32_t const v407 = *v419;
float const v408 = __shfl_xor_sync(__activemask(), v406, v407);
float const v410 = v406 + v408;
*v418 = v410;
uint32_t const v412 = 2;
uint32_t const v413 = v407 * v412;
*v419 = v413;
}
float const v414 = *v418;
if(v268) {
uint64_t const v425 = 0;
float * const v282 = &(*v296)[v425];
*v282 = v414;
}
}
__syncthreads();
uint64_t const v423 = 0;
float * const v284 = &(*v296)[v423];
float const v285 = *v284;
bool const v286 = v9 == v10;
if(v286) {
uint64_t const v287 = uint64_t(v7);
float * const v288 = &v2[v287];
*v288 = v285;
}
}
return;
}
