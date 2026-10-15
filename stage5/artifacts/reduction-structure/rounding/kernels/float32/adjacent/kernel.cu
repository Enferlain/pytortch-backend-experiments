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

struct info_st {
    int64_t scalars_i64[3];uint64_t static_meta[2];
};
extern "C" __global__ void __launch_bounds__(256)  cubecl5_206ec91c4bfb25e3(float const* const __restrict__ v1, float * const __restrict__ v2, info_st const* const __restrict__ v285) {
extern __shared__ __align__(4) char v291[];uint64_t const* const dynamic_meta_v287 = reinterpret_cast<uint64_t const*>(v285 + 1);
const info_st& info_v286 = *v285;int64_t const v289 = info_v286.scalars_i64[1];
int64_t const v290 = info_v286.scalars_i64[2];
uint64_t const v6 = blockIdx.x;
int64_t const v7 = int64_t(v6);
uint64_t const v8 = threadIdx.x;
int64_t const v9 = int64_t(v8);
int64_t const v10 = 0;
int64_t const v11 = 1;
int64_t const v15 = 256;
int64_t const v16 = 9;
int64_t const v17 = v16 * v290;
int64_t const v18 = 8;
int64_t const v19 = v18 + v289;
int64_t const v21 = v19 / v16;
int64_t const v22 = v19 % v16;
bool const v25 = v22 < v10;
int64_t const v26 = v21 - v11;
int64_t const v27 = v25 ? v26 : v21;
array<float, 8> * const v292 = reinterpret_cast<array<float, 8> *>(&v291[0]);
bool const v29 = v7 < v17;
if(v29) {
float v294_store;
float * v294 = &v294_store;float const v31 = 0.0;
*v294 = v31;
float v295_store;
float * v295 = &v295_store;*v295 = v31;
float v296_store;
float * v296 = &v296_store;*v296 = v31;
float v297_store;
float * v297 = &v297_store;*v297 = v31;
int64_t const v420 = 1024;
for(int64_t v40 = v9; v40 < v27; v40 += v420) {
bool const v43 = v40 < v27;
float v298_store;
float * v298 = &v298_store;*v298 = v31;
if(v43) {
int64_t const v299 = v7 % v16;
int64_t const v301 = v299 ^ v16;
bool const v302 = v301 < v10;
bool const v303 = v299 != v10;
bool const v304 = v303 && v302;
int64_t const v305 = v304 ? v16 : v10;
int64_t const v306 = v299 + v305;
int64_t const v58 = v27 * v306;
int64_t const v59 = v40 + v58;
bool const v60 = v59 < v289;
float v307_store;
float * v307 = &v307_store;if(v60) {
int64_t const v63 = v7 / v16;
bool const v67 = v299 < v10;
int64_t const v68 = v63 - v11;
int64_t const v69 = v67 ? v68 : v63;
int64_t const v70 = v289 * v69;
int64_t const v71 = v40 + v70;
int64_t const v85 = v71 + v58;
uint64_t const v86 = uint64_t(v85);
float const* const v87 = &v1[v86];
float const v88 = *v87;
*v307 = v88;
}
 else {
*v307 = v31;
}
float const v90 = *v307;
*v298 = v90;
}
int64_t const v92 = v40 + v15;
bool const v93 = v92 < v27;
float v316_store;
float * v316 = &v316_store;*v316 = v31;
if(v93) {
int64_t const v317 = v7 % v16;
int64_t const v319 = v317 ^ v16;
bool const v320 = v319 < v10;
bool const v321 = v317 != v10;
bool const v322 = v321 && v320;
int64_t const v323 = v322 ? v16 : v10;
int64_t const v324 = v317 + v323;
int64_t const v108 = v27 * v324;
int64_t const v109 = v92 + v108;
bool const v110 = v109 < v289;
float v325_store;
float * v325 = &v325_store;if(v110) {
int64_t const v113 = v7 / v16;
bool const v117 = v317 < v10;
int64_t const v118 = v113 - v11;
int64_t const v119 = v117 ? v118 : v113;
int64_t const v120 = v289 * v119;
int64_t const v121 = v92 + v120;
int64_t const v135 = v121 + v108;
uint64_t const v136 = uint64_t(v135);
float const* const v137 = &v1[v136];
float const v138 = *v137;
*v325 = v138;
}
 else {
*v325 = v31;
}
float const v140 = *v325;
*v316 = v140;
}
int64_t const v141 = 512;
int64_t const v142 = v40 + v141;
bool const v143 = v142 < v27;
float v334_store;
float * v334 = &v334_store;*v334 = v31;
if(v143) {
int64_t const v335 = v7 % v16;
int64_t const v337 = v335 ^ v16;
bool const v338 = v337 < v10;
bool const v339 = v335 != v10;
bool const v340 = v339 && v338;
int64_t const v341 = v340 ? v16 : v10;
int64_t const v342 = v335 + v341;
int64_t const v158 = v27 * v342;
int64_t const v159 = v142 + v158;
bool const v160 = v159 < v289;
float v343_store;
float * v343 = &v343_store;if(v160) {
int64_t const v163 = v7 / v16;
bool const v167 = v335 < v10;
int64_t const v168 = v163 - v11;
int64_t const v169 = v167 ? v168 : v163;
int64_t const v170 = v289 * v169;
int64_t const v171 = v142 + v170;
int64_t const v185 = v171 + v158;
uint64_t const v186 = uint64_t(v185);
float const* const v187 = &v1[v186];
float const v188 = *v187;
*v343 = v188;
}
 else {
*v343 = v31;
}
float const v190 = *v343;
*v334 = v190;
}
int64_t const v191 = 768;
int64_t const v192 = v40 + v191;
bool const v193 = v192 < v27;
float v352_store;
float * v352 = &v352_store;*v352 = v31;
if(v193) {
int64_t const v353 = v7 % v16;
int64_t const v355 = v353 ^ v16;
bool const v356 = v355 < v10;
bool const v357 = v353 != v10;
bool const v358 = v357 && v356;
int64_t const v359 = v358 ? v16 : v10;
int64_t const v360 = v353 + v359;
int64_t const v208 = v27 * v360;
int64_t const v209 = v192 + v208;
bool const v210 = v209 < v289;
float v361_store;
float * v361 = &v361_store;if(v210) {
int64_t const v213 = v7 / v16;
bool const v217 = v353 < v10;
int64_t const v218 = v213 - v11;
int64_t const v219 = v217 ? v218 : v213;
int64_t const v220 = v289 * v219;
int64_t const v221 = v192 + v220;
int64_t const v235 = v221 + v208;
uint64_t const v236 = uint64_t(v235);
float const* const v237 = &v1[v236];
float const v238 = *v237;
*v361 = v238;
}
 else {
*v361 = v31;
}
float const v240 = *v361;
*v352 = v240;
}
float const v241 = *v298;
float const v242 = *v294;
float const v243 = v242 + v241;
*v294 = v243;
float const v244 = *v316;
float const v245 = *v295;
float const v246 = v245 + v244;
*v295 = v246;
float const v247 = *v334;
float const v248 = *v296;
float const v249 = v248 + v247;
*v296 = v249;
float const v250 = *v352;
float const v251 = *v297;
float const v252 = v251 + v250;
*v297 = v252;
}
float const v253 = *v294;
float const v254 = *v295;
float const v255 = *v296;
float const v256 = *v297;
float const v257 = v253 + v254;
float const v258 = v255 + v256;
float const v259 = v257 + v258;
uint32_t const v370 = warpSize;
uint32_t const v417 = 256;
uint32_t const v372 = min(v370, v417);
float v411_store;
float * v411 = &v411_store;*v411 = v259;
uint32_t const v374 = 1;
uint32_t v412_store;
uint32_t * v412 = &v412_store;*v412 = v374;
while([&]{
uint32_t const v377 = *v412;
bool const v378 = v377 < v372;
return v378;
}()) {
float const v381 = *v411;
uint32_t const v382 = *v412;
float const v383 = __shfl_xor_sync(__activemask(), v381, v382);
float const v385 = v381 + v383;
*v411 = v385;
uint32_t const v387 = 2;
uint32_t const v388 = v382 * v387;
*v412 = v388;
}
float const v389 = *v411;
int64_t const v261 = 32;
int64_t const v262 = v9 / v261;
int64_t const v263 = v9 % v261;
bool const v264 = v263 == v10;
if(v264) {
uint64_t const v265 = uint64_t(v262);
float * const v266 = &(*v292)[v265];
*v266 = v389;
}
__syncthreads();
bool const v267 = v262 == v10;
if(v267) {
float v390_store;
float * v390 = &v390_store;*v390 = v31;
bool const v271 = v263 < v18;
if(v271) {
uint64_t const v272 = uint64_t(v263);
float * const v273 = &(*v292)[v272];
float const v274 = *v273;
*v390 = v274;
}
float const v275 = *v390;
float v414_store;
float * v414 = &v414_store;*v414 = v275;
uint32_t v415_store;
uint32_t * v415 = &v415_store;*v415 = v374;
while([&]{
uint32_t const v398 = *v415;
bool const v399 = v398 < v372;
return v399;
}()) {
float const v402 = *v414;
uint32_t const v403 = *v415;
float const v404 = __shfl_xor_sync(__activemask(), v402, v403);
float const v406 = v402 + v404;
*v414 = v406;
uint32_t const v408 = 2;
uint32_t const v409 = v403 * v408;
*v415 = v409;
}
float const v410 = *v414;
if(v264) {
uint64_t const v421 = 0;
float * const v278 = &(*v292)[v421];
*v278 = v410;
}
}
__syncthreads();
uint64_t const v419 = 0;
float * const v280 = &(*v292)[v419];
float const v281 = *v280;
bool const v282 = v9 == v10;
if(v282) {
uint64_t const v283 = uint64_t(v7);
float * const v284 = &v2[v283];
*v284 = v281;
}
}
return;
}
