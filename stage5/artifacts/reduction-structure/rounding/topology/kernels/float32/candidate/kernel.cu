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
extern "C" __global__ void __launch_bounds__(512)  cubecl5_42026f8357481503(float const* const __restrict__ v1, float * const __restrict__ v2, info_st const* const __restrict__ v310) {
extern __shared__ __align__(4) char v316[];uint64_t const* const dynamic_meta_v312 = reinterpret_cast<uint64_t const*>(v310 + 1);
const info_st& info_v311 = *v310;int64_t const v314 = info_v311.scalars_i64[1];
int64_t const v315 = info_v311.scalars_i64[2];
uint64_t const v6 = blockIdx.x;
int64_t const v7 = int64_t(v6);
uint64_t const v8 = threadIdx.x;
int64_t const v9 = int64_t(v8);
int64_t const v10 = 0;
int64_t const v11 = 1;
int64_t const v15 = 512;
int64_t const v16 = 9;
int64_t const v17 = v16 * v315;
int64_t const v18 = 8;
int64_t const v19 = v18 + v314;
int64_t const v21 = v19 / v16;
int64_t const v22 = v19 % v16;
bool const v25 = v22 < v10;
int64_t const v26 = v21 - v11;
int64_t const v27 = v25 ? v26 : v21;
array<float, 16> * const v317 = reinterpret_cast<array<float, 16> *>(&v316[0]);
bool const v29 = v7 < v17;
if(v29) {
float v319_store;
float * v319 = &v319_store;float const v31 = 0.0;
*v319 = v31;
float v320_store;
float * v320 = &v320_store;*v320 = v31;
float v321_store;
float * v321 = &v321_store;*v321 = v31;
float v322_store;
float * v322 = &v322_store;*v322 = v31;
int64_t const v397 = 2048;
for(int64_t v40 = v9; v40 < v27; v40 += v397) {
bool const v43 = v40 < v27;
float v323_store;
float * v323 = &v323_store;*v323 = v31;
if(v43) {
int64_t const v324 = v7 % v16;
int64_t const v326 = v324 ^ v16;
bool const v327 = v326 < v10;
bool const v328 = v324 != v10;
bool const v329 = v328 && v327;
int64_t const v330 = v329 ? v16 : v10;
int64_t const v331 = v324 + v330;
int64_t const v58 = v27 * v331;
int64_t const v59 = v40 + v58;
bool const v60 = v59 < v314;
float v332_store;
float * v332 = &v332_store;if(v60) {
int64_t const v63 = v7 / v16;
bool const v67 = v324 < v10;
int64_t const v68 = v63 - v11;
int64_t const v69 = v67 ? v68 : v63;
int64_t const v70 = v314 * v69;
int64_t const v71 = v40 + v70;
int64_t const v85 = v71 + v58;
uint64_t const v86 = uint64_t(v85);
float const* const v87 = &v1[v86];
float const v88 = *v87;
*v332 = v88;
}
 else {
*v332 = v31;
}
float const v90 = *v332;
*v323 = v90;
}
int64_t const v92 = v40 + v15;
bool const v93 = v92 < v27;
float v341_store;
float * v341 = &v341_store;*v341 = v31;
if(v93) {
int64_t const v342 = v7 % v16;
int64_t const v344 = v342 ^ v16;
bool const v345 = v344 < v10;
bool const v346 = v342 != v10;
bool const v347 = v346 && v345;
int64_t const v348 = v347 ? v16 : v10;
int64_t const v349 = v342 + v348;
int64_t const v108 = v27 * v349;
int64_t const v109 = v92 + v108;
bool const v110 = v109 < v314;
float v350_store;
float * v350 = &v350_store;if(v110) {
int64_t const v113 = v7 / v16;
bool const v117 = v342 < v10;
int64_t const v118 = v113 - v11;
int64_t const v119 = v117 ? v118 : v113;
int64_t const v120 = v314 * v119;
int64_t const v121 = v92 + v120;
int64_t const v135 = v121 + v108;
uint64_t const v136 = uint64_t(v135);
float const* const v137 = &v1[v136];
float const v138 = *v137;
*v350 = v138;
}
 else {
*v350 = v31;
}
float const v140 = *v350;
*v341 = v140;
}
int64_t const v141 = 1024;
int64_t const v142 = v40 + v141;
bool const v143 = v142 < v27;
float v359_store;
float * v359 = &v359_store;*v359 = v31;
if(v143) {
int64_t const v360 = v7 % v16;
int64_t const v362 = v360 ^ v16;
bool const v363 = v362 < v10;
bool const v364 = v360 != v10;
bool const v365 = v364 && v363;
int64_t const v366 = v365 ? v16 : v10;
int64_t const v367 = v360 + v366;
int64_t const v158 = v27 * v367;
int64_t const v159 = v142 + v158;
bool const v160 = v159 < v314;
float v368_store;
float * v368 = &v368_store;if(v160) {
int64_t const v163 = v7 / v16;
bool const v167 = v360 < v10;
int64_t const v168 = v163 - v11;
int64_t const v169 = v167 ? v168 : v163;
int64_t const v170 = v314 * v169;
int64_t const v171 = v142 + v170;
int64_t const v185 = v171 + v158;
uint64_t const v186 = uint64_t(v185);
float const* const v187 = &v1[v186];
float const v188 = *v187;
*v368 = v188;
}
 else {
*v368 = v31;
}
float const v190 = *v368;
*v359 = v190;
}
int64_t const v191 = 1536;
int64_t const v192 = v40 + v191;
bool const v193 = v192 < v27;
float v377_store;
float * v377 = &v377_store;*v377 = v31;
if(v193) {
int64_t const v378 = v7 % v16;
int64_t const v380 = v378 ^ v16;
bool const v381 = v380 < v10;
bool const v382 = v378 != v10;
bool const v383 = v382 && v381;
int64_t const v384 = v383 ? v16 : v10;
int64_t const v385 = v378 + v384;
int64_t const v208 = v27 * v385;
int64_t const v209 = v192 + v208;
bool const v210 = v209 < v314;
float v386_store;
float * v386 = &v386_store;if(v210) {
int64_t const v213 = v7 / v16;
bool const v217 = v378 < v10;
int64_t const v218 = v213 - v11;
int64_t const v219 = v217 ? v218 : v213;
int64_t const v220 = v314 * v219;
int64_t const v221 = v192 + v220;
int64_t const v235 = v221 + v208;
uint64_t const v236 = uint64_t(v235);
float const* const v237 = &v1[v236];
float const v238 = *v237;
*v386 = v238;
}
 else {
*v386 = v31;
}
float const v240 = *v386;
*v377 = v240;
}
float const v241 = *v323;
float const v242 = *v319;
float const v243 = v242 + v241;
*v319 = v243;
float const v244 = *v341;
float const v245 = *v320;
float const v246 = v245 + v244;
*v320 = v246;
float const v247 = *v359;
float const v248 = *v321;
float const v249 = v248 + v247;
*v321 = v249;
float const v250 = *v377;
float const v251 = *v322;
float const v252 = v251 + v250;
*v322 = v252;
}
float const v253 = *v319;
float const v254 = *v320;
float const v255 = *v321;
float const v256 = *v322;
float const v257 = v253 + v254;
float const v258 = v255 + v256;
float const v259 = v257 + v258;
int32_t const v260 = 16;
float const v261 = __shfl_xor_sync(__activemask(), v259, v260);
float const v262 = v259 + v261;
int32_t const v263 = 8;
float const v264 = __shfl_xor_sync(__activemask(), v262, v263);
float const v265 = v262 + v264;
int32_t const v266 = 4;
float const v267 = __shfl_xor_sync(__activemask(), v265, v266);
float const v268 = v265 + v267;
int32_t const v269 = 2;
float const v270 = __shfl_xor_sync(__activemask(), v268, v269);
float const v271 = v268 + v270;
int32_t const v272 = 1;
float const v273 = __shfl_xor_sync(__activemask(), v271, v272);
float const v274 = v271 + v273;
int64_t const v275 = 32;
int64_t const v276 = v9 / v275;
int64_t const v277 = v9 % v275;
bool const v278 = v277 == v10;
if(v278) {
uint64_t const v279 = uint64_t(v276);
float * const v280 = &(*v317)[v279];
*v280 = v274;
}
__syncthreads();
bool const v281 = v276 == v10;
if(v281) {
float v395_store;
float * v395 = &v395_store;*v395 = v31;
int64_t const v284 = 16;
bool const v285 = v277 < v284;
if(v285) {
uint64_t const v286 = uint64_t(v277);
float * const v287 = &(*v317)[v286];
float const v288 = *v287;
*v395 = v288;
}
float const v289 = *v395;
float const v291 = __shfl_xor_sync(__activemask(), v289, v263);
float const v292 = v289 + v291;
float const v294 = __shfl_xor_sync(__activemask(), v292, v266);
float const v295 = v292 + v294;
float const v297 = __shfl_xor_sync(__activemask(), v295, v269);
float const v298 = v295 + v297;
float const v300 = __shfl_xor_sync(__activemask(), v298, v272);
float const v301 = v298 + v300;
if(v278) {
uint64_t const v398 = 0;
float * const v303 = &(*v317)[v398];
*v303 = v301;
}
}
__syncthreads();
uint64_t const v396 = 0;
float * const v305 = &(*v317)[v396];
float const v306 = *v305;
bool const v307 = v9 == v10;
if(v307) {
uint64_t const v308 = uint64_t(v7);
float * const v309 = &v2[v308];
*v309 = v306;
}
}
return;
}
