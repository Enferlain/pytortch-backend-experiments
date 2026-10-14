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
extern "C" __global__ void __launch_bounds__(256)  cubecl5_e00bbcddb3e9231a(__half const* const __restrict__ v1, float * const __restrict__ v2, info_st const* const __restrict__ v425) {
extern __shared__ __align__(4) char v431[];uint64_t const* const dynamic_meta_v427 = reinterpret_cast<uint64_t const*>(v425 + 1);
const info_st& info_v426 = *v425;int64_t const v429 = info_v426.scalars_i64[1];
int64_t const v430 = info_v426.scalars_i64[2];
uint64_t const v6 = blockIdx.x;
int64_t const v7 = int64_t(v6);
uint64_t const v8 = threadIdx.x;
int64_t const v9 = int64_t(v8);
int64_t const v10 = 0;
int64_t const v11 = 1;
int64_t const v15 = 256;
int64_t const v16 = 9;
int64_t const v17 = v16 * v430;
int64_t const v18 = 8;
int64_t const v19 = v18 + v429;
int64_t const v21 = v19 / v16;
int64_t const v22 = v19 % v16;
bool const v25 = v22 < v10;
int64_t const v26 = v21 - v11;
int64_t const v27 = v25 ? v26 : v21;
array<float, 512> * const v432 = reinterpret_cast<array<float, 512> *>(&v431[0]);
bool const v29 = v7 < v17;
if(v29) {
float v434_store;
float * v434 = &v434_store;float const v31 = 0.0;
*v434 = v31;
float v435_store;
float * v435 = &v435_store;*v435 = v31;
for(int64_t v37 = v9; v37 < v27; v37 += v15) {
int64_t const v436 = v7 % v16;
int64_t const v438 = v436 ^ v16;
bool const v439 = v438 < v10;
bool const v440 = v436 != v10;
bool const v441 = v440 && v439;
int64_t const v442 = v441 ? v16 : v10;
int64_t const v443 = v436 + v442;
int64_t const v50 = v27 * v443;
int64_t const v51 = v37 + v50;
bool const v52 = v51 < v429;
float v444_store;
float * v444 = &v444_store;if(v52) {
int64_t const v55 = v7 / v16;
bool const v59 = v436 < v10;
int64_t const v60 = v55 - v11;
int64_t const v61 = v59 ? v60 : v55;
int64_t const v62 = v429 * v61;
int64_t const v63 = v37 + v62;
int64_t const v77 = v63 + v50;
uint64_t const v78 = uint64_t(v77);
__half const* const v79 = &v1[v78];
__half const v80 = *v79;
float const v81 = float(v80);
*v444 = v81;
}
 else {
*v444 = v31;
}
float const v83 = *v444;
float const v84 = *v435;
float const v85 = *v434;
float const v86 = v84 + v83;
float const v87 = v86 - v84;
float const v88 = v86 - v87;
float const v89 = v84 - v88;
float const v90 = v83 - v87;
float const v91 = v89 + v90;
float const v92 = v85 + v91;
float const v93 = fabs(v86);
float const v94 = 3.402823470e+38;
bool const v95 = v93 <= v94;
float const v97 = v95 ? v92 : v31;
*v434 = v97;
*v435 = v86;
}
float const v98 = *v435;
uint64_t const v102 = uint64_t(v9);
float * const v103 = &(*v432)[v102];
*v103 = v98;
float const v104 = *v434;
int64_t const v106 = v9 + v15;
uint64_t const v107 = uint64_t(v106);
float * const v108 = &(*v432)[v107];
*v108 = v104;
__syncthreads();
int64_t const v109 = 128;
bool const v110 = v9 < v109;
if(v110) {
int64_t const v111 = v9 + v109;
uint64_t const v120 = uint64_t(v111);
float * const v121 = &(*v432)[v120];
float const v122 = *v103;
float const v123 = *v121;
float const v128 = *v108;
int64_t const v130 = v111 + v15;
uint64_t const v131 = uint64_t(v130);
float * const v132 = &(*v432)[v131];
float const v133 = *v132;
float const v134 = v122 + v123;
float const v135 = v134 - v122;
float const v136 = v134 - v135;
float const v137 = v122 - v136;
float const v138 = v123 - v135;
float const v139 = v137 + v138;
float const v140 = v128 + v139;
float const v141 = fabs(v134);
float const v142 = 3.402823470e+38;
bool const v143 = v141 <= v142;
float const v145 = v143 ? v140 : v31;
float const v146 = v145 + v133;
*v108 = v146;
*v103 = v134;
}
__syncthreads();
int64_t const v147 = 64;
bool const v148 = v9 < v147;
if(v148) {
int64_t const v149 = v9 + v147;
uint64_t const v158 = uint64_t(v149);
float * const v159 = &(*v432)[v158];
float const v160 = *v103;
float const v161 = *v159;
float const v166 = *v108;
int64_t const v168 = v149 + v15;
uint64_t const v169 = uint64_t(v168);
float * const v170 = &(*v432)[v169];
float const v171 = *v170;
float const v172 = v160 + v161;
float const v173 = v172 - v160;
float const v174 = v172 - v173;
float const v175 = v160 - v174;
float const v176 = v161 - v173;
float const v177 = v175 + v176;
float const v178 = v166 + v177;
float const v179 = fabs(v172);
float const v180 = 3.402823470e+38;
bool const v181 = v179 <= v180;
float const v183 = v181 ? v178 : v31;
float const v184 = v183 + v171;
*v108 = v184;
*v103 = v172;
}
__syncthreads();
int64_t const v185 = 32;
bool const v186 = v9 < v185;
if(v186) {
int64_t const v187 = v9 + v185;
uint64_t const v196 = uint64_t(v187);
float * const v197 = &(*v432)[v196];
float const v198 = *v103;
float const v199 = *v197;
float const v204 = *v108;
int64_t const v206 = v187 + v15;
uint64_t const v207 = uint64_t(v206);
float * const v208 = &(*v432)[v207];
float const v209 = *v208;
float const v210 = v198 + v199;
float const v211 = v210 - v198;
float const v212 = v210 - v211;
float const v213 = v198 - v212;
float const v214 = v199 - v211;
float const v215 = v213 + v214;
float const v216 = v204 + v215;
float const v217 = fabs(v210);
float const v218 = 3.402823470e+38;
bool const v219 = v217 <= v218;
float const v221 = v219 ? v216 : v31;
float const v222 = v221 + v209;
*v108 = v222;
*v103 = v210;
}
__syncthreads();
int64_t const v223 = 16;
bool const v224 = v9 < v223;
if(v224) {
int64_t const v225 = v9 + v223;
uint64_t const v234 = uint64_t(v225);
float * const v235 = &(*v432)[v234];
float const v236 = *v103;
float const v237 = *v235;
float const v242 = *v108;
int64_t const v244 = v225 + v15;
uint64_t const v245 = uint64_t(v244);
float * const v246 = &(*v432)[v245];
float const v247 = *v246;
float const v248 = v236 + v237;
float const v249 = v248 - v236;
float const v250 = v248 - v249;
float const v251 = v236 - v250;
float const v252 = v237 - v249;
float const v253 = v251 + v252;
float const v254 = v242 + v253;
float const v255 = fabs(v248);
float const v256 = 3.402823470e+38;
bool const v257 = v255 <= v256;
float const v259 = v257 ? v254 : v31;
float const v260 = v259 + v247;
*v108 = v260;
*v103 = v248;
}
__syncthreads();
bool const v262 = v9 < v18;
if(v262) {
int64_t const v263 = v9 + v18;
uint64_t const v272 = uint64_t(v263);
float * const v273 = &(*v432)[v272];
float const v274 = *v103;
float const v275 = *v273;
float const v280 = *v108;
int64_t const v282 = v263 + v15;
uint64_t const v283 = uint64_t(v282);
float * const v284 = &(*v432)[v283];
float const v285 = *v284;
float const v286 = v274 + v275;
float const v287 = v286 - v274;
float const v288 = v286 - v287;
float const v289 = v274 - v288;
float const v290 = v275 - v287;
float const v291 = v289 + v290;
float const v292 = v280 + v291;
float const v293 = fabs(v286);
float const v294 = 3.402823470e+38;
bool const v295 = v293 <= v294;
float const v297 = v295 ? v292 : v31;
float const v298 = v297 + v285;
*v108 = v298;
*v103 = v286;
}
__syncthreads();
int64_t const v299 = 4;
bool const v300 = v9 < v299;
if(v300) {
int64_t const v301 = v9 + v299;
uint64_t const v310 = uint64_t(v301);
float * const v311 = &(*v432)[v310];
float const v312 = *v103;
float const v313 = *v311;
float const v318 = *v108;
int64_t const v320 = v301 + v15;
uint64_t const v321 = uint64_t(v320);
float * const v322 = &(*v432)[v321];
float const v323 = *v322;
float const v324 = v312 + v313;
float const v325 = v324 - v312;
float const v326 = v324 - v325;
float const v327 = v312 - v326;
float const v328 = v313 - v325;
float const v329 = v327 + v328;
float const v330 = v318 + v329;
float const v331 = fabs(v324);
float const v332 = 3.402823470e+38;
bool const v333 = v331 <= v332;
float const v335 = v333 ? v330 : v31;
float const v336 = v335 + v323;
*v108 = v336;
*v103 = v324;
}
__syncthreads();
int64_t const v337 = 2;
bool const v338 = v9 < v337;
if(v338) {
int64_t const v339 = v9 + v337;
uint64_t const v348 = uint64_t(v339);
float * const v349 = &(*v432)[v348];
float const v350 = *v103;
float const v351 = *v349;
float const v356 = *v108;
int64_t const v358 = v339 + v15;
uint64_t const v359 = uint64_t(v358);
float * const v360 = &(*v432)[v359];
float const v361 = *v360;
float const v362 = v350 + v351;
float const v363 = v362 - v350;
float const v364 = v362 - v363;
float const v365 = v350 - v364;
float const v366 = v351 - v363;
float const v367 = v365 + v366;
float const v368 = v356 + v367;
float const v369 = fabs(v362);
float const v370 = 3.402823470e+38;
bool const v371 = v369 <= v370;
float const v373 = v371 ? v368 : v31;
float const v374 = v373 + v361;
*v108 = v374;
*v103 = v362;
}
__syncthreads();
bool const v376 = v9 < v11;
if(v376) {
int64_t const v377 = v9 + v11;
uint64_t const v386 = uint64_t(v377);
float * const v387 = &(*v432)[v386];
float const v388 = *v103;
float const v389 = *v387;
float const v394 = *v108;
int64_t const v396 = v377 + v15;
uint64_t const v397 = uint64_t(v396);
float * const v398 = &(*v432)[v397];
float const v399 = *v398;
float const v400 = v388 + v389;
float const v401 = v400 - v388;
float const v402 = v400 - v401;
float const v403 = v388 - v402;
float const v404 = v389 - v401;
float const v405 = v403 + v404;
float const v406 = v394 + v405;
float const v407 = fabs(v400);
float const v408 = 3.402823470e+38;
bool const v409 = v407 <= v408;
float const v411 = v409 ? v406 : v31;
float const v412 = v411 + v399;
*v108 = v412;
*v103 = v400;
}
__syncthreads();
uint64_t const v455 = 0;
float * const v414 = &(*v432)[v455];
float const v415 = *v414;
uint64_t const v453 = 256;
float * const v419 = &(*v432)[v453];
float const v420 = *v419;
float const v421 = v415 + v420;
__syncthreads();
bool const v422 = v9 == v10;
if(v422) {
uint64_t const v423 = uint64_t(v7);
float * const v424 = &v2[v423];
*v424 = v421;
}
}
return;
}
