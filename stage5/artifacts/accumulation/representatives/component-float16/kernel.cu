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
extern "C" __global__ void __launch_bounds__(256)  cubecl5_44b91070cd716a74(__half const* const __restrict__ v1, __half * const __restrict__ v2, info_st const* const __restrict__ v376) {
extern __shared__ __align__(4) char v382[];uint64_t const* const dynamic_meta_v378 = reinterpret_cast<uint64_t const*>(v376 + 1);
const info_st& info_v377 = *v376;int64_t const v380 = info_v377.scalars_i64[1];
int64_t const v381 = info_v377.scalars_i64[2];
uint64_t const v6 = blockIdx.x;
int64_t const v7 = int64_t(v6);
uint64_t const v8 = threadIdx.x;
int64_t const v9 = int64_t(v8);
int64_t const v10 = 0;
int64_t const v11 = 1;
int64_t const v15 = 256;
array<float, 512> * const v383 = reinterpret_cast<array<float, 512> *>(&v382[0]);
bool const v17 = v7 < v381;
if(v17) {
float v385_store;
float * v385 = &v385_store;float const v19 = 0.0;
*v385 = v19;
float v386_store;
float * v386 = &v386_store;*v386 = v19;
for(int64_t v25 = v9; v25 < v380; v25 += v15) {
int64_t const v26 = v7 * v380;
int64_t const v27 = v25 + v26;
uint64_t const v28 = uint64_t(v27);
__half const* const v29 = &v1[v28];
__half const v30 = *v29;
float const v31 = float(v30);
float const v32 = *v386;
float const v33 = *v385;
float const v34 = v32 + v31;
float const v35 = v34 - v32;
float const v36 = v34 - v35;
float const v37 = v32 - v36;
float const v38 = v31 - v35;
float const v39 = v37 + v38;
float const v40 = v33 + v39;
float const v41 = fabs(v34);
float const v42 = 3.402823470e+38;
bool const v43 = v41 <= v42;
float const v45 = v43 ? v40 : v19;
*v385 = v45;
*v386 = v34;
}
float const v46 = *v386;
uint64_t const v50 = uint64_t(v9);
float * const v51 = &(*v383)[v50];
*v51 = v46;
float const v52 = *v385;
int64_t const v54 = v9 + v15;
uint64_t const v55 = uint64_t(v54);
float * const v56 = &(*v383)[v55];
*v56 = v52;
__syncthreads();
int64_t const v57 = 128;
bool const v58 = v9 < v57;
if(v58) {
int64_t const v59 = v9 + v57;
uint64_t const v68 = uint64_t(v59);
float * const v69 = &(*v383)[v68];
float const v70 = *v51;
float const v71 = *v69;
float const v76 = *v56;
int64_t const v78 = v59 + v15;
uint64_t const v79 = uint64_t(v78);
float * const v80 = &(*v383)[v79];
float const v81 = *v80;
float const v82 = v70 + v71;
float const v83 = v82 - v70;
float const v84 = v82 - v83;
float const v85 = v70 - v84;
float const v86 = v71 - v83;
float const v87 = v85 + v86;
float const v88 = v76 + v87;
float const v89 = fabs(v82);
float const v90 = 3.402823470e+38;
bool const v91 = v89 <= v90;
float const v93 = v91 ? v88 : v19;
float const v94 = v93 + v81;
*v56 = v94;
*v51 = v82;
}
__syncthreads();
int64_t const v95 = 64;
bool const v96 = v9 < v95;
if(v96) {
int64_t const v97 = v9 + v95;
uint64_t const v106 = uint64_t(v97);
float * const v107 = &(*v383)[v106];
float const v108 = *v51;
float const v109 = *v107;
float const v114 = *v56;
int64_t const v116 = v97 + v15;
uint64_t const v117 = uint64_t(v116);
float * const v118 = &(*v383)[v117];
float const v119 = *v118;
float const v120 = v108 + v109;
float const v121 = v120 - v108;
float const v122 = v120 - v121;
float const v123 = v108 - v122;
float const v124 = v109 - v121;
float const v125 = v123 + v124;
float const v126 = v114 + v125;
float const v127 = fabs(v120);
float const v128 = 3.402823470e+38;
bool const v129 = v127 <= v128;
float const v131 = v129 ? v126 : v19;
float const v132 = v131 + v119;
*v56 = v132;
*v51 = v120;
}
__syncthreads();
int64_t const v133 = 32;
bool const v134 = v9 < v133;
if(v134) {
int64_t const v135 = v9 + v133;
uint64_t const v144 = uint64_t(v135);
float * const v145 = &(*v383)[v144];
float const v146 = *v51;
float const v147 = *v145;
float const v152 = *v56;
int64_t const v154 = v135 + v15;
uint64_t const v155 = uint64_t(v154);
float * const v156 = &(*v383)[v155];
float const v157 = *v156;
float const v158 = v146 + v147;
float const v159 = v158 - v146;
float const v160 = v158 - v159;
float const v161 = v146 - v160;
float const v162 = v147 - v159;
float const v163 = v161 + v162;
float const v164 = v152 + v163;
float const v165 = fabs(v158);
float const v166 = 3.402823470e+38;
bool const v167 = v165 <= v166;
float const v169 = v167 ? v164 : v19;
float const v170 = v169 + v157;
*v56 = v170;
*v51 = v158;
}
__syncthreads();
int64_t const v171 = 16;
bool const v172 = v9 < v171;
if(v172) {
int64_t const v173 = v9 + v171;
uint64_t const v182 = uint64_t(v173);
float * const v183 = &(*v383)[v182];
float const v184 = *v51;
float const v185 = *v183;
float const v190 = *v56;
int64_t const v192 = v173 + v15;
uint64_t const v193 = uint64_t(v192);
float * const v194 = &(*v383)[v193];
float const v195 = *v194;
float const v196 = v184 + v185;
float const v197 = v196 - v184;
float const v198 = v196 - v197;
float const v199 = v184 - v198;
float const v200 = v185 - v197;
float const v201 = v199 + v200;
float const v202 = v190 + v201;
float const v203 = fabs(v196);
float const v204 = 3.402823470e+38;
bool const v205 = v203 <= v204;
float const v207 = v205 ? v202 : v19;
float const v208 = v207 + v195;
*v56 = v208;
*v51 = v196;
}
__syncthreads();
int64_t const v209 = 8;
bool const v210 = v9 < v209;
if(v210) {
int64_t const v211 = v9 + v209;
uint64_t const v220 = uint64_t(v211);
float * const v221 = &(*v383)[v220];
float const v222 = *v51;
float const v223 = *v221;
float const v228 = *v56;
int64_t const v230 = v211 + v15;
uint64_t const v231 = uint64_t(v230);
float * const v232 = &(*v383)[v231];
float const v233 = *v232;
float const v234 = v222 + v223;
float const v235 = v234 - v222;
float const v236 = v234 - v235;
float const v237 = v222 - v236;
float const v238 = v223 - v235;
float const v239 = v237 + v238;
float const v240 = v228 + v239;
float const v241 = fabs(v234);
float const v242 = 3.402823470e+38;
bool const v243 = v241 <= v242;
float const v245 = v243 ? v240 : v19;
float const v246 = v245 + v233;
*v56 = v246;
*v51 = v234;
}
__syncthreads();
int64_t const v247 = 4;
bool const v248 = v9 < v247;
if(v248) {
int64_t const v249 = v9 + v247;
uint64_t const v258 = uint64_t(v249);
float * const v259 = &(*v383)[v258];
float const v260 = *v51;
float const v261 = *v259;
float const v266 = *v56;
int64_t const v268 = v249 + v15;
uint64_t const v269 = uint64_t(v268);
float * const v270 = &(*v383)[v269];
float const v271 = *v270;
float const v272 = v260 + v261;
float const v273 = v272 - v260;
float const v274 = v272 - v273;
float const v275 = v260 - v274;
float const v276 = v261 - v273;
float const v277 = v275 + v276;
float const v278 = v266 + v277;
float const v279 = fabs(v272);
float const v280 = 3.402823470e+38;
bool const v281 = v279 <= v280;
float const v283 = v281 ? v278 : v19;
float const v284 = v283 + v271;
*v56 = v284;
*v51 = v272;
}
__syncthreads();
int64_t const v285 = 2;
bool const v286 = v9 < v285;
if(v286) {
int64_t const v287 = v9 + v285;
uint64_t const v296 = uint64_t(v287);
float * const v297 = &(*v383)[v296];
float const v298 = *v51;
float const v299 = *v297;
float const v304 = *v56;
int64_t const v306 = v287 + v15;
uint64_t const v307 = uint64_t(v306);
float * const v308 = &(*v383)[v307];
float const v309 = *v308;
float const v310 = v298 + v299;
float const v311 = v310 - v298;
float const v312 = v310 - v311;
float const v313 = v298 - v312;
float const v314 = v299 - v311;
float const v315 = v313 + v314;
float const v316 = v304 + v315;
float const v317 = fabs(v310);
float const v318 = 3.402823470e+38;
bool const v319 = v317 <= v318;
float const v321 = v319 ? v316 : v19;
float const v322 = v321 + v309;
*v56 = v322;
*v51 = v310;
}
__syncthreads();
bool const v324 = v9 < v11;
if(v324) {
int64_t const v325 = v9 + v11;
uint64_t const v334 = uint64_t(v325);
float * const v335 = &(*v383)[v334];
float const v336 = *v51;
float const v337 = *v335;
float const v342 = *v56;
int64_t const v344 = v325 + v15;
uint64_t const v345 = uint64_t(v344);
float * const v346 = &(*v383)[v345];
float const v347 = *v346;
float const v348 = v336 + v337;
float const v349 = v348 - v336;
float const v350 = v348 - v349;
float const v351 = v336 - v350;
float const v352 = v337 - v349;
float const v353 = v351 + v352;
float const v354 = v342 + v353;
float const v355 = fabs(v348);
float const v356 = 3.402823470e+38;
bool const v357 = v355 <= v356;
float const v359 = v357 ? v354 : v19;
float const v360 = v359 + v347;
*v56 = v360;
*v51 = v348;
}
__syncthreads();
uint64_t const v389 = 0;
float * const v362 = &(*v383)[v389];
float const v363 = *v362;
uint64_t const v387 = 256;
float * const v367 = &(*v383)[v387];
float const v368 = *v367;
float const v369 = v363 + v368;
__syncthreads();
bool const v370 = v9 == v10;
if(v370) {
__half const v371 = __half(v369);
float const v372 = float(v371);
__half const v373 = __half(v372);
uint64_t const v374 = uint64_t(v7);
__half * const v375 = &v2[v374];
*v375 = v373;
}
}
return;
}
