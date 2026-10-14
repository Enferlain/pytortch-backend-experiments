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
extern "C" __global__ void __launch_bounds__(256)  cubecl5_82747aaa29e9ab62(__nv_bfloat16 const* const __restrict__ v1, float * const __restrict__ v2, info_st const* const __restrict__ v382) {
extern __shared__ __align__(4) char v388[];uint64_t const* const dynamic_meta_v384 = reinterpret_cast<uint64_t const*>(v382 + 1);
const info_st& info_v383 = *v382;int64_t const v386 = info_v383.scalars_i64[1];
int64_t const v387 = info_v383.scalars_i64[2];
uint64_t const v6 = blockIdx.x;
int64_t const v7 = int64_t(v6);
uint64_t const v8 = threadIdx.x;
int64_t const v9 = int64_t(v8);
int64_t const v10 = 0;
int64_t const v11 = 1;
int64_t const v15 = 256;
array<float, 512> * const v389 = reinterpret_cast<array<float, 512> *>(&v388[0]);
bool const v17 = v7 < v387;
if(v17) {
float v391_store;
float * v391 = &v391_store;float const v19 = 0.0;
*v391 = v19;
float v392_store;
float * v392 = &v392_store;*v392 = v19;
for(int64_t v25 = v9; v25 < v386; v25 += v15) {
int64_t const v26 = v7 * v386;
int64_t const v27 = v25 + v26;
uint64_t const v28 = uint64_t(v27);
__nv_bfloat16 const* const v29 = &v1[v28];
__nv_bfloat16 const v30 = *v29;
float const v31 = float(v30);
__nv_bfloat16 const v32 = __nv_bfloat16(v31);
float const v33 = float(v32);
float const v35 = v33 * v33;
float const v36 = *v392;
float const v37 = *v391;
float const v38 = v36 + v35;
float const v39 = v38 - v36;
float const v40 = v38 - v39;
float const v41 = v36 - v40;
float const v42 = v35 - v39;
float const v43 = v41 + v42;
float const v44 = v37 + v43;
float const v45 = fabs(v38);
float const v46 = 3.402823470e+38;
bool const v47 = v45 <= v46;
float const v49 = v47 ? v44 : v19;
*v391 = v49;
*v392 = v38;
}
float const v50 = *v392;
uint64_t const v54 = uint64_t(v9);
float * const v55 = &(*v389)[v54];
*v55 = v50;
float const v56 = *v391;
int64_t const v58 = v9 + v15;
uint64_t const v59 = uint64_t(v58);
float * const v60 = &(*v389)[v59];
*v60 = v56;
__syncthreads();
int64_t const v61 = 128;
bool const v62 = v9 < v61;
if(v62) {
int64_t const v63 = v9 + v61;
uint64_t const v72 = uint64_t(v63);
float * const v73 = &(*v389)[v72];
float const v74 = *v55;
float const v75 = *v73;
float const v80 = *v60;
int64_t const v82 = v63 + v15;
uint64_t const v83 = uint64_t(v82);
float * const v84 = &(*v389)[v83];
float const v85 = *v84;
float const v86 = v74 + v75;
float const v87 = v86 - v74;
float const v88 = v86 - v87;
float const v89 = v74 - v88;
float const v90 = v75 - v87;
float const v91 = v89 + v90;
float const v92 = v80 + v91;
float const v93 = fabs(v86);
float const v94 = 3.402823470e+38;
bool const v95 = v93 <= v94;
float const v97 = v95 ? v92 : v19;
float const v98 = v97 + v85;
*v60 = v98;
*v55 = v86;
}
__syncthreads();
int64_t const v99 = 64;
bool const v100 = v9 < v99;
if(v100) {
int64_t const v101 = v9 + v99;
uint64_t const v110 = uint64_t(v101);
float * const v111 = &(*v389)[v110];
float const v112 = *v55;
float const v113 = *v111;
float const v118 = *v60;
int64_t const v120 = v101 + v15;
uint64_t const v121 = uint64_t(v120);
float * const v122 = &(*v389)[v121];
float const v123 = *v122;
float const v124 = v112 + v113;
float const v125 = v124 - v112;
float const v126 = v124 - v125;
float const v127 = v112 - v126;
float const v128 = v113 - v125;
float const v129 = v127 + v128;
float const v130 = v118 + v129;
float const v131 = fabs(v124);
float const v132 = 3.402823470e+38;
bool const v133 = v131 <= v132;
float const v135 = v133 ? v130 : v19;
float const v136 = v135 + v123;
*v60 = v136;
*v55 = v124;
}
__syncthreads();
int64_t const v137 = 32;
bool const v138 = v9 < v137;
if(v138) {
int64_t const v139 = v9 + v137;
uint64_t const v148 = uint64_t(v139);
float * const v149 = &(*v389)[v148];
float const v150 = *v55;
float const v151 = *v149;
float const v156 = *v60;
int64_t const v158 = v139 + v15;
uint64_t const v159 = uint64_t(v158);
float * const v160 = &(*v389)[v159];
float const v161 = *v160;
float const v162 = v150 + v151;
float const v163 = v162 - v150;
float const v164 = v162 - v163;
float const v165 = v150 - v164;
float const v166 = v151 - v163;
float const v167 = v165 + v166;
float const v168 = v156 + v167;
float const v169 = fabs(v162);
float const v170 = 3.402823470e+38;
bool const v171 = v169 <= v170;
float const v173 = v171 ? v168 : v19;
float const v174 = v173 + v161;
*v60 = v174;
*v55 = v162;
}
__syncthreads();
int64_t const v175 = 16;
bool const v176 = v9 < v175;
if(v176) {
int64_t const v177 = v9 + v175;
uint64_t const v186 = uint64_t(v177);
float * const v187 = &(*v389)[v186];
float const v188 = *v55;
float const v189 = *v187;
float const v194 = *v60;
int64_t const v196 = v177 + v15;
uint64_t const v197 = uint64_t(v196);
float * const v198 = &(*v389)[v197];
float const v199 = *v198;
float const v200 = v188 + v189;
float const v201 = v200 - v188;
float const v202 = v200 - v201;
float const v203 = v188 - v202;
float const v204 = v189 - v201;
float const v205 = v203 + v204;
float const v206 = v194 + v205;
float const v207 = fabs(v200);
float const v208 = 3.402823470e+38;
bool const v209 = v207 <= v208;
float const v211 = v209 ? v206 : v19;
float const v212 = v211 + v199;
*v60 = v212;
*v55 = v200;
}
__syncthreads();
int64_t const v213 = 8;
bool const v214 = v9 < v213;
if(v214) {
int64_t const v215 = v9 + v213;
uint64_t const v224 = uint64_t(v215);
float * const v225 = &(*v389)[v224];
float const v226 = *v55;
float const v227 = *v225;
float const v232 = *v60;
int64_t const v234 = v215 + v15;
uint64_t const v235 = uint64_t(v234);
float * const v236 = &(*v389)[v235];
float const v237 = *v236;
float const v238 = v226 + v227;
float const v239 = v238 - v226;
float const v240 = v238 - v239;
float const v241 = v226 - v240;
float const v242 = v227 - v239;
float const v243 = v241 + v242;
float const v244 = v232 + v243;
float const v245 = fabs(v238);
float const v246 = 3.402823470e+38;
bool const v247 = v245 <= v246;
float const v249 = v247 ? v244 : v19;
float const v250 = v249 + v237;
*v60 = v250;
*v55 = v238;
}
__syncthreads();
int64_t const v251 = 4;
bool const v252 = v9 < v251;
if(v252) {
int64_t const v253 = v9 + v251;
uint64_t const v262 = uint64_t(v253);
float * const v263 = &(*v389)[v262];
float const v264 = *v55;
float const v265 = *v263;
float const v270 = *v60;
int64_t const v272 = v253 + v15;
uint64_t const v273 = uint64_t(v272);
float * const v274 = &(*v389)[v273];
float const v275 = *v274;
float const v276 = v264 + v265;
float const v277 = v276 - v264;
float const v278 = v276 - v277;
float const v279 = v264 - v278;
float const v280 = v265 - v277;
float const v281 = v279 + v280;
float const v282 = v270 + v281;
float const v283 = fabs(v276);
float const v284 = 3.402823470e+38;
bool const v285 = v283 <= v284;
float const v287 = v285 ? v282 : v19;
float const v288 = v287 + v275;
*v60 = v288;
*v55 = v276;
}
__syncthreads();
int64_t const v289 = 2;
bool const v290 = v9 < v289;
if(v290) {
int64_t const v291 = v9 + v289;
uint64_t const v300 = uint64_t(v291);
float * const v301 = &(*v389)[v300];
float const v302 = *v55;
float const v303 = *v301;
float const v308 = *v60;
int64_t const v310 = v291 + v15;
uint64_t const v311 = uint64_t(v310);
float * const v312 = &(*v389)[v311];
float const v313 = *v312;
float const v314 = v302 + v303;
float const v315 = v314 - v302;
float const v316 = v314 - v315;
float const v317 = v302 - v316;
float const v318 = v303 - v315;
float const v319 = v317 + v318;
float const v320 = v308 + v319;
float const v321 = fabs(v314);
float const v322 = 3.402823470e+38;
bool const v323 = v321 <= v322;
float const v325 = v323 ? v320 : v19;
float const v326 = v325 + v313;
*v60 = v326;
*v55 = v314;
}
__syncthreads();
bool const v328 = v9 < v11;
if(v328) {
int64_t const v329 = v9 + v11;
uint64_t const v338 = uint64_t(v329);
float * const v339 = &(*v389)[v338];
float const v340 = *v55;
float const v341 = *v339;
float const v346 = *v60;
int64_t const v348 = v329 + v15;
uint64_t const v349 = uint64_t(v348);
float * const v350 = &(*v389)[v349];
float const v351 = *v350;
float const v352 = v340 + v341;
float const v353 = v352 - v340;
float const v354 = v352 - v353;
float const v355 = v340 - v354;
float const v356 = v341 - v353;
float const v357 = v355 + v356;
float const v358 = v346 + v357;
float const v359 = fabs(v352);
float const v360 = 3.402823470e+38;
bool const v361 = v359 <= v360;
float const v363 = v361 ? v358 : v19;
float const v364 = v363 + v351;
*v60 = v364;
*v55 = v352;
}
__syncthreads();
uint64_t const v395 = 0;
float * const v366 = &(*v389)[v395];
float const v367 = *v366;
uint64_t const v393 = 256;
float * const v371 = &(*v389)[v393];
float const v372 = *v371;
float const v373 = v367 + v372;
__syncthreads();
bool const v374 = v9 == v10;
if(v374) {
float const v375 = float(v386);
float const v376 = v373 / v375;
float const v377 = 9.999999740e-06;
float const v378 = v376 + v377;
float const v379 = rsqrt(v378);uint64_t const v380 = uint64_t(v7);
float * const v381 = &v2[v380];
*v381 = v379;
}
}
return;
}
