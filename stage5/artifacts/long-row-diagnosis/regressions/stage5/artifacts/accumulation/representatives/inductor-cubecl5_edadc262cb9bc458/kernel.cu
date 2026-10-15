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
extern "C" __global__ void __launch_bounds__(256)  cubecl5_edadc262cb9bc458(float const* const __restrict__ v1, float * const __restrict__ v2, info_st const* const __restrict__ v387) {
extern __shared__ __align__(4) char v393[];uint64_t const* const dynamic_meta_v389 = reinterpret_cast<uint64_t const*>(v387 + 1);
const info_st& info_v388 = *v387;int64_t const v391 = info_v388.scalars_i64[1];
int64_t const v392 = info_v388.scalars_i64[2];
uint64_t const v6 = blockIdx.x;
int64_t const v7 = int64_t(v6);
uint64_t const v8 = threadIdx.x;
int64_t const v9 = int64_t(v8);
int64_t const v10 = 0;
int64_t const v11 = 1;
int64_t const v15 = 256;
int64_t const v17 = v11 + v391;
array<float, 512> * const v394 = reinterpret_cast<array<float, 512> *>(&v393[0]);
bool const v19 = v7 < v392;
if(v19) {
float v396_store;
float * v396 = &v396_store;float const v21 = 0.0;
*v396 = v21;
float v397_store;
float * v397 = &v397_store;*v397 = v21;
for(int64_t v27 = v9; v27 < v17; v27 += v15) {
bool const v28 = v27 < v391;
float v398_store;
float * v398 = &v398_store;if(v28) {
int64_t const v30 = v7 * v391;
int64_t const v31 = v30 + v27;
uint64_t const v32 = uint64_t(v31);
float const* const v33 = &v1[v32];
float const v34 = *v33;
*v398 = v34;
}
 else {
*v398 = v21;
}
float const v36 = *v398;
bool const v37 = v27 >= v391;
float v399_store;
float * v399 = &v399_store;if(v37) {
int64_t const v39 = v7 * v391;
uint64_t const v40 = uint64_t(v39);
float const* const v41 = &v1[v40];
float const v42 = *v41;
*v399 = v42;
}
 else {
*v399 = v21;
}
float const v44 = *v399;
float const v45 = v28 ? v36 : v44;
float const v46 = *v397;
float const v47 = *v396;
float const v48 = v46 + v45;
float const v49 = v48 - v46;
float const v50 = v48 - v49;
float const v51 = v46 - v50;
float const v52 = v45 - v49;
float const v53 = v51 + v52;
float const v54 = v47 + v53;
float const v55 = fabs(v48);
float const v56 = 3.402823470e+38;
bool const v57 = v55 <= v56;
float const v59 = v57 ? v54 : v21;
*v396 = v59;
*v397 = v48;
}
float const v60 = *v397;
uint64_t const v64 = uint64_t(v9);
float * const v65 = &(*v394)[v64];
*v65 = v60;
float const v66 = *v396;
int64_t const v68 = v9 + v15;
uint64_t const v69 = uint64_t(v68);
float * const v70 = &(*v394)[v69];
*v70 = v66;
__syncthreads();
int64_t const v71 = 128;
bool const v72 = v9 < v71;
if(v72) {
int64_t const v73 = v9 + v71;
uint64_t const v82 = uint64_t(v73);
float * const v83 = &(*v394)[v82];
float const v84 = *v65;
float const v85 = *v83;
float const v90 = *v70;
int64_t const v92 = v73 + v15;
uint64_t const v93 = uint64_t(v92);
float * const v94 = &(*v394)[v93];
float const v95 = *v94;
float const v96 = v84 + v85;
float const v97 = v96 - v84;
float const v98 = v96 - v97;
float const v99 = v84 - v98;
float const v100 = v85 - v97;
float const v101 = v99 + v100;
float const v102 = v90 + v101;
float const v103 = fabs(v96);
float const v104 = 3.402823470e+38;
bool const v105 = v103 <= v104;
float const v107 = v105 ? v102 : v21;
float const v108 = v107 + v95;
*v70 = v108;
*v65 = v96;
}
__syncthreads();
int64_t const v109 = 64;
bool const v110 = v9 < v109;
if(v110) {
int64_t const v111 = v9 + v109;
uint64_t const v120 = uint64_t(v111);
float * const v121 = &(*v394)[v120];
float const v122 = *v65;
float const v123 = *v121;
float const v128 = *v70;
int64_t const v130 = v111 + v15;
uint64_t const v131 = uint64_t(v130);
float * const v132 = &(*v394)[v131];
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
float const v145 = v143 ? v140 : v21;
float const v146 = v145 + v133;
*v70 = v146;
*v65 = v134;
}
__syncthreads();
int64_t const v147 = 32;
bool const v148 = v9 < v147;
if(v148) {
int64_t const v149 = v9 + v147;
uint64_t const v158 = uint64_t(v149);
float * const v159 = &(*v394)[v158];
float const v160 = *v65;
float const v161 = *v159;
float const v166 = *v70;
int64_t const v168 = v149 + v15;
uint64_t const v169 = uint64_t(v168);
float * const v170 = &(*v394)[v169];
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
float const v183 = v181 ? v178 : v21;
float const v184 = v183 + v171;
*v70 = v184;
*v65 = v172;
}
__syncthreads();
int64_t const v185 = 16;
bool const v186 = v9 < v185;
if(v186) {
int64_t const v187 = v9 + v185;
uint64_t const v196 = uint64_t(v187);
float * const v197 = &(*v394)[v196];
float const v198 = *v65;
float const v199 = *v197;
float const v204 = *v70;
int64_t const v206 = v187 + v15;
uint64_t const v207 = uint64_t(v206);
float * const v208 = &(*v394)[v207];
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
float const v221 = v219 ? v216 : v21;
float const v222 = v221 + v209;
*v70 = v222;
*v65 = v210;
}
__syncthreads();
int64_t const v223 = 8;
bool const v224 = v9 < v223;
if(v224) {
int64_t const v225 = v9 + v223;
uint64_t const v234 = uint64_t(v225);
float * const v235 = &(*v394)[v234];
float const v236 = *v65;
float const v237 = *v235;
float const v242 = *v70;
int64_t const v244 = v225 + v15;
uint64_t const v245 = uint64_t(v244);
float * const v246 = &(*v394)[v245];
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
float const v259 = v257 ? v254 : v21;
float const v260 = v259 + v247;
*v70 = v260;
*v65 = v248;
}
__syncthreads();
int64_t const v261 = 4;
bool const v262 = v9 < v261;
if(v262) {
int64_t const v263 = v9 + v261;
uint64_t const v272 = uint64_t(v263);
float * const v273 = &(*v394)[v272];
float const v274 = *v65;
float const v275 = *v273;
float const v280 = *v70;
int64_t const v282 = v263 + v15;
uint64_t const v283 = uint64_t(v282);
float * const v284 = &(*v394)[v283];
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
float const v297 = v295 ? v292 : v21;
float const v298 = v297 + v285;
*v70 = v298;
*v65 = v286;
}
__syncthreads();
int64_t const v299 = 2;
bool const v300 = v9 < v299;
if(v300) {
int64_t const v301 = v9 + v299;
uint64_t const v310 = uint64_t(v301);
float * const v311 = &(*v394)[v310];
float const v312 = *v65;
float const v313 = *v311;
float const v318 = *v70;
int64_t const v320 = v301 + v15;
uint64_t const v321 = uint64_t(v320);
float * const v322 = &(*v394)[v321];
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
float const v335 = v333 ? v330 : v21;
float const v336 = v335 + v323;
*v70 = v336;
*v65 = v324;
}
__syncthreads();
bool const v338 = v9 < v11;
if(v338) {
int64_t const v339 = v9 + v11;
uint64_t const v348 = uint64_t(v339);
float * const v349 = &(*v394)[v348];
float const v350 = *v65;
float const v351 = *v349;
float const v356 = *v70;
int64_t const v358 = v339 + v15;
uint64_t const v359 = uint64_t(v358);
float * const v360 = &(*v394)[v359];
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
float const v373 = v371 ? v368 : v21;
float const v374 = v373 + v361;
*v70 = v374;
*v65 = v362;
}
__syncthreads();
uint64_t const v402 = 0;
float * const v376 = &(*v394)[v402];
float const v377 = *v376;
uint64_t const v400 = 256;
float * const v381 = &(*v394)[v400];
float const v382 = *v381;
float const v383 = v377 + v382;
bool const v384 = v9 == v10;
if(v384) {
uint64_t const v385 = uint64_t(v7);
float * const v386 = &v2[v385];
*v386 = v383;
}
}
return;
}
