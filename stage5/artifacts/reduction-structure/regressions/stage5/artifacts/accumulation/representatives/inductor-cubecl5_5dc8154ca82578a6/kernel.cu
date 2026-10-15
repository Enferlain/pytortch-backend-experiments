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
extern "C" __global__ void __launch_bounds__(256)  cubecl5_5dc8154ca82578a6(float const* const __restrict__ v1, float * const __restrict__ v2, info_st const* const __restrict__ v424) {
extern __shared__ __align__(4) char v430[];uint64_t const* const dynamic_meta_v426 = reinterpret_cast<uint64_t const*>(v424 + 1);
const info_st& info_v425 = *v424;int64_t const v428 = info_v425.scalars_i64[1];
int64_t const v429 = info_v425.scalars_i64[2];
uint64_t const v6 = blockIdx.x;
int64_t const v7 = int64_t(v6);
uint64_t const v8 = threadIdx.x;
int64_t const v9 = int64_t(v8);
int64_t const v10 = 0;
int64_t const v11 = 1;
int64_t const v15 = 256;
int64_t const v16 = 9;
int64_t const v17 = v16 * v429;
int64_t const v18 = 8;
int64_t const v19 = v18 + v428;
int64_t const v21 = v19 / v16;
int64_t const v22 = v19 % v16;
bool const v25 = v22 < v10;
int64_t const v26 = v21 - v11;
int64_t const v27 = v25 ? v26 : v21;
array<float, 512> * const v431 = reinterpret_cast<array<float, 512> *>(&v430[0]);
bool const v29 = v7 < v17;
if(v29) {
float v433_store;
float * v433 = &v433_store;float const v31 = 0.0;
*v433 = v31;
float v434_store;
float * v434 = &v434_store;*v434 = v31;
for(int64_t v37 = v9; v37 < v27; v37 += v15) {
int64_t const v435 = v7 % v16;
int64_t const v437 = v435 ^ v16;
bool const v438 = v437 < v10;
bool const v439 = v435 != v10;
bool const v440 = v439 && v438;
int64_t const v441 = v440 ? v16 : v10;
int64_t const v442 = v435 + v441;
int64_t const v50 = v27 * v442;
int64_t const v51 = v37 + v50;
bool const v52 = v51 < v428;
float v443_store;
float * v443 = &v443_store;if(v52) {
int64_t const v55 = v7 / v16;
bool const v59 = v435 < v10;
int64_t const v60 = v55 - v11;
int64_t const v61 = v59 ? v60 : v55;
int64_t const v62 = v428 * v61;
int64_t const v63 = v37 + v62;
int64_t const v77 = v63 + v50;
uint64_t const v78 = uint64_t(v77);
float const* const v79 = &v1[v78];
float const v80 = *v79;
*v443 = v80;
}
 else {
*v443 = v31;
}
float const v82 = *v443;
float const v83 = *v434;
float const v84 = *v433;
float const v85 = v83 + v82;
float const v86 = v85 - v83;
float const v87 = v85 - v86;
float const v88 = v83 - v87;
float const v89 = v82 - v86;
float const v90 = v88 + v89;
float const v91 = v84 + v90;
float const v92 = fabs(v85);
float const v93 = 3.402823470e+38;
bool const v94 = v92 <= v93;
float const v96 = v94 ? v91 : v31;
*v433 = v96;
*v434 = v85;
}
float const v97 = *v434;
uint64_t const v101 = uint64_t(v9);
float * const v102 = &(*v431)[v101];
*v102 = v97;
float const v103 = *v433;
int64_t const v105 = v9 + v15;
uint64_t const v106 = uint64_t(v105);
float * const v107 = &(*v431)[v106];
*v107 = v103;
__syncthreads();
int64_t const v108 = 128;
bool const v109 = v9 < v108;
if(v109) {
int64_t const v110 = v9 + v108;
uint64_t const v119 = uint64_t(v110);
float * const v120 = &(*v431)[v119];
float const v121 = *v102;
float const v122 = *v120;
float const v127 = *v107;
int64_t const v129 = v110 + v15;
uint64_t const v130 = uint64_t(v129);
float * const v131 = &(*v431)[v130];
float const v132 = *v131;
float const v133 = v121 + v122;
float const v134 = v133 - v121;
float const v135 = v133 - v134;
float const v136 = v121 - v135;
float const v137 = v122 - v134;
float const v138 = v136 + v137;
float const v139 = v127 + v138;
float const v140 = fabs(v133);
float const v141 = 3.402823470e+38;
bool const v142 = v140 <= v141;
float const v144 = v142 ? v139 : v31;
float const v145 = v144 + v132;
*v107 = v145;
*v102 = v133;
}
__syncthreads();
int64_t const v146 = 64;
bool const v147 = v9 < v146;
if(v147) {
int64_t const v148 = v9 + v146;
uint64_t const v157 = uint64_t(v148);
float * const v158 = &(*v431)[v157];
float const v159 = *v102;
float const v160 = *v158;
float const v165 = *v107;
int64_t const v167 = v148 + v15;
uint64_t const v168 = uint64_t(v167);
float * const v169 = &(*v431)[v168];
float const v170 = *v169;
float const v171 = v159 + v160;
float const v172 = v171 - v159;
float const v173 = v171 - v172;
float const v174 = v159 - v173;
float const v175 = v160 - v172;
float const v176 = v174 + v175;
float const v177 = v165 + v176;
float const v178 = fabs(v171);
float const v179 = 3.402823470e+38;
bool const v180 = v178 <= v179;
float const v182 = v180 ? v177 : v31;
float const v183 = v182 + v170;
*v107 = v183;
*v102 = v171;
}
__syncthreads();
int64_t const v184 = 32;
bool const v185 = v9 < v184;
if(v185) {
int64_t const v186 = v9 + v184;
uint64_t const v195 = uint64_t(v186);
float * const v196 = &(*v431)[v195];
float const v197 = *v102;
float const v198 = *v196;
float const v203 = *v107;
int64_t const v205 = v186 + v15;
uint64_t const v206 = uint64_t(v205);
float * const v207 = &(*v431)[v206];
float const v208 = *v207;
float const v209 = v197 + v198;
float const v210 = v209 - v197;
float const v211 = v209 - v210;
float const v212 = v197 - v211;
float const v213 = v198 - v210;
float const v214 = v212 + v213;
float const v215 = v203 + v214;
float const v216 = fabs(v209);
float const v217 = 3.402823470e+38;
bool const v218 = v216 <= v217;
float const v220 = v218 ? v215 : v31;
float const v221 = v220 + v208;
*v107 = v221;
*v102 = v209;
}
__syncthreads();
int64_t const v222 = 16;
bool const v223 = v9 < v222;
if(v223) {
int64_t const v224 = v9 + v222;
uint64_t const v233 = uint64_t(v224);
float * const v234 = &(*v431)[v233];
float const v235 = *v102;
float const v236 = *v234;
float const v241 = *v107;
int64_t const v243 = v224 + v15;
uint64_t const v244 = uint64_t(v243);
float * const v245 = &(*v431)[v244];
float const v246 = *v245;
float const v247 = v235 + v236;
float const v248 = v247 - v235;
float const v249 = v247 - v248;
float const v250 = v235 - v249;
float const v251 = v236 - v248;
float const v252 = v250 + v251;
float const v253 = v241 + v252;
float const v254 = fabs(v247);
float const v255 = 3.402823470e+38;
bool const v256 = v254 <= v255;
float const v258 = v256 ? v253 : v31;
float const v259 = v258 + v246;
*v107 = v259;
*v102 = v247;
}
__syncthreads();
bool const v261 = v9 < v18;
if(v261) {
int64_t const v262 = v9 + v18;
uint64_t const v271 = uint64_t(v262);
float * const v272 = &(*v431)[v271];
float const v273 = *v102;
float const v274 = *v272;
float const v279 = *v107;
int64_t const v281 = v262 + v15;
uint64_t const v282 = uint64_t(v281);
float * const v283 = &(*v431)[v282];
float const v284 = *v283;
float const v285 = v273 + v274;
float const v286 = v285 - v273;
float const v287 = v285 - v286;
float const v288 = v273 - v287;
float const v289 = v274 - v286;
float const v290 = v288 + v289;
float const v291 = v279 + v290;
float const v292 = fabs(v285);
float const v293 = 3.402823470e+38;
bool const v294 = v292 <= v293;
float const v296 = v294 ? v291 : v31;
float const v297 = v296 + v284;
*v107 = v297;
*v102 = v285;
}
__syncthreads();
int64_t const v298 = 4;
bool const v299 = v9 < v298;
if(v299) {
int64_t const v300 = v9 + v298;
uint64_t const v309 = uint64_t(v300);
float * const v310 = &(*v431)[v309];
float const v311 = *v102;
float const v312 = *v310;
float const v317 = *v107;
int64_t const v319 = v300 + v15;
uint64_t const v320 = uint64_t(v319);
float * const v321 = &(*v431)[v320];
float const v322 = *v321;
float const v323 = v311 + v312;
float const v324 = v323 - v311;
float const v325 = v323 - v324;
float const v326 = v311 - v325;
float const v327 = v312 - v324;
float const v328 = v326 + v327;
float const v329 = v317 + v328;
float const v330 = fabs(v323);
float const v331 = 3.402823470e+38;
bool const v332 = v330 <= v331;
float const v334 = v332 ? v329 : v31;
float const v335 = v334 + v322;
*v107 = v335;
*v102 = v323;
}
__syncthreads();
int64_t const v336 = 2;
bool const v337 = v9 < v336;
if(v337) {
int64_t const v338 = v9 + v336;
uint64_t const v347 = uint64_t(v338);
float * const v348 = &(*v431)[v347];
float const v349 = *v102;
float const v350 = *v348;
float const v355 = *v107;
int64_t const v357 = v338 + v15;
uint64_t const v358 = uint64_t(v357);
float * const v359 = &(*v431)[v358];
float const v360 = *v359;
float const v361 = v349 + v350;
float const v362 = v361 - v349;
float const v363 = v361 - v362;
float const v364 = v349 - v363;
float const v365 = v350 - v362;
float const v366 = v364 + v365;
float const v367 = v355 + v366;
float const v368 = fabs(v361);
float const v369 = 3.402823470e+38;
bool const v370 = v368 <= v369;
float const v372 = v370 ? v367 : v31;
float const v373 = v372 + v360;
*v107 = v373;
*v102 = v361;
}
__syncthreads();
bool const v375 = v9 < v11;
if(v375) {
int64_t const v376 = v9 + v11;
uint64_t const v385 = uint64_t(v376);
float * const v386 = &(*v431)[v385];
float const v387 = *v102;
float const v388 = *v386;
float const v393 = *v107;
int64_t const v395 = v376 + v15;
uint64_t const v396 = uint64_t(v395);
float * const v397 = &(*v431)[v396];
float const v398 = *v397;
float const v399 = v387 + v388;
float const v400 = v399 - v387;
float const v401 = v399 - v400;
float const v402 = v387 - v401;
float const v403 = v388 - v400;
float const v404 = v402 + v403;
float const v405 = v393 + v404;
float const v406 = fabs(v399);
float const v407 = 3.402823470e+38;
bool const v408 = v406 <= v407;
float const v410 = v408 ? v405 : v31;
float const v411 = v410 + v398;
*v107 = v411;
*v102 = v399;
}
__syncthreads();
uint64_t const v454 = 0;
float * const v413 = &(*v431)[v454];
float const v414 = *v413;
uint64_t const v452 = 256;
float * const v418 = &(*v431)[v452];
float const v419 = *v418;
float const v420 = v414 + v419;
bool const v421 = v9 == v10;
if(v421) {
uint64_t const v422 = uint64_t(v7);
float * const v423 = &v2[v422];
*v423 = v420;
}
}
return;
}
