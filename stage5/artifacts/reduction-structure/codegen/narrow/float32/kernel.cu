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
extern "C" __global__ void __launch_bounds__(512)  cubecl5_42026f8357481503(float const* const __restrict__ v1, float * const __restrict__ v2, info_st const* const __restrict__ v314) {
extern __shared__ __align__(4) char v320[];uint64_t const* const dynamic_meta_v316 = reinterpret_cast<uint64_t const*>(v314 + 1);
const info_st& info_v315 = *v314;int64_t const v318 = info_v315.scalars_i64[1];
int64_t const v319 = info_v315.scalars_i64[2];
uint64_t const v6 = blockIdx.x;
int64_t const v7 = int64_t(v6);
uint64_t const v8 = threadIdx.x;
int64_t const v9 = int64_t(v8);
int64_t const v10 = 0;
int64_t const v11 = 1;
int64_t const v15 = 512;
int64_t const v16 = 9;
int64_t const v17 = v16 * v319;
int64_t const v18 = 8;
int64_t const v19 = v18 + v318;
int64_t const v21 = v19 / v16;
int64_t const v22 = v19 % v16;
bool const v25 = v22 < v10;
int64_t const v26 = v21 - v11;
int64_t const v27 = v25 ? v26 : v21;
array<float, 16> * const v321 = reinterpret_cast<array<float, 16> *>(&v320[0]);
bool const v29 = v7 < v17;
if(v29) {
float v323_store;
float * v323 = &v323_store;float const v31 = 0.0;
*v323 = v31;
float v324_store;
float * v324 = &v324_store;*v324 = v31;
float v325_store;
float * v325 = &v325_store;*v325 = v31;
float v326_store;
float * v326 = &v326_store;*v326 = v31;
uint32_t const v40 = uint32_t(v9);
uint32_t const v41 = uint32_t(v27);
uint32_t const v401 = 2048;
for(uint32_t v43 = v40; v43 < v41; v43 += v401) {
int64_t const v44 = int64_t(v43);
bool const v47 = v44 < v27;
float v327_store;
float * v327 = &v327_store;*v327 = v31;
if(v47) {
int64_t const v328 = v7 % v16;
int64_t const v330 = v328 ^ v16;
bool const v331 = v330 < v10;
bool const v332 = v328 != v10;
bool const v333 = v332 && v331;
int64_t const v334 = v333 ? v16 : v10;
int64_t const v335 = v328 + v334;
int64_t const v62 = v27 * v335;
int64_t const v63 = v44 + v62;
bool const v64 = v63 < v318;
float v336_store;
float * v336 = &v336_store;if(v64) {
int64_t const v67 = v7 / v16;
bool const v71 = v328 < v10;
int64_t const v72 = v67 - v11;
int64_t const v73 = v71 ? v72 : v67;
int64_t const v74 = v318 * v73;
int64_t const v75 = v44 + v74;
int64_t const v89 = v75 + v62;
uint64_t const v90 = uint64_t(v89);
float const* const v91 = &v1[v90];
float const v92 = *v91;
*v336 = v92;
}
 else {
*v336 = v31;
}
float const v94 = *v336;
*v327 = v94;
}
int64_t const v96 = v44 + v15;
bool const v97 = v96 < v27;
float v345_store;
float * v345 = &v345_store;*v345 = v31;
if(v97) {
int64_t const v346 = v7 % v16;
int64_t const v348 = v346 ^ v16;
bool const v349 = v348 < v10;
bool const v350 = v346 != v10;
bool const v351 = v350 && v349;
int64_t const v352 = v351 ? v16 : v10;
int64_t const v353 = v346 + v352;
int64_t const v112 = v27 * v353;
int64_t const v113 = v96 + v112;
bool const v114 = v113 < v318;
float v354_store;
float * v354 = &v354_store;if(v114) {
int64_t const v117 = v7 / v16;
bool const v121 = v346 < v10;
int64_t const v122 = v117 - v11;
int64_t const v123 = v121 ? v122 : v117;
int64_t const v124 = v318 * v123;
int64_t const v125 = v96 + v124;
int64_t const v139 = v125 + v112;
uint64_t const v140 = uint64_t(v139);
float const* const v141 = &v1[v140];
float const v142 = *v141;
*v354 = v142;
}
 else {
*v354 = v31;
}
float const v144 = *v354;
*v345 = v144;
}
int64_t const v145 = 1024;
int64_t const v146 = v44 + v145;
bool const v147 = v146 < v27;
float v363_store;
float * v363 = &v363_store;*v363 = v31;
if(v147) {
int64_t const v364 = v7 % v16;
int64_t const v366 = v364 ^ v16;
bool const v367 = v366 < v10;
bool const v368 = v364 != v10;
bool const v369 = v368 && v367;
int64_t const v370 = v369 ? v16 : v10;
int64_t const v371 = v364 + v370;
int64_t const v162 = v27 * v371;
int64_t const v163 = v146 + v162;
bool const v164 = v163 < v318;
float v372_store;
float * v372 = &v372_store;if(v164) {
int64_t const v167 = v7 / v16;
bool const v171 = v364 < v10;
int64_t const v172 = v167 - v11;
int64_t const v173 = v171 ? v172 : v167;
int64_t const v174 = v318 * v173;
int64_t const v175 = v146 + v174;
int64_t const v189 = v175 + v162;
uint64_t const v190 = uint64_t(v189);
float const* const v191 = &v1[v190];
float const v192 = *v191;
*v372 = v192;
}
 else {
*v372 = v31;
}
float const v194 = *v372;
*v363 = v194;
}
int64_t const v195 = 1536;
int64_t const v196 = v44 + v195;
bool const v197 = v196 < v27;
float v381_store;
float * v381 = &v381_store;*v381 = v31;
if(v197) {
int64_t const v382 = v7 % v16;
int64_t const v384 = v382 ^ v16;
bool const v385 = v384 < v10;
bool const v386 = v382 != v10;
bool const v387 = v386 && v385;
int64_t const v388 = v387 ? v16 : v10;
int64_t const v389 = v382 + v388;
int64_t const v212 = v27 * v389;
int64_t const v213 = v196 + v212;
bool const v214 = v213 < v318;
float v390_store;
float * v390 = &v390_store;if(v214) {
int64_t const v217 = v7 / v16;
bool const v221 = v382 < v10;
int64_t const v222 = v217 - v11;
int64_t const v223 = v221 ? v222 : v217;
int64_t const v224 = v318 * v223;
int64_t const v225 = v196 + v224;
int64_t const v239 = v225 + v212;
uint64_t const v240 = uint64_t(v239);
float const* const v241 = &v1[v240];
float const v242 = *v241;
*v390 = v242;
}
 else {
*v390 = v31;
}
float const v244 = *v390;
*v381 = v244;
}
float const v245 = *v327;
float const v246 = *v323;
float const v247 = v246 + v245;
*v323 = v247;
float const v248 = *v345;
float const v249 = *v324;
float const v250 = v249 + v248;
*v324 = v250;
float const v251 = *v363;
float const v252 = *v325;
float const v253 = v252 + v251;
*v325 = v253;
float const v254 = *v381;
float const v255 = *v326;
float const v256 = v255 + v254;
*v326 = v256;
}
float const v257 = *v323;
float const v258 = *v324;
float const v259 = *v325;
float const v260 = *v326;
float const v261 = v257 + v258;
float const v262 = v259 + v260;
float const v263 = v261 + v262;
int32_t const v264 = 16;
float const v265 = __shfl_xor_sync(__activemask(), v263, v264);
float const v266 = v263 + v265;
int32_t const v267 = 8;
float const v268 = __shfl_xor_sync(__activemask(), v266, v267);
float const v269 = v266 + v268;
int32_t const v270 = 4;
float const v271 = __shfl_xor_sync(__activemask(), v269, v270);
float const v272 = v269 + v271;
int32_t const v273 = 2;
float const v274 = __shfl_xor_sync(__activemask(), v272, v273);
float const v275 = v272 + v274;
int32_t const v276 = 1;
float const v277 = __shfl_xor_sync(__activemask(), v275, v276);
float const v278 = v275 + v277;
int64_t const v279 = 32;
int64_t const v280 = v9 / v279;
int64_t const v281 = v9 % v279;
bool const v282 = v281 == v10;
if(v282) {
uint64_t const v283 = uint64_t(v280);
float * const v284 = &(*v321)[v283];
*v284 = v278;
}
__syncthreads();
bool const v285 = v280 == v10;
if(v285) {
float v399_store;
float * v399 = &v399_store;*v399 = v31;
int64_t const v288 = 16;
bool const v289 = v281 < v288;
if(v289) {
uint64_t const v290 = uint64_t(v281);
float * const v291 = &(*v321)[v290];
float const v292 = *v291;
*v399 = v292;
}
float const v293 = *v399;
float const v295 = __shfl_xor_sync(__activemask(), v293, v267);
float const v296 = v293 + v295;
float const v298 = __shfl_xor_sync(__activemask(), v296, v270);
float const v299 = v296 + v298;
float const v301 = __shfl_xor_sync(__activemask(), v299, v273);
float const v302 = v299 + v301;
float const v304 = __shfl_xor_sync(__activemask(), v302, v276);
float const v305 = v302 + v304;
if(v282) {
uint64_t const v403 = 0;
float * const v307 = &(*v321)[v403];
*v307 = v305;
}
}
__syncthreads();
uint64_t const v400 = 0;
float * const v309 = &(*v321)[v400];
float const v310 = *v309;
bool const v311 = v9 == v10;
if(v311) {
uint64_t const v312 = uint64_t(v7);
float * const v313 = &v2[v312];
*v313 = v310;
}
}
return;
}
