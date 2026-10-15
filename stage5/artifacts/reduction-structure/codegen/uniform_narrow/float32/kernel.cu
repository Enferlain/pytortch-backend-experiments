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
extern "C" __global__ void __launch_bounds__(512)  cubecl5_42026f8357481503(float const* const __restrict__ v1, float * const __restrict__ v2, info_st const* const __restrict__ v316) {
extern __shared__ __align__(4) char v322[];uint64_t const* const dynamic_meta_v318 = reinterpret_cast<uint64_t const*>(v316 + 1);
const info_st& info_v317 = *v316;int64_t const v320 = info_v317.scalars_i64[1];
int64_t const v321 = info_v317.scalars_i64[2];
uint64_t const v6 = blockIdx.x;
int64_t const v7 = int64_t(v6);
uint64_t const v8 = threadIdx.x;
int64_t const v9 = int64_t(v8);
int64_t const v10 = 0;
int64_t const v11 = 1;
int64_t const v15 = 512;
int64_t const v16 = 9;
int64_t const v17 = v16 * v321;
int64_t const v18 = 8;
int64_t const v19 = v18 + v320;
int64_t const v21 = v19 / v16;
int64_t const v22 = v19 % v16;
bool const v25 = v22 < v10;
int64_t const v26 = v21 - v11;
int64_t const v27 = v25 ? v26 : v21;
array<float, 16> * const v323 = reinterpret_cast<array<float, 16> *>(&v322[0]);
bool const v29 = v7 < v17;
if(v29) {
float v325_store;
float * v325 = &v325_store;float const v31 = 0.0;
*v325 = v31;
float v326_store;
float * v326 = &v326_store;*v326 = v31;
float v327_store;
float * v327 = &v327_store;*v327 = v31;
float v328_store;
float * v328 = &v328_store;*v328 = v31;
uint32_t const v404 = 0;
uint32_t const v41 = uint32_t(v27);
uint32_t const v403 = 2048;
for(uint32_t v43 = v404; v43 < v41; v43 += v403) {
int64_t const v44 = int64_t(v43);
int64_t const v45 = v44 + v9;
bool const v48 = v45 < v27;
float v329_store;
float * v329 = &v329_store;*v329 = v31;
if(v48) {
int64_t const v330 = v7 % v16;
int64_t const v332 = v330 ^ v16;
bool const v333 = v332 < v10;
bool const v334 = v330 != v10;
bool const v335 = v334 && v333;
int64_t const v336 = v335 ? v16 : v10;
int64_t const v337 = v330 + v336;
int64_t const v63 = v27 * v337;
int64_t const v64 = v45 + v63;
bool const v65 = v64 < v320;
float v338_store;
float * v338 = &v338_store;if(v65) {
int64_t const v68 = v7 / v16;
bool const v72 = v330 < v10;
int64_t const v73 = v68 - v11;
int64_t const v74 = v72 ? v73 : v68;
int64_t const v75 = v320 * v74;
int64_t const v76 = v45 + v75;
int64_t const v90 = v76 + v63;
uint64_t const v91 = uint64_t(v90);
float const* const v92 = &v1[v91];
float const v93 = *v92;
*v338 = v93;
}
 else {
*v338 = v31;
}
float const v95 = *v338;
*v329 = v95;
}
int64_t const v97 = v45 + v15;
bool const v98 = v97 < v27;
float v347_store;
float * v347 = &v347_store;*v347 = v31;
if(v98) {
int64_t const v348 = v7 % v16;
int64_t const v350 = v348 ^ v16;
bool const v351 = v350 < v10;
bool const v352 = v348 != v10;
bool const v353 = v352 && v351;
int64_t const v354 = v353 ? v16 : v10;
int64_t const v355 = v348 + v354;
int64_t const v113 = v27 * v355;
int64_t const v114 = v97 + v113;
bool const v115 = v114 < v320;
float v356_store;
float * v356 = &v356_store;if(v115) {
int64_t const v118 = v7 / v16;
bool const v122 = v348 < v10;
int64_t const v123 = v118 - v11;
int64_t const v124 = v122 ? v123 : v118;
int64_t const v125 = v320 * v124;
int64_t const v126 = v97 + v125;
int64_t const v140 = v126 + v113;
uint64_t const v141 = uint64_t(v140);
float const* const v142 = &v1[v141];
float const v143 = *v142;
*v356 = v143;
}
 else {
*v356 = v31;
}
float const v145 = *v356;
*v347 = v145;
}
int64_t const v146 = 1024;
int64_t const v147 = v45 + v146;
bool const v148 = v147 < v27;
float v365_store;
float * v365 = &v365_store;*v365 = v31;
if(v148) {
int64_t const v366 = v7 % v16;
int64_t const v368 = v366 ^ v16;
bool const v369 = v368 < v10;
bool const v370 = v366 != v10;
bool const v371 = v370 && v369;
int64_t const v372 = v371 ? v16 : v10;
int64_t const v373 = v366 + v372;
int64_t const v163 = v27 * v373;
int64_t const v164 = v147 + v163;
bool const v165 = v164 < v320;
float v374_store;
float * v374 = &v374_store;if(v165) {
int64_t const v168 = v7 / v16;
bool const v172 = v366 < v10;
int64_t const v173 = v168 - v11;
int64_t const v174 = v172 ? v173 : v168;
int64_t const v175 = v320 * v174;
int64_t const v176 = v147 + v175;
int64_t const v190 = v176 + v163;
uint64_t const v191 = uint64_t(v190);
float const* const v192 = &v1[v191];
float const v193 = *v192;
*v374 = v193;
}
 else {
*v374 = v31;
}
float const v195 = *v374;
*v365 = v195;
}
int64_t const v196 = 1536;
int64_t const v197 = v45 + v196;
bool const v198 = v197 < v27;
float v383_store;
float * v383 = &v383_store;*v383 = v31;
if(v198) {
int64_t const v384 = v7 % v16;
int64_t const v386 = v384 ^ v16;
bool const v387 = v386 < v10;
bool const v388 = v384 != v10;
bool const v389 = v388 && v387;
int64_t const v390 = v389 ? v16 : v10;
int64_t const v391 = v384 + v390;
int64_t const v213 = v27 * v391;
int64_t const v214 = v197 + v213;
bool const v215 = v214 < v320;
float v392_store;
float * v392 = &v392_store;if(v215) {
int64_t const v218 = v7 / v16;
bool const v222 = v384 < v10;
int64_t const v223 = v218 - v11;
int64_t const v224 = v222 ? v223 : v218;
int64_t const v225 = v320 * v224;
int64_t const v226 = v197 + v225;
int64_t const v240 = v226 + v213;
uint64_t const v241 = uint64_t(v240);
float const* const v242 = &v1[v241];
float const v243 = *v242;
*v392 = v243;
}
 else {
*v392 = v31;
}
float const v245 = *v392;
*v383 = v245;
}
if(v48) {
float const v247 = *v329;
float const v248 = *v325;
float const v249 = v248 + v247;
*v325 = v249;
float const v250 = *v347;
float const v251 = *v326;
float const v252 = v251 + v250;
*v326 = v252;
float const v253 = *v365;
float const v254 = *v327;
float const v255 = v254 + v253;
*v327 = v255;
float const v256 = *v383;
float const v257 = *v328;
float const v258 = v257 + v256;
*v328 = v258;
}
}
float const v259 = *v325;
float const v260 = *v326;
float const v261 = *v327;
float const v262 = *v328;
float const v263 = v259 + v260;
float const v264 = v261 + v262;
float const v265 = v263 + v264;
int32_t const v266 = 16;
float const v267 = __shfl_xor_sync(__activemask(), v265, v266);
float const v268 = v265 + v267;
int32_t const v269 = 8;
float const v270 = __shfl_xor_sync(__activemask(), v268, v269);
float const v271 = v268 + v270;
int32_t const v272 = 4;
float const v273 = __shfl_xor_sync(__activemask(), v271, v272);
float const v274 = v271 + v273;
int32_t const v275 = 2;
float const v276 = __shfl_xor_sync(__activemask(), v274, v275);
float const v277 = v274 + v276;
int32_t const v278 = 1;
float const v279 = __shfl_xor_sync(__activemask(), v277, v278);
float const v280 = v277 + v279;
int64_t const v281 = 32;
int64_t const v282 = v9 / v281;
int64_t const v283 = v9 % v281;
bool const v284 = v283 == v10;
if(v284) {
uint64_t const v285 = uint64_t(v282);
float * const v286 = &(*v323)[v285];
*v286 = v280;
}
__syncthreads();
bool const v287 = v282 == v10;
if(v287) {
float v401_store;
float * v401 = &v401_store;*v401 = v31;
int64_t const v290 = 16;
bool const v291 = v283 < v290;
if(v291) {
uint64_t const v292 = uint64_t(v283);
float * const v293 = &(*v323)[v292];
float const v294 = *v293;
*v401 = v294;
}
float const v295 = *v401;
float const v297 = __shfl_xor_sync(__activemask(), v295, v269);
float const v298 = v295 + v297;
float const v300 = __shfl_xor_sync(__activemask(), v298, v272);
float const v301 = v298 + v300;
float const v303 = __shfl_xor_sync(__activemask(), v301, v275);
float const v304 = v301 + v303;
float const v306 = __shfl_xor_sync(__activemask(), v304, v278);
float const v307 = v304 + v306;
if(v284) {
uint64_t const v406 = 0;
float * const v309 = &(*v323)[v406];
*v309 = v307;
}
}
__syncthreads();
uint64_t const v402 = 0;
float * const v311 = &(*v323)[v402];
float const v312 = *v311;
bool const v313 = v9 == v10;
if(v313) {
uint64_t const v314 = uint64_t(v7);
float * const v315 = &v2[v314];
*v315 = v312;
}
}
return;
}
