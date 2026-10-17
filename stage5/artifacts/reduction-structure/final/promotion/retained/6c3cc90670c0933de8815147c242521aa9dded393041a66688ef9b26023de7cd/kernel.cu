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
extern "C" __global__ void __launch_bounds__(512)  cubecl5_42026f8357481503(float const* const __restrict__ v1, float * const __restrict__ v2, info_st const* const __restrict__ v312) {
extern __shared__ __align__(4) char v318[];uint64_t const* const dynamic_meta_v314 = reinterpret_cast<uint64_t const*>(v312 + 1);
const info_st& info_v313 = *v312;int64_t const v316 = info_v313.scalars_i64[1];
int64_t const v317 = info_v313.scalars_i64[2];
uint64_t const v6 = blockIdx.x;
int64_t const v7 = int64_t(v6);
uint64_t const v8 = threadIdx.x;
int64_t const v9 = int64_t(v8);
int64_t const v10 = 0;
int64_t const v11 = 1;
int64_t const v15 = 512;
int64_t const v16 = 9;
int64_t const v17 = v16 * v317;
int64_t const v18 = 8;
int64_t const v19 = v18 + v316;
int64_t const v21 = v19 / v16;
int64_t const v22 = v19 % v16;
bool const v25 = v22 < v10;
int64_t const v26 = v21 - v11;
int64_t const v27 = v25 ? v26 : v21;
array<float, 16> * const v319 = reinterpret_cast<array<float, 16> *>(&v318[0]);
bool const v29 = v7 < v17;
if(v29) {
float v321_store;
float * v321 = &v321_store;float const v31 = 0.0;
*v321 = v31;
float v322_store;
float * v322 = &v322_store;*v322 = v31;
float v323_store;
float * v323 = &v323_store;*v323 = v31;
float v324_store;
float * v324 = &v324_store;*v324 = v31;
int64_t const v399 = 2048;
for(int64_t v40 = v10; v40 < v27; v40 += v399) {
int64_t const v41 = v40 + v9;
bool const v44 = v41 < v27;
float v325_store;
float * v325 = &v325_store;*v325 = v31;
if(v44) {
int64_t const v326 = v7 % v16;
int64_t const v328 = v326 ^ v16;
bool const v329 = v328 < v10;
bool const v330 = v326 != v10;
bool const v331 = v330 && v329;
int64_t const v332 = v331 ? v16 : v10;
int64_t const v333 = v326 + v332;
int64_t const v59 = v27 * v333;
int64_t const v60 = v41 + v59;
bool const v61 = v60 < v316;
float v334_store;
float * v334 = &v334_store;if(v61) {
int64_t const v64 = v7 / v16;
bool const v68 = v326 < v10;
int64_t const v69 = v64 - v11;
int64_t const v70 = v68 ? v69 : v64;
int64_t const v71 = v316 * v70;
int64_t const v72 = v41 + v71;
int64_t const v86 = v72 + v59;
uint64_t const v87 = uint64_t(v86);
float const* const v88 = &v1[v87];
float const v89 = *v88;
*v334 = v89;
}
 else {
*v334 = v31;
}
float const v91 = *v334;
*v325 = v91;
}
int64_t const v93 = v41 + v15;
bool const v94 = v93 < v27;
float v343_store;
float * v343 = &v343_store;*v343 = v31;
if(v94) {
int64_t const v344 = v7 % v16;
int64_t const v346 = v344 ^ v16;
bool const v347 = v346 < v10;
bool const v348 = v344 != v10;
bool const v349 = v348 && v347;
int64_t const v350 = v349 ? v16 : v10;
int64_t const v351 = v344 + v350;
int64_t const v109 = v27 * v351;
int64_t const v110 = v93 + v109;
bool const v111 = v110 < v316;
float v352_store;
float * v352 = &v352_store;if(v111) {
int64_t const v114 = v7 / v16;
bool const v118 = v344 < v10;
int64_t const v119 = v114 - v11;
int64_t const v120 = v118 ? v119 : v114;
int64_t const v121 = v316 * v120;
int64_t const v122 = v93 + v121;
int64_t const v136 = v122 + v109;
uint64_t const v137 = uint64_t(v136);
float const* const v138 = &v1[v137];
float const v139 = *v138;
*v352 = v139;
}
 else {
*v352 = v31;
}
float const v141 = *v352;
*v343 = v141;
}
int64_t const v142 = 1024;
int64_t const v143 = v41 + v142;
bool const v144 = v143 < v27;
float v361_store;
float * v361 = &v361_store;*v361 = v31;
if(v144) {
int64_t const v362 = v7 % v16;
int64_t const v364 = v362 ^ v16;
bool const v365 = v364 < v10;
bool const v366 = v362 != v10;
bool const v367 = v366 && v365;
int64_t const v368 = v367 ? v16 : v10;
int64_t const v369 = v362 + v368;
int64_t const v159 = v27 * v369;
int64_t const v160 = v143 + v159;
bool const v161 = v160 < v316;
float v370_store;
float * v370 = &v370_store;if(v161) {
int64_t const v164 = v7 / v16;
bool const v168 = v362 < v10;
int64_t const v169 = v164 - v11;
int64_t const v170 = v168 ? v169 : v164;
int64_t const v171 = v316 * v170;
int64_t const v172 = v143 + v171;
int64_t const v186 = v172 + v159;
uint64_t const v187 = uint64_t(v186);
float const* const v188 = &v1[v187];
float const v189 = *v188;
*v370 = v189;
}
 else {
*v370 = v31;
}
float const v191 = *v370;
*v361 = v191;
}
int64_t const v192 = 1536;
int64_t const v193 = v41 + v192;
bool const v194 = v193 < v27;
float v379_store;
float * v379 = &v379_store;*v379 = v31;
if(v194) {
int64_t const v380 = v7 % v16;
int64_t const v382 = v380 ^ v16;
bool const v383 = v382 < v10;
bool const v384 = v380 != v10;
bool const v385 = v384 && v383;
int64_t const v386 = v385 ? v16 : v10;
int64_t const v387 = v380 + v386;
int64_t const v209 = v27 * v387;
int64_t const v210 = v193 + v209;
bool const v211 = v210 < v316;
float v388_store;
float * v388 = &v388_store;if(v211) {
int64_t const v214 = v7 / v16;
bool const v218 = v380 < v10;
int64_t const v219 = v214 - v11;
int64_t const v220 = v218 ? v219 : v214;
int64_t const v221 = v316 * v220;
int64_t const v222 = v193 + v221;
int64_t const v236 = v222 + v209;
uint64_t const v237 = uint64_t(v236);
float const* const v238 = &v1[v237];
float const v239 = *v238;
*v388 = v239;
}
 else {
*v388 = v31;
}
float const v241 = *v388;
*v379 = v241;
}
if(v44) {
float const v243 = *v325;
float const v244 = *v321;
float const v245 = v244 + v243;
*v321 = v245;
float const v246 = *v343;
float const v247 = *v322;
float const v248 = v247 + v246;
*v322 = v248;
float const v249 = *v361;
float const v250 = *v323;
float const v251 = v250 + v249;
*v323 = v251;
float const v252 = *v379;
float const v253 = *v324;
float const v254 = v253 + v252;
*v324 = v254;
}
}
float const v255 = *v321;
float const v256 = *v322;
float const v257 = *v323;
float const v258 = *v324;
float const v259 = v255 + v256;
float const v260 = v257 + v258;
float const v261 = v259 + v260;
int32_t const v262 = 16;
float const v263 = __shfl_xor_sync(__activemask(), v261, v262);
float const v264 = v261 + v263;
int32_t const v265 = 8;
float const v266 = __shfl_xor_sync(__activemask(), v264, v265);
float const v267 = v264 + v266;
int32_t const v268 = 4;
float const v269 = __shfl_xor_sync(__activemask(), v267, v268);
float const v270 = v267 + v269;
int32_t const v271 = 2;
float const v272 = __shfl_xor_sync(__activemask(), v270, v271);
float const v273 = v270 + v272;
int32_t const v274 = 1;
float const v275 = __shfl_xor_sync(__activemask(), v273, v274);
float const v276 = v273 + v275;
int64_t const v277 = 32;
int64_t const v278 = v9 / v277;
int64_t const v279 = v9 % v277;
bool const v280 = v279 == v10;
if(v280) {
uint64_t const v281 = uint64_t(v278);
float * const v282 = &(*v319)[v281];
*v282 = v276;
}
__syncthreads();
bool const v283 = v278 == v10;
if(v283) {
float v397_store;
float * v397 = &v397_store;*v397 = v31;
int64_t const v286 = 16;
bool const v287 = v279 < v286;
if(v287) {
uint64_t const v288 = uint64_t(v279);
float * const v289 = &(*v319)[v288];
float const v290 = *v289;
*v397 = v290;
}
float const v291 = *v397;
float const v293 = __shfl_xor_sync(__activemask(), v291, v265);
float const v294 = v291 + v293;
float const v296 = __shfl_xor_sync(__activemask(), v294, v268);
float const v297 = v294 + v296;
float const v299 = __shfl_xor_sync(__activemask(), v297, v271);
float const v300 = v297 + v299;
float const v302 = __shfl_xor_sync(__activemask(), v300, v274);
float const v303 = v300 + v302;
if(v280) {
uint64_t const v400 = 0;
float * const v305 = &(*v319)[v400];
*v305 = v303;
}
}
__syncthreads();
uint64_t const v398 = 0;
float * const v307 = &(*v319)[v398];
float const v308 = *v307;
bool const v309 = v9 == v10;
if(v309) {
uint64_t const v310 = uint64_t(v7);
float * const v311 = &v2[v310];
*v311 = v308;
}
}
return;
}
