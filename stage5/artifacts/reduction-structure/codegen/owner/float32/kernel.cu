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
extern "C" __global__ void __launch_bounds__(512)  cubecl5_42026f8357481503(float const* const __restrict__ v1, float * const __restrict__ v2, info_st const* const __restrict__ v308) {
extern __shared__ __align__(4) char v314[];uint64_t const* const dynamic_meta_v310 = reinterpret_cast<uint64_t const*>(v308 + 1);
const info_st& info_v309 = *v308;int64_t const v312 = info_v309.scalars_i64[1];
int64_t const v313 = info_v309.scalars_i64[2];
uint64_t const v6 = blockIdx.x;
int64_t const v7 = int64_t(v6);
uint64_t const v8 = threadIdx.x;
int64_t const v9 = int64_t(v8);
int64_t const v10 = 0;
int64_t const v11 = 1;
int64_t const v15 = 512;
int64_t const v16 = 9;
int64_t const v17 = v16 * v313;
int64_t const v18 = 8;
int64_t const v19 = v18 + v312;
int64_t const v21 = v19 / v16;
int64_t const v22 = v19 % v16;
bool const v25 = v22 < v10;
int64_t const v26 = v21 - v11;
int64_t const v27 = v25 ? v26 : v21;
array<float, 16> * const v315 = reinterpret_cast<array<float, 16> *>(&v314[0]);
bool const v29 = v7 < v17;
if(v29) {
float v317_store;
float * v317 = &v317_store;float const v31 = 0.0;
*v317 = v31;
float v318_store;
float * v318 = &v318_store;*v318 = v31;
float v319_store;
float * v319 = &v319_store;*v319 = v31;
float v320_store;
float * v320 = &v320_store;*v320 = v31;
int64_t const v395 = 2048;
for(int64_t v40 = v9; v40 < v27; v40 += v395) {
bool const v43 = v40 < v27;
float v321_store;
float * v321 = &v321_store;*v321 = v31;
if(v43) {
int64_t const v322 = v7 % v16;
int64_t const v324 = v322 ^ v16;
bool const v325 = v324 < v10;
bool const v326 = v322 != v10;
bool const v327 = v326 && v325;
int64_t const v328 = v327 ? v16 : v10;
int64_t const v329 = v322 + v328;
int64_t const v58 = v27 * v329;
int64_t const v59 = v40 + v58;
bool const v60 = v59 < v312;
float v330_store;
float * v330 = &v330_store;if(v60) {
int64_t const v63 = v7 / v16;
bool const v67 = v322 < v10;
int64_t const v68 = v63 - v11;
int64_t const v69 = v67 ? v68 : v63;
int64_t const v70 = v312 * v69;
int64_t const v71 = v40 + v70;
int64_t const v85 = v71 + v58;
uint64_t const v86 = uint64_t(v85);
float const* const v87 = &v1[v86];
float const v88 = *v87;
*v330 = v88;
}
 else {
*v330 = v31;
}
float const v90 = *v330;
*v321 = v90;
}
int64_t const v92 = v40 + v15;
bool const v93 = v92 < v27;
float v339_store;
float * v339 = &v339_store;*v339 = v31;
if(v93) {
int64_t const v340 = v7 % v16;
int64_t const v342 = v340 ^ v16;
bool const v343 = v342 < v10;
bool const v344 = v340 != v10;
bool const v345 = v344 && v343;
int64_t const v346 = v345 ? v16 : v10;
int64_t const v347 = v340 + v346;
int64_t const v108 = v27 * v347;
int64_t const v109 = v92 + v108;
bool const v110 = v109 < v312;
float v348_store;
float * v348 = &v348_store;if(v110) {
int64_t const v113 = v7 / v16;
bool const v117 = v340 < v10;
int64_t const v118 = v113 - v11;
int64_t const v119 = v117 ? v118 : v113;
int64_t const v120 = v312 * v119;
int64_t const v121 = v92 + v120;
int64_t const v135 = v121 + v108;
uint64_t const v136 = uint64_t(v135);
float const* const v137 = &v1[v136];
float const v138 = *v137;
*v348 = v138;
}
 else {
*v348 = v31;
}
float const v140 = *v348;
*v339 = v140;
}
int64_t const v141 = 1024;
int64_t const v142 = v40 + v141;
bool const v143 = v142 < v27;
float v357_store;
float * v357 = &v357_store;*v357 = v31;
if(v143) {
int64_t const v358 = v7 % v16;
int64_t const v360 = v358 ^ v16;
bool const v361 = v360 < v10;
bool const v362 = v358 != v10;
bool const v363 = v362 && v361;
int64_t const v364 = v363 ? v16 : v10;
int64_t const v365 = v358 + v364;
int64_t const v158 = v27 * v365;
int64_t const v159 = v142 + v158;
bool const v160 = v159 < v312;
float v366_store;
float * v366 = &v366_store;if(v160) {
int64_t const v163 = v7 / v16;
bool const v167 = v358 < v10;
int64_t const v168 = v163 - v11;
int64_t const v169 = v167 ? v168 : v163;
int64_t const v170 = v312 * v169;
int64_t const v171 = v142 + v170;
int64_t const v185 = v171 + v158;
uint64_t const v186 = uint64_t(v185);
float const* const v187 = &v1[v186];
float const v188 = *v187;
*v366 = v188;
}
 else {
*v366 = v31;
}
float const v190 = *v366;
*v357 = v190;
}
int64_t const v191 = 1536;
int64_t const v192 = v40 + v191;
bool const v193 = v192 < v27;
float v375_store;
float * v375 = &v375_store;*v375 = v31;
if(v193) {
int64_t const v376 = v7 % v16;
int64_t const v378 = v376 ^ v16;
bool const v379 = v378 < v10;
bool const v380 = v376 != v10;
bool const v381 = v380 && v379;
int64_t const v382 = v381 ? v16 : v10;
int64_t const v383 = v376 + v382;
int64_t const v208 = v27 * v383;
int64_t const v209 = v192 + v208;
bool const v210 = v209 < v312;
float v384_store;
float * v384 = &v384_store;if(v210) {
int64_t const v213 = v7 / v16;
bool const v217 = v376 < v10;
int64_t const v218 = v213 - v11;
int64_t const v219 = v217 ? v218 : v213;
int64_t const v220 = v312 * v219;
int64_t const v221 = v192 + v220;
int64_t const v235 = v221 + v208;
uint64_t const v236 = uint64_t(v235);
float const* const v237 = &v1[v236];
float const v238 = *v237;
*v384 = v238;
}
 else {
*v384 = v31;
}
float const v240 = *v384;
*v375 = v240;
}
float const v241 = *v321;
float const v242 = *v317;
float const v243 = v242 + v241;
*v317 = v243;
float const v244 = *v339;
float const v245 = *v318;
float const v246 = v245 + v244;
*v318 = v246;
float const v247 = *v357;
float const v248 = *v319;
float const v249 = v248 + v247;
*v319 = v249;
float const v250 = *v375;
float const v251 = *v320;
float const v252 = v251 + v250;
*v320 = v252;
}
float const v253 = *v317;
float const v254 = *v318;
float const v255 = *v319;
float const v256 = *v320;
float const v257 = v253 + v254;
float const v258 = v255 + v256;
float const v259 = v257 + v258;
float v393_store;
float * v393 = &v393_store;*v393 = v31;
int32_t const v262 = 16;
float const v263 = __shfl_xor_sync(__activemask(), v259, v262);
float const v264 = v259 + v263;
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
float * const v282 = &(*v315)[v281];
*v282 = v276;
}
__syncthreads();
bool const v283 = v278 == v10;
if(v283) {
float v394_store;
float * v394 = &v394_store;*v394 = v31;
int64_t const v286 = 16;
bool const v287 = v279 < v286;
if(v287) {
uint64_t const v288 = uint64_t(v279);
float * const v289 = &(*v315)[v288];
float const v290 = *v289;
*v394 = v290;
}
float const v291 = *v394;
float const v293 = __shfl_xor_sync(__activemask(), v291, v265);
float const v294 = v291 + v293;
float const v296 = __shfl_xor_sync(__activemask(), v294, v268);
float const v297 = v294 + v296;
float const v299 = __shfl_xor_sync(__activemask(), v297, v271);
float const v300 = v297 + v299;
float const v302 = __shfl_xor_sync(__activemask(), v300, v274);
float const v303 = v300 + v302;
*v393 = v303;
}
float const v304 = *v393;
bool const v305 = v9 == v10;
if(v305) {
uint64_t const v306 = uint64_t(v7);
float * const v307 = &v2[v306];
*v307 = v304;
}
}
return;
}
