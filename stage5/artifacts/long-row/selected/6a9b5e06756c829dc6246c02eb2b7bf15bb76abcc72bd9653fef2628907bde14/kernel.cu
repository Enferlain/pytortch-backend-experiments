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
extern "C" __global__ void __launch_bounds__(256)  cubecl5_414f996f5c38ad89(float const* const __restrict__ v1, float * const __restrict__ v2, info_st const* const __restrict__ v273) {
extern __shared__ __align__(4) char v279[];uint64_t const* const dynamic_meta_v275 = reinterpret_cast<uint64_t const*>(v273 + 1);
const info_st& info_v274 = *v273;int64_t const v277 = info_v274.scalars_i64[1];
int64_t const v278 = info_v274.scalars_i64[2];
uint64_t const v6 = blockIdx.x;
int64_t const v7 = int64_t(v6);
uint64_t const v8 = threadIdx.x;
int64_t const v9 = int64_t(v8);
int64_t const v10 = 0;
int64_t const v11 = 1;
int64_t const v15 = 256;
int64_t const v16 = 32;
int64_t const v17 = v16 * v278;
int64_t const v18 = 31;
int64_t const v19 = v18 + v277;
int64_t const v21 = v19 / v16;
int64_t const v22 = v19 % v16;
bool const v25 = v22 < v10;
int64_t const v26 = v21 - v11;
int64_t const v27 = v25 ? v26 : v21;
array<float, 8> * const v280 = reinterpret_cast<array<float, 8> *>(&v279[0]);
bool const v29 = v7 < v17;
if(v29) {
float v282_store;
float * v282 = &v282_store;float const v31 = 0.0;
*v282 = v31;
float v283_store;
float * v283 = &v283_store;*v283 = v31;
float v284_store;
float * v284 = &v284_store;*v284 = v31;
float v285_store;
float * v285 = &v285_store;*v285 = v31;
int64_t const v404 = 1024;
for(int64_t v40 = v9; v40 < v27; v40 += v404) {
bool const v43 = v40 < v27;
if(v43) {
int64_t const v286 = v7 % v16;
int64_t const v288 = v286 ^ v16;
bool const v289 = v288 < v10;
bool const v290 = v286 != v10;
bool const v291 = v290 && v289;
int64_t const v292 = v291 ? v16 : v10;
int64_t const v293 = v286 + v292;
int64_t const v56 = v27 * v293;
int64_t const v57 = v40 + v56;
bool const v58 = v57 < v277;
float v294_store;
float * v294 = &v294_store;if(v58) {
int64_t const v61 = v7 / v16;
bool const v65 = v286 < v10;
int64_t const v66 = v61 - v11;
int64_t const v67 = v65 ? v66 : v61;
int64_t const v68 = v277 * v67;
int64_t const v69 = v40 + v68;
int64_t const v83 = v69 + v56;
uint64_t const v84 = uint64_t(v83);
float const* const v85 = &v1[v84];
float const v86 = *v85;
*v294 = v86;
}
 else {
*v294 = v31;
}
float const v88 = *v294;
float const v89 = *v282;
float const v90 = v89 + v88;
*v282 = v90;
}
int64_t const v92 = v40 + v15;
bool const v93 = v92 < v27;
if(v93) {
int64_t const v303 = v7 % v16;
int64_t const v305 = v303 ^ v16;
bool const v306 = v305 < v10;
bool const v307 = v303 != v10;
bool const v308 = v307 && v306;
int64_t const v309 = v308 ? v16 : v10;
int64_t const v310 = v303 + v309;
int64_t const v106 = v27 * v310;
int64_t const v107 = v92 + v106;
bool const v108 = v107 < v277;
float v311_store;
float * v311 = &v311_store;if(v108) {
int64_t const v111 = v7 / v16;
bool const v115 = v303 < v10;
int64_t const v116 = v111 - v11;
int64_t const v117 = v115 ? v116 : v111;
int64_t const v118 = v277 * v117;
int64_t const v119 = v92 + v118;
int64_t const v133 = v119 + v106;
uint64_t const v134 = uint64_t(v133);
float const* const v135 = &v1[v134];
float const v136 = *v135;
*v311 = v136;
}
 else {
*v311 = v31;
}
float const v138 = *v311;
float const v139 = *v283;
float const v140 = v139 + v138;
*v283 = v140;
}
int64_t const v141 = 512;
int64_t const v142 = v40 + v141;
bool const v143 = v142 < v27;
if(v143) {
int64_t const v320 = v7 % v16;
int64_t const v322 = v320 ^ v16;
bool const v323 = v322 < v10;
bool const v324 = v320 != v10;
bool const v325 = v324 && v323;
int64_t const v326 = v325 ? v16 : v10;
int64_t const v327 = v320 + v326;
int64_t const v156 = v27 * v327;
int64_t const v157 = v142 + v156;
bool const v158 = v157 < v277;
float v328_store;
float * v328 = &v328_store;if(v158) {
int64_t const v161 = v7 / v16;
bool const v165 = v320 < v10;
int64_t const v166 = v161 - v11;
int64_t const v167 = v165 ? v166 : v161;
int64_t const v168 = v277 * v167;
int64_t const v169 = v142 + v168;
int64_t const v183 = v169 + v156;
uint64_t const v184 = uint64_t(v183);
float const* const v185 = &v1[v184];
float const v186 = *v185;
*v328 = v186;
}
 else {
*v328 = v31;
}
float const v188 = *v328;
float const v189 = *v284;
float const v190 = v189 + v188;
*v284 = v190;
}
int64_t const v191 = 768;
int64_t const v192 = v40 + v191;
bool const v193 = v192 < v27;
if(v193) {
int64_t const v337 = v7 % v16;
int64_t const v339 = v337 ^ v16;
bool const v340 = v339 < v10;
bool const v341 = v337 != v10;
bool const v342 = v341 && v340;
int64_t const v343 = v342 ? v16 : v10;
int64_t const v344 = v337 + v343;
int64_t const v206 = v27 * v344;
int64_t const v207 = v192 + v206;
bool const v208 = v207 < v277;
float v345_store;
float * v345 = &v345_store;if(v208) {
int64_t const v211 = v7 / v16;
bool const v215 = v337 < v10;
int64_t const v216 = v211 - v11;
int64_t const v217 = v215 ? v216 : v211;
int64_t const v218 = v277 * v217;
int64_t const v219 = v192 + v218;
int64_t const v233 = v219 + v206;
uint64_t const v234 = uint64_t(v233);
float const* const v235 = &v1[v234];
float const v236 = *v235;
*v345 = v236;
}
 else {
*v345 = v31;
}
float const v238 = *v345;
float const v239 = *v285;
float const v240 = v239 + v238;
*v285 = v240;
}
}
float const v241 = *v282;
float const v242 = *v283;
float const v243 = *v284;
float const v244 = *v285;
float const v245 = v241 + v243;
float const v246 = v242 + v244;
float const v247 = v245 + v246;
uint32_t const v354 = warpSize;
uint32_t const v401 = 256;
uint32_t const v356 = min(v354, v401);
float v395_store;
float * v395 = &v395_store;*v395 = v247;
uint32_t const v358 = 1;
uint32_t v396_store;
uint32_t * v396 = &v396_store;*v396 = v358;
while([&]{
uint32_t const v361 = *v396;
bool const v362 = v361 < v356;
return v362;
}()) {
float const v365 = *v395;
uint32_t const v366 = *v396;
float const v367 = __shfl_xor_sync(__activemask(), v365, v366);
float const v369 = v365 + v367;
*v395 = v369;
uint32_t const v371 = 2;
uint32_t const v372 = v366 * v371;
*v396 = v372;
}
float const v373 = *v395;
int64_t const v250 = v9 / v16;
int64_t const v251 = v9 % v16;
bool const v252 = v251 == v10;
if(v252) {
uint64_t const v253 = uint64_t(v250);
float * const v254 = &(*v280)[v253];
*v254 = v373;
}
__syncthreads();
bool const v255 = v250 == v10;
if(v255) {
float v374_store;
float * v374 = &v374_store;*v374 = v31;
int64_t const v258 = 8;
bool const v259 = v251 < v258;
if(v259) {
uint64_t const v260 = uint64_t(v251);
float * const v261 = &(*v280)[v260];
float const v262 = *v261;
*v374 = v262;
}
float const v263 = *v374;
float v398_store;
float * v398 = &v398_store;*v398 = v263;
uint32_t v399_store;
uint32_t * v399 = &v399_store;*v399 = v358;
while([&]{
uint32_t const v382 = *v399;
bool const v383 = v382 < v356;
return v383;
}()) {
float const v386 = *v398;
uint32_t const v387 = *v399;
float const v388 = __shfl_xor_sync(__activemask(), v386, v387);
float const v390 = v386 + v388;
*v398 = v390;
uint32_t const v392 = 2;
uint32_t const v393 = v387 * v392;
*v399 = v393;
}
float const v394 = *v398;
if(v252) {
uint64_t const v405 = 0;
float * const v266 = &(*v280)[v405];
*v266 = v394;
}
}
__syncthreads();
uint64_t const v403 = 0;
float * const v268 = &(*v280)[v403];
float const v269 = *v268;
__syncthreads();
bool const v270 = v9 == v10;
if(v270) {
uint64_t const v271 = uint64_t(v7);
float * const v272 = &v2[v271];
*v272 = v269;
}
}
return;
}
