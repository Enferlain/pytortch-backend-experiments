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
struct alignas(16) float_4 { float i_0; float i_1; float i_2; float i_3; };

struct info_st {
    int64_t scalars_i64[3];uint64_t static_meta[3];
};
extern "C" __global__ void __launch_bounds__(256)  cubecl5_rms_cubek_vec4_observe_float16(__half const* const __restrict__ v1, float * const __restrict__ v2, __half * const __restrict__ v3, info_st const* const __restrict__ v299) {
uint64_t const* const dynamic_meta_v301 = reinterpret_cast<uint64_t const*>(v299 + 1);
const info_st& info_v300 = *v299;int64_t const v303 = info_v300.scalars_i64[1];
int64_t const v304 = info_v300.scalars_i64[2];
uint64_t const v7 = blockIdx.x;
int64_t const v8 = int64_t(v7);
uint64_t const v9 = threadIdx.y;
int64_t const v10 = int64_t(v9);
int64_t const v11 = 8;
int64_t const v12 = v8 * v11;
int64_t const v13 = v12 + v10;
uint64_t const v14 = uint64_t(v13);
uint64_t const v15 = uint64_t(v303);
uint64_t const v16 = uint64_t(v304);
bool const v17 = v14 < v16;
if(v17) {
uint64_t const v19 = 0;
uint64_t v306_store;
uint64_t * v306 = &v306_store;*v306 = v19;
uint64_t const v23 = 2;
uint64_t const v24 = 1;
for(uint64_t v25 = v19; v25 < v23; v25 += v24) {
int64_t const v28 = int64_t(v25);
int64_t const v29 = 0;
bool const v30 = v28 == v29;
int64_t const v31 = 1;
int64_t const v40 = v30 ? v304 : v31;
uint64_t const v41 = uint64_t(v40);
uint64_t const v42 = v14 % v41;
int64_t const v47 = v30 ? v303 : v31;
uint64_t const v48 = uint64_t(v47);
uint64_t const v49 = v42 * v48;
uint64_t const v50 = *v306;
uint64_t const v51 = v50 + v49;
*v306 = v51;
}
uint64_t const v52 = *v306;
uint64_t const v53 = 4;
uint64_t const v54 = v52 / v53;
*v306 = v54;
int64_t const v402 = 1;
uint64_t const v62 = v15 / v53;
float const v63 = 0.0;
float_4 const v64 = float_4{v63, v63, v63, v63};
int64_t const v69 = 4;
float_4 v311_store;
float_4 * v311 = &v311_store;*v311 = v64;
uint64_t const v400 = 32;
uint64_t const v79 = v62 + v400;
uint64_t const v81 = v79 - v24;
uint64_t const v82 = v81 / v400;
for(uint64_t v85 = v19; v85 < v82; v85 += v24) {
uint64_t const v89 = v85 * v400;
uint32_t const v90 = threadIdx.x;
uint64_t const v91 = uint64_t(v90);
uint64_t const v92 = v89 + v91;
uint64_t const v93 = *v306;
uint64_t const v94 = v92 + v93;
bool const v95 = v92 < v62;
float_4 v314_store;
float_4 * v314 = &v314_store;if(v95) {
int64_t const v96 = int64_t(v94);
int64_t const v98 = v96 * v69;
uint64_t const v101 = uint64_t(v98);
int64_t const v102 = int64_t(v101);
int64_t const v103 = v13 * v303;
int64_t const v104 = v102 - v103;
int64_t const v106 = v104 + v103;
uint64_t const v107 = uint64_t(v106);
__half const* const v108 = &v1[v107];
__half const v109 = *v108;
float const v110 = float(v109);
__half const v111 = __half(v110);
float const v112 = float(v111);
float const v114 = v112 * v112;
int64_t const v116 = v98 + v402;
uint64_t const v117 = uint64_t(v116);
int64_t const v118 = int64_t(v117);
int64_t const v120 = v118 - v103;
int64_t const v122 = v120 + v103;
uint64_t const v123 = uint64_t(v122);
__half const* const v124 = &v1[v123];
__half const v125 = *v124;
float const v126 = float(v125);
__half const v127 = __half(v126);
float const v128 = float(v127);
float const v130 = v128 * v128;
int64_t const v131 = 2;
int64_t const v132 = v98 + v131;
uint64_t const v133 = uint64_t(v132);
int64_t const v134 = int64_t(v133);
int64_t const v136 = v134 - v103;
int64_t const v138 = v136 + v103;
uint64_t const v139 = uint64_t(v138);
__half const* const v140 = &v1[v139];
__half const v141 = *v140;
float const v142 = float(v141);
__half const v143 = __half(v142);
float const v144 = float(v143);
float const v146 = v144 * v144;
int64_t const v147 = 3;
int64_t const v148 = v98 + v147;
uint64_t const v149 = uint64_t(v148);
int64_t const v150 = int64_t(v149);
int64_t const v152 = v150 - v103;
int64_t const v154 = v152 + v103;
uint64_t const v155 = uint64_t(v154);
__half const* const v156 = &v1[v155];
__half const v157 = *v156;
float const v158 = float(v157);
__half const v159 = __half(v158);
float const v160 = float(v159);
float const v162 = v160 * v160;
float_4 const v163 = float_4{v114, v130, v146, v162};
*v314 = v163;
}
 else {
*v314 = v64;
}
uint64_t const v173 = v53 * v15;
uint64_t const v175 = v173 + v400;
uint64_t const v176 = v14 * v175;
uint64_t const v178 = v400 * v85;
uint64_t const v181 = v178 + v91;
uint64_t const v183 = v181 * v53;
uint64_t const v184 = v176 + v183;
float_4 const v187 = *v314;
float const v188 = v187.i_0;
float * const v189 = &v2[v184];
*v189 = v188;
uint64_t const v204 = v184 + v24;
float const v206 = v187.i_1;
float * const v207 = &v2[v204];
*v207 = v206;
uint64_t const v222 = v184 + v23;
float const v224 = v187.i_2;
float * const v225 = &v2[v222];
*v225 = v224;
uint64_t const v239 = 3;
uint64_t const v240 = v184 + v239;
float const v242 = v187.i_3;
float * const v243 = &v2[v240];
*v243 = v242;
float_4 const v245 = *v311;
float const v363 = v245.i_0;
float const v365 = v363 + v188;
float const v366 = v245.i_1;
float const v368 = v366 + v206;
float const v369 = v245.i_2;
float const v371 = v369 + v224;
float const v372 = v245.i_3;
float const v374 = v372 + v242;
float_4 const v375 = float_4{v365, v368, v371, v374};
*v311 = v375;
}
float_4 const v247 = *v311;
uint32_t const v317 = warpSize;
uint32_t const v399 = 256;
uint32_t const v319 = min(v317, v399);
float_4 v357_store;
float_4 * v357 = &v357_store;*v357 = v247;
uint32_t const v321 = 1;
uint32_t v358_store;
uint32_t * v358 = &v358_store;*v358 = v321;
while([&]{
uint32_t const v324 = *v358;
bool const v325 = v324 < v319;
return v325;
}()) {
float_4 const v328 = *v357;
uint32_t const v329 = *v358;
float const v376 = v328.i_0;
float const v377 = __shfl_xor_sync(__activemask(), v376, v329);
float const v378 = v328.i_1;
float const v379 = __shfl_xor_sync(__activemask(), v378, v329);
float const v380 = v328.i_2;
float const v381 = __shfl_xor_sync(__activemask(), v380, v329);
float const v382 = v328.i_3;
float const v383 = __shfl_xor_sync(__activemask(), v382, v329);
float const v387 = v376 + v377;
float const v390 = v378 + v379;
float const v393 = v380 + v381;
float const v396 = v382 + v383;
float_4 const v397 = float_4{v387, v390, v393, v396};
*v357 = v397;
uint32_t const v334 = 2;
uint32_t const v335 = v329 * v334;
*v358 = v335;
}
float_4 const v336 = *v357;
*v311 = v336;
float_4 const v249 = *v311;
float const v342 = v249.i_0;
float const v344 = v63 + v342;
float const v345 = v249.i_1;
float const v347 = v344 + v345;
float const v348 = v249.i_2;
float const v350 = v347 + v348;
float const v351 = v249.i_3;
float const v353 = v350 + v351;
uint64_t const v252 = v53 * v15;
uint64_t const v254 = v252 + v400;
uint64_t const v255 = v14 * v254;
uint64_t const v258 = v255 + v252;
uint32_t const v259 = threadIdx.x;
uint64_t const v260 = uint64_t(v259);
uint64_t const v261 = v258 + v260;
float * const v262 = &v2[v261];
*v262 = v353;
for(uint64_t v266 = v260; v266 < v15; v266 += v400) {
uint64_t const v269 = v14 * v15;
uint64_t const v270 = v269 + v266;
int64_t const v271 = int64_t(v270);
int64_t const v272 = v13 * v303;
int64_t const v273 = v271 - v272;
int64_t const v275 = v273 + v272;
uint64_t const v276 = uint64_t(v275);
__half const* const v277 = &v1[v276];
__half const v278 = *v277;
float const v279 = float(v278);
__half const v280 = __half(v279);
float const v281 = float(v280);
float const v284 = float(v303);
float const v285 = v353 / v284;
float const v286 = 9.999999740e-06;
float const v287 = v285 + v286;
float const v288 = rsqrt(v287);float const v289 = v281 * v288;
__half const v290 = __half(v289);
float const v291 = float(v290);
__half const v292 = __half(v291);
float const v293 = float(v292);
__half const v294 = __half(v293);
__half * const v298 = &v3[v276];
*v298 = v294;
}
}
return;
}
