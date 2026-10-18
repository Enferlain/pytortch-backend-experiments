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
struct alignas(16) float_4 { float i_0; float i_1; float i_2; float i_3; };

struct info_st {
    int64_t scalars_i64[3];uint64_t static_meta[3];
};
extern "C" __global__ void __launch_bounds__(256)  cubecl5_rms_cubek_vec4_observe_float32(float const* const __restrict__ v1, float * const __restrict__ v2, float * const __restrict__ v3, info_st const* const __restrict__ v274) {
uint64_t const* const dynamic_meta_v276 = reinterpret_cast<uint64_t const*>(v274 + 1);
const info_st& info_v275 = *v274;int64_t const v278 = info_v275.scalars_i64[1];
int64_t const v279 = info_v275.scalars_i64[2];
uint64_t const v7 = blockIdx.x;
int64_t const v8 = int64_t(v7);
uint64_t const v9 = threadIdx.y;
int64_t const v10 = int64_t(v9);
int64_t const v11 = 8;
int64_t const v12 = v8 * v11;
int64_t const v13 = v12 + v10;
uint64_t const v14 = uint64_t(v13);
uint64_t const v15 = uint64_t(v278);
uint64_t const v16 = uint64_t(v279);
bool const v17 = v14 < v16;
if(v17) {
uint64_t const v19 = 0;
uint64_t v281_store;
uint64_t * v281 = &v281_store;*v281 = v19;
uint64_t const v23 = 2;
uint64_t const v24 = 1;
for(uint64_t v25 = v19; v25 < v23; v25 += v24) {
int64_t const v28 = int64_t(v25);
int64_t const v29 = 0;
bool const v30 = v28 == v29;
int64_t const v31 = 1;
int64_t const v40 = v30 ? v279 : v31;
uint64_t const v41 = uint64_t(v40);
uint64_t const v42 = v14 % v41;
int64_t const v47 = v30 ? v278 : v31;
uint64_t const v48 = uint64_t(v47);
uint64_t const v49 = v42 * v48;
uint64_t const v50 = *v281;
uint64_t const v51 = v50 + v49;
*v281 = v51;
}
uint64_t const v52 = *v281;
uint64_t const v53 = 4;
uint64_t const v54 = v52 / v53;
*v281 = v54;
int64_t const v377 = 1;
uint64_t const v62 = v15 / v53;
float const v63 = 0.0;
float_4 const v64 = float_4{v63, v63, v63, v63};
int64_t const v69 = 4;
float_4 v286_store;
float_4 * v286 = &v286_store;*v286 = v64;
uint64_t const v375 = 32;
uint64_t const v79 = v62 + v375;
uint64_t const v81 = v79 - v24;
uint64_t const v82 = v81 / v375;
for(uint64_t v85 = v19; v85 < v82; v85 += v24) {
uint64_t const v89 = v85 * v375;
uint32_t const v90 = threadIdx.x;
uint64_t const v91 = uint64_t(v90);
uint64_t const v92 = v89 + v91;
uint64_t const v93 = *v281;
uint64_t const v94 = v92 + v93;
bool const v95 = v92 < v62;
float_4 v289_store;
float_4 * v289 = &v289_store;if(v95) {
int64_t const v96 = int64_t(v94);
int64_t const v98 = v96 * v69;
uint64_t const v101 = uint64_t(v98);
int64_t const v102 = int64_t(v101);
int64_t const v103 = v13 * v278;
int64_t const v104 = v102 - v103;
int64_t const v106 = v104 + v103;
uint64_t const v107 = uint64_t(v106);
float const* const v108 = &v1[v107];
float const v109 = *v108;
float const v110 = v109 * v109;
int64_t const v112 = v98 + v377;
uint64_t const v113 = uint64_t(v112);
int64_t const v114 = int64_t(v113);
int64_t const v116 = v114 - v103;
int64_t const v118 = v116 + v103;
uint64_t const v119 = uint64_t(v118);
float const* const v120 = &v1[v119];
float const v121 = *v120;
float const v122 = v121 * v121;
int64_t const v123 = 2;
int64_t const v124 = v98 + v123;
uint64_t const v125 = uint64_t(v124);
int64_t const v126 = int64_t(v125);
int64_t const v128 = v126 - v103;
int64_t const v130 = v128 + v103;
uint64_t const v131 = uint64_t(v130);
float const* const v132 = &v1[v131];
float const v133 = *v132;
float const v134 = v133 * v133;
int64_t const v135 = 3;
int64_t const v136 = v98 + v135;
uint64_t const v137 = uint64_t(v136);
int64_t const v138 = int64_t(v137);
int64_t const v140 = v138 - v103;
int64_t const v142 = v140 + v103;
uint64_t const v143 = uint64_t(v142);
float const* const v144 = &v1[v143];
float const v145 = *v144;
float const v146 = v145 * v145;
float_4 const v147 = float_4{v110, v122, v134, v146};
*v289 = v147;
}
 else {
*v289 = v64;
}
uint64_t const v157 = v53 * v15;
uint64_t const v159 = v157 + v375;
uint64_t const v160 = v14 * v159;
uint64_t const v162 = v375 * v85;
uint64_t const v165 = v162 + v91;
uint64_t const v167 = v165 * v53;
uint64_t const v168 = v160 + v167;
float_4 const v171 = *v289;
float const v172 = v171.i_0;
float * const v173 = &v2[v168];
*v173 = v172;
uint64_t const v188 = v168 + v24;
float const v190 = v171.i_1;
float * const v191 = &v2[v188];
*v191 = v190;
uint64_t const v206 = v168 + v23;
float const v208 = v171.i_2;
float * const v209 = &v2[v206];
*v209 = v208;
uint64_t const v223 = 3;
uint64_t const v224 = v168 + v223;
float const v226 = v171.i_3;
float * const v227 = &v2[v224];
*v227 = v226;
float_4 const v229 = *v286;
float const v338 = v229.i_0;
float const v340 = v338 + v172;
float const v341 = v229.i_1;
float const v343 = v341 + v190;
float const v344 = v229.i_2;
float const v346 = v344 + v208;
float const v347 = v229.i_3;
float const v349 = v347 + v226;
float_4 const v350 = float_4{v340, v343, v346, v349};
*v286 = v350;
}
float_4 const v231 = *v286;
uint32_t const v292 = warpSize;
uint32_t const v374 = 256;
uint32_t const v294 = min(v292, v374);
float_4 v332_store;
float_4 * v332 = &v332_store;*v332 = v231;
uint32_t const v296 = 1;
uint32_t v333_store;
uint32_t * v333 = &v333_store;*v333 = v296;
while([&]{
uint32_t const v299 = *v333;
bool const v300 = v299 < v294;
return v300;
}()) {
float_4 const v303 = *v332;
uint32_t const v304 = *v333;
float const v351 = v303.i_0;
float const v352 = __shfl_xor_sync(__activemask(), v351, v304);
float const v353 = v303.i_1;
float const v354 = __shfl_xor_sync(__activemask(), v353, v304);
float const v355 = v303.i_2;
float const v356 = __shfl_xor_sync(__activemask(), v355, v304);
float const v357 = v303.i_3;
float const v358 = __shfl_xor_sync(__activemask(), v357, v304);
float const v362 = v351 + v352;
float const v365 = v353 + v354;
float const v368 = v355 + v356;
float const v371 = v357 + v358;
float_4 const v372 = float_4{v362, v365, v368, v371};
*v332 = v372;
uint32_t const v309 = 2;
uint32_t const v310 = v304 * v309;
*v333 = v310;
}
float_4 const v311 = *v332;
*v286 = v311;
float_4 const v233 = *v286;
float const v317 = v233.i_0;
float const v319 = v63 + v317;
float const v320 = v233.i_1;
float const v322 = v319 + v320;
float const v323 = v233.i_2;
float const v325 = v322 + v323;
float const v326 = v233.i_3;
float const v328 = v325 + v326;
uint64_t const v236 = v53 * v15;
uint64_t const v238 = v236 + v375;
uint64_t const v239 = v14 * v238;
uint64_t const v242 = v239 + v236;
uint32_t const v243 = threadIdx.x;
uint64_t const v244 = uint64_t(v243);
uint64_t const v245 = v242 + v244;
float * const v246 = &v2[v245];
*v246 = v328;
for(uint64_t v250 = v244; v250 < v15; v250 += v375) {
uint64_t const v253 = v14 * v15;
uint64_t const v254 = v253 + v250;
int64_t const v255 = int64_t(v254);
int64_t const v256 = v13 * v278;
int64_t const v257 = v255 - v256;
int64_t const v259 = v257 + v256;
uint64_t const v260 = uint64_t(v259);
float const* const v261 = &v1[v260];
float const v262 = *v261;
float const v264 = float(v278);
float const v265 = v328 / v264;
float const v266 = 9.999999740e-06;
float const v267 = v265 + v266;
float const v268 = rsqrt(v267);float const v269 = v262 * v268;
float * const v273 = &v3[v260];
*v273 = v269;
}
}
return;
}
