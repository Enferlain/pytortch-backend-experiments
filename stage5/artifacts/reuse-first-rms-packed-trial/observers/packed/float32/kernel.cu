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
extern "C" __global__ void __launch_bounds__(256)  cubecl5_rms_cubek_pack4_observe_float32(float const* const __restrict__ v1, float * const __restrict__ v2, float * const __restrict__ v3, info_st const* const __restrict__ v268) {
uint64_t const* const dynamic_meta_v270 = reinterpret_cast<uint64_t const*>(v268 + 1);
const info_st& info_v269 = *v268;int64_t const v272 = info_v269.scalars_i64[1];
int64_t const v273 = info_v269.scalars_i64[2];
uint64_t const v7 = blockIdx.x;
int64_t const v8 = int64_t(v7);
uint64_t const v9 = threadIdx.y;
int64_t const v10 = int64_t(v9);
int64_t const v11 = 8;
int64_t const v12 = v8 * v11;
int64_t const v13 = v12 + v10;
uint64_t const v14 = uint64_t(v13);
uint64_t const v15 = uint64_t(v272);
uint64_t const v16 = uint64_t(v273);
bool const v17 = v14 < v16;
if(v17) {
uint64_t const v19 = 0;
uint64_t v275_store;
uint64_t * v275 = &v275_store;*v275 = v19;
uint64_t const v23 = 2;
uint64_t const v24 = 1;
for(uint64_t v25 = v19; v25 < v23; v25 += v24) {
int64_t const v28 = int64_t(v25);
int64_t const v29 = 0;
bool const v30 = v28 == v29;
int64_t const v31 = 1;
int64_t const v40 = v30 ? v273 : v31;
uint64_t const v41 = uint64_t(v40);
uint64_t const v42 = v14 % v41;
int64_t const v47 = v30 ? v272 : v31;
uint64_t const v48 = uint64_t(v47);
uint64_t const v49 = v42 * v48;
uint64_t const v50 = *v275;
uint64_t const v51 = v50 + v49;
*v275 = v51;
}
uint64_t const v52 = *v275;
uint64_t const v53 = 4;
uint64_t const v54 = v52 / v53;
*v275 = v54;
uint64_t const v62 = v15 / v53;
float const v63 = 0.0;
float_4 const v64 = float_4{v63, v63, v63, v63};
int64_t const v69 = 4;
float_4 v280_store;
float_4 * v280 = &v280_store;*v280 = v64;
uint64_t const v369 = 32;
uint64_t const v79 = v62 + v369;
uint64_t const v81 = v79 - v24;
uint64_t const v82 = v81 / v369;
for(uint64_t v85 = v19; v85 < v82; v85 += v24) {
uint64_t const v89 = v85 * v369;
uint32_t const v90 = threadIdx.x;
uint64_t const v91 = uint64_t(v90);
uint64_t const v92 = v89 + v91;
uint64_t const v93 = *v275;
uint64_t const v94 = v92 + v93;
bool const v95 = v92 < v62;
float_4 v283_store;
float_4 * v283 = &v283_store;if(v95) {
int64_t const v96 = int64_t(v94);
int64_t const v98 = v96 * v69;
uint64_t const v99 = uint64_t(v98);
int64_t const v100 = int64_t(v99);
int64_t const v101 = v13 * v272;
int64_t const v102 = v100 - v101;
int64_t const v104 = v102 + v101;
uint64_t const v105 = uint64_t(v104);
float const* const v106 = &v1[v105];
float_4 const* const v107 = reinterpret_cast<float_4 const*>(v106);
float_4 const v108 = *v107;
float const v109 = v108.i_0;
float const v116 = v109 * v109;
float const v117 = v108.i_1;
float const v124 = v117 * v117;
float const v125 = v108.i_2;
float const v132 = v125 * v125;
float const v133 = v108.i_3;
float const v140 = v133 * v133;
float_4 const v141 = float_4{v116, v124, v132, v140};
*v283 = v141;
}
 else {
*v283 = v64;
}
uint64_t const v151 = v53 * v15;
uint64_t const v153 = v151 + v369;
uint64_t const v154 = v14 * v153;
uint64_t const v156 = v369 * v85;
uint64_t const v159 = v156 + v91;
uint64_t const v161 = v159 * v53;
uint64_t const v162 = v154 + v161;
float_4 const v165 = *v283;
float const v166 = v165.i_0;
float * const v167 = &v2[v162];
*v167 = v166;
uint64_t const v182 = v162 + v24;
float const v184 = v165.i_1;
float * const v185 = &v2[v182];
*v185 = v184;
uint64_t const v200 = v162 + v23;
float const v202 = v165.i_2;
float * const v203 = &v2[v200];
*v203 = v202;
uint64_t const v217 = 3;
uint64_t const v218 = v162 + v217;
float const v220 = v165.i_3;
float * const v221 = &v2[v218];
*v221 = v220;
float_4 const v223 = *v280;
float const v332 = v223.i_0;
float const v334 = v332 + v166;
float const v335 = v223.i_1;
float const v337 = v335 + v184;
float const v338 = v223.i_2;
float const v340 = v338 + v202;
float const v341 = v223.i_3;
float const v343 = v341 + v220;
float_4 const v344 = float_4{v334, v337, v340, v343};
*v280 = v344;
}
float_4 const v225 = *v280;
uint32_t const v286 = warpSize;
uint32_t const v368 = 256;
uint32_t const v288 = min(v286, v368);
float_4 v326_store;
float_4 * v326 = &v326_store;*v326 = v225;
uint32_t const v290 = 1;
uint32_t v327_store;
uint32_t * v327 = &v327_store;*v327 = v290;
while([&]{
uint32_t const v293 = *v327;
bool const v294 = v293 < v288;
return v294;
}()) {
float_4 const v297 = *v326;
uint32_t const v298 = *v327;
float const v345 = v297.i_0;
float const v346 = __shfl_xor_sync(__activemask(), v345, v298);
float const v347 = v297.i_1;
float const v348 = __shfl_xor_sync(__activemask(), v347, v298);
float const v349 = v297.i_2;
float const v350 = __shfl_xor_sync(__activemask(), v349, v298);
float const v351 = v297.i_3;
float const v352 = __shfl_xor_sync(__activemask(), v351, v298);
float const v356 = v345 + v346;
float const v359 = v347 + v348;
float const v362 = v349 + v350;
float const v365 = v351 + v352;
float_4 const v366 = float_4{v356, v359, v362, v365};
*v326 = v366;
uint32_t const v303 = 2;
uint32_t const v304 = v298 * v303;
*v327 = v304;
}
float_4 const v305 = *v326;
*v280 = v305;
float_4 const v227 = *v280;
float const v311 = v227.i_0;
float const v313 = v63 + v311;
float const v314 = v227.i_1;
float const v316 = v313 + v314;
float const v317 = v227.i_2;
float const v319 = v316 + v317;
float const v320 = v227.i_3;
float const v322 = v319 + v320;
uint64_t const v230 = v53 * v15;
uint64_t const v232 = v230 + v369;
uint64_t const v233 = v14 * v232;
uint64_t const v236 = v233 + v230;
uint32_t const v237 = threadIdx.x;
uint64_t const v238 = uint64_t(v237);
uint64_t const v239 = v236 + v238;
float * const v240 = &v2[v239];
*v240 = v322;
for(uint64_t v244 = v238; v244 < v15; v244 += v369) {
uint64_t const v247 = v14 * v15;
uint64_t const v248 = v247 + v244;
int64_t const v249 = int64_t(v248);
int64_t const v250 = v13 * v272;
int64_t const v251 = v249 - v250;
int64_t const v253 = v251 + v250;
uint64_t const v254 = uint64_t(v253);
float const* const v255 = &v1[v254];
float const v256 = *v255;
float const v258 = float(v272);
float const v259 = v322 / v258;
float const v260 = 9.999999740e-06;
float const v261 = v259 + v260;
float const v262 = rsqrt(v261);float const v263 = v256 * v262;
float * const v267 = &v3[v254];
*v267 = v263;
}
}
return;
}
