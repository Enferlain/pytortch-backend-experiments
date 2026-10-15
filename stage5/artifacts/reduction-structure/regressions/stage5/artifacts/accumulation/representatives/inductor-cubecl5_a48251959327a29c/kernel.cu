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
extern "C" __global__ void __launch_bounds__(256)  cubecl5_a48251959327a29c(float const* const __restrict__ v1, float * const __restrict__ v2, info_st const* const __restrict__ v372) {
extern __shared__ __align__(4) char v378[];uint64_t const* const dynamic_meta_v374 = reinterpret_cast<uint64_t const*>(v372 + 1);
const info_st& info_v373 = *v372;int64_t const v376 = info_v373.scalars_i64[1];
int64_t const v377 = info_v373.scalars_i64[2];
uint64_t const v6 = blockIdx.x;
int64_t const v7 = int64_t(v6);
uint64_t const v8 = threadIdx.x;
int64_t const v9 = int64_t(v8);
int64_t const v10 = 0;
int64_t const v11 = 1;
int64_t const v15 = 256;
array<float, 512> * const v379 = reinterpret_cast<array<float, 512> *>(&v378[0]);
bool const v17 = v7 < v376;
if(v17) {
float v381_store;
float * v381 = &v381_store;float const v19 = 0.0;
*v381 = v19;
float v382_store;
float * v382 = &v382_store;*v382 = v19;
for(int64_t v25 = v9; v25 < v377; v25 += v15) {
int64_t const v26 = v25 * v376;
int64_t const v27 = v7 + v26;
uint64_t const v28 = uint64_t(v27);
float const* const v29 = &v1[v28];
float const v30 = *v29;
float const v31 = *v382;
float const v32 = *v381;
float const v33 = v31 + v30;
float const v34 = v33 - v31;
float const v35 = v33 - v34;
float const v36 = v31 - v35;
float const v37 = v30 - v34;
float const v38 = v36 + v37;
float const v39 = v32 + v38;
float const v40 = fabs(v33);
float const v41 = 3.402823470e+38;
bool const v42 = v40 <= v41;
float const v44 = v42 ? v39 : v19;
*v381 = v44;
*v382 = v33;
}
float const v45 = *v382;
uint64_t const v49 = uint64_t(v9);
float * const v50 = &(*v379)[v49];
*v50 = v45;
float const v51 = *v381;
int64_t const v53 = v9 + v15;
uint64_t const v54 = uint64_t(v53);
float * const v55 = &(*v379)[v54];
*v55 = v51;
__syncthreads();
int64_t const v56 = 128;
bool const v57 = v9 < v56;
if(v57) {
int64_t const v58 = v9 + v56;
uint64_t const v67 = uint64_t(v58);
float * const v68 = &(*v379)[v67];
float const v69 = *v50;
float const v70 = *v68;
float const v75 = *v55;
int64_t const v77 = v58 + v15;
uint64_t const v78 = uint64_t(v77);
float * const v79 = &(*v379)[v78];
float const v80 = *v79;
float const v81 = v69 + v70;
float const v82 = v81 - v69;
float const v83 = v81 - v82;
float const v84 = v69 - v83;
float const v85 = v70 - v82;
float const v86 = v84 + v85;
float const v87 = v75 + v86;
float const v88 = fabs(v81);
float const v89 = 3.402823470e+38;
bool const v90 = v88 <= v89;
float const v92 = v90 ? v87 : v19;
float const v93 = v92 + v80;
*v55 = v93;
*v50 = v81;
}
__syncthreads();
int64_t const v94 = 64;
bool const v95 = v9 < v94;
if(v95) {
int64_t const v96 = v9 + v94;
uint64_t const v105 = uint64_t(v96);
float * const v106 = &(*v379)[v105];
float const v107 = *v50;
float const v108 = *v106;
float const v113 = *v55;
int64_t const v115 = v96 + v15;
uint64_t const v116 = uint64_t(v115);
float * const v117 = &(*v379)[v116];
float const v118 = *v117;
float const v119 = v107 + v108;
float const v120 = v119 - v107;
float const v121 = v119 - v120;
float const v122 = v107 - v121;
float const v123 = v108 - v120;
float const v124 = v122 + v123;
float const v125 = v113 + v124;
float const v126 = fabs(v119);
float const v127 = 3.402823470e+38;
bool const v128 = v126 <= v127;
float const v130 = v128 ? v125 : v19;
float const v131 = v130 + v118;
*v55 = v131;
*v50 = v119;
}
__syncthreads();
int64_t const v132 = 32;
bool const v133 = v9 < v132;
if(v133) {
int64_t const v134 = v9 + v132;
uint64_t const v143 = uint64_t(v134);
float * const v144 = &(*v379)[v143];
float const v145 = *v50;
float const v146 = *v144;
float const v151 = *v55;
int64_t const v153 = v134 + v15;
uint64_t const v154 = uint64_t(v153);
float * const v155 = &(*v379)[v154];
float const v156 = *v155;
float const v157 = v145 + v146;
float const v158 = v157 - v145;
float const v159 = v157 - v158;
float const v160 = v145 - v159;
float const v161 = v146 - v158;
float const v162 = v160 + v161;
float const v163 = v151 + v162;
float const v164 = fabs(v157);
float const v165 = 3.402823470e+38;
bool const v166 = v164 <= v165;
float const v168 = v166 ? v163 : v19;
float const v169 = v168 + v156;
*v55 = v169;
*v50 = v157;
}
__syncthreads();
int64_t const v170 = 16;
bool const v171 = v9 < v170;
if(v171) {
int64_t const v172 = v9 + v170;
uint64_t const v181 = uint64_t(v172);
float * const v182 = &(*v379)[v181];
float const v183 = *v50;
float const v184 = *v182;
float const v189 = *v55;
int64_t const v191 = v172 + v15;
uint64_t const v192 = uint64_t(v191);
float * const v193 = &(*v379)[v192];
float const v194 = *v193;
float const v195 = v183 + v184;
float const v196 = v195 - v183;
float const v197 = v195 - v196;
float const v198 = v183 - v197;
float const v199 = v184 - v196;
float const v200 = v198 + v199;
float const v201 = v189 + v200;
float const v202 = fabs(v195);
float const v203 = 3.402823470e+38;
bool const v204 = v202 <= v203;
float const v206 = v204 ? v201 : v19;
float const v207 = v206 + v194;
*v55 = v207;
*v50 = v195;
}
__syncthreads();
int64_t const v208 = 8;
bool const v209 = v9 < v208;
if(v209) {
int64_t const v210 = v9 + v208;
uint64_t const v219 = uint64_t(v210);
float * const v220 = &(*v379)[v219];
float const v221 = *v50;
float const v222 = *v220;
float const v227 = *v55;
int64_t const v229 = v210 + v15;
uint64_t const v230 = uint64_t(v229);
float * const v231 = &(*v379)[v230];
float const v232 = *v231;
float const v233 = v221 + v222;
float const v234 = v233 - v221;
float const v235 = v233 - v234;
float const v236 = v221 - v235;
float const v237 = v222 - v234;
float const v238 = v236 + v237;
float const v239 = v227 + v238;
float const v240 = fabs(v233);
float const v241 = 3.402823470e+38;
bool const v242 = v240 <= v241;
float const v244 = v242 ? v239 : v19;
float const v245 = v244 + v232;
*v55 = v245;
*v50 = v233;
}
__syncthreads();
int64_t const v246 = 4;
bool const v247 = v9 < v246;
if(v247) {
int64_t const v248 = v9 + v246;
uint64_t const v257 = uint64_t(v248);
float * const v258 = &(*v379)[v257];
float const v259 = *v50;
float const v260 = *v258;
float const v265 = *v55;
int64_t const v267 = v248 + v15;
uint64_t const v268 = uint64_t(v267);
float * const v269 = &(*v379)[v268];
float const v270 = *v269;
float const v271 = v259 + v260;
float const v272 = v271 - v259;
float const v273 = v271 - v272;
float const v274 = v259 - v273;
float const v275 = v260 - v272;
float const v276 = v274 + v275;
float const v277 = v265 + v276;
float const v278 = fabs(v271);
float const v279 = 3.402823470e+38;
bool const v280 = v278 <= v279;
float const v282 = v280 ? v277 : v19;
float const v283 = v282 + v270;
*v55 = v283;
*v50 = v271;
}
__syncthreads();
int64_t const v284 = 2;
bool const v285 = v9 < v284;
if(v285) {
int64_t const v286 = v9 + v284;
uint64_t const v295 = uint64_t(v286);
float * const v296 = &(*v379)[v295];
float const v297 = *v50;
float const v298 = *v296;
float const v303 = *v55;
int64_t const v305 = v286 + v15;
uint64_t const v306 = uint64_t(v305);
float * const v307 = &(*v379)[v306];
float const v308 = *v307;
float const v309 = v297 + v298;
float const v310 = v309 - v297;
float const v311 = v309 - v310;
float const v312 = v297 - v311;
float const v313 = v298 - v310;
float const v314 = v312 + v313;
float const v315 = v303 + v314;
float const v316 = fabs(v309);
float const v317 = 3.402823470e+38;
bool const v318 = v316 <= v317;
float const v320 = v318 ? v315 : v19;
float const v321 = v320 + v308;
*v55 = v321;
*v50 = v309;
}
__syncthreads();
bool const v323 = v9 < v11;
if(v323) {
int64_t const v324 = v9 + v11;
uint64_t const v333 = uint64_t(v324);
float * const v334 = &(*v379)[v333];
float const v335 = *v50;
float const v336 = *v334;
float const v341 = *v55;
int64_t const v343 = v324 + v15;
uint64_t const v344 = uint64_t(v343);
float * const v345 = &(*v379)[v344];
float const v346 = *v345;
float const v347 = v335 + v336;
float const v348 = v347 - v335;
float const v349 = v347 - v348;
float const v350 = v335 - v349;
float const v351 = v336 - v348;
float const v352 = v350 + v351;
float const v353 = v341 + v352;
float const v354 = fabs(v347);
float const v355 = 3.402823470e+38;
bool const v356 = v354 <= v355;
float const v358 = v356 ? v353 : v19;
float const v359 = v358 + v346;
*v55 = v359;
*v50 = v347;
}
__syncthreads();
uint64_t const v385 = 0;
float * const v361 = &(*v379)[v385];
float const v362 = *v361;
uint64_t const v383 = 256;
float * const v366 = &(*v379)[v383];
float const v367 = *v366;
float const v368 = v362 + v367;
bool const v369 = v9 == v10;
if(v369) {
uint64_t const v370 = uint64_t(v7);
float * const v371 = &v2[v370];
*v371 = v368;
}
}
return;
}
