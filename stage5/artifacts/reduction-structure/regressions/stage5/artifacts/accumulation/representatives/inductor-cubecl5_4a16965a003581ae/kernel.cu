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
    int64_t scalars_i64[3];uint64_t static_meta[4];
};
extern "C" __global__ void __launch_bounds__(256)  cubecl5_4a16965a003581ae(float const* const __restrict__ v1, float const* const __restrict__ v2, float const* const __restrict__ v3, float * const __restrict__ v4, info_st const* const __restrict__ v406) {
extern __shared__ __align__(4) char v412[];uint64_t const* const dynamic_meta_v408 = reinterpret_cast<uint64_t const*>(v406 + 1);
const info_st& info_v407 = *v406;int64_t const v410 = info_v407.scalars_i64[1];
int64_t const v411 = info_v407.scalars_i64[2];
uint64_t const v8 = blockIdx.x;
int64_t const v9 = int64_t(v8);
uint64_t const v10 = threadIdx.x;
int64_t const v11 = int64_t(v10);
int64_t const v13 = 1;
int64_t const v17 = 256;
array<float, 512> * const v413 = reinterpret_cast<array<float, 512> *>(&v412[0]);
bool const v19 = v9 < v411;
if(v19) {
float v415_store;
float * v415 = &v415_store;float const v21 = 0.0;
*v415 = v21;
float v416_store;
float * v416 = &v416_store;*v416 = v21;
for(int64_t v27 = v11; v27 < v410; v27 += v17) {
int64_t const v28 = v9 * v410;
int64_t const v29 = v27 + v28;
uint64_t const v30 = uint64_t(v29);
float const* const v31 = &v1[v30];
float const v32 = *v31;
float const* const v36 = &v2[v30];
float const v37 = *v36;
float const v38 = v32 * v37;
float const v39 = *v416;
float const v40 = *v415;
float const v41 = v39 + v38;
float const v42 = v41 - v39;
float const v43 = v41 - v42;
float const v44 = v39 - v43;
float const v45 = v38 - v42;
float const v46 = v44 + v45;
float const v47 = v40 + v46;
float const v48 = fabs(v41);
float const v49 = 3.402823470e+38;
bool const v50 = v48 <= v49;
float const v52 = v50 ? v47 : v21;
*v415 = v52;
*v416 = v41;
}
float const v53 = *v416;
uint64_t const v57 = uint64_t(v11);
float * const v58 = &(*v413)[v57];
*v58 = v53;
float const v59 = *v415;
int64_t const v61 = v11 + v17;
uint64_t const v62 = uint64_t(v61);
float * const v63 = &(*v413)[v62];
*v63 = v59;
__syncthreads();
int64_t const v64 = 128;
bool const v65 = v11 < v64;
if(v65) {
int64_t const v66 = v11 + v64;
uint64_t const v75 = uint64_t(v66);
float * const v76 = &(*v413)[v75];
float const v77 = *v58;
float const v78 = *v76;
float const v83 = *v63;
int64_t const v85 = v66 + v17;
uint64_t const v86 = uint64_t(v85);
float * const v87 = &(*v413)[v86];
float const v88 = *v87;
float const v89 = v77 + v78;
float const v90 = v89 - v77;
float const v91 = v89 - v90;
float const v92 = v77 - v91;
float const v93 = v78 - v90;
float const v94 = v92 + v93;
float const v95 = v83 + v94;
float const v96 = fabs(v89);
float const v97 = 3.402823470e+38;
bool const v98 = v96 <= v97;
float const v100 = v98 ? v95 : v21;
float const v101 = v100 + v88;
*v63 = v101;
*v58 = v89;
}
__syncthreads();
int64_t const v102 = 64;
bool const v103 = v11 < v102;
if(v103) {
int64_t const v104 = v11 + v102;
uint64_t const v113 = uint64_t(v104);
float * const v114 = &(*v413)[v113];
float const v115 = *v58;
float const v116 = *v114;
float const v121 = *v63;
int64_t const v123 = v104 + v17;
uint64_t const v124 = uint64_t(v123);
float * const v125 = &(*v413)[v124];
float const v126 = *v125;
float const v127 = v115 + v116;
float const v128 = v127 - v115;
float const v129 = v127 - v128;
float const v130 = v115 - v129;
float const v131 = v116 - v128;
float const v132 = v130 + v131;
float const v133 = v121 + v132;
float const v134 = fabs(v127);
float const v135 = 3.402823470e+38;
bool const v136 = v134 <= v135;
float const v138 = v136 ? v133 : v21;
float const v139 = v138 + v126;
*v63 = v139;
*v58 = v127;
}
__syncthreads();
int64_t const v140 = 32;
bool const v141 = v11 < v140;
if(v141) {
int64_t const v142 = v11 + v140;
uint64_t const v151 = uint64_t(v142);
float * const v152 = &(*v413)[v151];
float const v153 = *v58;
float const v154 = *v152;
float const v159 = *v63;
int64_t const v161 = v142 + v17;
uint64_t const v162 = uint64_t(v161);
float * const v163 = &(*v413)[v162];
float const v164 = *v163;
float const v165 = v153 + v154;
float const v166 = v165 - v153;
float const v167 = v165 - v166;
float const v168 = v153 - v167;
float const v169 = v154 - v166;
float const v170 = v168 + v169;
float const v171 = v159 + v170;
float const v172 = fabs(v165);
float const v173 = 3.402823470e+38;
bool const v174 = v172 <= v173;
float const v176 = v174 ? v171 : v21;
float const v177 = v176 + v164;
*v63 = v177;
*v58 = v165;
}
__syncthreads();
int64_t const v178 = 16;
bool const v179 = v11 < v178;
if(v179) {
int64_t const v180 = v11 + v178;
uint64_t const v189 = uint64_t(v180);
float * const v190 = &(*v413)[v189];
float const v191 = *v58;
float const v192 = *v190;
float const v197 = *v63;
int64_t const v199 = v180 + v17;
uint64_t const v200 = uint64_t(v199);
float * const v201 = &(*v413)[v200];
float const v202 = *v201;
float const v203 = v191 + v192;
float const v204 = v203 - v191;
float const v205 = v203 - v204;
float const v206 = v191 - v205;
float const v207 = v192 - v204;
float const v208 = v206 + v207;
float const v209 = v197 + v208;
float const v210 = fabs(v203);
float const v211 = 3.402823470e+38;
bool const v212 = v210 <= v211;
float const v214 = v212 ? v209 : v21;
float const v215 = v214 + v202;
*v63 = v215;
*v58 = v203;
}
__syncthreads();
int64_t const v216 = 8;
bool const v217 = v11 < v216;
if(v217) {
int64_t const v218 = v11 + v216;
uint64_t const v227 = uint64_t(v218);
float * const v228 = &(*v413)[v227];
float const v229 = *v58;
float const v230 = *v228;
float const v235 = *v63;
int64_t const v237 = v218 + v17;
uint64_t const v238 = uint64_t(v237);
float * const v239 = &(*v413)[v238];
float const v240 = *v239;
float const v241 = v229 + v230;
float const v242 = v241 - v229;
float const v243 = v241 - v242;
float const v244 = v229 - v243;
float const v245 = v230 - v242;
float const v246 = v244 + v245;
float const v247 = v235 + v246;
float const v248 = fabs(v241);
float const v249 = 3.402823470e+38;
bool const v250 = v248 <= v249;
float const v252 = v250 ? v247 : v21;
float const v253 = v252 + v240;
*v63 = v253;
*v58 = v241;
}
__syncthreads();
int64_t const v254 = 4;
bool const v255 = v11 < v254;
if(v255) {
int64_t const v256 = v11 + v254;
uint64_t const v265 = uint64_t(v256);
float * const v266 = &(*v413)[v265];
float const v267 = *v58;
float const v268 = *v266;
float const v273 = *v63;
int64_t const v275 = v256 + v17;
uint64_t const v276 = uint64_t(v275);
float * const v277 = &(*v413)[v276];
float const v278 = *v277;
float const v279 = v267 + v268;
float const v280 = v279 - v267;
float const v281 = v279 - v280;
float const v282 = v267 - v281;
float const v283 = v268 - v280;
float const v284 = v282 + v283;
float const v285 = v273 + v284;
float const v286 = fabs(v279);
float const v287 = 3.402823470e+38;
bool const v288 = v286 <= v287;
float const v290 = v288 ? v285 : v21;
float const v291 = v290 + v278;
*v63 = v291;
*v58 = v279;
}
__syncthreads();
int64_t const v292 = 2;
bool const v293 = v11 < v292;
if(v293) {
int64_t const v294 = v11 + v292;
uint64_t const v303 = uint64_t(v294);
float * const v304 = &(*v413)[v303];
float const v305 = *v58;
float const v306 = *v304;
float const v311 = *v63;
int64_t const v313 = v294 + v17;
uint64_t const v314 = uint64_t(v313);
float * const v315 = &(*v413)[v314];
float const v316 = *v315;
float const v317 = v305 + v306;
float const v318 = v317 - v305;
float const v319 = v317 - v318;
float const v320 = v305 - v319;
float const v321 = v306 - v318;
float const v322 = v320 + v321;
float const v323 = v311 + v322;
float const v324 = fabs(v317);
float const v325 = 3.402823470e+38;
bool const v326 = v324 <= v325;
float const v328 = v326 ? v323 : v21;
float const v329 = v328 + v316;
*v63 = v329;
*v58 = v317;
}
__syncthreads();
bool const v331 = v11 < v13;
if(v331) {
int64_t const v332 = v11 + v13;
uint64_t const v341 = uint64_t(v332);
float * const v342 = &(*v413)[v341];
float const v343 = *v58;
float const v344 = *v342;
float const v349 = *v63;
int64_t const v351 = v332 + v17;
uint64_t const v352 = uint64_t(v351);
float * const v353 = &(*v413)[v352];
float const v354 = *v353;
float const v355 = v343 + v344;
float const v356 = v355 - v343;
float const v357 = v355 - v356;
float const v358 = v343 - v357;
float const v359 = v344 - v356;
float const v360 = v358 + v359;
float const v361 = v349 + v360;
float const v362 = fabs(v355);
float const v363 = 3.402823470e+38;
bool const v364 = v362 <= v363;
float const v366 = v364 ? v361 : v21;
float const v367 = v366 + v354;
*v63 = v367;
*v58 = v355;
}
__syncthreads();
uint64_t const v419 = 0;
float * const v369 = &(*v413)[v419];
float const v370 = *v369;
uint64_t const v417 = 256;
float * const v374 = &(*v413)[v417];
float const v375 = *v374;
float const v376 = v370 + v375;
for(int64_t v377 = v11; v377 < v410; v377 += v17) {
int64_t const v378 = v9 * v410;
int64_t const v379 = v377 + v378;
uint64_t const v380 = uint64_t(v379);
float const* const v381 = &v1[v380];
float const v382 = *v381;
float const* const v386 = &v2[v380];
float const v387 = *v386;
uint64_t const v388 = uint64_t(v9);
float const* const v389 = &v3[v388];
float const v390 = *v389;
float const v391 = v382 * v390;
float const v392 = -0.5;
float const v393 = v376 * v392;
float const v394 = v390 * v390;
float const v395 = v394 * v390;
float const v396 = v393 * v395;
float const v397 = float(v410);
float const v398 = v396 / v397;
float const v399 = v398 * v387;
float const v400 = v391 + v399;
float const v401 = v400 + v399;
float * const v405 = &v4[v380];
*v405 = v401;
}
}
return;
}
