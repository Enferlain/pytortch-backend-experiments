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
    int64_t scalars_i64[3];uint64_t static_meta[2];
};
extern "C" __global__ void __launch_bounds__(256)  cubecl5_rms_cubek_vec4_b07174f9020a573a(__half const* const __restrict__ v1, __half * const __restrict__ v2, info_st const* const __restrict__ v212) {
uint64_t const* const dynamic_meta_v214 = reinterpret_cast<uint64_t const*>(v212 + 1);
const info_st& info_v213 = *v212;int64_t const v216 = info_v213.scalars_i64[1];
int64_t const v217 = info_v213.scalars_i64[2];
uint64_t const v6 = blockIdx.x;
int64_t const v7 = int64_t(v6);
uint64_t const v8 = threadIdx.y;
int64_t const v9 = int64_t(v8);
int64_t const v10 = 8;
int64_t const v11 = v7 * v10;
int64_t const v12 = v11 + v9;
uint64_t const v13 = uint64_t(v12);
uint64_t const v14 = uint64_t(v216);
uint64_t const v15 = uint64_t(v217);
bool const v16 = v13 < v15;
if(v16) {
uint64_t const v18 = 0;
uint64_t v219_store;
uint64_t * v219 = &v219_store;*v219 = v18;
uint64_t const v22 = 2;
uint64_t const v23 = 1;
for(uint64_t v24 = v18; v24 < v22; v24 += v23) {
int64_t const v27 = int64_t(v24);
int64_t const v28 = 0;
bool const v29 = v27 == v28;
int64_t const v30 = 1;
int64_t const v39 = v29 ? v217 : v30;
uint64_t const v40 = uint64_t(v39);
uint64_t const v41 = v13 % v40;
int64_t const v46 = v29 ? v216 : v30;
uint64_t const v47 = uint64_t(v46);
uint64_t const v48 = v41 * v47;
uint64_t const v49 = *v219;
uint64_t const v50 = v49 + v48;
*v219 = v50;
}
uint64_t const v51 = *v219;
uint64_t const v52 = 4;
uint64_t const v53 = v51 / v52;
*v219 = v53;
int64_t const v313 = 1;
uint64_t const v61 = v14 / v52;
float const v62 = 0.0;
float_4 const v63 = float_4{v62, v62, v62, v62};
int64_t const v68 = 4;
float_4 v224_store;
float_4 * v224 = &v224_store;*v224 = v63;
uint64_t const v311 = 32;
uint64_t const v78 = v61 + v311;
uint64_t const v80 = v78 - v23;
uint64_t const v81 = v80 / v311;
for(uint64_t v84 = v18; v84 < v81; v84 += v23) {
uint64_t const v88 = v84 * v311;
uint32_t const v89 = threadIdx.x;
uint64_t const v90 = uint64_t(v89);
uint64_t const v91 = v88 + v90;
uint64_t const v92 = *v219;
uint64_t const v93 = v91 + v92;
bool const v94 = v91 < v61;
float_4 v227_store;
float_4 * v227 = &v227_store;if(v94) {
int64_t const v95 = int64_t(v93);
int64_t const v97 = v95 * v68;
uint64_t const v100 = uint64_t(v97);
int64_t const v101 = int64_t(v100);
int64_t const v102 = v12 * v216;
int64_t const v103 = v101 - v102;
int64_t const v105 = v103 + v102;
uint64_t const v106 = uint64_t(v105);
__half const* const v107 = &v1[v106];
__half const v108 = *v107;
float const v109 = float(v108);
__half const v110 = __half(v109);
float const v111 = float(v110);
float const v113 = v111 * v111;
int64_t const v115 = v97 + v313;
uint64_t const v116 = uint64_t(v115);
int64_t const v117 = int64_t(v116);
int64_t const v119 = v117 - v102;
int64_t const v121 = v119 + v102;
uint64_t const v122 = uint64_t(v121);
__half const* const v123 = &v1[v122];
__half const v124 = *v123;
float const v125 = float(v124);
__half const v126 = __half(v125);
float const v127 = float(v126);
float const v129 = v127 * v127;
int64_t const v130 = 2;
int64_t const v131 = v97 + v130;
uint64_t const v132 = uint64_t(v131);
int64_t const v133 = int64_t(v132);
int64_t const v135 = v133 - v102;
int64_t const v137 = v135 + v102;
uint64_t const v138 = uint64_t(v137);
__half const* const v139 = &v1[v138];
__half const v140 = *v139;
float const v141 = float(v140);
__half const v142 = __half(v141);
float const v143 = float(v142);
float const v145 = v143 * v143;
int64_t const v146 = 3;
int64_t const v147 = v97 + v146;
uint64_t const v148 = uint64_t(v147);
int64_t const v149 = int64_t(v148);
int64_t const v151 = v149 - v102;
int64_t const v153 = v151 + v102;
uint64_t const v154 = uint64_t(v153);
__half const* const v155 = &v1[v154];
__half const v156 = *v155;
float const v157 = float(v156);
__half const v158 = __half(v157);
float const v159 = float(v158);
float const v161 = v159 * v159;
float_4 const v162 = float_4{v113, v129, v145, v161};
*v227 = v162;
}
 else {
*v227 = v63;
}
float_4 const v169 = *v227;
float_4 const v170 = *v224;
float const v274 = v170.i_0;
float const v275 = v169.i_0;
float const v276 = v274 + v275;
float const v277 = v170.i_1;
float const v278 = v169.i_1;
float const v279 = v277 + v278;
float const v280 = v170.i_2;
float const v281 = v169.i_2;
float const v282 = v280 + v281;
float const v283 = v170.i_3;
float const v284 = v169.i_3;
float const v285 = v283 + v284;
float_4 const v286 = float_4{v276, v279, v282, v285};
*v224 = v286;
}
float_4 const v172 = *v224;
uint32_t const v228 = warpSize;
uint32_t const v310 = 256;
uint32_t const v230 = min(v228, v310);
float_4 v268_store;
float_4 * v268 = &v268_store;*v268 = v172;
uint32_t const v232 = 1;
uint32_t v269_store;
uint32_t * v269 = &v269_store;*v269 = v232;
while([&]{
uint32_t const v235 = *v269;
bool const v236 = v235 < v230;
return v236;
}()) {
float_4 const v239 = *v268;
uint32_t const v240 = *v269;
float const v287 = v239.i_0;
float const v288 = __shfl_xor_sync(__activemask(), v287, v240);
float const v289 = v239.i_1;
float const v290 = __shfl_xor_sync(__activemask(), v289, v240);
float const v291 = v239.i_2;
float const v292 = __shfl_xor_sync(__activemask(), v291, v240);
float const v293 = v239.i_3;
float const v294 = __shfl_xor_sync(__activemask(), v293, v240);
float const v298 = v287 + v288;
float const v301 = v289 + v290;
float const v304 = v291 + v292;
float const v307 = v293 + v294;
float_4 const v308 = float_4{v298, v301, v304, v307};
*v268 = v308;
uint32_t const v245 = 2;
uint32_t const v246 = v240 * v245;
*v269 = v246;
}
float_4 const v247 = *v268;
*v224 = v247;
float_4 const v174 = *v224;
float const v253 = v174.i_0;
float const v255 = v62 + v253;
float const v256 = v174.i_1;
float const v258 = v255 + v256;
float const v259 = v174.i_2;
float const v261 = v258 + v259;
float const v262 = v174.i_3;
float const v264 = v261 + v262;
uint32_t const v176 = threadIdx.x;
uint64_t const v177 = uint64_t(v176);
for(uint64_t v179 = v177; v179 < v14; v179 += v311) {
uint64_t const v182 = v13 * v14;
uint64_t const v183 = v182 + v179;
int64_t const v184 = int64_t(v183);
int64_t const v185 = v12 * v216;
int64_t const v186 = v184 - v185;
int64_t const v188 = v186 + v185;
uint64_t const v189 = uint64_t(v188);
__half const* const v190 = &v1[v189];
__half const v191 = *v190;
float const v192 = float(v191);
__half const v193 = __half(v192);
float const v194 = float(v193);
float const v197 = float(v216);
float const v198 = v264 / v197;
float const v199 = 9.999999740e-06;
float const v200 = v198 + v199;
float const v201 = rsqrt(v200);float const v202 = v194 * v201;
__half const v203 = __half(v202);
float const v204 = float(v203);
__half const v205 = __half(v204);
float const v206 = float(v205);
__half const v207 = __half(v206);
__half * const v211 = &v2[v189];
*v211 = v207;
}
}
return;
}
