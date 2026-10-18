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
    int64_t scalars_i64[3];uint64_t static_meta[2];
};
extern "C" __global__ void __launch_bounds__(256)  cubecl5_rms_cubek_vec4_5c853a16b2fca1e9(float const* const __restrict__ v1, float * const __restrict__ v2, info_st const* const __restrict__ v187) {
uint64_t const* const dynamic_meta_v189 = reinterpret_cast<uint64_t const*>(v187 + 1);
const info_st& info_v188 = *v187;int64_t const v191 = info_v188.scalars_i64[1];
int64_t const v192 = info_v188.scalars_i64[2];
uint64_t const v6 = blockIdx.x;
int64_t const v7 = int64_t(v6);
uint64_t const v8 = threadIdx.y;
int64_t const v9 = int64_t(v8);
int64_t const v10 = 8;
int64_t const v11 = v7 * v10;
int64_t const v12 = v11 + v9;
uint64_t const v13 = uint64_t(v12);
uint64_t const v14 = uint64_t(v191);
uint64_t const v15 = uint64_t(v192);
bool const v16 = v13 < v15;
if(v16) {
uint64_t const v18 = 0;
uint64_t v194_store;
uint64_t * v194 = &v194_store;*v194 = v18;
uint64_t const v22 = 2;
uint64_t const v23 = 1;
for(uint64_t v24 = v18; v24 < v22; v24 += v23) {
int64_t const v27 = int64_t(v24);
int64_t const v28 = 0;
bool const v29 = v27 == v28;
int64_t const v30 = 1;
int64_t const v39 = v29 ? v192 : v30;
uint64_t const v40 = uint64_t(v39);
uint64_t const v41 = v13 % v40;
int64_t const v46 = v29 ? v191 : v30;
uint64_t const v47 = uint64_t(v46);
uint64_t const v48 = v41 * v47;
uint64_t const v49 = *v194;
uint64_t const v50 = v49 + v48;
*v194 = v50;
}
uint64_t const v51 = *v194;
uint64_t const v52 = 4;
uint64_t const v53 = v51 / v52;
*v194 = v53;
int64_t const v288 = 1;
uint64_t const v61 = v14 / v52;
float const v62 = 0.0;
float_4 const v63 = float_4{v62, v62, v62, v62};
int64_t const v68 = 4;
float_4 v199_store;
float_4 * v199 = &v199_store;*v199 = v63;
uint64_t const v286 = 32;
uint64_t const v78 = v61 + v286;
uint64_t const v80 = v78 - v23;
uint64_t const v81 = v80 / v286;
for(uint64_t v84 = v18; v84 < v81; v84 += v23) {
uint64_t const v88 = v84 * v286;
uint32_t const v89 = threadIdx.x;
uint64_t const v90 = uint64_t(v89);
uint64_t const v91 = v88 + v90;
uint64_t const v92 = *v194;
uint64_t const v93 = v91 + v92;
bool const v94 = v91 < v61;
float_4 v202_store;
float_4 * v202 = &v202_store;if(v94) {
int64_t const v95 = int64_t(v93);
int64_t const v97 = v95 * v68;
uint64_t const v100 = uint64_t(v97);
int64_t const v101 = int64_t(v100);
int64_t const v102 = v12 * v191;
int64_t const v103 = v101 - v102;
int64_t const v105 = v103 + v102;
uint64_t const v106 = uint64_t(v105);
float const* const v107 = &v1[v106];
float const v108 = *v107;
float const v109 = v108 * v108;
int64_t const v111 = v97 + v288;
uint64_t const v112 = uint64_t(v111);
int64_t const v113 = int64_t(v112);
int64_t const v115 = v113 - v102;
int64_t const v117 = v115 + v102;
uint64_t const v118 = uint64_t(v117);
float const* const v119 = &v1[v118];
float const v120 = *v119;
float const v121 = v120 * v120;
int64_t const v122 = 2;
int64_t const v123 = v97 + v122;
uint64_t const v124 = uint64_t(v123);
int64_t const v125 = int64_t(v124);
int64_t const v127 = v125 - v102;
int64_t const v129 = v127 + v102;
uint64_t const v130 = uint64_t(v129);
float const* const v131 = &v1[v130];
float const v132 = *v131;
float const v133 = v132 * v132;
int64_t const v134 = 3;
int64_t const v135 = v97 + v134;
uint64_t const v136 = uint64_t(v135);
int64_t const v137 = int64_t(v136);
int64_t const v139 = v137 - v102;
int64_t const v141 = v139 + v102;
uint64_t const v142 = uint64_t(v141);
float const* const v143 = &v1[v142];
float const v144 = *v143;
float const v145 = v144 * v144;
float_4 const v146 = float_4{v109, v121, v133, v145};
*v202 = v146;
}
 else {
*v202 = v63;
}
float_4 const v153 = *v202;
float_4 const v154 = *v199;
float const v249 = v154.i_0;
float const v250 = v153.i_0;
float const v251 = v249 + v250;
float const v252 = v154.i_1;
float const v253 = v153.i_1;
float const v254 = v252 + v253;
float const v255 = v154.i_2;
float const v256 = v153.i_2;
float const v257 = v255 + v256;
float const v258 = v154.i_3;
float const v259 = v153.i_3;
float const v260 = v258 + v259;
float_4 const v261 = float_4{v251, v254, v257, v260};
*v199 = v261;
}
float_4 const v156 = *v199;
uint32_t const v203 = warpSize;
uint32_t const v285 = 256;
uint32_t const v205 = min(v203, v285);
float_4 v243_store;
float_4 * v243 = &v243_store;*v243 = v156;
uint32_t const v207 = 1;
uint32_t v244_store;
uint32_t * v244 = &v244_store;*v244 = v207;
while([&]{
uint32_t const v210 = *v244;
bool const v211 = v210 < v205;
return v211;
}()) {
float_4 const v214 = *v243;
uint32_t const v215 = *v244;
float const v262 = v214.i_0;
float const v263 = __shfl_xor_sync(__activemask(), v262, v215);
float const v264 = v214.i_1;
float const v265 = __shfl_xor_sync(__activemask(), v264, v215);
float const v266 = v214.i_2;
float const v267 = __shfl_xor_sync(__activemask(), v266, v215);
float const v268 = v214.i_3;
float const v269 = __shfl_xor_sync(__activemask(), v268, v215);
float const v273 = v262 + v263;
float const v276 = v264 + v265;
float const v279 = v266 + v267;
float const v282 = v268 + v269;
float_4 const v283 = float_4{v273, v276, v279, v282};
*v243 = v283;
uint32_t const v220 = 2;
uint32_t const v221 = v215 * v220;
*v244 = v221;
}
float_4 const v222 = *v243;
*v199 = v222;
float_4 const v158 = *v199;
float const v228 = v158.i_0;
float const v230 = v62 + v228;
float const v231 = v158.i_1;
float const v233 = v230 + v231;
float const v234 = v158.i_2;
float const v236 = v233 + v234;
float const v237 = v158.i_3;
float const v239 = v236 + v237;
uint32_t const v160 = threadIdx.x;
uint64_t const v161 = uint64_t(v160);
for(uint64_t v163 = v161; v163 < v14; v163 += v286) {
uint64_t const v166 = v13 * v14;
uint64_t const v167 = v166 + v163;
int64_t const v168 = int64_t(v167);
int64_t const v169 = v12 * v191;
int64_t const v170 = v168 - v169;
int64_t const v172 = v170 + v169;
uint64_t const v173 = uint64_t(v172);
float const* const v174 = &v1[v173];
float const v175 = *v174;
float const v177 = float(v191);
float const v178 = v239 / v177;
float const v179 = 9.999999740e-06;
float const v180 = v178 + v179;
float const v181 = rsqrt(v180);float const v182 = v175 * v181;
float * const v186 = &v2[v173];
*v186 = v182;
}
}
return;
}
