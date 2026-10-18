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
extern "C" __global__ void __launch_bounds__(256)  cubecl5_rms_cubek_pack4_835adf5c9c772d31(float const* const __restrict__ v1, float * const __restrict__ v2, info_st const* const __restrict__ v181) {
uint64_t const* const dynamic_meta_v183 = reinterpret_cast<uint64_t const*>(v181 + 1);
const info_st& info_v182 = *v181;int64_t const v185 = info_v182.scalars_i64[1];
int64_t const v186 = info_v182.scalars_i64[2];
uint64_t const v6 = blockIdx.x;
int64_t const v7 = int64_t(v6);
uint64_t const v8 = threadIdx.y;
int64_t const v9 = int64_t(v8);
int64_t const v10 = 8;
int64_t const v11 = v7 * v10;
int64_t const v12 = v11 + v9;
uint64_t const v13 = uint64_t(v12);
uint64_t const v14 = uint64_t(v185);
uint64_t const v15 = uint64_t(v186);
bool const v16 = v13 < v15;
if(v16) {
uint64_t const v18 = 0;
uint64_t v188_store;
uint64_t * v188 = &v188_store;*v188 = v18;
uint64_t const v22 = 2;
uint64_t const v23 = 1;
for(uint64_t v24 = v18; v24 < v22; v24 += v23) {
int64_t const v27 = int64_t(v24);
int64_t const v28 = 0;
bool const v29 = v27 == v28;
int64_t const v30 = 1;
int64_t const v39 = v29 ? v186 : v30;
uint64_t const v40 = uint64_t(v39);
uint64_t const v41 = v13 % v40;
int64_t const v46 = v29 ? v185 : v30;
uint64_t const v47 = uint64_t(v46);
uint64_t const v48 = v41 * v47;
uint64_t const v49 = *v188;
uint64_t const v50 = v49 + v48;
*v188 = v50;
}
uint64_t const v51 = *v188;
uint64_t const v52 = 4;
uint64_t const v53 = v51 / v52;
*v188 = v53;
uint64_t const v61 = v14 / v52;
float const v62 = 0.0;
float_4 const v63 = float_4{v62, v62, v62, v62};
int64_t const v68 = 4;
float_4 v193_store;
float_4 * v193 = &v193_store;*v193 = v63;
uint64_t const v280 = 32;
uint64_t const v78 = v61 + v280;
uint64_t const v80 = v78 - v23;
uint64_t const v81 = v80 / v280;
for(uint64_t v84 = v18; v84 < v81; v84 += v23) {
uint64_t const v88 = v84 * v280;
uint32_t const v89 = threadIdx.x;
uint64_t const v90 = uint64_t(v89);
uint64_t const v91 = v88 + v90;
uint64_t const v92 = *v188;
uint64_t const v93 = v91 + v92;
bool const v94 = v91 < v61;
float_4 v196_store;
float_4 * v196 = &v196_store;if(v94) {
int64_t const v95 = int64_t(v93);
int64_t const v97 = v95 * v68;
uint64_t const v98 = uint64_t(v97);
int64_t const v99 = int64_t(v98);
int64_t const v100 = v12 * v185;
int64_t const v101 = v99 - v100;
int64_t const v103 = v101 + v100;
uint64_t const v104 = uint64_t(v103);
float const* const v105 = &v1[v104];
float_4 const* const v106 = reinterpret_cast<float_4 const*>(v105);
float_4 const v107 = *v106;
float const v108 = v107.i_0;
float const v115 = v108 * v108;
float const v116 = v107.i_1;
float const v123 = v116 * v116;
float const v124 = v107.i_2;
float const v131 = v124 * v124;
float const v132 = v107.i_3;
float const v139 = v132 * v132;
float_4 const v140 = float_4{v115, v123, v131, v139};
*v196 = v140;
}
 else {
*v196 = v63;
}
float_4 const v147 = *v196;
float_4 const v148 = *v193;
float const v243 = v148.i_0;
float const v244 = v147.i_0;
float const v245 = v243 + v244;
float const v246 = v148.i_1;
float const v247 = v147.i_1;
float const v248 = v246 + v247;
float const v249 = v148.i_2;
float const v250 = v147.i_2;
float const v251 = v249 + v250;
float const v252 = v148.i_3;
float const v253 = v147.i_3;
float const v254 = v252 + v253;
float_4 const v255 = float_4{v245, v248, v251, v254};
*v193 = v255;
}
float_4 const v150 = *v193;
uint32_t const v197 = warpSize;
uint32_t const v279 = 256;
uint32_t const v199 = min(v197, v279);
float_4 v237_store;
float_4 * v237 = &v237_store;*v237 = v150;
uint32_t const v201 = 1;
uint32_t v238_store;
uint32_t * v238 = &v238_store;*v238 = v201;
while([&]{
uint32_t const v204 = *v238;
bool const v205 = v204 < v199;
return v205;
}()) {
float_4 const v208 = *v237;
uint32_t const v209 = *v238;
float const v256 = v208.i_0;
float const v257 = __shfl_xor_sync(__activemask(), v256, v209);
float const v258 = v208.i_1;
float const v259 = __shfl_xor_sync(__activemask(), v258, v209);
float const v260 = v208.i_2;
float const v261 = __shfl_xor_sync(__activemask(), v260, v209);
float const v262 = v208.i_3;
float const v263 = __shfl_xor_sync(__activemask(), v262, v209);
float const v267 = v256 + v257;
float const v270 = v258 + v259;
float const v273 = v260 + v261;
float const v276 = v262 + v263;
float_4 const v277 = float_4{v267, v270, v273, v276};
*v237 = v277;
uint32_t const v214 = 2;
uint32_t const v215 = v209 * v214;
*v238 = v215;
}
float_4 const v216 = *v237;
*v193 = v216;
float_4 const v152 = *v193;
float const v222 = v152.i_0;
float const v224 = v62 + v222;
float const v225 = v152.i_1;
float const v227 = v224 + v225;
float const v228 = v152.i_2;
float const v230 = v227 + v228;
float const v231 = v152.i_3;
float const v233 = v230 + v231;
uint32_t const v154 = threadIdx.x;
uint64_t const v155 = uint64_t(v154);
for(uint64_t v157 = v155; v157 < v14; v157 += v280) {
uint64_t const v160 = v13 * v14;
uint64_t const v161 = v160 + v157;
int64_t const v162 = int64_t(v161);
int64_t const v163 = v12 * v185;
int64_t const v164 = v162 - v163;
int64_t const v166 = v164 + v163;
uint64_t const v167 = uint64_t(v166);
float const* const v168 = &v1[v167];
float const v169 = *v168;
float const v171 = float(v185);
float const v172 = v233 / v171;
float const v173 = 9.999999740e-06;
float const v174 = v172 + v173;
float const v175 = rsqrt(v174);float const v176 = v169 * v175;
float * const v180 = &v2[v167];
*v180 = v176;
}
}
return;
}
