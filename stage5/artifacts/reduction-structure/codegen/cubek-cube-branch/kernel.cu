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
    uint64_t scalars_usize[2];uint64_t static_meta[6];
};
extern "C" __global__ void __launch_bounds__(512)  cubek_reader_cube_branch(float const* const __restrict__ v1, float * const __restrict__ v2, info_st const* const __restrict__ v180) {
extern __shared__ __align__(4) char v199[];uint64_t const* const dynamic_meta_v182 = reinterpret_cast<uint64_t const*>(v180 + 1);
const info_st& info_v181 = *v180;uint64_t const v184 = info_v181.scalars_usize[1];
uint64_t const v186 = info_v181.static_meta[1];
uint64_t const v7 = 0;
uint64_t const v13 = 2;
uint64_t v206_store;
uint64_t * v206 = &v206_store;*v206 = v184;
uint32_t const v250 = blockIdx.z;
uint64_t const v251 = uint64_t(v250);
uint32_t const v252 = gridDim.y;
uint64_t const v253 = uint64_t(v252);
uint64_t const v254 = v251 * v253;
uint32_t const v255 = gridDim.x;
uint64_t const v256 = uint64_t(v255);
uint64_t const v257 = v254 * v256;
uint32_t const v258 = blockIdx.y;
uint64_t const v259 = uint64_t(v258);
uint64_t const v262 = v259 * v256;
uint64_t const v263 = v257 + v262;
uint32_t const v264 = blockIdx.x;
uint64_t const v265 = uint64_t(v264);
uint64_t const v266 = v263 + v265;
uint64_t const v187 = info_v181.static_meta[3];
uint64_t const v29 = 1;
float const v31 = 0.0;
uint64_t const v189 = info_v181.static_meta[2];
uint64_t const v190 = dynamic_meta_v182[v189 + v29];
float v209_store;
float * v209 = &v209_store;*v209 = v31;
uint64_t v211_store;
uint64_t * v211 = &v211_store;*v211 = v266;
uint32_t const v40 = threadIdx.y;
uint64_t const v41 = uint64_t(v40);
uint32_t const v267 = 32;
uint64_t v214_store;
uint64_t * v214 = &v214_store;*v214 = v7;
for(uint64_t v56 = v7; v56 < v13; v56 += v29) {
uint64_t const v195 = info_v181.static_meta[5];
uint64_t const v196 = dynamic_meta_v182[v195 + v56];
uint64_t const v61 = v266 / v196;
uint64_t const v192 = dynamic_meta_v182[v187 + v56];
uint64_t const v64 = v61 % v192;
uint64_t const v197 = info_v181.static_meta[4];
uint64_t const v198 = dynamic_meta_v182[v197 + v56];
uint64_t const v67 = v64 * v198;
uint64_t const v68 = *v214;
uint64_t const v69 = v68 + v67;
*v214 = v69;
}
uint64_t const v70 = *v214;
*v214 = v70;
float v219_store;
float * v219 = &v219_store;*v219 = v31;
uint32_t const v268 = 512;
uint64_t const v286 = 512;
uint64_t const v87 = v190 + v286;
uint64_t const v89 = v87 - v29;
uint64_t const v90 = v89 / v286;
for(uint64_t v93 = v7; v93 < v90; v93 += v29) {
uint64_t const v98 = v93 * v286;
uint32_t const v270 = threadIdx.x;
uint32_t const v273 = v40 * v267;
uint32_t const v274 = v270 + v273;
uint64_t const v100 = uint64_t(v274);
uint64_t const v101 = v98 + v100;
uint64_t const v102 = *v214;
uint64_t const v103 = v101 + v102;
bool const v104 = v101 < v190;
float v222_store;
float * v222 = &v222_store;if(v104) {
float const* const v108 = &v1[v103];
float const v109 = *v108;
*v222 = v109;
}
 else {
*v222 = v31;
}
float const v116 = *v222;
float const v117 = *v219;
float const v118 = v117 + v116;
*v219 = v118;
}
float const v121 = *v219;
uint32_t const v223 = warpSize;
uint32_t const v225 = min(v223, v268);
float v247_store;
float * v247 = &v247_store;*v247 = v121;
uint32_t const v227 = 1;
uint32_t v248_store;
uint32_t * v248 = &v248_store;*v248 = v227;
while([&]{
uint32_t const v230 = *v248;
bool const v231 = v230 < v225;
return v231;
}()) {
float const v234 = *v247;
uint32_t const v235 = *v248;
float const v236 = __shfl_xor_sync(__activemask(), v234, v235);
float const v238 = v234 + v236;
*v247 = v238;
uint32_t const v240 = 2;
uint32_t const v241 = v235 * v240;
*v248 = v241;
}
float const v242 = *v247;
*v219 = v242;
array<float, 16> * const v200 = reinterpret_cast<array<float, 16> *>(&v199[0]);
uint64_t const v125 = 16;
float * const v130 = &(*v200)[v41];
float const v131 = *v219;
*v130 = v131;
__syncthreads();
float v244_store;
float * v244 = &v244_store;*v244 = v31;
bool const v137 = v41 == v7;
if(v137) {
for(uint64_t v141 = v7; v141 < v125; v141 += v29) {
float * const v147 = &(*v200)[v141];
float const v148 = *v147;
float const v149 = *v244;
float const v150 = v149 + v148;
*v244 = v150;
}
float const v151 = *v244;
*v209 = v151;
}
__syncthreads();
if(v137) {
uint64_t const v155 = *v211;
uint32_t const v156 = uint32_t(v155);
uint64_t const v157 = uint64_t(v156);
uint64_t const v171 = *v206;
bool const v172 = v157 < v171;
if(v172) {
bool const v174 = v157 < v186;
if(v174) {
float * const v178 = &v2[v157];
float const v179 = *v209;
*v178 = v179;
}
}
}
return;
}
