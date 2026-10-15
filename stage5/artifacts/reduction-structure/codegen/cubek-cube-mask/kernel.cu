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
extern "C" __global__ void __launch_bounds__(512)  cubek_reader_cube_mask(float const* const __restrict__ v1, float * const __restrict__ v2, info_st const* const __restrict__ v181) {
extern __shared__ __align__(4) char v200[];uint64_t const* const dynamic_meta_v183 = reinterpret_cast<uint64_t const*>(v181 + 1);
const info_st& info_v182 = *v181;uint64_t const v185 = info_v182.scalars_usize[1];
uint64_t const v187 = info_v182.static_meta[1];
uint64_t const v7 = 0;
uint64_t const v13 = 2;
uint64_t v207_store;
uint64_t * v207 = &v207_store;*v207 = v185;
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
uint64_t const v188 = info_v182.static_meta[3];
uint64_t const v29 = 1;
float const v31 = 0.0;
uint64_t const v190 = info_v182.static_meta[2];
uint64_t const v191 = dynamic_meta_v183[v190 + v29];
float v210_store;
float * v210 = &v210_store;*v210 = v31;
uint64_t v212_store;
uint64_t * v212 = &v212_store;*v212 = v266;
uint32_t const v40 = threadIdx.y;
uint64_t const v41 = uint64_t(v40);
uint32_t const v267 = 32;
uint64_t v215_store;
uint64_t * v215 = &v215_store;*v215 = v7;
for(uint64_t v56 = v7; v56 < v13; v56 += v29) {
uint64_t const v196 = info_v182.static_meta[5];
uint64_t const v197 = dynamic_meta_v183[v196 + v56];
uint64_t const v61 = v266 / v197;
uint64_t const v193 = dynamic_meta_v183[v188 + v56];
uint64_t const v64 = v61 % v193;
uint64_t const v198 = info_v182.static_meta[4];
uint64_t const v199 = dynamic_meta_v183[v198 + v56];
uint64_t const v67 = v64 * v199;
uint64_t const v68 = *v215;
uint64_t const v69 = v68 + v67;
*v215 = v69;
}
uint64_t const v70 = *v215;
*v215 = v70;
float v220_store;
float * v220 = &v220_store;*v220 = v31;
uint32_t const v268 = 512;
uint64_t const v286 = 512;
uint64_t const v87 = v191 + v286;
uint64_t const v89 = v87 - v29;
uint64_t const v90 = v89 / v286;
for(uint64_t v93 = v7; v93 < v90; v93 += v29) {
uint64_t const v98 = v93 * v286;
uint32_t const v270 = threadIdx.x;
uint32_t const v273 = v40 * v267;
uint32_t const v274 = v270 + v273;
uint64_t const v100 = uint64_t(v274);
uint64_t const v101 = v98 + v100;
uint64_t const v102 = *v215;
uint64_t const v103 = v101 + v102;
bool const v104 = v101 < v191;
uint64_t const v105 = uint64_t(v104);
uint64_t const v106 = v103 * v105;
float const* const v110 = &v1[v106];
float const v111 = *v110;
float const v112 = v104 ? v111 : v31;
float const v118 = *v220;
float const v119 = v118 + v112;
*v220 = v119;
}
float const v122 = *v220;
uint32_t const v223 = warpSize;
uint32_t const v225 = min(v223, v268);
float v247_store;
float * v247 = &v247_store;*v247 = v122;
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
*v220 = v242;
array<float, 16> * const v201 = reinterpret_cast<array<float, 16> *>(&v200[0]);
uint64_t const v126 = 16;
float * const v131 = &(*v201)[v41];
float const v132 = *v220;
*v131 = v132;
__syncthreads();
float v244_store;
float * v244 = &v244_store;*v244 = v31;
bool const v138 = v41 == v7;
if(v138) {
for(uint64_t v142 = v7; v142 < v126; v142 += v29) {
float * const v148 = &(*v201)[v142];
float const v149 = *v148;
float const v150 = *v244;
float const v151 = v150 + v149;
*v244 = v151;
}
float const v152 = *v244;
*v210 = v152;
}
__syncthreads();
if(v138) {
uint64_t const v156 = *v212;
uint32_t const v157 = uint32_t(v156);
uint64_t const v158 = uint64_t(v157);
uint64_t const v172 = *v207;
bool const v173 = v158 < v172;
if(v173) {
bool const v175 = v158 < v187;
if(v175) {
float * const v179 = &v2[v158];
float const v180 = *v210;
*v179 = v180;
}
}
}
return;
}
