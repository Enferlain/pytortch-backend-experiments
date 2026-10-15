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
extern "C" __global__ void __launch_bounds__(256)  cubek_reader_plane_branch(float const* const __restrict__ v1, float * const __restrict__ v2, info_st const* const __restrict__ v156) {
uint64_t const* const dynamic_meta_v158 = reinterpret_cast<uint64_t const*>(v156 + 1);
const info_st& info_v157 = *v156;uint64_t const v160 = info_v157.scalars_usize[1];
uint64_t const v162 = info_v157.static_meta[1];
uint64_t const v7 = 0;
uint64_t const v13 = 2;
uint64_t v180_store;
uint64_t * v180 = &v180_store;*v180 = v160;
uint32_t const v220 = blockIdx.z;
uint64_t const v221 = uint64_t(v220);
uint32_t const v222 = gridDim.y;
uint64_t const v223 = uint64_t(v222);
uint64_t const v224 = v221 * v223;
uint32_t const v225 = gridDim.x;
uint64_t const v226 = uint64_t(v225);
uint64_t const v227 = v224 * v226;
uint32_t const v228 = blockIdx.y;
uint64_t const v229 = uint64_t(v228);
uint64_t const v232 = v229 * v226;
uint64_t const v233 = v227 + v232;
uint32_t const v234 = blockIdx.x;
uint64_t const v235 = uint64_t(v234);
uint64_t const v236 = v233 + v235;
uint64_t const v241 = 8;
uint64_t const v27 = v236 * v241;
uint32_t const v28 = threadIdx.y;
uint64_t const v29 = uint64_t(v28);
uint64_t const v30 = v27 + v29;
uint64_t const v163 = info_v157.static_meta[3];
uint64_t const v35 = 1;
float const v37 = 0.0;
uint64_t const v165 = info_v157.static_meta[2];
uint64_t const v166 = dynamic_meta_v158[v165 + v35];
float v183_store;
float * v183 = &v183_store;*v183 = v37;
uint64_t v185_store;
uint64_t * v185 = &v185_store;*v185 = v30;
uint64_t v188_store;
uint64_t * v188 = &v188_store;*v188 = v7;
for(uint64_t v60 = v7; v60 < v13; v60 += v35) {
uint64_t const v171 = info_v157.static_meta[5];
uint64_t const v172 = dynamic_meta_v158[v171 + v60];
uint64_t const v65 = v30 / v172;
uint64_t const v168 = dynamic_meta_v158[v163 + v60];
uint64_t const v68 = v65 % v168;
uint64_t const v173 = info_v157.static_meta[4];
uint64_t const v174 = dynamic_meta_v158[v173 + v60];
uint64_t const v71 = v68 * v174;
uint64_t const v72 = *v188;
uint64_t const v73 = v72 + v71;
*v188 = v73;
}
uint64_t const v74 = *v188;
*v188 = v74;
float v193_store;
float * v193 = &v193_store;*v193 = v37;
uint64_t const v240 = 32;
uint64_t const v90 = v166 + v240;
uint64_t const v92 = v90 - v35;
uint64_t const v93 = v92 / v240;
for(uint64_t v96 = v7; v96 < v93; v96 += v35) {
uint64_t const v100 = v96 * v240;
uint32_t const v101 = threadIdx.x;
uint64_t const v102 = uint64_t(v101);
uint64_t const v103 = v100 + v102;
uint64_t const v104 = *v188;
uint64_t const v105 = v103 + v104;
bool const v106 = v103 < v166;
float v196_store;
float * v196 = &v196_store;if(v106) {
float const* const v110 = &v1[v105];
float const v111 = *v110;
*v196 = v111;
}
 else {
*v196 = v37;
}
float const v118 = *v196;
float const v119 = *v193;
float const v120 = v119 + v118;
*v193 = v120;
}
float const v121 = *v193;
uint32_t const v197 = warpSize;
uint32_t const v239 = 256;
uint32_t const v199 = min(v197, v239);
float v217_store;
float * v217 = &v217_store;*v217 = v121;
uint32_t const v201 = 1;
uint32_t v218_store;
uint32_t * v218 = &v218_store;*v218 = v201;
while([&]{
uint32_t const v204 = *v218;
bool const v205 = v204 < v199;
return v205;
}()) {
float const v208 = *v217;
uint32_t const v209 = *v218;
float const v210 = __shfl_xor_sync(__activemask(), v208, v209);
float const v212 = v208 + v210;
*v217 = v212;
uint32_t const v214 = 2;
uint32_t const v215 = v209 * v214;
*v218 = v215;
}
float const v216 = *v217;
*v193 = v216;
uint32_t const v123 = threadIdx.x;
uint32_t const v124 = 0;
bool const v125 = v123 == v124;
if(v125) {
float const v126 = *v193;
*v183 = v126;
}
if(v125) {
uint64_t const v131 = *v185;
uint32_t const v132 = uint32_t(v131);
uint64_t const v133 = uint64_t(v132);
uint64_t const v147 = *v180;
bool const v148 = v133 < v147;
if(v148) {
bool const v150 = v133 < v162;
if(v150) {
float * const v154 = &v2[v133];
float const v155 = *v183;
*v154 = v155;
}
}
}
return;
}
