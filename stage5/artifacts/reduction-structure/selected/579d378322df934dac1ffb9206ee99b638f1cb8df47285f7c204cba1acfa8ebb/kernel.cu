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
extern "C" __global__ void __launch_bounds__(256)  cubecl5_d3c6e8bc4b1f2f4d(float const* const __restrict__ v1, float * const __restrict__ v2, info_st const* const __restrict__ v223) {
extern __shared__ __align__(4) char v229[];uint64_t const* const dynamic_meta_v225 = reinterpret_cast<uint64_t const*>(v223 + 1);
const info_st& info_v224 = *v223;int64_t const v227 = info_v224.scalars_i64[1];
int64_t const v228 = info_v224.scalars_i64[2];
uint64_t const v6 = blockIdx.x;
int64_t const v7 = int64_t(v6);
uint64_t const v8 = threadIdx.x;
int64_t const v9 = int64_t(v8);
int64_t const v10 = 0;
int64_t const v11 = 1;
int64_t const v15 = 256;
int64_t const v16 = 9;
int64_t const v17 = v16 * v228;
int64_t const v18 = 8;
int64_t const v19 = v18 + v227;
int64_t const v21 = v19 / v16;
int64_t const v22 = v19 % v16;
bool const v25 = v22 < v10;
int64_t const v26 = v21 - v11;
int64_t const v27 = v25 ? v26 : v21;
array<float, 256> * const v230 = reinterpret_cast<array<float, 256> *>(&v229[0]);
bool const v29 = v7 < v17;
if(v29) {
float v232_store;
float * v232 = &v232_store;float const v31 = 0.0;
*v232 = v31;
for(int64_t v35 = v9; v35 < v27; v35 += v15) {
int64_t const v233 = v7 % v16;
int64_t const v235 = v233 ^ v16;
bool const v236 = v235 < v10;
bool const v237 = v233 != v10;
bool const v238 = v237 && v236;
int64_t const v239 = v238 ? v16 : v10;
int64_t const v240 = v233 + v239;
int64_t const v48 = v27 * v240;
int64_t const v49 = v35 + v48;
bool const v50 = v49 < v227;
float v241_store;
float * v241 = &v241_store;if(v50) {
int64_t const v53 = v7 / v16;
bool const v57 = v233 < v10;
int64_t const v58 = v53 - v11;
int64_t const v59 = v57 ? v58 : v53;
int64_t const v60 = v227 * v59;
int64_t const v61 = v35 + v60;
int64_t const v75 = v61 + v48;
uint64_t const v76 = uint64_t(v75);
float const* const v77 = &v1[v76];
float const v78 = *v77;
*v241 = v78;
}
 else {
*v241 = v31;
}
float const v80 = *v241;
float const v81 = *v232;
float const v82 = v81 + v80;
*v232 = v82;
}
float const v83 = *v232;
uint64_t const v87 = uint64_t(v9);
float * const v88 = &(*v230)[v87];
*v88 = v83;
__syncthreads();
int64_t const v89 = 128;
bool const v90 = v9 < v89;
if(v90) {
int64_t const v91 = v9 + v89;
uint64_t const v100 = uint64_t(v91);
float * const v101 = &(*v230)[v100];
float const v102 = *v88;
float const v103 = *v101;
float const v104 = v102 + v103;
*v88 = v104;
}
__syncthreads();
int64_t const v105 = 64;
bool const v106 = v9 < v105;
if(v106) {
int64_t const v107 = v9 + v105;
uint64_t const v116 = uint64_t(v107);
float * const v117 = &(*v230)[v116];
float const v118 = *v88;
float const v119 = *v117;
float const v120 = v118 + v119;
*v88 = v120;
}
__syncthreads();
int64_t const v121 = 32;
bool const v122 = v9 < v121;
if(v122) {
int64_t const v123 = v9 + v121;
uint64_t const v132 = uint64_t(v123);
float * const v133 = &(*v230)[v132];
float const v134 = *v88;
float const v135 = *v133;
float const v136 = v134 + v135;
*v88 = v136;
}
__syncthreads();
int64_t const v137 = 16;
bool const v138 = v9 < v137;
if(v138) {
int64_t const v139 = v9 + v137;
uint64_t const v148 = uint64_t(v139);
float * const v149 = &(*v230)[v148];
float const v150 = *v88;
float const v151 = *v149;
float const v152 = v150 + v151;
*v88 = v152;
}
__syncthreads();
bool const v154 = v9 < v18;
if(v154) {
int64_t const v155 = v9 + v18;
uint64_t const v164 = uint64_t(v155);
float * const v165 = &(*v230)[v164];
float const v166 = *v88;
float const v167 = *v165;
float const v168 = v166 + v167;
*v88 = v168;
}
__syncthreads();
int64_t const v169 = 4;
bool const v170 = v9 < v169;
if(v170) {
int64_t const v171 = v9 + v169;
uint64_t const v180 = uint64_t(v171);
float * const v181 = &(*v230)[v180];
float const v182 = *v88;
float const v183 = *v181;
float const v184 = v182 + v183;
*v88 = v184;
}
__syncthreads();
int64_t const v185 = 2;
bool const v186 = v9 < v185;
if(v186) {
int64_t const v187 = v9 + v185;
uint64_t const v196 = uint64_t(v187);
float * const v197 = &(*v230)[v196];
float const v198 = *v88;
float const v199 = *v197;
float const v200 = v198 + v199;
*v88 = v200;
}
__syncthreads();
bool const v202 = v9 < v11;
if(v202) {
int64_t const v203 = v9 + v11;
uint64_t const v212 = uint64_t(v203);
float * const v213 = &(*v230)[v212];
float const v214 = *v88;
float const v215 = *v213;
float const v216 = v214 + v215;
*v88 = v216;
}
__syncthreads();
uint64_t const v250 = 0;
float * const v218 = &(*v230)[v250];
float const v219 = *v218;
bool const v220 = v9 == v10;
if(v220) {
uint64_t const v221 = uint64_t(v7);
float * const v222 = &v2[v221];
*v222 = v219;
}
}
return;
}
