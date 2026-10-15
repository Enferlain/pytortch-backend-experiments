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
extern "C" __global__ void __launch_bounds__(256)  cubecl5_5406b3e834b065d8(float const* const __restrict__ v1, float * const __restrict__ v2, info_st const* const __restrict__ v109) {
extern __shared__ __align__(4) char v115[];uint64_t const* const dynamic_meta_v111 = reinterpret_cast<uint64_t const*>(v109 + 1);
const info_st& info_v110 = *v109;int64_t const v113 = info_v110.scalars_i64[1];
int64_t const v114 = info_v110.scalars_i64[2];
uint64_t const v6 = blockIdx.x;
int64_t const v7 = int64_t(v6);
uint64_t const v8 = threadIdx.x;
int64_t const v9 = int64_t(v8);
int64_t const v10 = 0;
int64_t const v11 = 1;
int64_t const v15 = 256;
int64_t const v16 = 9;
int64_t const v17 = v16 * v114;
int64_t const v18 = 8;
int64_t const v19 = v18 + v113;
int64_t const v21 = v19 / v16;
int64_t const v22 = v19 % v16;
bool const v25 = v22 < v10;
int64_t const v26 = v21 - v11;
int64_t const v27 = v25 ? v26 : v21;
array<float, 8> * const v116 = reinterpret_cast<array<float, 8> *>(&v115[0]);
bool const v29 = v7 < v17;
if(v29) {
float v118_store;
float * v118 = &v118_store;float const v31 = 0.0;
*v118 = v31;
for(int64_t v35 = v9; v35 < v27; v35 += v15) {
int64_t const v119 = v7 % v16;
int64_t const v121 = v119 ^ v16;
bool const v122 = v121 < v10;
bool const v123 = v119 != v10;
bool const v124 = v123 && v122;
int64_t const v125 = v124 ? v16 : v10;
int64_t const v126 = v119 + v125;
int64_t const v48 = v27 * v126;
int64_t const v49 = v35 + v48;
bool const v50 = v49 < v113;
float v127_store;
float * v127 = &v127_store;if(v50) {
int64_t const v53 = v7 / v16;
bool const v57 = v119 < v10;
int64_t const v58 = v53 - v11;
int64_t const v59 = v57 ? v58 : v53;
int64_t const v60 = v113 * v59;
int64_t const v61 = v35 + v60;
int64_t const v75 = v61 + v48;
uint64_t const v76 = uint64_t(v75);
float const* const v77 = &v1[v76];
float const v78 = *v77;
*v127 = v78;
}
 else {
*v127 = v31;
}
float const v80 = *v127;
float const v81 = *v118;
float const v82 = v81 + v80;
*v118 = v82;
}
float const v83 = *v118;
uint32_t const v136 = warpSize;
uint32_t const v183 = 256;
uint32_t const v138 = min(v136, v183);
float v177_store;
float * v177 = &v177_store;*v177 = v83;
uint32_t const v140 = 1;
uint32_t v178_store;
uint32_t * v178 = &v178_store;*v178 = v140;
while([&]{
uint32_t const v143 = *v178;
bool const v144 = v143 < v138;
return v144;
}()) {
float const v147 = *v177;
uint32_t const v148 = *v178;
float const v149 = __shfl_xor_sync(__activemask(), v147, v148);
float const v151 = v147 + v149;
*v177 = v151;
uint32_t const v153 = 2;
uint32_t const v154 = v148 * v153;
*v178 = v154;
}
float const v155 = *v177;
int64_t const v85 = 32;
int64_t const v86 = v9 / v85;
int64_t const v87 = v9 % v85;
bool const v88 = v87 == v10;
if(v88) {
uint64_t const v89 = uint64_t(v86);
float * const v90 = &(*v116)[v89];
*v90 = v155;
}
__syncthreads();
bool const v91 = v86 == v10;
if(v91) {
float v156_store;
float * v156 = &v156_store;*v156 = v31;
bool const v95 = v87 < v18;
if(v95) {
uint64_t const v96 = uint64_t(v87);
float * const v97 = &(*v116)[v96];
float const v98 = *v97;
*v156 = v98;
}
float const v99 = *v156;
float v180_store;
float * v180 = &v180_store;*v180 = v99;
uint32_t v181_store;
uint32_t * v181 = &v181_store;*v181 = v140;
while([&]{
uint32_t const v164 = *v181;
bool const v165 = v164 < v138;
return v165;
}()) {
float const v168 = *v180;
uint32_t const v169 = *v181;
float const v170 = __shfl_xor_sync(__activemask(), v168, v169);
float const v172 = v168 + v170;
*v180 = v172;
uint32_t const v174 = 2;
uint32_t const v175 = v169 * v174;
*v181 = v175;
}
float const v176 = *v180;
if(v88) {
uint64_t const v187 = 0;
float * const v102 = &(*v116)[v187];
*v102 = v176;
}
}
__syncthreads();
uint64_t const v185 = 0;
float * const v104 = &(*v116)[v185];
float const v105 = *v104;
bool const v106 = v9 == v10;
if(v106) {
uint64_t const v107 = uint64_t(v7);
float * const v108 = &v2[v107];
*v108 = v105;
}
}
return;
}
