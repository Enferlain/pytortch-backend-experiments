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
extern "C" __global__ void __launch_bounds__(256)  cubecl5_rms_cubek_6dd137ece77b52c9(float const* const __restrict__ v1, float * const __restrict__ v2, info_st const* const __restrict__ v137) {
uint64_t const* const dynamic_meta_v139 = reinterpret_cast<uint64_t const*>(v137 + 1);
const info_st& info_v138 = *v137;int64_t const v141 = info_v138.scalars_i64[1];
int64_t const v142 = info_v138.scalars_i64[2];
uint64_t const v6 = blockIdx.x;
int64_t const v7 = int64_t(v6);
uint64_t const v8 = threadIdx.y;
int64_t const v9 = int64_t(v8);
int64_t const v10 = 8;
int64_t const v11 = v7 * v10;
int64_t const v12 = v11 + v9;
uint64_t const v13 = uint64_t(v12);
uint64_t const v14 = uint64_t(v141);
uint64_t const v15 = uint64_t(v142);
bool const v16 = v13 < v15;
if(v16) {
uint64_t const v18 = 0;
uint64_t v144_store;
uint64_t * v144 = &v144_store;*v144 = v18;
uint64_t const v22 = 2;
uint64_t const v23 = 1;
for(uint64_t v24 = v18; v24 < v22; v24 += v23) {
int64_t const v27 = int64_t(v24);
int64_t const v28 = 0;
bool const v29 = v27 == v28;
int64_t const v30 = 1;
int64_t const v39 = v29 ? v142 : v30;
uint64_t const v40 = uint64_t(v39);
uint64_t const v41 = v13 % v40;
int64_t const v46 = v29 ? v141 : v30;
uint64_t const v47 = uint64_t(v46);
uint64_t const v48 = v41 * v47;
uint64_t const v49 = *v144;
uint64_t const v50 = v49 + v48;
*v144 = v50;
}
uint64_t const v51 = *v144;
*v144 = v51;
float const v62 = 0.0;
float v149_store;
float * v149 = &v149_store;*v149 = v62;
uint64_t const v180 = 32;
uint64_t const v72 = v14 + v180;
uint64_t const v74 = v72 - v23;
uint64_t const v75 = v74 / v180;
for(uint64_t v78 = v18; v78 < v75; v78 += v23) {
uint64_t const v82 = v78 * v180;
uint32_t const v83 = threadIdx.x;
uint64_t const v84 = uint64_t(v83);
uint64_t const v85 = v82 + v84;
uint64_t const v86 = *v144;
uint64_t const v87 = v85 + v86;
bool const v88 = v85 < v14;
float v152_store;
float * v152 = &v152_store;if(v88) {
int64_t const v89 = int64_t(v87);
int64_t const v90 = v12 * v141;
int64_t const v91 = v89 - v90;
int64_t const v93 = v91 + v90;
uint64_t const v94 = uint64_t(v93);
float const* const v95 = &v1[v94];
float const v96 = *v95;
float const v97 = v96 * v96;
*v152 = v97;
}
 else {
*v152 = v62;
}
float const v104 = *v152;
float const v105 = *v149;
float const v106 = v105 + v104;
*v149 = v106;
}
float const v107 = *v149;
uint32_t const v153 = warpSize;
uint32_t const v179 = 256;
uint32_t const v155 = min(v153, v179);
float v175_store;
float * v175 = &v175_store;*v175 = v107;
uint32_t const v157 = 1;
uint32_t v176_store;
uint32_t * v176 = &v176_store;*v176 = v157;
while([&]{
uint32_t const v160 = *v176;
bool const v161 = v160 < v155;
return v161;
}()) {
float const v164 = *v175;
uint32_t const v165 = *v176;
float const v166 = __shfl_xor_sync(__activemask(), v164, v165);
float const v168 = v164 + v166;
*v175 = v168;
uint32_t const v170 = 2;
uint32_t const v171 = v165 * v170;
*v176 = v171;
}
float const v172 = *v175;
*v149 = v172;
uint32_t const v109 = threadIdx.x;
uint64_t const v110 = uint64_t(v109);
for(uint64_t v112 = v110; v112 < v14; v112 += v180) {
uint64_t const v115 = v13 * v14;
uint64_t const v116 = v115 + v112;
float const v117 = *v149;
int64_t const v118 = int64_t(v116);
int64_t const v119 = v12 * v141;
int64_t const v120 = v118 - v119;
int64_t const v122 = v120 + v119;
uint64_t const v123 = uint64_t(v122);
float const* const v124 = &v1[v123];
float const v125 = *v124;
float const v127 = float(v141);
float const v128 = v117 / v127;
float const v129 = 9.999999740e-06;
float const v130 = v128 + v129;
float const v131 = rsqrt(v130);float const v132 = v125 * v131;
float * const v136 = &v2[v123];
*v136 = v132;
}
}
return;
}
