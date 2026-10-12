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
    int64_t scalars_i64[5];uint64_t static_meta[2];
};
extern "C" __global__ void __launch_bounds__(256)  cubecl3_3b8848ac6d58f4ab(float const* const __restrict__ v1, float * const __restrict__ v2, info_st const* const __restrict__ v169) {
extern __shared__ __align__(4) char v177[];uint64_t const* const dynamic_meta_v171 = reinterpret_cast<uint64_t const*>(v169 + 1);
const info_st& info_v170 = *v169;int64_t const v173 = info_v170.scalars_i64[1];
int64_t const v174 = info_v170.scalars_i64[2];
int64_t const v175 = info_v170.scalars_i64[3];
int64_t const v176 = info_v170.scalars_i64[4];
uint64_t const v8 = blockIdx.x;
int64_t const v9 = int64_t(v8);
uint64_t const v10 = threadIdx.x;
int64_t const v11 = int64_t(v10);
int64_t const v12 = 0;
int64_t const v13 = 256;
int64_t const v14 = 3;
int64_t const v15 = v14 * v176;
int64_t const v16 = 2;
int64_t const v17 = v16 + v173;
int64_t const v19 = v17 / v14;
int64_t const v20 = v17 % v14;
int64_t const v22 = 1;
bool const v23 = v20 < v12;
int64_t const v24 = v19 - v22;
int64_t const v25 = v23 ? v24 : v19;
array<float, 256> * const v178 = reinterpret_cast<array<float, 256> *>(&v177[0]);
bool const v27 = v9 < v15;
if(v27) {
float v180_store;
float * v180 = &v180_store;float const v29 = 0.0;
*v180 = v29;
for(int64_t v30 = v11; v30 < v25; v30 += v13) {
int64_t const v181 = v9 % v14;
int64_t const v183 = v181 ^ v14;
bool const v184 = v183 < v12;
bool const v185 = v181 != v12;
bool const v186 = v185 && v184;
int64_t const v187 = v186 ? v14 : v12;
int64_t const v188 = v181 + v187;
int64_t const v43 = v25 * v188;
int64_t const v44 = v30 + v43;
bool const v45 = v44 < v173;
float v189_store;
float * v189 = &v189_store;if(v45) {
int64_t const v47 = v30 * v174;
int64_t const v49 = v9 / v14;
bool const v53 = v181 < v12;
int64_t const v54 = v49 - v22;
int64_t const v55 = v53 ? v54 : v49;
int64_t const v56 = v175 * v55;
int64_t const v57 = v47 + v56;
int64_t const v68 = v174 * v25;
int64_t const v71 = v68 * v188;
int64_t const v72 = v57 + v71;
uint64_t const v73 = uint64_t(v72);
float const* const v74 = &v1[v73];
float const v75 = *v74;
*v189 = v75;
}
 else {
*v189 = v29;
}
float const v77 = *v189;
float const v78 = *v180;
float const v79 = v78 + v77;
*v180 = v79;
}
float const v80 = *v180;
uint64_t const v81 = uint64_t(v11);
float * const v82 = &(*v178)[v81];
*v82 = v80;
__syncthreads();
int64_t const v83 = 128;
bool const v84 = v11 < v83;
if(v84) {
int64_t const v85 = v11 + v83;
uint64_t const v88 = uint64_t(v85);
float * const v89 = &(*v178)[v88];
float const v90 = *v82;
float const v91 = *v89;
float const v92 = v90 + v91;
*v82 = v92;
}
__syncthreads();
int64_t const v93 = 64;
bool const v94 = v11 < v93;
if(v94) {
int64_t const v95 = v11 + v93;
uint64_t const v98 = uint64_t(v95);
float * const v99 = &(*v178)[v98];
float const v100 = *v82;
float const v101 = *v99;
float const v102 = v100 + v101;
*v82 = v102;
}
__syncthreads();
int64_t const v103 = 32;
bool const v104 = v11 < v103;
if(v104) {
int64_t const v105 = v11 + v103;
uint64_t const v108 = uint64_t(v105);
float * const v109 = &(*v178)[v108];
float const v110 = *v82;
float const v111 = *v109;
float const v112 = v110 + v111;
*v82 = v112;
}
__syncthreads();
int64_t const v113 = 16;
bool const v114 = v11 < v113;
if(v114) {
int64_t const v115 = v11 + v113;
uint64_t const v118 = uint64_t(v115);
float * const v119 = &(*v178)[v118];
float const v120 = *v82;
float const v121 = *v119;
float const v122 = v120 + v121;
*v82 = v122;
}
__syncthreads();
int64_t const v123 = 8;
bool const v124 = v11 < v123;
if(v124) {
int64_t const v125 = v11 + v123;
uint64_t const v128 = uint64_t(v125);
float * const v129 = &(*v178)[v128];
float const v130 = *v82;
float const v131 = *v129;
float const v132 = v130 + v131;
*v82 = v132;
}
__syncthreads();
int64_t const v133 = 4;
bool const v134 = v11 < v133;
if(v134) {
int64_t const v135 = v11 + v133;
uint64_t const v138 = uint64_t(v135);
float * const v139 = &(*v178)[v138];
float const v140 = *v82;
float const v141 = *v139;
float const v142 = v140 + v141;
*v82 = v142;
}
__syncthreads();
bool const v144 = v11 < v16;
if(v144) {
int64_t const v145 = v11 + v16;
uint64_t const v148 = uint64_t(v145);
float * const v149 = &(*v178)[v148];
float const v150 = *v82;
float const v151 = *v149;
float const v152 = v150 + v151;
*v82 = v152;
}
__syncthreads();
bool const v154 = v11 < v22;
if(v154) {
int64_t const v155 = v11 + v22;
uint64_t const v158 = uint64_t(v155);
float * const v159 = &(*v178)[v158];
float const v160 = *v82;
float const v161 = *v159;
float const v162 = v160 + v161;
*v82 = v162;
}
__syncthreads();
uint64_t const v198 = 0;
float * const v164 = &(*v178)[v198];
float const v165 = *v164;
__syncthreads();
bool const v166 = v11 == v12;
if(v166) {
uint64_t const v167 = uint64_t(v9);
float * const v168 = &v2[v167];
*v168 = v165;
}
}
return;
}
