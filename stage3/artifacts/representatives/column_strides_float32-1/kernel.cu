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
extern "C" __global__ void __launch_bounds__(256)  cubecl3_d846b09db5f634cf(float const* const __restrict__ v1, float * const __restrict__ v2, info_st const* const __restrict__ v169) {
extern __shared__ __align__(4) char v175[];uint64_t const* const dynamic_meta_v171 = reinterpret_cast<uint64_t const*>(v169 + 1);
const info_st& info_v170 = *v169;int64_t const v173 = info_v170.scalars_i64[1];
int64_t const v174 = info_v170.scalars_i64[2];
uint64_t const v6 = blockIdx.x;
int64_t const v7 = int64_t(v6);
uint64_t const v8 = threadIdx.x;
int64_t const v9 = int64_t(v8);
int64_t const v10 = 0;
int64_t const v11 = 256;
int64_t const v12 = 3;
int64_t const v13 = v12 * v173;
int64_t const v14 = 2;
int64_t const v15 = v14 + v174;
int64_t const v17 = v15 / v12;
int64_t const v18 = v15 % v12;
int64_t const v20 = 1;
bool const v21 = v18 < v10;
int64_t const v22 = v17 - v20;
int64_t const v23 = v21 ? v22 : v17;
array<float, 256> * const v176 = reinterpret_cast<array<float, 256> *>(&v175[0]);
bool const v25 = v7 < v13;
if(v25) {
float v178_store;
float * v178 = &v178_store;float const v27 = 0.0;
*v178 = v27;
for(int64_t v28 = v9; v28 < v23; v28 += v11) {
int64_t const v29 = v7 / v173;
int64_t const v30 = v7 % v173;
bool const v33 = v30 < v10;
int64_t const v34 = v29 - v20;
int64_t const v35 = v33 ? v34 : v29;
int64_t const v46 = v35 * v23;
int64_t const v47 = v28 + v46;
bool const v48 = v47 < v174;
float v179_store;
float * v179 = &v179_store;if(v48) {
int64_t const v50 = v28 * v173;
int64_t const v58 = v173 * v35;
int64_t const v69 = v58 * v23;
int64_t const v70 = v50 + v69;
int64_t const v182 = v30 ^ v173;
bool const v183 = v182 < v10;
bool const v184 = v30 != v10;
bool const v185 = v184 && v183;
int64_t const v186 = v185 ? v173 : v10;
int64_t const v187 = v30 + v186;
int64_t const v72 = v70 + v187;
uint64_t const v73 = uint64_t(v72);
float const* const v74 = &v1[v73];
float const v75 = *v74;
*v179 = v75;
}
 else {
*v179 = v27;
}
float const v77 = *v179;
float const v78 = *v178;
float const v79 = v78 + v77;
*v178 = v79;
}
float const v80 = *v178;
uint64_t const v81 = uint64_t(v9);
float * const v82 = &(*v176)[v81];
*v82 = v80;
__syncthreads();
int64_t const v83 = 128;
bool const v84 = v9 < v83;
if(v84) {
int64_t const v85 = v9 + v83;
uint64_t const v88 = uint64_t(v85);
float * const v89 = &(*v176)[v88];
float const v90 = *v82;
float const v91 = *v89;
float const v92 = v90 + v91;
*v82 = v92;
}
__syncthreads();
int64_t const v93 = 64;
bool const v94 = v9 < v93;
if(v94) {
int64_t const v95 = v9 + v93;
uint64_t const v98 = uint64_t(v95);
float * const v99 = &(*v176)[v98];
float const v100 = *v82;
float const v101 = *v99;
float const v102 = v100 + v101;
*v82 = v102;
}
__syncthreads();
int64_t const v103 = 32;
bool const v104 = v9 < v103;
if(v104) {
int64_t const v105 = v9 + v103;
uint64_t const v108 = uint64_t(v105);
float * const v109 = &(*v176)[v108];
float const v110 = *v82;
float const v111 = *v109;
float const v112 = v110 + v111;
*v82 = v112;
}
__syncthreads();
int64_t const v113 = 16;
bool const v114 = v9 < v113;
if(v114) {
int64_t const v115 = v9 + v113;
uint64_t const v118 = uint64_t(v115);
float * const v119 = &(*v176)[v118];
float const v120 = *v82;
float const v121 = *v119;
float const v122 = v120 + v121;
*v82 = v122;
}
__syncthreads();
int64_t const v123 = 8;
bool const v124 = v9 < v123;
if(v124) {
int64_t const v125 = v9 + v123;
uint64_t const v128 = uint64_t(v125);
float * const v129 = &(*v176)[v128];
float const v130 = *v82;
float const v131 = *v129;
float const v132 = v130 + v131;
*v82 = v132;
}
__syncthreads();
int64_t const v133 = 4;
bool const v134 = v9 < v133;
if(v134) {
int64_t const v135 = v9 + v133;
uint64_t const v138 = uint64_t(v135);
float * const v139 = &(*v176)[v138];
float const v140 = *v82;
float const v141 = *v139;
float const v142 = v140 + v141;
*v82 = v142;
}
__syncthreads();
bool const v144 = v9 < v14;
if(v144) {
int64_t const v145 = v9 + v14;
uint64_t const v148 = uint64_t(v145);
float * const v149 = &(*v176)[v148];
float const v150 = *v82;
float const v151 = *v149;
float const v152 = v150 + v151;
*v82 = v152;
}
__syncthreads();
bool const v154 = v9 < v20;
if(v154) {
int64_t const v155 = v9 + v20;
uint64_t const v158 = uint64_t(v155);
float * const v159 = &(*v176)[v158];
float const v160 = *v82;
float const v161 = *v159;
float const v162 = v160 + v161;
*v82 = v162;
}
__syncthreads();
uint64_t const v188 = 0;
float * const v164 = &(*v176)[v188];
float const v165 = *v164;
__syncthreads();
bool const v166 = v9 == v10;
if(v166) {
uint64_t const v167 = uint64_t(v7);
float * const v168 = &v2[v167];
*v168 = v165;
}
}
return;
}
