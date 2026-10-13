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
    int64_t scalars_i64[3];uint64_t static_meta[4];
};
extern "C" __global__ void __launch_bounds__(256)  cubecl5_674cb26a733d539d(float const* const __restrict__ v1, float const* const __restrict__ v2, float const* const __restrict__ v3, float * const __restrict__ v4, info_st const* const __restrict__ v205) {
extern __shared__ __align__(4) char v211[];uint64_t const* const dynamic_meta_v207 = reinterpret_cast<uint64_t const*>(v205 + 1);
const info_st& info_v206 = *v205;int64_t const v209 = info_v206.scalars_i64[1];
int64_t const v210 = info_v206.scalars_i64[2];
uint64_t const v8 = blockIdx.x;
int64_t const v9 = int64_t(v8);
uint64_t const v10 = threadIdx.x;
int64_t const v11 = int64_t(v10);
int64_t const v13 = 1;
int64_t const v17 = 256;
array<float, 256> * const v212 = reinterpret_cast<array<float, 256> *>(&v211[0]);
bool const v19 = v9 < v210;
if(v19) {
float v214_store;
float * v214 = &v214_store;float const v21 = 0.0;
*v214 = v21;
for(int64_t v25 = v11; v25 < v209; v25 += v17) {
int64_t const v26 = v9 * v209;
int64_t const v27 = v25 + v26;
uint64_t const v28 = uint64_t(v27);
float const* const v29 = &v1[v28];
float const v30 = *v29;
float const* const v34 = &v2[v28];
float const v35 = *v34;
float const v36 = v30 * v35;
float const v37 = *v214;
float const v38 = v37 + v36;
*v214 = v38;
}
float const v39 = *v214;
uint64_t const v43 = uint64_t(v11);
float * const v44 = &(*v212)[v43];
*v44 = v39;
__syncthreads();
int64_t const v45 = 128;
bool const v46 = v11 < v45;
if(v46) {
int64_t const v47 = v11 + v45;
uint64_t const v56 = uint64_t(v47);
float * const v57 = &(*v212)[v56];
float const v58 = *v44;
float const v59 = *v57;
float const v60 = v58 + v59;
*v44 = v60;
}
__syncthreads();
int64_t const v61 = 64;
bool const v62 = v11 < v61;
if(v62) {
int64_t const v63 = v11 + v61;
uint64_t const v72 = uint64_t(v63);
float * const v73 = &(*v212)[v72];
float const v74 = *v44;
float const v75 = *v73;
float const v76 = v74 + v75;
*v44 = v76;
}
__syncthreads();
int64_t const v77 = 32;
bool const v78 = v11 < v77;
if(v78) {
int64_t const v79 = v11 + v77;
uint64_t const v88 = uint64_t(v79);
float * const v89 = &(*v212)[v88];
float const v90 = *v44;
float const v91 = *v89;
float const v92 = v90 + v91;
*v44 = v92;
}
__syncthreads();
int64_t const v93 = 16;
bool const v94 = v11 < v93;
if(v94) {
int64_t const v95 = v11 + v93;
uint64_t const v104 = uint64_t(v95);
float * const v105 = &(*v212)[v104];
float const v106 = *v44;
float const v107 = *v105;
float const v108 = v106 + v107;
*v44 = v108;
}
__syncthreads();
int64_t const v109 = 8;
bool const v110 = v11 < v109;
if(v110) {
int64_t const v111 = v11 + v109;
uint64_t const v120 = uint64_t(v111);
float * const v121 = &(*v212)[v120];
float const v122 = *v44;
float const v123 = *v121;
float const v124 = v122 + v123;
*v44 = v124;
}
__syncthreads();
int64_t const v125 = 4;
bool const v126 = v11 < v125;
if(v126) {
int64_t const v127 = v11 + v125;
uint64_t const v136 = uint64_t(v127);
float * const v137 = &(*v212)[v136];
float const v138 = *v44;
float const v139 = *v137;
float const v140 = v138 + v139;
*v44 = v140;
}
__syncthreads();
int64_t const v141 = 2;
bool const v142 = v11 < v141;
if(v142) {
int64_t const v143 = v11 + v141;
uint64_t const v152 = uint64_t(v143);
float * const v153 = &(*v212)[v152];
float const v154 = *v44;
float const v155 = *v153;
float const v156 = v154 + v155;
*v44 = v156;
}
__syncthreads();
bool const v158 = v11 < v13;
if(v158) {
int64_t const v159 = v11 + v13;
uint64_t const v168 = uint64_t(v159);
float * const v169 = &(*v212)[v168];
float const v170 = *v44;
float const v171 = *v169;
float const v172 = v170 + v171;
*v44 = v172;
}
__syncthreads();
uint64_t const v215 = 0;
float * const v174 = &(*v212)[v215];
float const v175 = *v174;
__syncthreads();
for(int64_t v176 = v11; v176 < v209; v176 += v17) {
int64_t const v177 = v9 * v209;
int64_t const v178 = v176 + v177;
uint64_t const v179 = uint64_t(v178);
float const* const v180 = &v1[v179];
float const v181 = *v180;
float const* const v185 = &v2[v179];
float const v186 = *v185;
uint64_t const v187 = uint64_t(v9);
float const* const v188 = &v3[v187];
float const v189 = *v188;
float const v190 = v181 * v189;
float const v191 = -0.5;
float const v192 = v175 * v191;
float const v193 = v189 * v189;
float const v194 = v193 * v189;
float const v195 = v192 * v194;
float const v196 = float(v209);
float const v197 = v195 / v196;
float const v198 = v197 * v186;
float const v199 = v190 + v198;
float const v200 = v199 + v198;
float * const v204 = &v4[v179];
*v204 = v200;
}
}
return;
}
