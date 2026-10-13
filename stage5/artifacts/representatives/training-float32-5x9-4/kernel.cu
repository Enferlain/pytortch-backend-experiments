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
    int64_t scalars_i64[2];uint64_t static_meta[4];
};
extern "C" __global__ void __launch_bounds__(256)  cubecl5_81110a964cd193a4(bool const* const __restrict__ v1, float const* const __restrict__ v2, float const* const __restrict__ v3, float * const __restrict__ v4, info_st const* const __restrict__ v198) {
extern __shared__ __align__(4) char v203[];uint64_t const* const dynamic_meta_v200 = reinterpret_cast<uint64_t const*>(v198 + 1);
const info_st& info_v199 = *v198;int64_t const v202 = info_v199.scalars_i64[1];
uint64_t const v7 = blockIdx.x;
int64_t const v8 = int64_t(v7);
uint64_t const v9 = threadIdx.x;
int64_t const v10 = int64_t(v9);
int64_t const v11 = 0;
int64_t const v12 = 1;
int64_t const v16 = 256;
int64_t const v18 = 9;
int64_t const v19 = v18 * v202;
array<float, 256> * const v204 = reinterpret_cast<array<float, 256> *>(&v203[0]);
bool const v21 = v8 < v12;
if(v21) {
float v206_store;
float * v206 = &v206_store;float const v23 = 0.0;
*v206 = v23;
for(int64_t v27 = v10; v27 < v19; v27 += v16) {
uint64_t const v28 = uint64_t(v27);
bool const* const v29 = &v1[v28];
bool const v30 = *v29;
float const v31 = float(v30);
float const* const v33 = &v2[v28];
float const v34 = *v33;
int64_t const v36 = v27 / v18;
int64_t const v37 = v27 % v18;
bool const v40 = v37 < v11;
int64_t const v41 = v36 - v12;
int64_t const v42 = v40 ? v41 : v36;
uint64_t const v43 = uint64_t(v42);
float const* const v44 = &v3[v43];
float const v45 = *v44;
float const v46 = v34 * v45;
float const v47 = v31 * v46;
float const v48 = 1.33333337;
float const v49 = v47 * v48;
float const v50 = v49 * v49;
float const v51 = *v206;
float const v52 = v51 + v50;
*v206 = v52;
}
float const v53 = *v206;
uint64_t const v57 = uint64_t(v10);
float * const v58 = &(*v204)[v57];
*v58 = v53;
__syncthreads();
int64_t const v59 = 128;
bool const v60 = v10 < v59;
if(v60) {
int64_t const v61 = v10 + v59;
uint64_t const v70 = uint64_t(v61);
float * const v71 = &(*v204)[v70];
float const v72 = *v58;
float const v73 = *v71;
float const v74 = v72 + v73;
*v58 = v74;
}
__syncthreads();
int64_t const v75 = 64;
bool const v76 = v10 < v75;
if(v76) {
int64_t const v77 = v10 + v75;
uint64_t const v86 = uint64_t(v77);
float * const v87 = &(*v204)[v86];
float const v88 = *v58;
float const v89 = *v87;
float const v90 = v88 + v89;
*v58 = v90;
}
__syncthreads();
int64_t const v91 = 32;
bool const v92 = v10 < v91;
if(v92) {
int64_t const v93 = v10 + v91;
uint64_t const v102 = uint64_t(v93);
float * const v103 = &(*v204)[v102];
float const v104 = *v58;
float const v105 = *v103;
float const v106 = v104 + v105;
*v58 = v106;
}
__syncthreads();
int64_t const v107 = 16;
bool const v108 = v10 < v107;
if(v108) {
int64_t const v109 = v10 + v107;
uint64_t const v118 = uint64_t(v109);
float * const v119 = &(*v204)[v118];
float const v120 = *v58;
float const v121 = *v119;
float const v122 = v120 + v121;
*v58 = v122;
}
__syncthreads();
int64_t const v123 = 8;
bool const v124 = v10 < v123;
if(v124) {
int64_t const v125 = v10 + v123;
uint64_t const v134 = uint64_t(v125);
float * const v135 = &(*v204)[v134];
float const v136 = *v58;
float const v137 = *v135;
float const v138 = v136 + v137;
*v58 = v138;
}
__syncthreads();
int64_t const v139 = 4;
bool const v140 = v10 < v139;
if(v140) {
int64_t const v141 = v10 + v139;
uint64_t const v150 = uint64_t(v141);
float * const v151 = &(*v204)[v150];
float const v152 = *v58;
float const v153 = *v151;
float const v154 = v152 + v153;
*v58 = v154;
}
__syncthreads();
int64_t const v155 = 2;
bool const v156 = v10 < v155;
if(v156) {
int64_t const v157 = v10 + v155;
uint64_t const v166 = uint64_t(v157);
float * const v167 = &(*v204)[v166];
float const v168 = *v58;
float const v169 = *v167;
float const v170 = v168 + v169;
*v58 = v170;
}
__syncthreads();
bool const v172 = v10 < v12;
if(v172) {
int64_t const v173 = v10 + v12;
uint64_t const v182 = uint64_t(v173);
float * const v183 = &(*v204)[v182];
float const v184 = *v58;
float const v185 = *v183;
float const v186 = v184 + v185;
*v58 = v186;
}
__syncthreads();
uint64_t const v207 = 0;
float * const v188 = &(*v204)[v207];
float const v189 = *v188;
__syncthreads();
bool const v190 = v10 == v11;
if(v190) {
float const v193 = float(v19);
float const v194 = v189 / v193;
float * const v197 = &v4[v207];
*v197 = v194;
}
}
return;
}
