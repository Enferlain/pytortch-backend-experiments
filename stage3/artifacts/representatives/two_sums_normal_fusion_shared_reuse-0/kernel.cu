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
extern "C" __global__ void __launch_bounds__(256)  cubecl3_6ac7fbc0084e2a4d(float const* const __restrict__ v1, float * const __restrict__ v2, info_st const* const __restrict__ v211) {
extern __shared__ __align__(4) char v217[];uint64_t const* const dynamic_meta_v213 = reinterpret_cast<uint64_t const*>(v211 + 1);
const info_st& info_v212 = *v211;int64_t const v215 = info_v212.scalars_i64[1];
int64_t const v216 = info_v212.scalars_i64[2];
uint64_t const v6 = blockIdx.x;
int64_t const v7 = int64_t(v6);
uint64_t const v8 = threadIdx.x;
int64_t const v9 = int64_t(v8);
int64_t const v10 = 0;
int64_t const v11 = 256;
array<float, 256> * const v218 = reinterpret_cast<array<float, 256> *>(&v217[0]);
bool const v13 = v7 < v216;
if(v13) {
float v220_store;
float * v220 = &v220_store;float const v15 = 0.0;
*v220 = v15;
for(int64_t v16 = v9; v16 < v215; v16 += v11) {
int64_t const v17 = v7 * v215;
int64_t const v18 = v16 + v17;
uint64_t const v19 = uint64_t(v18);
float const* const v20 = &v1[v19];
float const v21 = *v20;
float const v22 = *v220;
float const v23 = v22 + v21;
*v220 = v23;
}
float const v24 = *v220;
uint64_t const v25 = uint64_t(v9);
float * const v26 = &(*v218)[v25];
*v26 = v24;
__syncthreads();
int64_t const v27 = 128;
bool const v28 = v9 < v27;
if(v28) {
int64_t const v29 = v9 + v27;
uint64_t const v32 = uint64_t(v29);
float * const v33 = &(*v218)[v32];
float const v34 = *v26;
float const v35 = *v33;
float const v36 = v34 + v35;
*v26 = v36;
}
__syncthreads();
int64_t const v37 = 64;
bool const v38 = v9 < v37;
if(v38) {
int64_t const v39 = v9 + v37;
uint64_t const v42 = uint64_t(v39);
float * const v43 = &(*v218)[v42];
float const v44 = *v26;
float const v45 = *v43;
float const v46 = v44 + v45;
*v26 = v46;
}
__syncthreads();
int64_t const v47 = 32;
bool const v48 = v9 < v47;
if(v48) {
int64_t const v49 = v9 + v47;
uint64_t const v52 = uint64_t(v49);
float * const v53 = &(*v218)[v52];
float const v54 = *v26;
float const v55 = *v53;
float const v56 = v54 + v55;
*v26 = v56;
}
__syncthreads();
int64_t const v57 = 16;
bool const v58 = v9 < v57;
if(v58) {
int64_t const v59 = v9 + v57;
uint64_t const v62 = uint64_t(v59);
float * const v63 = &(*v218)[v62];
float const v64 = *v26;
float const v65 = *v63;
float const v66 = v64 + v65;
*v26 = v66;
}
__syncthreads();
int64_t const v67 = 8;
bool const v68 = v9 < v67;
if(v68) {
int64_t const v69 = v9 + v67;
uint64_t const v72 = uint64_t(v69);
float * const v73 = &(*v218)[v72];
float const v74 = *v26;
float const v75 = *v73;
float const v76 = v74 + v75;
*v26 = v76;
}
__syncthreads();
int64_t const v77 = 4;
bool const v78 = v9 < v77;
if(v78) {
int64_t const v79 = v9 + v77;
uint64_t const v82 = uint64_t(v79);
float * const v83 = &(*v218)[v82];
float const v84 = *v26;
float const v85 = *v83;
float const v86 = v84 + v85;
*v26 = v86;
}
__syncthreads();
int64_t const v87 = 2;
bool const v88 = v9 < v87;
if(v88) {
int64_t const v89 = v9 + v87;
uint64_t const v92 = uint64_t(v89);
float * const v93 = &(*v218)[v92];
float const v94 = *v26;
float const v95 = *v93;
float const v96 = v94 + v95;
*v26 = v96;
}
__syncthreads();
int64_t const v97 = 1;
bool const v98 = v9 < v97;
if(v98) {
int64_t const v99 = v9 + v97;
uint64_t const v102 = uint64_t(v99);
float * const v103 = &(*v218)[v102];
float const v104 = *v26;
float const v105 = *v103;
float const v106 = v104 + v105;
*v26 = v106;
}
__syncthreads();
uint64_t const v223 = 0;
float * const v108 = &(*v218)[v223];
float const v109 = *v108;
__syncthreads();
float v221_store;
float * v221 = &v221_store;*v221 = v15;
for(int64_t v112 = v9; v112 < v215; v112 += v11) {
int64_t const v113 = v7 * v215;
int64_t const v114 = v112 + v113;
uint64_t const v115 = uint64_t(v114);
float const* const v116 = &v1[v115];
float const v117 = *v116;
float const v118 = v117 * v117;
float const v119 = *v221;
float const v120 = v119 + v118;
*v221 = v120;
}
float const v121 = *v221;
*v26 = v121;
__syncthreads();
if(v28) {
int64_t const v126 = v9 + v27;
uint64_t const v129 = uint64_t(v126);
float * const v130 = &(*v218)[v129];
float const v131 = *v26;
float const v132 = *v130;
float const v133 = v131 + v132;
*v26 = v133;
}
__syncthreads();
if(v38) {
int64_t const v136 = v9 + v37;
uint64_t const v139 = uint64_t(v136);
float * const v140 = &(*v218)[v139];
float const v141 = *v26;
float const v142 = *v140;
float const v143 = v141 + v142;
*v26 = v143;
}
__syncthreads();
if(v48) {
int64_t const v146 = v9 + v47;
uint64_t const v149 = uint64_t(v146);
float * const v150 = &(*v218)[v149];
float const v151 = *v26;
float const v152 = *v150;
float const v153 = v151 + v152;
*v26 = v153;
}
__syncthreads();
if(v58) {
int64_t const v156 = v9 + v57;
uint64_t const v159 = uint64_t(v156);
float * const v160 = &(*v218)[v159];
float const v161 = *v26;
float const v162 = *v160;
float const v163 = v161 + v162;
*v26 = v163;
}
__syncthreads();
if(v68) {
int64_t const v166 = v9 + v67;
uint64_t const v169 = uint64_t(v166);
float * const v170 = &(*v218)[v169];
float const v171 = *v26;
float const v172 = *v170;
float const v173 = v171 + v172;
*v26 = v173;
}
__syncthreads();
if(v78) {
int64_t const v176 = v9 + v77;
uint64_t const v179 = uint64_t(v176);
float * const v180 = &(*v218)[v179];
float const v181 = *v26;
float const v182 = *v180;
float const v183 = v181 + v182;
*v26 = v183;
}
__syncthreads();
if(v88) {
int64_t const v186 = v9 + v87;
uint64_t const v189 = uint64_t(v186);
float * const v190 = &(*v218)[v189];
float const v191 = *v26;
float const v192 = *v190;
float const v193 = v191 + v192;
*v26 = v193;
}
__syncthreads();
if(v98) {
int64_t const v196 = v9 + v97;
uint64_t const v199 = uint64_t(v196);
float * const v200 = &(*v218)[v199];
float const v201 = *v26;
float const v202 = *v200;
float const v203 = v201 + v202;
*v26 = v203;
}
__syncthreads();
float const v206 = *v108;
__syncthreads();
bool const v207 = v9 == v10;
if(v207) {
float const v208 = v109 + v206;
uint64_t const v209 = uint64_t(v7);
float * const v210 = &v2[v209];
*v210 = v208;
}
}
return;
}
