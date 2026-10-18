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
    int64_t scalars_i64[3];uint64_t static_meta[3];
};
extern "C" __global__ void __launch_bounds__(64)  cubecl5_e50f059428be3b01(float const* const __restrict__ v1, float const* const __restrict__ v2, float * const __restrict__ v3, info_st const* const __restrict__ v162) {
extern __shared__ __align__(4) char v168[];uint64_t const* const dynamic_meta_v164 = reinterpret_cast<uint64_t const*>(v162 + 1);
const info_st& info_v163 = *v162;int64_t const v166 = info_v163.scalars_i64[1];
int64_t const v167 = info_v163.scalars_i64[2];
uint64_t const v7 = blockIdx.x;
int64_t const v8 = int64_t(v7);
uint64_t const v9 = threadIdx.x;
int64_t const v10 = int64_t(v9);
int64_t const v12 = 1;
int64_t const v16 = 64;
array<float, 64> * const v169 = reinterpret_cast<array<float, 64> *>(&v168[0]);
bool const v18 = v8 < v167;
if(v18) {
float v171_store;
float * v171 = &v171_store;float const v20 = 0.0;
*v171 = v20;
for(int64_t v24 = v10; v24 < v166; v24 += v16) {
int64_t const v25 = v8 * v166;
int64_t const v26 = v24 + v25;
uint64_t const v27 = uint64_t(v26);
float const* const v28 = &v1[v27];
float const v29 = *v28;
uint64_t const v30 = uint64_t(v8);
float const* const v31 = &v2[v30];
float const v32 = *v31;
float const v33 = v29 + v32;
float const v34 = v33 * v33;
float const v35 = *v171;
float const v36 = v35 + v34;
*v171 = v36;
}
float const v37 = *v171;
uint64_t const v41 = uint64_t(v10);
float * const v42 = &(*v169)[v41];
*v42 = v37;
__syncthreads();
int64_t const v43 = 32;
bool const v44 = v10 < v43;
if(v44) {
int64_t const v45 = v10 + v43;
uint64_t const v54 = uint64_t(v45);
float * const v55 = &(*v169)[v54];
float const v56 = *v42;
float const v57 = *v55;
float const v58 = v56 + v57;
*v42 = v58;
}
__syncthreads();
int64_t const v59 = 16;
bool const v60 = v10 < v59;
if(v60) {
int64_t const v61 = v10 + v59;
uint64_t const v70 = uint64_t(v61);
float * const v71 = &(*v169)[v70];
float const v72 = *v42;
float const v73 = *v71;
float const v74 = v72 + v73;
*v42 = v74;
}
__syncthreads();
int64_t const v75 = 8;
bool const v76 = v10 < v75;
if(v76) {
int64_t const v77 = v10 + v75;
uint64_t const v86 = uint64_t(v77);
float * const v87 = &(*v169)[v86];
float const v88 = *v42;
float const v89 = *v87;
float const v90 = v88 + v89;
*v42 = v90;
}
__syncthreads();
int64_t const v91 = 4;
bool const v92 = v10 < v91;
if(v92) {
int64_t const v93 = v10 + v91;
uint64_t const v102 = uint64_t(v93);
float * const v103 = &(*v169)[v102];
float const v104 = *v42;
float const v105 = *v103;
float const v106 = v104 + v105;
*v42 = v106;
}
__syncthreads();
int64_t const v107 = 2;
bool const v108 = v10 < v107;
if(v108) {
int64_t const v109 = v10 + v107;
uint64_t const v118 = uint64_t(v109);
float * const v119 = &(*v169)[v118];
float const v120 = *v42;
float const v121 = *v119;
float const v122 = v120 + v121;
*v42 = v122;
}
__syncthreads();
bool const v124 = v10 < v12;
if(v124) {
int64_t const v125 = v10 + v12;
uint64_t const v134 = uint64_t(v125);
float * const v135 = &(*v169)[v134];
float const v136 = *v42;
float const v137 = *v135;
float const v138 = v136 + v137;
*v42 = v138;
}
__syncthreads();
uint64_t const v172 = 0;
float * const v140 = &(*v169)[v172];
float const v141 = *v140;
for(int64_t v142 = v10; v142 < v166; v142 += v16) {
int64_t const v143 = v8 * v166;
int64_t const v144 = v142 + v143;
uint64_t const v145 = uint64_t(v144);
float const* const v146 = &v1[v145];
float const v147 = *v146;
uint64_t const v148 = uint64_t(v8);
float const* const v149 = &v2[v148];
float const v150 = *v149;
float const v151 = v147 + v150;
float const v152 = float(v166);
float const v153 = v141 / v152;
float const v154 = 9.999999740e-06;
float const v155 = v153 + v154;
float const v156 = rsqrt(v155);float const v157 = v151 * v156;
float * const v161 = &v3[v145];
*v161 = v157;
}
}
return;
}
