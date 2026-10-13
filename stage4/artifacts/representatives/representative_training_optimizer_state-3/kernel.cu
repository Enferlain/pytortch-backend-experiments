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
    int64_t scalars_i64[2];uint64_t static_meta[2];
};
extern "C" __global__ void __launch_bounds__(256)  cubecl3_4b9a15134b702ca6(float const* const __restrict__ v1, float * const __restrict__ v2, info_st const* const __restrict__ v120) {
extern __shared__ __align__(4) char v125[];uint64_t const* const dynamic_meta_v122 = reinterpret_cast<uint64_t const*>(v120 + 1);
const info_st& info_v121 = *v120;int64_t const v124 = info_v121.scalars_i64[1];
uint64_t const v5 = blockIdx.x;
int64_t const v6 = int64_t(v5);
uint64_t const v7 = threadIdx.x;
int64_t const v8 = int64_t(v7);
int64_t const v9 = 0;
int64_t const v10 = 256;
int64_t const v11 = 9;
array<float, 256> * const v126 = reinterpret_cast<array<float, 256> *>(&v125[0]);
bool const v13 = v6 < v124;
if(v13) {
float v128_store;
float * v128 = &v128_store;float const v15 = 0.0;
*v128 = v15;
for(int64_t v16 = v8; v16 < v11; v16 += v10) {
int64_t const v18 = v11 * v6;
int64_t const v19 = v16 + v18;
uint64_t const v20 = uint64_t(v19);
float const* const v21 = &v1[v20];
float const v22 = *v21;
float const v23 = v22 * v22;
float const v24 = *v128;
float const v25 = v24 + v23;
*v128 = v25;
}
float const v26 = *v128;
uint64_t const v27 = uint64_t(v8);
float * const v28 = &(*v126)[v27];
*v28 = v26;
__syncthreads();
int64_t const v29 = 128;
bool const v30 = v8 < v29;
if(v30) {
int64_t const v31 = v8 + v29;
uint64_t const v34 = uint64_t(v31);
float * const v35 = &(*v126)[v34];
float const v36 = *v28;
float const v37 = *v35;
float const v38 = v36 + v37;
*v28 = v38;
}
__syncthreads();
int64_t const v39 = 64;
bool const v40 = v8 < v39;
if(v40) {
int64_t const v41 = v8 + v39;
uint64_t const v44 = uint64_t(v41);
float * const v45 = &(*v126)[v44];
float const v46 = *v28;
float const v47 = *v45;
float const v48 = v46 + v47;
*v28 = v48;
}
__syncthreads();
int64_t const v49 = 32;
bool const v50 = v8 < v49;
if(v50) {
int64_t const v51 = v8 + v49;
uint64_t const v54 = uint64_t(v51);
float * const v55 = &(*v126)[v54];
float const v56 = *v28;
float const v57 = *v55;
float const v58 = v56 + v57;
*v28 = v58;
}
__syncthreads();
int64_t const v59 = 16;
bool const v60 = v8 < v59;
if(v60) {
int64_t const v61 = v8 + v59;
uint64_t const v64 = uint64_t(v61);
float * const v65 = &(*v126)[v64];
float const v66 = *v28;
float const v67 = *v65;
float const v68 = v66 + v67;
*v28 = v68;
}
__syncthreads();
int64_t const v69 = 8;
bool const v70 = v8 < v69;
if(v70) {
int64_t const v71 = v8 + v69;
uint64_t const v74 = uint64_t(v71);
float * const v75 = &(*v126)[v74];
float const v76 = *v28;
float const v77 = *v75;
float const v78 = v76 + v77;
*v28 = v78;
}
__syncthreads();
int64_t const v79 = 4;
bool const v80 = v8 < v79;
if(v80) {
int64_t const v81 = v8 + v79;
uint64_t const v84 = uint64_t(v81);
float * const v85 = &(*v126)[v84];
float const v86 = *v28;
float const v87 = *v85;
float const v88 = v86 + v87;
*v28 = v88;
}
__syncthreads();
int64_t const v89 = 2;
bool const v90 = v8 < v89;
if(v90) {
int64_t const v91 = v8 + v89;
uint64_t const v94 = uint64_t(v91);
float * const v95 = &(*v126)[v94];
float const v96 = *v28;
float const v97 = *v95;
float const v98 = v96 + v97;
*v28 = v98;
}
__syncthreads();
int64_t const v99 = 1;
bool const v100 = v8 < v99;
if(v100) {
int64_t const v101 = v8 + v99;
uint64_t const v104 = uint64_t(v101);
float * const v105 = &(*v126)[v104];
float const v106 = *v28;
float const v107 = *v105;
float const v108 = v106 + v107;
*v28 = v108;
}
__syncthreads();
uint64_t const v129 = 0;
float * const v110 = &(*v126)[v129];
float const v111 = *v110;
__syncthreads();
bool const v112 = v8 == v9;
if(v112) {
float const v113 = 9.0;
float const v114 = v111 / v113;
float const v115 = 9.999999740e-06;
float const v116 = v114 + v115;
float const v117 = rsqrt(v116);uint64_t const v118 = uint64_t(v6);
float * const v119 = &v2[v118];
*v119 = v117;
}
}
return;
}
