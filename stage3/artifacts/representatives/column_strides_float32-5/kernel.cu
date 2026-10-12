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
extern "C" __global__ void __launch_bounds__(256)  cubecl3_da158273147733f1(float const* const __restrict__ v1, float * const __restrict__ v2, info_st const* const __restrict__ v111) {
extern __shared__ __align__(4) char v117[];uint64_t const* const dynamic_meta_v113 = reinterpret_cast<uint64_t const*>(v111 + 1);
const info_st& info_v112 = *v111;int64_t const v115 = info_v112.scalars_i64[1];
int64_t const v116 = info_v112.scalars_i64[2];
uint64_t const v6 = blockIdx.x;
int64_t const v7 = int64_t(v6);
uint64_t const v8 = threadIdx.x;
int64_t const v9 = int64_t(v8);
int64_t const v10 = 0;
int64_t const v11 = 256;
array<float, 256> * const v118 = reinterpret_cast<array<float, 256> *>(&v117[0]);
bool const v13 = v7 < v115;
if(v13) {
float v120_store;
float * v120 = &v120_store;float const v15 = 0.0;
*v120 = v15;
for(int64_t v16 = v9; v16 < v116; v16 += v11) {
uint64_t const v17 = uint64_t(v7);
float const* const v18 = &v1[v17];
float const v19 = *v18;
float const v20 = *v120;
float const v21 = v20 + v19;
*v120 = v21;
}
float const v22 = *v120;
uint64_t const v23 = uint64_t(v9);
float * const v24 = &(*v118)[v23];
*v24 = v22;
__syncthreads();
int64_t const v25 = 128;
bool const v26 = v9 < v25;
if(v26) {
int64_t const v27 = v9 + v25;
uint64_t const v30 = uint64_t(v27);
float * const v31 = &(*v118)[v30];
float const v32 = *v24;
float const v33 = *v31;
float const v34 = v32 + v33;
*v24 = v34;
}
__syncthreads();
int64_t const v35 = 64;
bool const v36 = v9 < v35;
if(v36) {
int64_t const v37 = v9 + v35;
uint64_t const v40 = uint64_t(v37);
float * const v41 = &(*v118)[v40];
float const v42 = *v24;
float const v43 = *v41;
float const v44 = v42 + v43;
*v24 = v44;
}
__syncthreads();
int64_t const v45 = 32;
bool const v46 = v9 < v45;
if(v46) {
int64_t const v47 = v9 + v45;
uint64_t const v50 = uint64_t(v47);
float * const v51 = &(*v118)[v50];
float const v52 = *v24;
float const v53 = *v51;
float const v54 = v52 + v53;
*v24 = v54;
}
__syncthreads();
int64_t const v55 = 16;
bool const v56 = v9 < v55;
if(v56) {
int64_t const v57 = v9 + v55;
uint64_t const v60 = uint64_t(v57);
float * const v61 = &(*v118)[v60];
float const v62 = *v24;
float const v63 = *v61;
float const v64 = v62 + v63;
*v24 = v64;
}
__syncthreads();
int64_t const v65 = 8;
bool const v66 = v9 < v65;
if(v66) {
int64_t const v67 = v9 + v65;
uint64_t const v70 = uint64_t(v67);
float * const v71 = &(*v118)[v70];
float const v72 = *v24;
float const v73 = *v71;
float const v74 = v72 + v73;
*v24 = v74;
}
__syncthreads();
int64_t const v75 = 4;
bool const v76 = v9 < v75;
if(v76) {
int64_t const v77 = v9 + v75;
uint64_t const v80 = uint64_t(v77);
float * const v81 = &(*v118)[v80];
float const v82 = *v24;
float const v83 = *v81;
float const v84 = v82 + v83;
*v24 = v84;
}
__syncthreads();
int64_t const v85 = 2;
bool const v86 = v9 < v85;
if(v86) {
int64_t const v87 = v9 + v85;
uint64_t const v90 = uint64_t(v87);
float * const v91 = &(*v118)[v90];
float const v92 = *v24;
float const v93 = *v91;
float const v94 = v92 + v93;
*v24 = v94;
}
__syncthreads();
int64_t const v95 = 1;
bool const v96 = v9 < v95;
if(v96) {
int64_t const v97 = v9 + v95;
uint64_t const v100 = uint64_t(v97);
float * const v101 = &(*v118)[v100];
float const v102 = *v24;
float const v103 = *v101;
float const v104 = v102 + v103;
*v24 = v104;
}
__syncthreads();
uint64_t const v121 = 0;
float * const v106 = &(*v118)[v121];
float const v107 = *v106;
__syncthreads();
bool const v108 = v9 == v10;
if(v108) {
uint64_t const v109 = uint64_t(v7);
float * const v110 = &v2[v109];
*v110 = v107;
}
}
return;
}
