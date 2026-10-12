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
extern "C" __global__ void __launch_bounds__(256)  cubecl3_89ff183f2d821cf3(float const* const __restrict__ v1, float * const __restrict__ v2, info_st const* const __restrict__ v118) {
extern __shared__ __align__(4) char v124[];uint64_t const* const dynamic_meta_v120 = reinterpret_cast<uint64_t const*>(v118 + 1);
const info_st& info_v119 = *v118;int64_t const v122 = info_v119.scalars_i64[1];
int64_t const v123 = info_v119.scalars_i64[2];
uint64_t const v6 = blockIdx.x;
int64_t const v7 = int64_t(v6);
uint64_t const v8 = threadIdx.x;
int64_t const v9 = int64_t(v8);
int64_t const v10 = 0;
int64_t const v11 = 256;
array<float, 256> * const v125 = reinterpret_cast<array<float, 256> *>(&v124[0]);
bool const v13 = v7 < v122;
if(v13) {
float v127_store;
float * v127 = &v127_store;float const v15 = 0.0;
*v127 = v15;
for(int64_t v16 = v9; v16 < v123; v16 += v11) {
float const v17 = 7.0;
float const v26 = *v127;
float const v27 = v26 + v17;
*v127 = v27;
}
float const v28 = *v127;
uint64_t const v29 = uint64_t(v9);
float * const v30 = &(*v125)[v29];
*v30 = v28;
__syncthreads();
int64_t const v31 = 128;
bool const v32 = v9 < v31;
if(v32) {
int64_t const v33 = v9 + v31;
uint64_t const v36 = uint64_t(v33);
float * const v37 = &(*v125)[v36];
float const v38 = *v30;
float const v39 = *v37;
float const v40 = v38 + v39;
*v30 = v40;
}
__syncthreads();
int64_t const v41 = 64;
bool const v42 = v9 < v41;
if(v42) {
int64_t const v43 = v9 + v41;
uint64_t const v46 = uint64_t(v43);
float * const v47 = &(*v125)[v46];
float const v48 = *v30;
float const v49 = *v47;
float const v50 = v48 + v49;
*v30 = v50;
}
__syncthreads();
int64_t const v51 = 32;
bool const v52 = v9 < v51;
if(v52) {
int64_t const v53 = v9 + v51;
uint64_t const v56 = uint64_t(v53);
float * const v57 = &(*v125)[v56];
float const v58 = *v30;
float const v59 = *v57;
float const v60 = v58 + v59;
*v30 = v60;
}
__syncthreads();
int64_t const v61 = 16;
bool const v62 = v9 < v61;
if(v62) {
int64_t const v63 = v9 + v61;
uint64_t const v66 = uint64_t(v63);
float * const v67 = &(*v125)[v66];
float const v68 = *v30;
float const v69 = *v67;
float const v70 = v68 + v69;
*v30 = v70;
}
__syncthreads();
int64_t const v71 = 8;
bool const v72 = v9 < v71;
if(v72) {
int64_t const v73 = v9 + v71;
uint64_t const v76 = uint64_t(v73);
float * const v77 = &(*v125)[v76];
float const v78 = *v30;
float const v79 = *v77;
float const v80 = v78 + v79;
*v30 = v80;
}
__syncthreads();
int64_t const v81 = 4;
bool const v82 = v9 < v81;
if(v82) {
int64_t const v83 = v9 + v81;
uint64_t const v86 = uint64_t(v83);
float * const v87 = &(*v125)[v86];
float const v88 = *v30;
float const v89 = *v87;
float const v90 = v88 + v89;
*v30 = v90;
}
__syncthreads();
int64_t const v91 = 2;
bool const v92 = v9 < v91;
if(v92) {
int64_t const v93 = v9 + v91;
uint64_t const v96 = uint64_t(v93);
float * const v97 = &(*v125)[v96];
float const v98 = *v30;
float const v99 = *v97;
float const v100 = v98 + v99;
*v30 = v100;
}
__syncthreads();
int64_t const v101 = 1;
bool const v102 = v9 < v101;
if(v102) {
int64_t const v103 = v9 + v101;
uint64_t const v106 = uint64_t(v103);
float * const v107 = &(*v125)[v106];
float const v108 = *v30;
float const v109 = *v107;
float const v110 = v108 + v109;
*v30 = v110;
}
__syncthreads();
uint64_t const v130 = 0;
float * const v112 = &(*v125)[v130];
float const v113 = *v112;
__syncthreads();
bool const v114 = v9 == v10;
if(v114) {
uint64_t const v116 = uint64_t(v7);
float * const v117 = &v2[v116];
*v117 = v113;
}
}
return;
}
