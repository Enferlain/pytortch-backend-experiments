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
extern "C" __global__ void __launch_bounds__(256)  cubecl3_75da34b4c768db99(float const* const __restrict__ v1, float * const __restrict__ v2, info_st const* const __restrict__ v128) {
extern __shared__ __align__(4) char v134[];uint64_t const* const dynamic_meta_v130 = reinterpret_cast<uint64_t const*>(v128 + 1);
const info_st& info_v129 = *v128;int64_t const v132 = info_v129.scalars_i64[1];
int64_t const v133 = info_v129.scalars_i64[2];
uint64_t const v6 = blockIdx.x;
int64_t const v7 = int64_t(v6);
uint64_t const v8 = threadIdx.x;
int64_t const v9 = int64_t(v8);
int64_t const v10 = 0;
int64_t const v11 = 256;
int64_t const v12 = 1;
int64_t const v13 = v12 + v132;
array<float, 256> * const v135 = reinterpret_cast<array<float, 256> *>(&v134[0]);
bool const v15 = v7 < v133;
if(v15) {
float v137_store;
float * v137 = &v137_store;float const v17 = 0.0;
*v137 = v17;
for(int64_t v18 = v9; v18 < v13; v18 += v11) {
bool const v19 = v18 < v132;
float v138_store;
float * v138 = &v138_store;if(v19) {
int64_t const v21 = v7 * v132;
int64_t const v22 = v21 + v18;
uint64_t const v23 = uint64_t(v22);
float const* const v24 = &v1[v23];
float const v25 = *v24;
*v138 = v25;
}
 else {
*v138 = v17;
}
float const v27 = *v138;
bool const v28 = v18 >= v132;
float v139_store;
float * v139 = &v139_store;if(v28) {
int64_t const v30 = v7 * v132;
uint64_t const v31 = uint64_t(v30);
float const* const v32 = &v1[v31];
float const v33 = *v32;
*v139 = v33;
}
 else {
*v139 = v17;
}
float const v35 = *v139;
float const v36 = v19 ? v27 : v35;
float const v37 = *v137;
float const v38 = v37 + v36;
*v137 = v38;
}
float const v39 = *v137;
uint64_t const v40 = uint64_t(v9);
float * const v41 = &(*v135)[v40];
*v41 = v39;
__syncthreads();
int64_t const v42 = 128;
bool const v43 = v9 < v42;
if(v43) {
int64_t const v44 = v9 + v42;
uint64_t const v47 = uint64_t(v44);
float * const v48 = &(*v135)[v47];
float const v49 = *v41;
float const v50 = *v48;
float const v51 = v49 + v50;
*v41 = v51;
}
__syncthreads();
int64_t const v52 = 64;
bool const v53 = v9 < v52;
if(v53) {
int64_t const v54 = v9 + v52;
uint64_t const v57 = uint64_t(v54);
float * const v58 = &(*v135)[v57];
float const v59 = *v41;
float const v60 = *v58;
float const v61 = v59 + v60;
*v41 = v61;
}
__syncthreads();
int64_t const v62 = 32;
bool const v63 = v9 < v62;
if(v63) {
int64_t const v64 = v9 + v62;
uint64_t const v67 = uint64_t(v64);
float * const v68 = &(*v135)[v67];
float const v69 = *v41;
float const v70 = *v68;
float const v71 = v69 + v70;
*v41 = v71;
}
__syncthreads();
int64_t const v72 = 16;
bool const v73 = v9 < v72;
if(v73) {
int64_t const v74 = v9 + v72;
uint64_t const v77 = uint64_t(v74);
float * const v78 = &(*v135)[v77];
float const v79 = *v41;
float const v80 = *v78;
float const v81 = v79 + v80;
*v41 = v81;
}
__syncthreads();
int64_t const v82 = 8;
bool const v83 = v9 < v82;
if(v83) {
int64_t const v84 = v9 + v82;
uint64_t const v87 = uint64_t(v84);
float * const v88 = &(*v135)[v87];
float const v89 = *v41;
float const v90 = *v88;
float const v91 = v89 + v90;
*v41 = v91;
}
__syncthreads();
int64_t const v92 = 4;
bool const v93 = v9 < v92;
if(v93) {
int64_t const v94 = v9 + v92;
uint64_t const v97 = uint64_t(v94);
float * const v98 = &(*v135)[v97];
float const v99 = *v41;
float const v100 = *v98;
float const v101 = v99 + v100;
*v41 = v101;
}
__syncthreads();
int64_t const v102 = 2;
bool const v103 = v9 < v102;
if(v103) {
int64_t const v104 = v9 + v102;
uint64_t const v107 = uint64_t(v104);
float * const v108 = &(*v135)[v107];
float const v109 = *v41;
float const v110 = *v108;
float const v111 = v109 + v110;
*v41 = v111;
}
__syncthreads();
bool const v113 = v9 < v12;
if(v113) {
int64_t const v114 = v9 + v12;
uint64_t const v117 = uint64_t(v114);
float * const v118 = &(*v135)[v117];
float const v119 = *v41;
float const v120 = *v118;
float const v121 = v119 + v120;
*v41 = v121;
}
__syncthreads();
uint64_t const v140 = 0;
float * const v123 = &(*v135)[v140];
float const v124 = *v123;
__syncthreads();
bool const v125 = v9 == v10;
if(v125) {
uint64_t const v126 = uint64_t(v7);
float * const v127 = &v2[v126];
*v127 = v124;
}
}
return;
}
