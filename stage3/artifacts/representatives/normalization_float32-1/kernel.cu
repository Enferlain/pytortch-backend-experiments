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
extern "C" __global__ void __launch_bounds__(256)  cubecl3_bcdf26ae5a8328ab(float const* const __restrict__ v1, float * const __restrict__ v2, info_st const* const __restrict__ v119) {
extern __shared__ __align__(4) char v125[];uint64_t const* const dynamic_meta_v121 = reinterpret_cast<uint64_t const*>(v119 + 1);
const info_st& info_v120 = *v119;int64_t const v123 = info_v120.scalars_i64[1];
int64_t const v124 = info_v120.scalars_i64[2];
uint64_t const v6 = blockIdx.x;
int64_t const v7 = int64_t(v6);
uint64_t const v8 = threadIdx.x;
int64_t const v9 = int64_t(v8);
int64_t const v10 = 0;
int64_t const v11 = 256;
array<float, 256> * const v126 = reinterpret_cast<array<float, 256> *>(&v125[0]);
bool const v13 = v7 < v124;
if(v13) {
float v128_store;
float * v128 = &v128_store;float const v15 = 0.0;
*v128 = v15;
for(int64_t v16 = v9; v16 < v123; v16 += v11) {
int64_t const v17 = v7 * v123;
int64_t const v18 = v16 + v17;
uint64_t const v19 = uint64_t(v18);
float const* const v20 = &v1[v19];
float const v21 = *v20;
float const v22 = v21 * v21;
float const v23 = *v128;
float const v24 = v23 + v22;
*v128 = v24;
}
float const v25 = *v128;
uint64_t const v26 = uint64_t(v9);
float * const v27 = &(*v126)[v26];
*v27 = v25;
__syncthreads();
int64_t const v28 = 128;
bool const v29 = v9 < v28;
if(v29) {
int64_t const v30 = v9 + v28;
uint64_t const v33 = uint64_t(v30);
float * const v34 = &(*v126)[v33];
float const v35 = *v27;
float const v36 = *v34;
float const v37 = v35 + v36;
*v27 = v37;
}
__syncthreads();
int64_t const v38 = 64;
bool const v39 = v9 < v38;
if(v39) {
int64_t const v40 = v9 + v38;
uint64_t const v43 = uint64_t(v40);
float * const v44 = &(*v126)[v43];
float const v45 = *v27;
float const v46 = *v44;
float const v47 = v45 + v46;
*v27 = v47;
}
__syncthreads();
int64_t const v48 = 32;
bool const v49 = v9 < v48;
if(v49) {
int64_t const v50 = v9 + v48;
uint64_t const v53 = uint64_t(v50);
float * const v54 = &(*v126)[v53];
float const v55 = *v27;
float const v56 = *v54;
float const v57 = v55 + v56;
*v27 = v57;
}
__syncthreads();
int64_t const v58 = 16;
bool const v59 = v9 < v58;
if(v59) {
int64_t const v60 = v9 + v58;
uint64_t const v63 = uint64_t(v60);
float * const v64 = &(*v126)[v63];
float const v65 = *v27;
float const v66 = *v64;
float const v67 = v65 + v66;
*v27 = v67;
}
__syncthreads();
int64_t const v68 = 8;
bool const v69 = v9 < v68;
if(v69) {
int64_t const v70 = v9 + v68;
uint64_t const v73 = uint64_t(v70);
float * const v74 = &(*v126)[v73];
float const v75 = *v27;
float const v76 = *v74;
float const v77 = v75 + v76;
*v27 = v77;
}
__syncthreads();
int64_t const v78 = 4;
bool const v79 = v9 < v78;
if(v79) {
int64_t const v80 = v9 + v78;
uint64_t const v83 = uint64_t(v80);
float * const v84 = &(*v126)[v83];
float const v85 = *v27;
float const v86 = *v84;
float const v87 = v85 + v86;
*v27 = v87;
}
__syncthreads();
int64_t const v88 = 2;
bool const v89 = v9 < v88;
if(v89) {
int64_t const v90 = v9 + v88;
uint64_t const v93 = uint64_t(v90);
float * const v94 = &(*v126)[v93];
float const v95 = *v27;
float const v96 = *v94;
float const v97 = v95 + v96;
*v27 = v97;
}
__syncthreads();
int64_t const v98 = 1;
bool const v99 = v9 < v98;
if(v99) {
int64_t const v100 = v9 + v98;
uint64_t const v103 = uint64_t(v100);
float * const v104 = &(*v126)[v103];
float const v105 = *v27;
float const v106 = *v104;
float const v107 = v105 + v106;
*v27 = v107;
}
__syncthreads();
uint64_t const v129 = 0;
float * const v109 = &(*v126)[v129];
float const v110 = *v109;
__syncthreads();
bool const v111 = v9 == v10;
if(v111) {
float const v112 = float(v123);
float const v113 = v110 / v112;
float const v114 = 9.999999740e-06;
float const v115 = v113 + v114;
float const v116 = rsqrt(v115);uint64_t const v117 = uint64_t(v7);
float * const v118 = &v2[v117];
*v118 = v116;
}
}
return;
}
