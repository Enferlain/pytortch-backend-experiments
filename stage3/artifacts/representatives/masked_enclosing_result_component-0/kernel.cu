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
extern "C" __global__ void __launch_bounds__(256)  cubecl3_f45287afb4c1f976(float const* const __restrict__ v1, float * const __restrict__ v2, info_st const* const __restrict__ v114) {
extern __shared__ __align__(4) char v120[];uint64_t const* const dynamic_meta_v116 = reinterpret_cast<uint64_t const*>(v114 + 1);
const info_st& info_v115 = *v114;int64_t const v118 = info_v115.scalars_i64[1];
int64_t const v119 = info_v115.scalars_i64[2];
uint64_t const v6 = blockIdx.x;
int64_t const v7 = int64_t(v6);
uint64_t const v8 = threadIdx.x;
int64_t const v9 = int64_t(v8);
int64_t const v10 = 0;
int64_t const v11 = 256;
array<float, 256> * const v121 = reinterpret_cast<array<float, 256> *>(&v120[0]);
bool const v13 = v7 < v118;
if(v13) {
float v123_store;
float * v123 = &v123_store;float const v15 = 0.0;
*v123 = v15;
for(int64_t v16 = v9; v16 < v119; v16 += v11) {
float const v17 = 7.0;
float const v22 = *v123;
float const v23 = v22 + v17;
*v123 = v23;
}
float const v24 = *v123;
uint64_t const v25 = uint64_t(v9);
float * const v26 = &(*v121)[v25];
*v26 = v24;
__syncthreads();
int64_t const v27 = 128;
bool const v28 = v9 < v27;
if(v28) {
int64_t const v29 = v9 + v27;
uint64_t const v32 = uint64_t(v29);
float * const v33 = &(*v121)[v32];
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
float * const v43 = &(*v121)[v42];
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
float * const v53 = &(*v121)[v52];
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
float * const v63 = &(*v121)[v62];
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
float * const v73 = &(*v121)[v72];
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
float * const v83 = &(*v121)[v82];
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
float * const v93 = &(*v121)[v92];
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
float * const v103 = &(*v121)[v102];
float const v104 = *v26;
float const v105 = *v103;
float const v106 = v104 + v105;
*v26 = v106;
}
__syncthreads();
uint64_t const v125 = 0;
float * const v108 = &(*v121)[v125];
float const v109 = *v108;
__syncthreads();
bool const v110 = v9 == v10;
if(v110) {
uint64_t const v112 = uint64_t(v7);
float * const v113 = &v2[v112];
*v113 = v109;
}
}
return;
}
