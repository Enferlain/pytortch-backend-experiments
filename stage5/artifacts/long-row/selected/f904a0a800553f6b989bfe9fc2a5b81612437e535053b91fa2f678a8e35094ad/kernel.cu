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
extern "C" __global__ void __launch_bounds__(256)  cubecl5_9f083fc429d5c5ce(float const* const __restrict__ v1, float * const __restrict__ v2, info_st const* const __restrict__ v101) {
extern __shared__ __align__(4) char v107[];uint64_t const* const dynamic_meta_v103 = reinterpret_cast<uint64_t const*>(v101 + 1);
const info_st& info_v102 = *v101;int64_t const v105 = info_v102.scalars_i64[1];
int64_t const v106 = info_v102.scalars_i64[2];
uint64_t const v6 = blockIdx.x;
int64_t const v7 = int64_t(v6);
uint64_t const v8 = threadIdx.x;
int64_t const v9 = int64_t(v8);
int64_t const v10 = 0;
int64_t const v15 = 256;
array<float, 8> * const v108 = reinterpret_cast<array<float, 8> *>(&v107[0]);
bool const v17 = v7 < v106;
if(v17) {
float v110_store;
float * v110 = &v110_store;float const v19 = 0.0;
*v110 = v19;
float v111_store;
float * v111 = &v111_store;*v111 = v19;
float v112_store;
float * v112 = &v112_store;*v112 = v19;
float v113_store;
float * v113 = &v113_store;*v113 = v19;
int64_t const v164 = 1024;
for(int64_t v28 = v9; v28 < v105; v28 += v164) {
bool const v31 = v28 < v105;
if(v31) {
int64_t const v32 = v7 * v105;
int64_t const v33 = v28 + v32;
uint64_t const v34 = uint64_t(v33);
float const* const v35 = &v1[v34];
float const v36 = *v35;
float const v37 = *v110;
float const v38 = v37 + v36;
*v110 = v38;
}
int64_t const v40 = v28 + v15;
bool const v41 = v40 < v105;
if(v41) {
int64_t const v42 = v7 * v105;
int64_t const v43 = v40 + v42;
uint64_t const v44 = uint64_t(v43);
float const* const v45 = &v1[v44];
float const v46 = *v45;
float const v47 = *v111;
float const v48 = v47 + v46;
*v111 = v48;
}
int64_t const v49 = 512;
int64_t const v50 = v28 + v49;
bool const v51 = v50 < v105;
if(v51) {
int64_t const v52 = v7 * v105;
int64_t const v53 = v50 + v52;
uint64_t const v54 = uint64_t(v53);
float const* const v55 = &v1[v54];
float const v56 = *v55;
float const v57 = *v112;
float const v58 = v57 + v56;
*v112 = v58;
}
int64_t const v59 = 768;
int64_t const v60 = v28 + v59;
bool const v61 = v60 < v105;
if(v61) {
int64_t const v62 = v7 * v105;
int64_t const v63 = v60 + v62;
uint64_t const v64 = uint64_t(v63);
float const* const v65 = &v1[v64];
float const v66 = *v65;
float const v67 = *v113;
float const v68 = v67 + v66;
*v113 = v68;
}
}
float const v69 = *v110;
float const v70 = *v111;
float const v71 = *v112;
float const v72 = *v113;
float const v73 = v69 + v71;
float const v74 = v70 + v72;
float const v75 = v73 + v74;
uint32_t const v114 = warpSize;
uint32_t const v161 = 256;
uint32_t const v116 = min(v114, v161);
float v155_store;
float * v155 = &v155_store;*v155 = v75;
uint32_t const v118 = 1;
uint32_t v156_store;
uint32_t * v156 = &v156_store;*v156 = v118;
while([&]{
uint32_t const v121 = *v156;
bool const v122 = v121 < v116;
return v122;
}()) {
float const v125 = *v155;
uint32_t const v126 = *v156;
float const v127 = __shfl_xor_sync(__activemask(), v125, v126);
float const v129 = v125 + v127;
*v155 = v129;
uint32_t const v131 = 2;
uint32_t const v132 = v126 * v131;
*v156 = v132;
}
float const v133 = *v155;
int64_t const v77 = 32;
int64_t const v78 = v9 / v77;
int64_t const v79 = v9 % v77;
bool const v80 = v79 == v10;
if(v80) {
uint64_t const v81 = uint64_t(v78);
float * const v82 = &(*v108)[v81];
*v82 = v133;
}
__syncthreads();
bool const v83 = v78 == v10;
if(v83) {
float v134_store;
float * v134 = &v134_store;*v134 = v19;
int64_t const v86 = 8;
bool const v87 = v79 < v86;
if(v87) {
uint64_t const v88 = uint64_t(v79);
float * const v89 = &(*v108)[v88];
float const v90 = *v89;
*v134 = v90;
}
float const v91 = *v134;
float v158_store;
float * v158 = &v158_store;*v158 = v91;
uint32_t v159_store;
uint32_t * v159 = &v159_store;*v159 = v118;
while([&]{
uint32_t const v142 = *v159;
bool const v143 = v142 < v116;
return v143;
}()) {
float const v146 = *v158;
uint32_t const v147 = *v159;
float const v148 = __shfl_xor_sync(__activemask(), v146, v147);
float const v150 = v146 + v148;
*v158 = v150;
uint32_t const v152 = 2;
uint32_t const v153 = v147 * v152;
*v159 = v153;
}
float const v154 = *v158;
if(v80) {
uint64_t const v165 = 0;
float * const v94 = &(*v108)[v165];
*v94 = v154;
}
}
__syncthreads();
uint64_t const v163 = 0;
float * const v96 = &(*v108)[v163];
float const v97 = *v96;
__syncthreads();
bool const v98 = v9 == v10;
if(v98) {
uint64_t const v99 = uint64_t(v7);
float * const v100 = &v2[v99];
*v100 = v97;
}
}
return;
}
