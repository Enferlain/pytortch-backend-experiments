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
    uint64_t scalars_u64[2];uint64_t static_meta[2];
};
extern "C" __global__ void __launch_bounds__(512)  cubek_reader_mask(float const* const __restrict__ v1, float * const __restrict__ v2, info_st const* const __restrict__ v124) {
uint64_t const* const dynamic_meta_v126 = reinterpret_cast<uint64_t const*>(v124 + 1);
const info_st& info_v125 = *v124;uint64_t const v127 = info_v125.scalars_usize[0];
uint64_t const v128 = info_v125.scalars_usize[1];
uint64_t const v7 = 0;
float const v11 = 0.0;
uint32_t const v15 = blockIdx.x;
uint64_t const v16 = uint64_t(v15);
uint32_t const v139 = threadIdx.x;
uint64_t const v18 = uint64_t(v139);
bool const v19 = v16 < v128;
if(v19) {
float v133_store;
float * v133 = &v133_store;*v133 = v11;
float v134_store;
float * v134 = &v134_store;*v134 = v11;
float v135_store;
float * v135 = &v135_store;*v135 = v11;
float v136_store;
float * v136 = &v136_store;*v136 = v11;
uint64_t const v32 = 2048;
uint64_t const v33 = v127 + v32;
uint64_t const v34 = 1;
uint64_t const v35 = v33 - v34;
uint64_t const v37 = v35 / v32;
for(uint64_t v40 = v7; v40 < v37; v40 += v34) {
uint64_t const v44 = v40 * v32;
uint64_t const v45 = v44 + v18;
uint64_t const v46 = v16 * v127;
uint64_t const v47 = v46 + v45;
bool const v48 = v45 < v127;
uint64_t const v49 = uint64_t(v48);
uint64_t const v50 = v47 * v49;
float const* const v54 = &v1[v50];
float const v55 = *v54;
float const v56 = v48 ? v55 : v11;
uint64_t const v57 = 512;
uint64_t const v58 = v45 + v57;
uint64_t const v62 = v47 + v57;
bool const v63 = v58 < v127;
uint64_t const v64 = uint64_t(v63);
uint64_t const v65 = v62 * v64;
float const* const v69 = &v1[v65];
float const v70 = *v69;
float const v71 = v63 ? v70 : v11;
uint64_t const v72 = 1024;
uint64_t const v73 = v45 + v72;
uint64_t const v77 = v47 + v72;
bool const v78 = v73 < v127;
uint64_t const v79 = uint64_t(v78);
uint64_t const v80 = v77 * v79;
float const* const v84 = &v1[v80];
float const v85 = *v84;
float const v86 = v78 ? v85 : v11;
uint64_t const v87 = 1536;
uint64_t const v88 = v45 + v87;
uint64_t const v92 = v47 + v87;
bool const v93 = v88 < v127;
uint64_t const v94 = uint64_t(v93);
uint64_t const v95 = v92 * v94;
float const* const v99 = &v1[v95];
float const v100 = *v99;
float const v101 = v93 ? v100 : v11;
float const v102 = *v133;
float const v103 = v102 + v56;
*v133 = v103;
float const v104 = *v134;
float const v105 = v104 + v71;
*v134 = v105;
float const v106 = *v135;
float const v107 = v106 + v86;
*v135 = v107;
float const v108 = *v136;
float const v109 = v108 + v101;
*v136 = v109;
}
float const v110 = *v133;
float const v111 = *v134;
float const v112 = v110 + v111;
float const v113 = *v135;
float const v114 = *v136;
float const v115 = v113 + v114;
float const v116 = v112 + v115;
uint64_t const v117 = 512;
uint64_t const v118 = v16 * v117;
uint64_t const v119 = v118 + v18;
float * const v123 = &v2[v119];
*v123 = v116;
}
return;
}
