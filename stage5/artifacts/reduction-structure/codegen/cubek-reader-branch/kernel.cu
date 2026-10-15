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
extern "C" __global__ void __launch_bounds__(512)  cubek_reader_branch(float const* const __restrict__ v1, float * const __restrict__ v2, info_st const* const __restrict__ v120) {
uint64_t const* const dynamic_meta_v122 = reinterpret_cast<uint64_t const*>(v120 + 1);
const info_st& info_v121 = *v120;uint64_t const v123 = info_v121.scalars_usize[0];
uint64_t const v124 = info_v121.scalars_usize[1];
uint64_t const v7 = 0;
float const v11 = 0.0;
uint32_t const v15 = blockIdx.x;
uint64_t const v16 = uint64_t(v15);
uint32_t const v139 = threadIdx.x;
uint64_t const v18 = uint64_t(v139);
bool const v19 = v16 < v124;
if(v19) {
float v129_store;
float * v129 = &v129_store;*v129 = v11;
float v130_store;
float * v130 = &v130_store;*v130 = v11;
float v131_store;
float * v131 = &v131_store;*v131 = v11;
float v132_store;
float * v132 = &v132_store;*v132 = v11;
uint64_t const v32 = 2048;
uint64_t const v33 = v123 + v32;
uint64_t const v34 = 1;
uint64_t const v35 = v33 - v34;
uint64_t const v37 = v35 / v32;
for(uint64_t v40 = v7; v40 < v37; v40 += v34) {
uint64_t const v44 = v40 * v32;
uint64_t const v45 = v44 + v18;
uint64_t const v46 = v16 * v123;
uint64_t const v47 = v46 + v45;
bool const v48 = v45 < v123;
float v135_store;
float * v135 = &v135_store;if(v48) {
float const* const v52 = &v1[v47];
float const v53 = *v52;
*v135 = v53;
}
 else {
*v135 = v11;
}
uint64_t const v55 = 512;
uint64_t const v56 = v45 + v55;
uint64_t const v60 = v47 + v55;
bool const v61 = v56 < v123;
float v136_store;
float * v136 = &v136_store;if(v61) {
float const* const v65 = &v1[v60];
float const v66 = *v65;
*v136 = v66;
}
 else {
*v136 = v11;
}
uint64_t const v68 = 1024;
uint64_t const v69 = v45 + v68;
uint64_t const v73 = v47 + v68;
bool const v74 = v69 < v123;
float v137_store;
float * v137 = &v137_store;if(v74) {
float const* const v78 = &v1[v73];
float const v79 = *v78;
*v137 = v79;
}
 else {
*v137 = v11;
}
uint64_t const v81 = 1536;
uint64_t const v82 = v45 + v81;
uint64_t const v86 = v47 + v81;
bool const v87 = v82 < v123;
float v138_store;
float * v138 = &v138_store;if(v87) {
float const* const v91 = &v1[v86];
float const v92 = *v91;
*v138 = v92;
}
 else {
*v138 = v11;
}
float const v94 = *v135;
float const v95 = *v129;
float const v96 = v95 + v94;
*v129 = v96;
float const v97 = *v136;
float const v98 = *v130;
float const v99 = v98 + v97;
*v130 = v99;
float const v100 = *v137;
float const v101 = *v131;
float const v102 = v101 + v100;
*v131 = v102;
float const v103 = *v138;
float const v104 = *v132;
float const v105 = v104 + v103;
*v132 = v105;
}
float const v106 = *v129;
float const v107 = *v130;
float const v108 = v106 + v107;
float const v109 = *v131;
float const v110 = *v132;
float const v111 = v109 + v110;
float const v112 = v108 + v111;
uint64_t const v113 = 512;
uint64_t const v114 = v16 * v113;
uint64_t const v115 = v114 + v18;
float * const v119 = &v2[v115];
*v119 = v112;
}
return;
}
