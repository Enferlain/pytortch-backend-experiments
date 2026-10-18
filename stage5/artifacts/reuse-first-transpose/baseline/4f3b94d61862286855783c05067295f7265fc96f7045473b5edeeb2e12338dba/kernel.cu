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
extern "C" __global__ void __launch_bounds__(128)  cubecl5_dbe19b7c970725e8(float const* const __restrict__ v1, float * const __restrict__ v2, info_st const* const __restrict__ v110) {
uint64_t const* const dynamic_meta_v112 = reinterpret_cast<uint64_t const*>(v110 + 1);
const info_st& info_v111 = *v110;int64_t const v113 = info_v111.scalars_i64[0];
int64_t const v114 = info_v111.scalars_i64[1];
int64_t const v115 = info_v111.scalars_i64[2];
uint64_t const v6 = blockIdx.x;
int64_t const v7 = int64_t(v6);
uint64_t const v8 = threadIdx.x;
int64_t const v9 = int64_t(v8);
int64_t const v10 = 128;
int64_t const v11 = v7 * v10;
int64_t const v12 = 4;
int64_t const v13 = v11 * v12;
int64_t const v14 = 0;
int64_t const v16 = 1;
int64_t const v18 = v13 + v9;
bool const v19 = v18 < v113;
if(v19) {
int64_t const v117 = v18 % v115;
int64_t const v119 = v117 ^ v115;
bool const v120 = v119 < v14;
bool const v121 = v117 != v14;
bool const v122 = v121 && v120;
int64_t const v123 = v122 ? v115 : v14;
int64_t const v124 = v117 + v123;
int64_t const v21 = v114 * v124;
int64_t const v22 = v18 / v115;
bool const v26 = v117 < v14;
int64_t const v27 = v22 - v16;
int64_t const v28 = v26 ? v27 : v22;
int64_t const v29 = v21 + v28;
uint64_t const v30 = uint64_t(v29);
float const* const v31 = &v1[v30];
float const v32 = *v31;
float const v33 = sin(v32);float const v34 = 1.0;
float const v35 = v33 + v34;
uint64_t const v36 = uint64_t(v18);
float * const v37 = &v2[v36];
*v37 = v35;
}
int64_t const v39 = v9 + v10;
int64_t const v42 = v13 + v39;
bool const v43 = v42 < v113;
if(v43) {
int64_t const v125 = v42 % v115;
int64_t const v127 = v125 ^ v115;
bool const v128 = v127 < v14;
bool const v129 = v125 != v14;
bool const v130 = v129 && v128;
int64_t const v131 = v130 ? v115 : v14;
int64_t const v132 = v125 + v131;
int64_t const v45 = v114 * v132;
int64_t const v46 = v42 / v115;
bool const v50 = v125 < v14;
int64_t const v51 = v46 - v16;
int64_t const v52 = v50 ? v51 : v46;
int64_t const v53 = v45 + v52;
uint64_t const v54 = uint64_t(v53);
float const* const v55 = &v1[v54];
float const v56 = *v55;
float const v57 = sin(v56);float const v58 = 1.0;
float const v59 = v57 + v58;
uint64_t const v60 = uint64_t(v42);
float * const v61 = &v2[v60];
*v61 = v59;
}
int64_t const v62 = 256;
int64_t const v63 = v9 + v62;
int64_t const v66 = v13 + v63;
bool const v67 = v66 < v113;
if(v67) {
int64_t const v133 = v66 % v115;
int64_t const v135 = v133 ^ v115;
bool const v136 = v135 < v14;
bool const v137 = v133 != v14;
bool const v138 = v137 && v136;
int64_t const v139 = v138 ? v115 : v14;
int64_t const v140 = v133 + v139;
int64_t const v69 = v114 * v140;
int64_t const v70 = v66 / v115;
bool const v74 = v133 < v14;
int64_t const v75 = v70 - v16;
int64_t const v76 = v74 ? v75 : v70;
int64_t const v77 = v69 + v76;
uint64_t const v78 = uint64_t(v77);
float const* const v79 = &v1[v78];
float const v80 = *v79;
float const v81 = sin(v80);float const v82 = 1.0;
float const v83 = v81 + v82;
uint64_t const v84 = uint64_t(v66);
float * const v85 = &v2[v84];
*v85 = v83;
}
int64_t const v86 = 384;
int64_t const v87 = v9 + v86;
int64_t const v90 = v13 + v87;
bool const v91 = v90 < v113;
if(v91) {
int64_t const v141 = v90 % v115;
int64_t const v143 = v141 ^ v115;
bool const v144 = v143 < v14;
bool const v145 = v141 != v14;
bool const v146 = v145 && v144;
int64_t const v147 = v146 ? v115 : v14;
int64_t const v148 = v141 + v147;
int64_t const v93 = v114 * v148;
int64_t const v94 = v90 / v115;
bool const v98 = v141 < v14;
int64_t const v99 = v94 - v16;
int64_t const v100 = v98 ? v99 : v94;
int64_t const v101 = v93 + v100;
uint64_t const v102 = uint64_t(v101);
float const* const v103 = &v1[v102];
float const v104 = *v103;
float const v105 = sin(v104);float const v106 = 1.0;
float const v107 = v105 + v106;
uint64_t const v108 = uint64_t(v90);
float * const v109 = &v2[v108];
*v109 = v107;
}
return;
}
