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
extern "C" __global__ void __launch_bounds__(128)  cubecl5_b25d5bd2f8183287(float const* const __restrict__ v1, float * const __restrict__ v2, info_st const* const __restrict__ v98) {
uint64_t const* const dynamic_meta_v100 = reinterpret_cast<uint64_t const*>(v98 + 1);
const info_st& info_v99 = *v98;int64_t const v101 = info_v99.scalars_i64[0];
int64_t const v102 = info_v99.scalars_i64[1];
int64_t const v103 = info_v99.scalars_i64[2];
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
bool const v19 = v18 < v101;
if(v19) {
int64_t const v105 = v18 % v103;
int64_t const v107 = v105 ^ v103;
bool const v108 = v107 < v14;
bool const v109 = v105 != v14;
bool const v110 = v109 && v108;
int64_t const v111 = v110 ? v103 : v14;
int64_t const v112 = v105 + v111;
int64_t const v21 = v102 * v112;
int64_t const v22 = v18 / v103;
bool const v26 = v105 < v14;
int64_t const v27 = v22 - v16;
int64_t const v28 = v26 ? v27 : v22;
int64_t const v29 = v21 + v28;
uint64_t const v30 = uint64_t(v29);
float const* const v31 = &v1[v30];
float const v32 = *v31;
uint64_t const v33 = uint64_t(v18);
float * const v34 = &v2[v33];
*v34 = v32;
}
int64_t const v36 = v9 + v10;
int64_t const v39 = v13 + v36;
bool const v40 = v39 < v101;
if(v40) {
int64_t const v113 = v39 % v103;
int64_t const v115 = v113 ^ v103;
bool const v116 = v115 < v14;
bool const v117 = v113 != v14;
bool const v118 = v117 && v116;
int64_t const v119 = v118 ? v103 : v14;
int64_t const v120 = v113 + v119;
int64_t const v42 = v102 * v120;
int64_t const v43 = v39 / v103;
bool const v47 = v113 < v14;
int64_t const v48 = v43 - v16;
int64_t const v49 = v47 ? v48 : v43;
int64_t const v50 = v42 + v49;
uint64_t const v51 = uint64_t(v50);
float const* const v52 = &v1[v51];
float const v53 = *v52;
uint64_t const v54 = uint64_t(v39);
float * const v55 = &v2[v54];
*v55 = v53;
}
int64_t const v56 = 256;
int64_t const v57 = v9 + v56;
int64_t const v60 = v13 + v57;
bool const v61 = v60 < v101;
if(v61) {
int64_t const v121 = v60 % v103;
int64_t const v123 = v121 ^ v103;
bool const v124 = v123 < v14;
bool const v125 = v121 != v14;
bool const v126 = v125 && v124;
int64_t const v127 = v126 ? v103 : v14;
int64_t const v128 = v121 + v127;
int64_t const v63 = v102 * v128;
int64_t const v64 = v60 / v103;
bool const v68 = v121 < v14;
int64_t const v69 = v64 - v16;
int64_t const v70 = v68 ? v69 : v64;
int64_t const v71 = v63 + v70;
uint64_t const v72 = uint64_t(v71);
float const* const v73 = &v1[v72];
float const v74 = *v73;
uint64_t const v75 = uint64_t(v60);
float * const v76 = &v2[v75];
*v76 = v74;
}
int64_t const v77 = 384;
int64_t const v78 = v9 + v77;
int64_t const v81 = v13 + v78;
bool const v82 = v81 < v101;
if(v82) {
int64_t const v129 = v81 % v103;
int64_t const v131 = v129 ^ v103;
bool const v132 = v131 < v14;
bool const v133 = v129 != v14;
bool const v134 = v133 && v132;
int64_t const v135 = v134 ? v103 : v14;
int64_t const v136 = v129 + v135;
int64_t const v84 = v102 * v136;
int64_t const v85 = v81 / v103;
bool const v89 = v129 < v14;
int64_t const v90 = v85 - v16;
int64_t const v91 = v89 ? v90 : v85;
int64_t const v92 = v84 + v91;
uint64_t const v93 = uint64_t(v92);
float const* const v94 = &v1[v93];
float const v95 = *v94;
uint64_t const v96 = uint64_t(v81);
float * const v97 = &v2[v96];
*v97 = v95;
}
return;
}
