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
    int64_t scalars_i64[2];uint64_t static_meta[5];
};
extern "C" __global__ void __launch_bounds__(128)  cubecl5_978c83e5edab78a4(float const* const __restrict__ v1, bool const* const __restrict__ v2, float const* const __restrict__ v3, float const* const __restrict__ v4, float * const __restrict__ v5, info_st const* const __restrict__ v116) {
uint64_t const* const dynamic_meta_v118 = reinterpret_cast<uint64_t const*>(v116 + 1);
const info_st& info_v117 = *v116;int64_t const v120 = info_v117.scalars_i64[1];
uint64_t const v8 = blockIdx.x;
int64_t const v9 = int64_t(v8);
uint64_t const v10 = threadIdx.x;
int64_t const v11 = int64_t(v10);
uint64_t const v12 = threadIdx.y;
int64_t const v13 = int64_t(v12);
int64_t const v14 = 4;
int64_t const v15 = v9 * v14;
int64_t const v16 = v15 + v13;
int64_t const v18 = 32;
int64_t const v19 = 9;
bool const v21 = v16 < v120;
if(v21) {
float v124_store;
float * v124 = &v124_store;float const v23 = 0.0;
*v124 = v23;
for(int64_t v27 = v11; v27 < v19; v27 += v18) {
uint64_t const v150 = 0;
float const* const v30 = &v1[v150];
float const v31 = *v30;
int64_t const v33 = v19 * v120;
float const v34 = float(v33);
float const v35 = v31 / v34;
int64_t const v37 = v19 * v16;
int64_t const v38 = v27 + v37;
uint64_t const v39 = uint64_t(v38);
bool const* const v40 = &v2[v39];
bool const v41 = *v40;
float const v42 = float(v41);
float const* const v47 = &v3[v39];
float const v48 = *v47;
uint64_t const v49 = uint64_t(v16);
float const* const v50 = &v4[v49];
float const v51 = *v50;
float const v52 = v48 * v51;
float const v53 = v42 * v52;
float const v54 = 1.33333337;
float const v55 = v53 * v54;
float const v56 = 2.0;
float const v57 = v55 * v56;
float const v58 = v35 * v57;
float const v59 = v42 * v54;
float const v60 = v58 * v59;
float const v61 = v60 * v48;
float const v62 = *v124;
float const v63 = v62 + v61;
*v124 = v63;
}
float const v64 = *v124;
uint32_t const v125 = warpSize;
uint32_t const v148 = 128;
uint32_t const v127 = min(v125, v148);
float v145_store;
float * v145 = &v145_store;*v145 = v64;
uint32_t const v129 = 1;
uint32_t v146_store;
uint32_t * v146 = &v146_store;*v146 = v129;
while([&]{
uint32_t const v132 = *v146;
bool const v133 = v132 < v127;
return v133;
}()) {
float const v136 = *v145;
uint32_t const v137 = *v146;
float const v138 = __shfl_xor_sync(__activemask(), v136, v137);
float const v140 = v136 + v138;
*v145 = v140;
uint32_t const v142 = 2;
uint32_t const v143 = v137 * v142;
*v146 = v143;
}
float const v144 = *v145;
for(int64_t v66 = v11; v66 < v19; v66 += v18) {
uint64_t const v151 = 0;
float const* const v69 = &v1[v151];
float const v70 = *v69;
int64_t const v72 = v19 * v120;
float const v73 = float(v72);
float const v74 = v70 / v73;
int64_t const v76 = v19 * v16;
int64_t const v77 = v66 + v76;
uint64_t const v78 = uint64_t(v77);
bool const* const v79 = &v2[v78];
bool const v80 = *v79;
float const v81 = float(v80);
float const* const v86 = &v3[v78];
float const v87 = *v86;
uint64_t const v88 = uint64_t(v16);
float const* const v89 = &v4[v88];
float const v90 = *v89;
float const v91 = v87 * v90;
float const v92 = v81 * v91;
float const v93 = 1.33333337;
float const v94 = v92 * v93;
float const v95 = 2.0;
float const v96 = v94 * v95;
float const v97 = v74 * v96;
float const v98 = v81 * v93;
float const v99 = v97 * v98;
float const v100 = v99 * v90;
float const v101 = -0.5;
float const v102 = v144 * v101;
float const v103 = v90 * v90;
float const v104 = v103 * v90;
float const v105 = v102 * v104;
float const v106 = 0.111111112;
float const v107 = v105 * v106;
float const v108 = v107 * v87;
float const v109 = v100 + v108;
float const v110 = v109 + v108;
float * const v115 = &v5[v78];
*v115 = v110;
}
}
return;
}
