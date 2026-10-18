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
extern "C" __global__ void __launch_bounds__(256)  cubecl5_transpose_33fd8fea8faab7fe(float const* const __restrict__ v1, float * const __restrict__ v2, info_st const* const __restrict__ v125) {
uint64_t const* const dynamic_meta_v127 = reinterpret_cast<uint64_t const*>(v125 + 1);
const info_st& info_v126 = *v125;bool inv_terminate_flag_v131_store = true;
bool * inv_terminate_flag_v131 = &inv_terminate_flag_v131_store;int64_t const v129 = info_v126.scalars_i64[1];
int64_t const v130 = info_v126.scalars_i64[2];
uint64_t const v6 = 2;
uint64_t const v7 = 1;
int64_t const v235 = 1;
int64_t const v10 = 0;
bool const v234 = false;
uint64_t const v13 = uint64_t(v130);
float const v18 = 0.0;
float v134_store;
float * v134 = &v134_store;*v134 = v18;
uint64_t const v26 = 0;
int64_t const v37 = v129 * v130;
uint64_t const v40 = uint64_t(v37);
uint64_t const v43 = v40 / v13;
uint32_t const v146 = blockIdx.z;
uint64_t const v147 = uint64_t(v146);
uint32_t const v148 = gridDim.y;
uint64_t const v149 = uint64_t(v148);
uint64_t const v150 = v147 * v149;
uint32_t const v151 = gridDim.x;
uint64_t const v152 = uint64_t(v151);
uint64_t const v153 = v150 * v152;
uint32_t const v154 = blockIdx.y;
uint64_t const v155 = uint64_t(v154);
uint64_t const v158 = v155 * v152;
uint64_t const v159 = v153 + v158;
uint32_t const v160 = blockIdx.x;
uint64_t const v161 = uint64_t(v160);
uint64_t const v162 = v159 + v161;
uint64_t const v229 = 256;
uint64_t const v165 = v162 * v229;
uint32_t const v193 = threadIdx.x;
uint64_t const v167 = uint64_t(v193);
uint64_t const v168 = v165 + v167;
bool const v45 = v168 >= v43;
if(v45) {
*inv_terminate_flag_v131 = v234;
}
bool const v47 = *inv_terminate_flag_v131;
if(v47) {
uint64_t const v49 = v168 * v13;
uint64_t const v50 = v49 / v13;
uint64_t const v51 = v49 % v13;
uint64_t const v52 = v50 * v13;
uint64_t const v53 = v52 + v51;
for(uint64_t v56 = v26; v56 < v13; v56 += v7) {
uint64_t const v59 = v53 + v56;
uint64_t v137_store;
uint64_t * v137 = &v137_store;*v137 = v26;
for(uint64_t v66 = v26; v66 < v6; v66 += v7) {
int64_t const v71 = int64_t(v66);
bool const v73 = v71 == v10;
int64_t const v75 = v73 ? v130 : v235;
uint64_t const v76 = uint64_t(v75);
uint64_t const v77 = v59 / v76;
int64_t const v81 = v73 ? v129 : v130;
uint64_t const v82 = uint64_t(v81);
uint64_t const v83 = v77 % v82;
int64_t const v88 = v73 ? v235 : v129;
uint64_t const v89 = uint64_t(v88);
uint64_t const v90 = v83 * v89;
uint64_t const v91 = *v137;
uint64_t const v92 = v91 + v90;
*v137 = v92;
}
uint64_t const v93 = *v137;
int64_t const v101 = int64_t(v93);
uint64_t const v104 = uint64_t(v101);
float const* const v105 = &v1[v104];
float const v107 = *v105;
*v134 = v107;
float const v116 = *v134;
int64_t const v117 = int64_t(v59);
uint64_t const v122 = uint64_t(v117);
float * const v123 = &v2[v122];
*v123 = v116;
}
}
return;
}
