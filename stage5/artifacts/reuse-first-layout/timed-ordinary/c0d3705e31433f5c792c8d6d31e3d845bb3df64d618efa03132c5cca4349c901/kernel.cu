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
#include <cuda_bf16.h>

struct info_st {
    int64_t scalars_i64[3];uint64_t static_meta[2];
};
extern "C" __global__ void __launch_bounds__(128)  cubecl5_11fd74db5e6b823c(__nv_bfloat16 const* const __restrict__ v1, __nv_bfloat16 * const __restrict__ v2, info_st const* const __restrict__ v114) {
uint64_t const* const dynamic_meta_v116 = reinterpret_cast<uint64_t const*>(v114 + 1);
const info_st& info_v115 = *v114;int64_t const v117 = info_v115.scalars_i64[0];
int64_t const v118 = info_v115.scalars_i64[1];
uint64_t const v6 = blockIdx.x;
int64_t const v7 = int64_t(v6);
uint64_t const v8 = threadIdx.x;
int64_t const v9 = int64_t(v8);
int64_t const v10 = 128;
int64_t const v11 = v7 * v10;
int64_t const v12 = 4;
int64_t const v13 = v11 * v12;
int64_t const v18 = v13 + v9;
bool const v19 = v18 < v117;
if(v19) {
int64_t const v20 = v18 * v118;
uint64_t const v21 = uint64_t(v20);
__nv_bfloat16 const* const v22 = &v1[v21];
__nv_bfloat16 const v23 = *v22;
float const v24 = float(v23);
__nv_bfloat16 const v25 = __nv_bfloat16(v24);
float const v26 = float(v25);
float const v27 = sin(v26);__nv_bfloat16 const v28 = __nv_bfloat16(v27);
float const v29 = float(v28);
__nv_bfloat16 const v30 = __nv_bfloat16(v29);
float const v31 = float(v30);
float const v32 = 1.0;
float const v33 = v31 + v32;
__nv_bfloat16 const v34 = __nv_bfloat16(v33);
float const v35 = float(v34);
__nv_bfloat16 const v36 = __nv_bfloat16(v35);
uint64_t const v37 = uint64_t(v18);
__nv_bfloat16 * const v38 = &v2[v37];
*v38 = v36;
}
int64_t const v40 = v9 + v10;
int64_t const v43 = v13 + v40;
bool const v44 = v43 < v117;
if(v44) {
int64_t const v45 = v43 * v118;
uint64_t const v46 = uint64_t(v45);
__nv_bfloat16 const* const v47 = &v1[v46];
__nv_bfloat16 const v48 = *v47;
float const v49 = float(v48);
__nv_bfloat16 const v50 = __nv_bfloat16(v49);
float const v51 = float(v50);
float const v52 = sin(v51);__nv_bfloat16 const v53 = __nv_bfloat16(v52);
float const v54 = float(v53);
__nv_bfloat16 const v55 = __nv_bfloat16(v54);
float const v56 = float(v55);
float const v57 = 1.0;
float const v58 = v56 + v57;
__nv_bfloat16 const v59 = __nv_bfloat16(v58);
float const v60 = float(v59);
__nv_bfloat16 const v61 = __nv_bfloat16(v60);
uint64_t const v62 = uint64_t(v43);
__nv_bfloat16 * const v63 = &v2[v62];
*v63 = v61;
}
int64_t const v64 = 256;
int64_t const v65 = v9 + v64;
int64_t const v68 = v13 + v65;
bool const v69 = v68 < v117;
if(v69) {
int64_t const v70 = v68 * v118;
uint64_t const v71 = uint64_t(v70);
__nv_bfloat16 const* const v72 = &v1[v71];
__nv_bfloat16 const v73 = *v72;
float const v74 = float(v73);
__nv_bfloat16 const v75 = __nv_bfloat16(v74);
float const v76 = float(v75);
float const v77 = sin(v76);__nv_bfloat16 const v78 = __nv_bfloat16(v77);
float const v79 = float(v78);
__nv_bfloat16 const v80 = __nv_bfloat16(v79);
float const v81 = float(v80);
float const v82 = 1.0;
float const v83 = v81 + v82;
__nv_bfloat16 const v84 = __nv_bfloat16(v83);
float const v85 = float(v84);
__nv_bfloat16 const v86 = __nv_bfloat16(v85);
uint64_t const v87 = uint64_t(v68);
__nv_bfloat16 * const v88 = &v2[v87];
*v88 = v86;
}
int64_t const v89 = 384;
int64_t const v90 = v9 + v89;
int64_t const v93 = v13 + v90;
bool const v94 = v93 < v117;
if(v94) {
int64_t const v95 = v93 * v118;
uint64_t const v96 = uint64_t(v95);
__nv_bfloat16 const* const v97 = &v1[v96];
__nv_bfloat16 const v98 = *v97;
float const v99 = float(v98);
__nv_bfloat16 const v100 = __nv_bfloat16(v99);
float const v101 = float(v100);
float const v102 = sin(v101);__nv_bfloat16 const v103 = __nv_bfloat16(v102);
float const v104 = float(v103);
__nv_bfloat16 const v105 = __nv_bfloat16(v104);
float const v106 = float(v105);
float const v107 = 1.0;
float const v108 = v106 + v107;
__nv_bfloat16 const v109 = __nv_bfloat16(v108);
float const v110 = float(v109);
__nv_bfloat16 const v111 = __nv_bfloat16(v110);
uint64_t const v112 = uint64_t(v93);
__nv_bfloat16 * const v113 = &v2[v112];
*v113 = v111;
}
return;
}
