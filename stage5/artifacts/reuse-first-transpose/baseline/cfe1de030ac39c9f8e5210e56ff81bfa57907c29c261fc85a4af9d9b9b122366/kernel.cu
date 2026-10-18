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
#include <cuda_fp16.h>
struct alignas(4) __half_2 { __half i_0; __half i_1; };

struct info_st {
    int64_t scalars_i64[3];uint64_t static_meta[2];
};
extern "C" __global__ void __launch_bounds__(256)  cubecl5_6b1cb950c3e68c8f(__half const* const __restrict__ v1, __half * const __restrict__ v2, info_st const* const __restrict__ v104) {
uint64_t const* const dynamic_meta_v106 = reinterpret_cast<uint64_t const*>(v104 + 1);
const info_st& info_v105 = *v104;int64_t const v107 = info_v105.scalars_i64[0];
uint64_t const v6 = blockIdx.x;
int64_t const v7 = int64_t(v6);
uint64_t const v8 = threadIdx.x;
int64_t const v9 = int64_t(v8);
int64_t const v10 = 256;
int64_t const v11 = v7 * v10;
int64_t const v12 = 2;
int64_t const v13 = v11 * v12;
int64_t const v17 = v9 * v12;
int64_t const v18 = v13 + v17;
int64_t const v19 = 1;
int64_t const v20 = v18 + v19;
bool const v21 = v20 < v107;
if(v21) {
uint64_t const v22 = uint64_t(v18);
__half const* const v23 = &v1[v22];
__half_2 const* const v24 = reinterpret_cast<__half_2 const*>(v23);
__half_2 const v25 = *v24;
__half const v28 = v25.i_0;
float const v29 = float(v28);
__half const v30 = __half(v29);
float const v31 = float(v30);
float const v32 = sin(v31);__half const v33 = __half(v32);
float const v34 = float(v33);
__half const v35 = __half(v34);
float const v36 = float(v35);
float const v37 = 1.0;
float const v38 = v36 + v37;
__half const v39 = __half(v38);
float const v40 = float(v39);
__half const v41 = __half(v40);
__half const v44 = v25.i_1;
float const v45 = float(v44);
__half const v46 = __half(v45);
float const v47 = float(v46);
float const v48 = sin(v47);__half const v49 = __half(v48);
float const v50 = float(v49);
__half const v51 = __half(v50);
float const v52 = float(v51);
float const v54 = v52 + v37;
__half const v55 = __half(v54);
float const v56 = float(v55);
__half const v57 = __half(v56);
__half * const v59 = &v2[v22];
__half_2 * const v60 = reinterpret_cast<__half_2 *>(v59);
__half_2 const v61 = __half_2{v41, v57};
*v60 = v61;
}
 else {
bool const v64 = v18 < v107;
if(v64) {
uint64_t const v65 = uint64_t(v18);
__half const* const v66 = &v1[v65];
__half const v67 = *v66;
float const v68 = float(v67);
__half const v69 = __half(v68);
float const v70 = float(v69);
float const v71 = sin(v70);__half const v72 = __half(v71);
float const v73 = float(v72);
__half const v74 = __half(v73);
float const v75 = float(v74);
float const v76 = 1.0;
float const v77 = v75 + v76;
__half const v78 = __half(v77);
float const v79 = float(v78);
__half const v80 = __half(v79);
__half * const v82 = &v2[v65];
*v82 = v80;
}
if(v21) {
uint64_t const v86 = uint64_t(v20);
__half const* const v87 = &v1[v86];
__half const v88 = *v87;
float const v89 = float(v88);
__half const v90 = __half(v89);
float const v91 = float(v90);
float const v92 = sin(v91);__half const v93 = __half(v92);
float const v94 = float(v93);
__half const v95 = __half(v94);
float const v96 = float(v95);
float const v97 = 1.0;
float const v98 = v96 + v97;
__half const v99 = __half(v98);
float const v100 = float(v99);
__half const v101 = __half(v100);
__half * const v103 = &v2[v86];
*v103 = v101;
}
}
return;
}
