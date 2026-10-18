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
struct alignas(16) float_4 { float i_0; float i_1; float i_2; float i_3; };

struct info_st {
    int64_t scalars_i64[2];uint64_t static_meta[2];
};
extern "C" __global__ void __launch_bounds__(128)  cubecl5_314dacfd1b0da4a5(float const* const __restrict__ v1, float * const __restrict__ v2, info_st const* const __restrict__ v89) {
uint64_t const* const dynamic_meta_v91 = reinterpret_cast<uint64_t const*>(v89 + 1);
const info_st& info_v90 = *v89;int64_t const v92 = info_v90.scalars_i64[0];
uint64_t const v5 = blockIdx.x;
int64_t const v6 = int64_t(v5);
uint64_t const v7 = threadIdx.x;
int64_t const v8 = int64_t(v7);
int64_t const v9 = 128;
int64_t const v10 = v6 * v9;
int64_t const v11 = 4;
int64_t const v12 = v10 * v11;
int64_t const v16 = v8 * v11;
int64_t const v17 = v12 + v16;
int64_t const v18 = 3;
int64_t const v19 = v17 + v18;
bool const v20 = v19 < v92;
if(v20) {
uint64_t const v21 = uint64_t(v17);
float const* const v22 = &v1[v21];
float_4 const* const v23 = reinterpret_cast<float_4 const*>(v22);
float_4 const v24 = *v23;
float const v27 = v24.i_0;
float const v28 = 1.0;
float const v29 = v27 + v28;
float const v32 = v24.i_1;
float const v34 = v32 + v28;
float const v37 = v24.i_2;
float const v39 = v37 + v28;
float const v42 = v24.i_3;
float const v44 = v42 + v28;
float * const v46 = &v2[v21];
float_4 * const v47 = reinterpret_cast<float_4 *>(v46);
float_4 const v48 = float_4{v29, v34, v39, v44};
*v47 = v48;
}
 else {
bool const v51 = v17 < v92;
if(v51) {
uint64_t const v52 = uint64_t(v17);
float const* const v53 = &v1[v52];
float const v54 = *v53;
float const v55 = 1.0;
float const v56 = v54 + v55;
float * const v58 = &v2[v52];
*v58 = v56;
}
int64_t const v59 = 1;
int64_t const v60 = v17 + v59;
bool const v61 = v60 < v92;
if(v61) {
uint64_t const v62 = uint64_t(v60);
float const* const v63 = &v1[v62];
float const v64 = *v63;
float const v65 = 1.0;
float const v66 = v64 + v65;
float * const v68 = &v2[v62];
*v68 = v66;
}
int64_t const v69 = 2;
int64_t const v70 = v17 + v69;
bool const v71 = v70 < v92;
if(v71) {
uint64_t const v72 = uint64_t(v70);
float const* const v73 = &v1[v72];
float const v74 = *v73;
float const v75 = 1.0;
float const v76 = v74 + v75;
float * const v78 = &v2[v72];
*v78 = v76;
}
if(v20) {
uint64_t const v82 = uint64_t(v19);
float const* const v83 = &v1[v82];
float const v84 = *v83;
float const v85 = 1.0;
float const v86 = v84 + v85;
float * const v88 = &v2[v82];
*v88 = v86;
}
}
return;
}
