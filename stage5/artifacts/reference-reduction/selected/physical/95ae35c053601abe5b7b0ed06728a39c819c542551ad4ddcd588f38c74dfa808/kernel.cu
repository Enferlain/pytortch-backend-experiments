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
extern "C" __global__ void __launch_bounds__(256)  cubecl5_79aafae2c4702ae7(float const* const __restrict__ v1, float * const __restrict__ v2, info_st const* const __restrict__ v57) {
extern __shared__ __align__(4) char v63[];uint64_t const* const dynamic_meta_v59 = reinterpret_cast<uint64_t const*>(v57 + 1);
const info_st& info_v58 = *v57;int64_t const v61 = info_v58.scalars_i64[1];
int64_t const v62 = info_v58.scalars_i64[2];
uint64_t const v6 = blockIdx.x;
int64_t const v7 = int64_t(v6);
uint64_t const v8 = threadIdx.x;
int64_t const v9 = int64_t(v8);
int64_t const v10 = 0;
int64_t const v15 = 256;
array<float, 8> * const v64 = reinterpret_cast<array<float, 8> *>(&v63[0]);
bool const v17 = v7 < v62;
if(v17) {
float v66_store;
float * v66 = &v66_store;float const v19 = 0.0;
*v66 = v19;
for(int64_t v23 = v9; v23 < v61; v23 += v15) {
int64_t const v24 = v7 * v61;
int64_t const v25 = v23 + v24;
uint64_t const v26 = uint64_t(v25);
float const* const v27 = &v1[v26];
float const v28 = *v27;
float const v29 = *v66;
float const v30 = v29 + v28;
*v66 = v30;
}
float const v31 = *v66;
uint32_t const v67 = warpSize;
uint32_t const v114 = 256;
uint32_t const v69 = min(v67, v114);
float v108_store;
float * v108 = &v108_store;*v108 = v31;
uint32_t const v71 = 1;
uint32_t v109_store;
uint32_t * v109 = &v109_store;*v109 = v71;
while([&]{
uint32_t const v74 = *v109;
bool const v75 = v74 < v69;
return v75;
}()) {
float const v78 = *v108;
uint32_t const v79 = *v109;
float const v80 = __shfl_xor_sync(__activemask(), v78, v79);
float const v82 = v78 + v80;
*v108 = v82;
uint32_t const v84 = 2;
uint32_t const v85 = v79 * v84;
*v109 = v85;
}
float const v86 = *v108;
int64_t const v33 = 32;
int64_t const v34 = v9 / v33;
int64_t const v35 = v9 % v33;
bool const v36 = v35 == v10;
if(v36) {
uint64_t const v37 = uint64_t(v34);
float * const v38 = &(*v64)[v37];
*v38 = v86;
}
__syncthreads();
bool const v39 = v34 == v10;
if(v39) {
float v87_store;
float * v87 = &v87_store;*v87 = v19;
int64_t const v42 = 8;
bool const v43 = v35 < v42;
if(v43) {
uint64_t const v44 = uint64_t(v35);
float * const v45 = &(*v64)[v44];
float const v46 = *v45;
*v87 = v46;
}
float const v47 = *v87;
float v111_store;
float * v111 = &v111_store;*v111 = v47;
uint32_t v112_store;
uint32_t * v112 = &v112_store;*v112 = v71;
while([&]{
uint32_t const v95 = *v112;
bool const v96 = v95 < v69;
return v96;
}()) {
float const v99 = *v111;
uint32_t const v100 = *v112;
float const v101 = __shfl_xor_sync(__activemask(), v99, v100);
float const v103 = v99 + v101;
*v111 = v103;
uint32_t const v105 = 2;
uint32_t const v106 = v100 * v105;
*v112 = v106;
}
float const v107 = *v111;
if(v36) {
uint64_t const v118 = 0;
float * const v50 = &(*v64)[v118];
*v50 = v107;
}
}
__syncthreads();
uint64_t const v116 = 0;
float * const v52 = &(*v64)[v116];
float const v53 = *v52;
__syncthreads();
bool const v54 = v9 == v10;
if(v54) {
uint64_t const v55 = uint64_t(v7);
float * const v56 = &v2[v55];
*v56 = v53;
}
}
return;
}
