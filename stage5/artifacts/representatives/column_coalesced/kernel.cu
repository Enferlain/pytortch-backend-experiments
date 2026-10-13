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
extern "C" __global__ void __launch_bounds__(256)  cubecl5_088d4e0e2e01f494(float const* const __restrict__ v1, float * const __restrict__ v2, info_st const* const __restrict__ v93) {
extern __shared__ __align__(4) char v99[];uint64_t const* const dynamic_meta_v95 = reinterpret_cast<uint64_t const*>(v93 + 1);
const info_st& info_v94 = *v93;int64_t const v97 = info_v94.scalars_i64[1];
int64_t const v98 = info_v94.scalars_i64[2];
uint64_t const v6 = blockIdx.x;
int64_t const v7 = int64_t(v6);
uint64_t const v8 = threadIdx.x;
int64_t const v9 = int64_t(v8);
uint64_t const v10 = threadIdx.y;
int64_t const v11 = int64_t(v10);
int64_t const v12 = 32;
int64_t const v13 = v7 * v12;
int64_t const v14 = v13 + v9;
int64_t const v15 = 0;
int64_t const v16 = 8;
array<float, 256> * const v100 = reinterpret_cast<array<float, 256> *>(&v99[0]);
bool const v18 = v14 < v97;
float v102_store;
float * v102 = &v102_store;float const v21 = 0.0;
*v102 = v21;
int64_t const v22 = 1;
for(int64_t v25 = v11; v25 < v98; v25 += v16) {
if(v18) {
int64_t const v26 = v25 * v97;
int64_t const v27 = v14 + v26;
uint64_t const v28 = uint64_t(v27);
float const* const v29 = &v1[v28];
float const v30 = *v29;
float const v31 = *v102;
float const v32 = v31 + v30;
*v102 = v32;
}
}
float const v33 = *v102;
int64_t const v35 = v11 * v12;
int64_t const v36 = v35 + v9;
uint64_t const v37 = uint64_t(v36);
float * const v38 = &(*v100)[v37];
*v38 = v33;
__syncthreads();
int64_t const v39 = 4;
bool const v40 = v11 < v39;
if(v40) {
int64_t const v41 = v11 + v39;
int64_t const v48 = v41 * v12;
int64_t const v49 = v48 + v9;
uint64_t const v50 = uint64_t(v49);
float * const v51 = &(*v100)[v50];
float const v52 = *v38;
float const v53 = *v51;
float const v54 = v52 + v53;
*v38 = v54;
}
__syncthreads();
int64_t const v55 = 2;
bool const v56 = v11 < v55;
if(v56) {
int64_t const v57 = v11 + v55;
int64_t const v64 = v57 * v12;
int64_t const v65 = v64 + v9;
uint64_t const v66 = uint64_t(v65);
float * const v67 = &(*v100)[v66];
float const v68 = *v38;
float const v69 = *v67;
float const v70 = v68 + v69;
*v38 = v70;
}
__syncthreads();
bool const v72 = v11 < v22;
if(v72) {
int64_t const v73 = v11 + v22;
int64_t const v80 = v73 * v12;
int64_t const v81 = v80 + v9;
uint64_t const v82 = uint64_t(v81);
float * const v83 = &(*v100)[v82];
float const v84 = *v38;
float const v85 = *v83;
float const v86 = v84 + v85;
*v38 = v86;
}
__syncthreads();
uint64_t const v87 = uint64_t(v9);
float * const v88 = &(*v100)[v87];
float const v89 = *v88;
__syncthreads();
bool const v90 = v11 == v15;
if(v90) {
if(v18) {
uint64_t const v91 = uint64_t(v14);
float * const v92 = &v2[v91];
*v92 = v89;
}
}
return;
}
