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
    int64_t scalars_i64[2];uint64_t static_meta[2];
};
extern "C" __global__ void __launch_bounds__(128)  cubecl5_2c49a01de0211073(float const* const __restrict__ v1, bool * const __restrict__ v2, info_st const* const __restrict__ v65) {
uint64_t const* const dynamic_meta_v67 = reinterpret_cast<uint64_t const*>(v65 + 1);
const info_st& info_v66 = *v65;int64_t const v68 = info_v66.scalars_i64[0];
uint64_t const v5 = blockIdx.x;
int64_t const v6 = int64_t(v5);
uint64_t const v7 = threadIdx.x;
int64_t const v8 = int64_t(v7);
int64_t const v9 = 128;
int64_t const v10 = v6 * v9;
int64_t const v11 = 4;
int64_t const v12 = v10 * v11;
int64_t const v17 = v12 + v8;
bool const v18 = v17 < v68;
if(v18) {
uint64_t const v19 = uint64_t(v17);
float const* const v20 = &v1[v19];
float const v21 = *v20;
float const v22 = 0.25;
bool const v23 = v21 > v22;
bool * const v25 = &v2[v19];
*v25 = v23;
}
int64_t const v27 = v8 + v9;
int64_t const v30 = v12 + v27;
bool const v31 = v30 < v68;
if(v31) {
uint64_t const v32 = uint64_t(v30);
float const* const v33 = &v1[v32];
float const v34 = *v33;
float const v35 = 0.25;
bool const v36 = v34 > v35;
bool * const v38 = &v2[v32];
*v38 = v36;
}
int64_t const v39 = 256;
int64_t const v40 = v8 + v39;
int64_t const v43 = v12 + v40;
bool const v44 = v43 < v68;
if(v44) {
uint64_t const v45 = uint64_t(v43);
float const* const v46 = &v1[v45];
float const v47 = *v46;
float const v48 = 0.25;
bool const v49 = v47 > v48;
bool * const v51 = &v2[v45];
*v51 = v49;
}
int64_t const v52 = 384;
int64_t const v53 = v8 + v52;
int64_t const v56 = v12 + v53;
bool const v57 = v56 < v68;
if(v57) {
uint64_t const v58 = uint64_t(v56);
float const* const v59 = &v1[v58];
float const v60 = *v59;
float const v61 = 0.25;
bool const v62 = v60 > v61;
bool * const v64 = &v2[v58];
*v64 = v62;
}
return;
}
