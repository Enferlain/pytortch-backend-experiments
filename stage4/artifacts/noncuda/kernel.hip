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


struct info_st {
    int64_t scalars_i64[3];uint64_t static_meta[3];
};
extern "C" __global__ void __launch_bounds__(256) cubecl4_23d35b5081231fe7(int64_t const* const __restrict__ v1, float const* const __restrict__ v2, float * const __restrict__ v3, info_st const* const __restrict__ v30) {
uint64_t const* const dynamic_meta_v32 = reinterpret_cast<uint64_t const*>(v30 + 1);
const info_st& info_v31 = *v30;int64_t const v33 = info_v31.scalars_i64[0];
int64_t const v34 = info_v31.scalars_i64[1];
int64_t const v35 = info_v31.scalars_i64[2];
uint64_t const v7 = blockIdx.x;
int64_t const v8 = int64_t(v7);
uint64_t const v9 = threadIdx.x;
int64_t const v10 = int64_t(v9);
int64_t const v11 = 256;
int64_t const v12 = v8 * v11;
int64_t const v13 = v12 + v10;
bool const v14 = v13 < v33;
if(v14) {
int64_t const v15 = 0;
uint64_t const v38 = 0;
int64_t const* const v17 = &v1[v38];
int64_t const v18 = *v17;
bool const v20 = v18 >= v15;
bool const v21 = v18 < v34;
bool const v22 = v20 & v21;
if (!v22) { __builtin_trap(); }
int64_t const v23 = v18 * v35;
uint64_t const v24 = uint64_t(v23);
float const* const v25 = &v2[v24];
float const v26 = *v25;
float * const v29 = &v3[v38];
*v29 = v26;
}
return;
}
