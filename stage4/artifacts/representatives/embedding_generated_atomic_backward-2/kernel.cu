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
    int64_t scalars_i64[2];uint64_t static_meta[3];
};
extern "C" __global__ void __launch_bounds__(256)  cubecl4_de5c7eb836a5d7f3(int64_t const* const __restrict__ v1, float const* const __restrict__ v2, float const* const __restrict__ v3, info_st const* const __restrict__ v48) {
uint64_t const* const dynamic_meta_v50 = reinterpret_cast<uint64_t const*>(v48 + 1);
const info_st& info_v49 = *v48;int64_t const v51 = info_v49.scalars_i64[0];
int64_t const v52 = info_v49.scalars_i64[1];
uint64_t const v6 = blockIdx.x;
int64_t const v7 = int64_t(v6);
uint64_t const v8 = threadIdx.x;
int64_t const v9 = int64_t(v8);
int64_t const v10 = 256;
int64_t const v11 = v7 * v10;
int64_t const v12 = v11 + v9;
bool const v13 = v12 < v51;
if(v13) {
int64_t const v14 = 0;
uint64_t const v55 = 0;
int64_t const* const v16 = &v1[v55];
int64_t const v17 = *v16;
int64_t const v18 = -1;
int64_t const v19 = v18 + v52;
int64_t const v20 = min(v19, v17);
int64_t const v22 = v18 * v52;
int64_t const v23 = max(v22, v20);
int64_t const v24 = v23 + v52;
bool const v26 = v23 < v14;
int64_t const v27 = v26 ? v24 : v23;
bool const v29 = v27 >= v14;
bool const v30 = v27 < v52;
bool const v31 = v29 & v30;
if (!v31) { __trap(); }
bool const v32 = v17 >= v14;
bool const v33 = v17 < v52;
bool const v34 = v32 & v33;
bool const v36 = v17 != v18;
bool const v37 = v34 & v36;
uint64_t const v38 = uint64_t(v12);
float const* const v39 = &v2[v38];
float const v40 = *v39;
float const v41 = 0.0;
float const v42 = v37 ? v40 : v41;
int64_t const v43 = v27 * v52;
int64_t const v44 = v12 + v43;
uint64_t const v45 = uint64_t(v44);
float const* const v46 = &v3[v45];
float const v54 = [&]{
float result;
asm volatile("atom.relaxed.add.f32 %0, [%1], %2;" : "=f"(result) : "l"(v46), "f"(v42) );
return result;
}();}
return;
}
