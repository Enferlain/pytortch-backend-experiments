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
    int64_t scalars_i64[4];uint64_t static_meta[3];
};
extern "C" __global__ void __launch_bounds__(256)  cubecl4_ca583ba330126438(int64_t const* const __restrict__ v1, float const* const __restrict__ v2, float const* const __restrict__ v3, info_st const* const __restrict__ v29) {
uint64_t const* const dynamic_meta_v31 = reinterpret_cast<uint64_t const*>(v29 + 1);
const info_st& info_v30 = *v29;int64_t const v32 = info_v30.scalars_i64[0];
int64_t const v33 = info_v30.scalars_i64[1];
uint64_t const v8 = blockIdx.x;
int64_t const v9 = int64_t(v8);
uint64_t const v10 = threadIdx.x;
int64_t const v11 = int64_t(v10);
int64_t const v12 = 256;
int64_t const v13 = v9 * v12;
int64_t const v14 = v13 + v11;
bool const v15 = v14 < v32;
if(v15) {
uint64_t const v16 = uint64_t(v14);
int64_t const* const v17 = &v1[v16];
int64_t const v18 = *v17;
int64_t const v19 = 0;
bool const v20 = v18 >= v19;
bool const v21 = v18 < v33;
bool const v22 = v20 & v21;
if (!v22) { __trap(); }
float const* const v24 = &v2[v16];
float const v25 = *v24;
uint64_t const v26 = uint64_t(v18);
float const* const v27 = &v3[v26];
float const v37 = [&]{
float result;
asm volatile("atom.relaxed.add.f32 %0, [%1], %2;" : "=f"(result) : "l"(v27), "f"(v25) );
return result;
}();}
return;
}
