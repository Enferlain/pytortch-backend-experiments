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
extern "C" __global__ void __launch_bounds__(256)  cubecl4_96d94ff198985051(int64_t const* const __restrict__ v1, float const* const __restrict__ v2, float const* const __restrict__ v3, info_st const* const __restrict__ v57) {
uint64_t const* const dynamic_meta_v59 = reinterpret_cast<uint64_t const*>(v57 + 1);
const info_st& info_v58 = *v57;int64_t const v60 = info_v58.scalars_i64[0];
int64_t const v61 = info_v58.scalars_i64[1];
int64_t const v62 = info_v58.scalars_i64[2];
uint64_t const v8 = blockIdx.x;
int64_t const v9 = int64_t(v8);
uint64_t const v10 = threadIdx.x;
int64_t const v11 = int64_t(v10);
int64_t const v12 = 256;
int64_t const v13 = v9 * v12;
int64_t const v14 = v13 + v11;
bool const v15 = v14 < v60;
if(v15) {
int64_t const v16 = v14 / v61;
int64_t const v17 = v14 % v61;
int64_t const v18 = 0;
int64_t const v19 = 1;
bool const v20 = v17 < v18;
int64_t const v21 = v16 - v19;
int64_t const v22 = v20 ? v21 : v16;
uint64_t const v23 = uint64_t(v22);
int64_t const* const v24 = &v1[v23];
int64_t const v25 = *v24;
int64_t const v26 = -1;
int64_t const v27 = v26 + v62;
int64_t const v28 = min(v27, v25);
int64_t const v30 = v26 * v62;
int64_t const v31 = max(v30, v28);
int64_t const v32 = v31 + v62;
bool const v34 = v31 < v18;
int64_t const v35 = v34 ? v32 : v31;
bool const v37 = v35 >= v18;
bool const v38 = v35 < v62;
bool const v39 = v37 & v38;
if (!v39) { __trap(); }
bool const v40 = v25 >= v18;
bool const v41 = v25 < v62;
bool const v42 = v40 & v41;
bool const v44 = v25 != v26;
bool const v45 = v42 & v44;
uint64_t const v46 = uint64_t(v14);
float const* const v47 = &v2[v46];
float const v48 = *v47;
float const v49 = 0.0;
float const v50 = v45 ? v48 : v49;
int64_t const v51 = v35 * v61;
int64_t const v67 = v17 ^ v61;
bool const v68 = v67 < v18;
bool const v69 = v17 != v18;
bool const v70 = v69 && v68;
int64_t const v71 = v70 ? v61 : v18;
int64_t const v72 = v17 + v71;
int64_t const v53 = v51 + v72;
uint64_t const v54 = uint64_t(v53);
float const* const v55 = &v3[v54];
float const v73 = [&]{
float result;
asm volatile("atom.relaxed.add.f32 %0, [%1], %2;" : "=f"(result) : "l"(v55), "f"(v50) );
return result;
}();}
return;
}
