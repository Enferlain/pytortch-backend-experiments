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
extern "C" __global__ void __launch_bounds__(256)  cubecl4_0a410df3b1f0b821(int64_t const* const __restrict__ v1, float const* const __restrict__ v2, float const* const __restrict__ v3, info_st const* const __restrict__ v59) {
uint64_t const* const dynamic_meta_v61 = reinterpret_cast<uint64_t const*>(v59 + 1);
const info_st& info_v60 = *v59;int64_t const v62 = info_v60.scalars_i64[0];
uint64_t const v6 = blockIdx.x;
int64_t const v7 = int64_t(v6);
uint64_t const v8 = threadIdx.x;
int64_t const v9 = int64_t(v8);
int64_t const v10 = 256;
int64_t const v11 = v7 * v10;
int64_t const v12 = v11 + v9;
bool const v13 = v12 < v62;
if(v13) {
int64_t const v14 = 7;
int64_t const v15 = v12 / v14;
int64_t const v16 = v12 % v14;
int64_t const v17 = 0;
int64_t const v18 = 1;
bool const v19 = v16 < v17;
int64_t const v20 = v15 - v18;
int64_t const v21 = v19 ? v20 : v15;
uint64_t const v22 = uint64_t(v21);
int64_t const* const v23 = &v1[v22];
int64_t const v24 = *v23;
int64_t const v25 = 16;
int64_t const v26 = min(v25, v24);
int64_t const v27 = -17;
int64_t const v28 = max(v27, v26);
int64_t const v29 = 17;
int64_t const v30 = v28 + v29;
bool const v32 = v28 < v17;
int64_t const v33 = v32 ? v30 : v28;
bool const v36 = v33 >= v17;
bool const v37 = v33 < v29;
bool const v38 = v36 & v37;
if (!v38) { __trap(); }
bool const v39 = v24 >= v17;
bool const v41 = v24 < v29;
bool const v42 = v39 & v41;
int64_t const v43 = -1;
bool const v44 = v24 != v43;
bool const v45 = v42 & v44;
uint64_t const v46 = uint64_t(v12);
float const* const v47 = &v2[v46];
float const v48 = *v47;
float const v49 = 0.0;
float const v50 = v45 ? v48 : v49;
int64_t const v52 = v14 * v33;
int64_t const v67 = v16 ^ v14;
bool const v68 = v67 < v17;
bool const v69 = v16 != v17;
bool const v70 = v69 && v68;
int64_t const v71 = v70 ? v14 : v17;
int64_t const v72 = v16 + v71;
int64_t const v55 = v52 + v72;
uint64_t const v56 = uint64_t(v55);
float const* const v57 = &v3[v56];
float const v73 = [&]{
float result;
asm volatile("atom.relaxed.add.f32 %0, [%1], %2;" : "=f"(result) : "l"(v57), "f"(v50) );
return result;
}();}
return;
}
