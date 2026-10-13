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
extern "C" __global__ void __launch_bounds__(256)  cubecl5_99751ad6858a6062(int64_t const* const __restrict__ v1, float const* const __restrict__ v2, float const* const __restrict__ v3, info_st const* const __restrict__ v65) {
uint64_t const* const dynamic_meta_v67 = reinterpret_cast<uint64_t const*>(v65 + 1);
const info_st& info_v66 = *v65;int64_t const v68 = info_v66.scalars_i64[0];
uint64_t const v6 = blockIdx.x;
int64_t const v7 = int64_t(v6);
uint64_t const v8 = threadIdx.x;
int64_t const v9 = int64_t(v8);
int64_t const v10 = 256;
int64_t const v11 = v7 * v10;
int64_t const v12 = 1;
int64_t const v14 = 0;
int64_t const v18 = v11 + v9;
bool const v19 = v18 < v68;
if(v19) {
int64_t const v20 = 7;
int64_t const v21 = v18 / v20;
int64_t const v22 = v18 % v20;
bool const v25 = v22 < v14;
int64_t const v26 = v21 - v12;
int64_t const v27 = v25 ? v26 : v21;
uint64_t const v28 = uint64_t(v27);
int64_t const* const v29 = &v1[v28];
int64_t const v30 = *v29;
int64_t const v31 = 16;
int64_t const v32 = min(v31, v30);
int64_t const v33 = -17;
int64_t const v34 = max(v33, v32);
int64_t const v35 = 17;
int64_t const v36 = v34 + v35;
bool const v38 = v34 < v14;
int64_t const v39 = v38 ? v36 : v34;
bool const v42 = v39 >= v14;
bool const v43 = v39 < v35;
bool const v44 = v42 & v43;
if (!v44) { __trap(); }
bool const v45 = v30 >= v14;
bool const v47 = v30 < v35;
bool const v48 = v45 & v47;
int64_t const v49 = -1;
bool const v50 = v30 != v49;
bool const v51 = v48 & v50;
uint64_t const v52 = uint64_t(v18);
float const* const v53 = &v2[v52];
float const v54 = *v53;
float const v55 = 0.0;
float const v56 = v51 ? v54 : v55;
int64_t const v58 = v20 * v39;
int64_t const v73 = v22 ^ v20;
bool const v74 = v73 < v14;
bool const v75 = v22 != v14;
bool const v76 = v75 && v74;
int64_t const v77 = v76 ? v20 : v14;
int64_t const v78 = v22 + v77;
int64_t const v61 = v58 + v78;
uint64_t const v62 = uint64_t(v61);
float const* const v63 = &v3[v62];
float const v79 = [&]{
float result;
asm volatile("atom.relaxed.add.f32 %0, [%1], %2;" : "=f"(result) : "l"(v63), "f"(v56) );
return result;
}();}
return;
}
