/* Reusable CUDA Driver launcher. This file contains no device kernel. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <stdatomic.h>

static _Atomic uint64_t launcher_calls = 0, kernel_launches = 0;
static _Atomic uint64_t reduction_launches = 0, reduction_shared_bytes = 0;
static _Atomic uint64_t stage4_launches = 0, stage4_five_read_launches = 0;
uint64_t stage4_kernel_launches(void) { return atomic_load(&stage4_launches); }
uint64_t stage4_five_input_launches(void) { return atomic_load(&stage4_five_read_launches); }
uint64_t stage0_launcher_calls(void) { return atomic_load(&launcher_calls); }
uint64_t stage0_kernel_launches(void) { return atomic_load(&kernel_launches); }
uint64_t stage3_kernel_launches(void) { return atomic_load(&reduction_launches); }
uint64_t stage3_total_shared_bytes(void) { return atomic_load(&reduction_shared_bytes); }

typedef struct {
    void *library, *module, *function, *context;
    int device;
    int (*get_context)(void **);
    int (*get_device)(int *);
    int (*launch)(void *, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
                  unsigned, void *, void **, void **);
    int (*error_string)(int, const char **);
} Kernel;

static int fail(Kernel *k, int code, const char *where, char *error, size_t cap) {
    const char *detail = "CUDA driver failure";
    if (k && k->error_string) k->error_string(code, &detail);
    if (!detail) detail = "unrecognized CUDA error";
    snprintf(error, cap, "%s: %d (%s)", where, code, detail);
    return code ? code : -1;
}

/* Caller must have initialized PyTorch CUDA and selected the intended device. */
int stage0_load(const char *ptx, const char *entry, void **result, char *error, size_t cap) {
    Kernel *k = calloc(1, sizeof(*k));
    if (!k) { snprintf(error, cap, "allocation failed"); return -1; }
    k->library = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!k->library) { snprintf(error, cap, "%s", dlerror()); free(k); return -1; }
#define RESOLVE(dst, name) do { *(void **)(&(dst)) = dlsym(k->library, name); \
    if (!(dst)) { snprintf(error, cap, "missing %s", name); dlclose(k->library); free(k); return -1; } } while (0)
    int (*load)(void **, const void *);
    int (*get_function)(void **, void *, const char *);
    int (*unload)(void *);
    RESOLVE(k->get_context, "cuCtxGetCurrent");
    RESOLVE(k->get_device, "cuCtxGetDevice");
    RESOLVE(k->launch, "cuLaunchKernel");
    RESOLVE(k->error_string, "cuGetErrorString");
    RESOLVE(load, "cuModuleLoadData");
    RESOLVE(get_function, "cuModuleGetFunction");
    RESOLVE(unload, "cuModuleUnload");
    int rc = k->get_context(&k->context);
    if (!rc && !k->context) { snprintf(error, cap, "no current CUDA context"); rc = -1; }
    else if (rc) fail(k, rc, "cuCtxGetCurrent", error, cap);
    if (!rc && (rc = k->get_device(&k->device))) fail(k, rc, "cuCtxGetDevice", error, cap);
    if (!rc && (rc = load(&k->module, ptx))) fail(k, rc, "cuModuleLoadData", error, cap);
    if (!rc && (rc = get_function(&k->function, k->module, entry))) fail(k, rc, "cuModuleGetFunction", error, cap);
    if (rc) {
        if (k->module) unload(k->module);
        dlclose(k->library); free(k); return rc;
    }
    /* Deliberately process-lifetime modules: asynchronous launches and graphs can
       outlive Python wrappers. No automatic unload or stream ownership. */
    *result = k;
    return 0;
}

static int launch_args(void *handle, void **args, uint64_t n,
                       unsigned shared, void *stream, char *error, size_t cap) {
    atomic_fetch_add(&launcher_calls, 1);
    Kernel *k = handle;
    if (!k) { snprintf(error, cap, "null kernel handle"); return -1; }
    void *context = NULL;
    int rc = k->get_context(&context);
    if (rc) return fail(k, rc, "cuCtxGetCurrent", error, cap);
    if (context != k->context) { snprintf(error, cap, "kernel context differs from current context"); return -1; }
    if (n == 0) return 0;
    /* CUDA driver kernelParams are HOST addresses of argument values. */
    if (n > (uint64_t)2147483647 * 256) { snprintf(error, cap, "grid exceeds CUDA x limit"); return -1; }
    unsigned grid = (unsigned)((n + 255) / 256);
    rc = k->launch(k->function, grid, 1, 1, 256, 1, 1, shared, stream, args, NULL);
    if (!rc) atomic_fetch_add(&kernel_launches, 1);
    return rc ? fail(k, rc, "cuLaunchKernel", error, cap) : 0;
}

/* Preserve the known-good Stage 0 entry; same common enqueue code. */
int stage0_launch(void *handle, uint64_t x, uint64_t y, uint64_t info,
                  uint64_t n, void *stream, char *error, size_t cap) {
    void *args[] = { &x, &y, &info };
    return launch_args(handle, args, n, 0, stream, error, cap);
}

/* Stage 1 only needs 2 or 3 tensor buffers plus the InfoBuilder metadata. */
int stage1_launch(void *handle, const uint64_t *pointers, size_t count,
                  uint64_t n, void *stream, char *error, size_t cap) {
    if (!pointers || count < 3 || count > 4) {
        snprintf(error, cap, "unsupported pointer argument count"); return -1;
    }
    void *args[4];
    for (size_t i = 0; i < count; ++i) args[i] = (void *)&pointers[i];
    return launch_args(handle, args, n, 0, stream, error, cap);
}

/* Stage 2 binary backward can also consume a live previous gradient buffer. */
int stage2_launch(void *handle, const uint64_t *pointers, size_t count,
                  uint64_t n, void *stream, char *error, size_t cap) {
    if (!pointers || count < 2 || count > 6) {
        snprintf(error, cap, "unsupported Stage 2 pointer argument count"); return -1;
    }
    void *args[6];
    for (size_t i = 0; i < count; ++i) args[i] = (void *)&pointers[i];
    return launch_args(handle, args, n, 0, stream, error, cap);
}

/* Stage 3 retains the same pointer ABI/context/stream/lifetime contract. The
   checked plan has 256 units and exactly 1024 bytes of shared FP32 scratch. */
int stage3_launch(void *handle, const uint64_t *pointers, size_t count,
                  uint64_t grid, unsigned shared, void *stream, char *error, size_t cap) {
    if (!pointers || count < 3 || count > 6 || shared != 1024 || grid > 2147483647) {
        snprintf(error, cap, "unsupported Stage 3 launch plan"); return -1;
    }
    void *args[6];
    for (size_t i = 0; i < count; ++i) args[i] = (void *)&pointers[i];
    int rc = launch_args(handle, args, grid * 256, shared, stream, error, cap);
    if (!rc && grid) {
        atomic_fetch_add(&reduction_launches, 1);
        atomic_fetch_add(&reduction_shared_bytes, shared);
    }
    return rc;
}

/* Read-only structural evidence from the driver's compiled function. */
int stage0_kernel_attributes(void *handle, int *values, char *error, size_t cap) {
    Kernel *k = handle;
    if (!k || !values) { snprintf(error, cap, "null attribute argument"); return -1; }
    int (*get_attribute)(int *, int, void *);
    *(void **)(&get_attribute) = dlsym(k->library, "cuFuncGetAttribute");
    if (!get_attribute) { snprintf(error, cap, "missing cuFuncGetAttribute"); return -1; }
    const int attributes[] = {4, 1, 3}; /* NUM_REGS, SHARED_SIZE_BYTES, LOCAL_SIZE_BYTES */
    for (size_t i = 0; i < 3; ++i) {
        int rc = get_attribute(&values[i], attributes[i], k->function);
        if (rc) return fail(k, rc, "cuFuncGetAttribute", error, cap);
    }
    return 0;
}

/* Stage4's observed norm/dropout backward requires exactly one more read. */
int stage4_launch(void *handle, const uint64_t *pointers, size_t count,
                  uint64_t n, void *stream, char *error, size_t cap) {
    if (!pointers || count < 2 || count > 7) {
        snprintf(error, cap, "unsupported Stage4 pointer argument count"); return -1;
    }
    void *args[7];
    for (size_t i=0; i<count; ++i) args[i]=(void *)&pointers[i];
    int rc = launch_args(handle,args,n,0,stream,error,cap);
    if (!rc && n) { atomic_fetch_add(&stage4_launches, 1); if (count == 7) atomic_fetch_add(&stage4_five_read_launches, 1); }
    return rc;
}
