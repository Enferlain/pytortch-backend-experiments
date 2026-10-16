// Diagnostic CPU delay through CUDA's existing stream host-function API.
// No GPU kernel, CUDA call inside the callback, or runtime ownership change.
#include <chrono>
#include <cstdint>
#include <cuda.h>
#include <new>

struct Delay {
  uint64_t nanoseconds;
  uint64_t *timestamps;
};

static uint64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

static void CUDA_CB idle_callback(void *opaque) {
  auto *delay = static_cast<Delay *>(opaque);
  uint64_t start = now_ns(), end;
  do {
    end = now_ns();
  } while (end - start < delay->nanoseconds);
  delay->timestamps[0] = start;
  delay->timestamps[1] = end;
  delete delay;
}

extern "C" int state_idle_enqueue(CUstream stream, uint64_t nanoseconds,
                                  uint64_t *timestamps) {
  if (!timestamps || nanoseconds > 2500000)
    return CUDA_ERROR_INVALID_VALUE;
  auto *delay = new (std::nothrow) Delay{nanoseconds, timestamps};
  if (!delay)
    return CUDA_ERROR_OUT_OF_MEMORY;
  CUresult status = cuLaunchHostFunc(stream, idle_callback, delay);
  if (status != CUDA_SUCCESS)
    delete delay;
  return status;
}
