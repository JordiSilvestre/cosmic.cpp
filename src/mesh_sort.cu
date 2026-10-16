// cosmic.cpp: GPU stable key/value radix sort for mesh adjacency (see mesh_sort.h).
#include <cstdint>
#include <cstdio>
#include <vector>

#if defined(__HIP__) || defined(__HIP_PLATFORM_AMD__) || defined(GGML_USE_HIP)
  #include <hip/hip_runtime.h>
  #include <hipcub/hipcub.hpp>
  namespace gpucub = hipcub;
  #define gpuError_t            hipError_t
  #define gpuSuccess            hipSuccess
  #define gpuMalloc             hipMalloc
  #define gpuFree               hipFree
  #define gpuMemcpy             hipMemcpy
  #define gpuMemcpyHostToDevice hipMemcpyHostToDevice
  #define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
  #define gpuGetDeviceCount     hipGetDeviceCount
  #define gpuGetErrorString     hipGetErrorString
#else
  #include <cuda_runtime.h>
  #include <cub/cub.cuh>
  namespace gpucub = cub;
  #define gpuError_t            cudaError_t
  #define gpuSuccess            cudaSuccess
  #define gpuMalloc             cudaMalloc
  #define gpuFree               cudaFree
  #define gpuMemcpy             cudaMemcpy
  #define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
  #define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
  #define gpuGetDeviceCount     cudaGetDeviceCount
  #define gpuGetErrorString     cudaGetErrorString
#endif

namespace trellis {

namespace {
// Device buffers released on every exit path.
struct DeviceBuffers {
    void* p[5] = {nullptr, nullptr, nullptr, nullptr, nullptr};
    ~DeviceBuffers() { for (void* q : p) if (q) gpuFree(q); }
};

bool ok(gpuError_t e, const char* what) {
    if (e == gpuSuccess) return true;
    fprintf(stderr, "[cosmic] mesh sort: %s failed: %s\n", what, gpuGetErrorString(e));
    return false;
}
}  // namespace

bool sort_u64_pairs_gpu(std::vector<uint64_t>& keys, std::vector<int32_t>& values) {
    const size_t n = keys.size();
    if (n == 0) return true;
    if (n != values.size() || n > (size_t)INT32_MAX) return false;
    int devices = 0;
    if (gpuGetDeviceCount(&devices) != gpuSuccess || devices == 0) return false;

    DeviceBuffers b;
    uint64_t *k_in, *k_out; int32_t *v_in, *v_out;
    if (!ok(gpuMalloc(&b.p[0], n * sizeof(uint64_t)), "alloc keys") ||
        !ok(gpuMalloc(&b.p[1], n * sizeof(uint64_t)), "alloc keys") ||
        !ok(gpuMalloc(&b.p[2], n * sizeof(int32_t)), "alloc values") ||
        !ok(gpuMalloc(&b.p[3], n * sizeof(int32_t)), "alloc values")) return false;
    k_in = (uint64_t*)b.p[0]; k_out = (uint64_t*)b.p[1];
    v_in = (int32_t*)b.p[2];  v_out = (int32_t*)b.p[3];
    if (!ok(gpuMemcpy(k_in, keys.data(), n * sizeof(uint64_t), gpuMemcpyHostToDevice), "upload") ||
        !ok(gpuMemcpy(v_in, values.data(), n * sizeof(int32_t), gpuMemcpyHostToDevice), "upload"))
        return false;

    size_t temp = 0;   // radix sort is stable: equal keys keep their input order
    if (!ok(gpucub::DeviceRadixSort::SortPairs(nullptr, temp, k_in, k_out, v_in, v_out, (int)n),
            "sort sizing") ||
        !ok(gpuMalloc(&b.p[4], temp), "alloc temp") ||
        !ok(gpucub::DeviceRadixSort::SortPairs(b.p[4], temp, k_in, k_out, v_in, v_out, (int)n),
            "sort")) return false;

    if (!ok(gpuMemcpy(keys.data(), k_out, n * sizeof(uint64_t), gpuMemcpyDeviceToHost), "download") ||
        !ok(gpuMemcpy(values.data(), v_out, n * sizeof(int32_t), gpuMemcpyDeviceToHost), "download"))
        return false;
    return true;
}

}  // namespace trellis
