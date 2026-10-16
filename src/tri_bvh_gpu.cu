// cosmic.cpp: batched closest-point queries over TriBvh on the GPU.
// One thread per query, the same traversal and closest_on_tri arithmetic as tri_bvh.cpp. Built
// with --fmad=false (CMake) so no multiply-add is fused: every float op rounds like the CPU path,
// giving bit-identical distances, faces and points.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>

#include <cuda_runtime.h>

#include "tri_bvh.h"

namespace trellis {
namespace {

struct DeviceBvh {
    TriBvh::Node* nodes = nullptr;
    int32_t* prim = nullptr;
    float* verts = nullptr;
    int32_t* faces = nullptr;
    ~DeviceBvh() { cudaFree(nodes); cudaFree(prim); cudaFree(verts); cudaFree(faces); }
};

bool ok(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return true;
    fprintf(stderr, "[cosmic] bvh gpu: %s failed: %s\n", what, cudaGetErrorString(e));
    return false;
}

__device__ __forceinline__ float dot3(const float* a, const float* b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }

// Ericson 5.1.5, identical to tri_bvh.cpp.
__device__ void closest_on_tri(const float* p, const float* a, const float* b, const float* c, float* out) {
    float ab[3], ac[3], ap[3];
    for (int k = 0; k < 3; ++k) { ab[k] = b[k]-a[k]; ac[k] = c[k]-a[k]; ap[k] = p[k]-a[k]; }
    const float d1 = dot3(ab, ap), d2 = dot3(ac, ap);
    if (d1 <= 0.f && d2 <= 0.f) { for (int k = 0; k < 3; ++k) out[k] = a[k]; return; }
    float bp[3];
    for (int k = 0; k < 3; ++k) bp[k] = p[k]-b[k];
    const float d3 = dot3(ab, bp), d4 = dot3(ac, bp);
    if (d3 >= 0.f && d4 <= d3) { for (int k = 0; k < 3; ++k) out[k] = b[k]; return; }
    const float vc = d1*d4 - d3*d2;
    if (vc <= 0.f && d1 >= 0.f && d3 <= 0.f) {
        const float v = d1 / (d1 - d3);
        for (int k = 0; k < 3; ++k) out[k] = a[k] + v*ab[k];
        return;
    }
    float cp[3];
    for (int k = 0; k < 3; ++k) cp[k] = p[k]-c[k];
    const float d5 = dot3(ab, cp), d6 = dot3(ac, cp);
    if (d6 >= 0.f && d5 <= d6) { for (int k = 0; k < 3; ++k) out[k] = c[k]; return; }
    const float vb = d5*d2 - d1*d6;
    if (vb <= 0.f && d2 >= 0.f && d6 <= 0.f) {
        const float w = d2 / (d2 - d6);
        for (int k = 0; k < 3; ++k) out[k] = a[k] + w*ac[k];
        return;
    }
    const float va = d3*d6 - d5*d4;
    if (va <= 0.f && (d4 - d3) >= 0.f && (d5 - d6) >= 0.f) {
        const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        for (int k = 0; k < 3; ++k) out[k] = b[k] + w*(c[k]-b[k]);
        return;
    }
    const float denom = 1.f / (va + vb + vc);
    const float v = vb * denom, w = vc * denom;
    for (int k = 0; k < 3; ++k) out[k] = a[k] + ab[k]*v + ac[k]*w;
}

__device__ __forceinline__ float box_dist2(const float* p, const float* bmin, const float* bmax) {
    float d2 = 0.f;
    for (int k = 0; k < 3; ++k) {
        const float v = p[k] < bmin[k] ? bmin[k] - p[k] : (p[k] > bmax[k] ? p[k] - bmax[k] : 0.f);
        d2 += v * v;
    }
    return d2;
}

__global__ void closest_kernel(const TriBvh::Node* __restrict__ nodes, const int32_t* __restrict__ prim,
                               const float* __restrict__ verts, const int32_t* __restrict__ faces,
                               const float* __restrict__ pts, int n, float max_d2,
                               float* __restrict__ out_d2, int32_t* __restrict__ out_face,
                               float* __restrict__ out_point) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= n) return;
    const float p[3] = {pts[3*q], pts[3*q+1], pts[3*q+2]};
    float best = max_d2, bp[3] = {0.f, 0.f, 0.f};
    int32_t best_face = -1;
    struct Entry { float d2; int32_t node; };
    Entry stack[64];
    int sp = 0;
    stack[sp++] = {box_dist2(p, nodes[0].bmin, nodes[0].bmax), 0};
    while (sp > 0) {
        const Entry e = stack[--sp];
        if (e.d2 >= best) continue;
        const TriBvh::Node n0 = nodes[e.node];
        if (n0.count > 0) {
            for (int32_t i = 0; i < n0.count; ++i) {
                const int32_t f = prim[n0.left + i];
                float c[3];
                closest_on_tri(p, &verts[3*faces[3*f]], &verts[3*faces[3*f+1]], &verts[3*faces[3*f+2]], c);
                const float dx = c[0]-p[0], dy = c[1]-p[1], dz = c[2]-p[2];
                const float d2 = dx*dx + dy*dy + dz*dz;
                if (d2 < best) { best = d2; best_face = f; bp[0] = c[0]; bp[1] = c[1]; bp[2] = c[2]; }
            }
            continue;
        }
        const int32_t l = n0.left, r = n0.left + 1;
        float dl = box_dist2(p, nodes[l].bmin, nodes[l].bmax);
        float dr = box_dist2(p, nodes[r].bmin, nodes[r].bmax);
        int32_t first = l, second = r;
        if (dr < dl) { const float t = dl; dl = dr; dr = t; first = r; second = l; }
        if (sp + 2 <= 64) {
            if (dr < best) stack[sp++] = {dr, second};
            if (dl < best) stack[sp++] = {dl, first};
        }
    }
    out_d2[q] = best;
    out_face[q] = best_face;
    if (out_point) { out_point[3*q] = bp[0]; out_point[3*q+1] = bp[1]; out_point[3*q+2] = bp[2]; }
}

}  // namespace

// Uploads the tree; returns nullptr (CPU fallback) when no device is usable.
std::shared_ptr<void> bvh_gpu_upload(const TriBvh::Node* nodes, int64_t nnodes, const int32_t* prim,
                                     int64_t nprim, const float* verts, int64_t V,
                                     const int32_t* faces, int64_t F) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return nullptr;
    auto d = std::make_shared<DeviceBvh>();
    if (!ok(cudaMalloc(&d->nodes, nnodes * sizeof(TriBvh::Node)), "alloc nodes") ||
        !ok(cudaMalloc(&d->prim, nprim * sizeof(int32_t)), "alloc prims") ||
        !ok(cudaMalloc(&d->verts, V * 3 * sizeof(float)), "alloc verts") ||
        !ok(cudaMalloc(&d->faces, F * 3 * sizeof(int32_t)), "alloc faces") ||
        !ok(cudaMemcpy(d->nodes, nodes, nnodes * sizeof(TriBvh::Node), cudaMemcpyHostToDevice), "upload") ||
        !ok(cudaMemcpy(d->prim, prim, nprim * sizeof(int32_t), cudaMemcpyHostToDevice), "upload") ||
        !ok(cudaMemcpy(d->verts, verts, V * 3 * sizeof(float), cudaMemcpyHostToDevice), "upload") ||
        !ok(cudaMemcpy(d->faces, faces, F * 3 * sizeof(int32_t), cudaMemcpyHostToDevice), "upload"))
        return nullptr;
    return d;
}

bool bvh_gpu_closest(const std::shared_ptr<void>& handle, const float* points, int64_t n, float max_dist,
                     float* dist2, int32_t* face, float* point) {
    const DeviceBvh* d = static_cast<const DeviceBvh*>(handle.get());
    if (!d) return false;
    const int64_t kChunk = 4 << 20;   // 4M queries: ~100 MB of device buffers per chunk
    const int64_t cap = n < kChunk ? n : kChunk;
    float *dp = nullptr, *dd = nullptr, *dpt = nullptr; int32_t* df = nullptr;
    struct Free { void* p[4]; ~Free() { for (void* q : p) cudaFree(q); } } guard{};
    if (!ok(cudaMalloc(&dp, cap * 3 * sizeof(float)), "alloc queries") ||
        !ok(cudaMalloc(&dd, cap * sizeof(float)), "alloc results") ||
        !ok(cudaMalloc(&df, cap * sizeof(int32_t)), "alloc results") ||
        (point && !ok(cudaMalloc(&dpt, cap * 3 * sizeof(float)), "alloc results")))
        return false;
    guard.p[0] = dp; guard.p[1] = dd; guard.p[2] = df; guard.p[3] = dpt;
    const float max_d2 = max_dist * max_dist;
    for (int64_t b = 0; b < n; b += kChunk) {
        const int m = (int)((n - b) < kChunk ? (n - b) : kChunk);
        if (!ok(cudaMemcpy(dp, points + 3*b, m * 3 * sizeof(float), cudaMemcpyHostToDevice), "upload queries"))
            return false;
        closest_kernel<<<(m + 127) / 128, 128>>>(d->nodes, d->prim, d->verts, d->faces, dp, m, max_d2,
                                                 dd, df, point ? dpt : nullptr);
        if (!ok(cudaGetLastError(), "kernel launch") ||
            !ok(cudaMemcpy(dist2 + b, dd, m * sizeof(float), cudaMemcpyDeviceToHost), "download") ||
            !ok(cudaMemcpy(face + b, df, m * sizeof(int32_t), cudaMemcpyDeviceToHost), "download") ||
            (point && !ok(cudaMemcpy(point + 3*b, dpt, m * 3 * sizeof(float), cudaMemcpyDeviceToHost), "download")))
            return false;
    }
    return true;
}

}  // namespace trellis
