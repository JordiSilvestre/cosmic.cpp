// Median-split AABB tree over triangles with exact closest-point queries.
// Non-owning: the caller's vertex/index arrays must outlive the tree. Used by
// the texture bake (per-texel snap to the original surface, mirroring the
// reference's cuBVH unsigned_distance) and by the narrow-band remesh (UDF
// evaluation).
#pragma once
#include <cstdint>
#include <memory>
#include <vector>

namespace trellis {

class TriBvh {
public:
    struct Hit {
        float dist2 = 1e30f;
        int32_t face = -1;
        float point[3] = {0, 0, 0};
    };

    static TriBvh build(const float* verts, int64_t V, const int32_t* faces, int64_t F);

    Hit closest(const float p[3], float max_dist = 1e30f) const;

    // cosmic.cpp: closest() for n points [3n] at once, same max_dist for all. Writes dist2[n]
    // and face[n] (face -1 = nothing within max_dist), and point[3n] when non-null. Runs on the
    // GPU when compiled with CUDA (the tree is uploaded once and reused by later batches),
    // matching closest() bit for bit; otherwise on CPU threads.
    void closest_batch(const float* points, int64_t n, float max_dist,
                       float* dist2, int32_t* face, float* point = nullptr) const;

    using PointFilter = bool (*)(void *, int32_t, const float[3]);
    Hit closest_filtered(const float p[3], PointFilter accept, void *context, float max_dist) const;
    using BoxVisitor = bool (*)(void *, int32_t);
    bool visit_overlapping(const float bmin[3], const float bmax[3], BoxVisitor visit, void *context) const;
    bool empty() const { return nodes_.empty(); }

    // Layout shared with tri_bvh_gpu.cu (32 bytes, uploaded as raw bytes).
    struct Node {
        float bmin[3], bmax[3];
        int32_t left;    // internal: index of left child (right = left+1); leaf: first prim index
        int32_t count;   // 0 for internal nodes; >0 = leaf primitive count
    };

private:
    std::vector<Node> nodes_;
    std::vector<int32_t> prim_;
    const float* verts_ = nullptr;
    const int32_t* faces_ = nullptr;
    int64_t nverts_ = 0, nfaces_ = 0;
    mutable std::shared_ptr<void> gpu_;   // device copy of the tree (tri_bvh_gpu.cu), made lazily
};

}  // namespace trellis
