// cosmic.cpp: key/value sort for mesh adjacency (edge key -> face slot).
#pragma once
#include <cstdint>
#include <vector>

namespace trellis {

// Sort (key, value) pairs by key, stable (equal keys keep their input order). Runs a CUB radix
// sort on the GPU when a CUDA/HIP backend was compiled in and a device is available; otherwise
// (or on any GPU error) a CPU stable sort. Replaces std::unordered_map<edge, std::vector<face>>
// adjacency, which made the post-process allocate one vector per edge (tens of millions).
void sort_u64_pairs(std::vector<uint64_t>& keys, std::vector<int32_t>& values);

// Undirected edge key of (a, b): min in the high 32 bits, max in the low 32 bits.
inline uint64_t undirected_edge_key(int32_t a, int32_t b) {
    const uint32_t lo = (uint32_t)(a < b ? a : b), hi = (uint32_t)(a < b ? b : a);
    return ((uint64_t)lo << 32) | hi;
}

// For every face corner slot s = 3*f + j (edge j runs faces[s] -> faces[3*f + (j+1)%3]), the
// face on the other side of that edge when the edge is manifold (exactly two face slots share
// it), else -1. One sort of 3F pairs instead of a hash map of vectors.
std::vector<int32_t> manifold_edge_neighbours(const std::vector<int32_t>& faces);

}  // namespace trellis
