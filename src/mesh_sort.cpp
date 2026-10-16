// cosmic.cpp: CPU side of mesh_sort.h (GPU sort in mesh_sort.cu when compiled in).
#include "mesh_sort.h"

#include <algorithm>
#include <cstdio>
#include <numeric>

namespace trellis {

#ifdef TRELLIS_HAVE_GPU_MESHSORT
bool sort_u64_pairs_gpu(std::vector<uint64_t>& keys, std::vector<int32_t>& values);
#endif

static void sort_u64_pairs_cpu(std::vector<uint64_t>& keys, std::vector<int32_t>& values) {
    std::vector<uint32_t> order(keys.size());
    std::iota(order.begin(), order.end(), 0u);
    std::stable_sort(order.begin(), order.end(),
                     [&keys](uint32_t a, uint32_t b) { return keys[a] < keys[b]; });
    std::vector<uint64_t> k(keys.size());
    std::vector<int32_t> v(values.size());
    for (size_t i = 0; i < order.size(); ++i) { k[i] = keys[order[i]]; v[i] = values[order[i]]; }
    keys.swap(k);
    values.swap(v);
}

void sort_u64_pairs(std::vector<uint64_t>& keys, std::vector<int32_t>& values) {
#ifdef TRELLIS_HAVE_GPU_MESHSORT
    if (sort_u64_pairs_gpu(keys, values)) return;
    fprintf(stderr, "[cosmic] GPU sort unavailable; sorting %zu pairs on the CPU\n", keys.size());
#endif
    sort_u64_pairs_cpu(keys, values);
}

std::vector<int32_t> manifold_edge_neighbours(const std::vector<int32_t>& faces) {
    const size_t S = faces.size();               // one slot per face corner = one per edge use
    std::vector<uint64_t> keys(S);
    std::vector<int32_t> slots(S);
    for (size_t f = 0; f + 2 < S; f += 3)
        for (int j = 0; j < 3; ++j) {
            keys[f + j] = undirected_edge_key(faces[f + j], faces[f + (j + 1) % 3]);
            slots[f + j] = (int32_t)(f + j);
        }
    sort_u64_pairs(keys, slots);
    std::vector<int32_t> nbr(S, -1);
    for (size_t i = 0; i < S;) {
        size_t j = i + 1;
        while (j < S && keys[j] == keys[i]) ++j;
        if (j - i == 2) {                        // manifold edge: link the two faces
            nbr[slots[i]] = slots[i + 1] / 3;
            nbr[slots[i + 1]] = slots[i] / 3;
        }
        i = j;
    }
    return nbr;
}

}  // namespace trellis
