// cosmic.cpp: regression test for the sort-based mesh adjacency (mesh_sort.h).
// Loads the mesh of a POST dump (trellis_cli TRELLIS_DUMP_POST) and checks that clean_mesh and
// drop_small_components give bit-identical faces to the previous unordered_map implementations,
// which are reproduced below as the reference.
//   cosmic-test-mesh-sort <dump.post>
#include "mesh_sort.h"
#include "tri_bvh.h"
#include "uv_bake.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <unordered_map>
#include <vector>

namespace {

double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Reference: clean_mesh before cosmic.cpp (unordered_map<edge, vector<face>>).
void clean_mesh_reference(std::vector<int32_t>& faces) {
    std::vector<int32_t> kept; kept.reserve(faces.size());
    for (size_t f = 0; f + 2 < faces.size(); f += 3) {
        int a = faces[f], b = faces[f+1], c = faces[f+2];
        if (a != b && b != c && a != c) { kept.push_back(a); kept.push_back(b); kept.push_back(c); }
    }
    faces.swap(kept);
    const int F = (int)(faces.size() / 3);
    if (F == 0) return;
    std::unordered_map<uint64_t, std::vector<int>> ef;
    ef.reserve((size_t)F * 3);
    auto ek = [](int a, int b) -> uint64_t {
        return ((uint64_t)(uint32_t)std::min(a, b) << 32) | (uint32_t)std::max(a, b);
    };
    for (int f = 0; f < F; ++f)
        for (int j = 0; j < 3; ++j) ef[ek(faces[3*f+j], faces[3*f+(j+1)%3])].push_back(f);
    std::vector<char> vis(F, 0), flip(F, 0);
    std::vector<int> st;
    for (int seed = 0; seed < F; ++seed) {
        if (vis[seed]) continue;
        vis[seed] = 1; st.push_back(seed);
        while (!st.empty()) {
            int f = st.back(); st.pop_back();
            int v[3] = { faces[3*f], faces[3*f+1], faces[3*f+2] };
            if (flip[f]) std::swap(v[1], v[2]);
            for (int j = 0; j < 3; ++j) {
                int a = v[j], b = v[(j+1)%3];
                auto it = ef.find(ek(a, b));
                if (it == ef.end() || it->second.size() != 2) continue;
                int g = it->second[0] == f ? it->second[1] : it->second[0];
                if (vis[g]) continue;
                bool same = false;
                for (int k = 0; k < 3; ++k)
                    if (faces[3*g+k] == a && faces[3*g+(k+1)%3] == b) { same = true; break; }
                if (same) flip[g] = 1;
                vis[g] = 1; st.push_back(g);
            }
        }
    }
    for (int f = 0; f < F; ++f) if (flip[f]) std::swap(faces[3*f+1], faces[3*f+2]);
}

// Reference: vertex-connected drop_small_components before cosmic.cpp (unordered_map counts).
int drop_reference(std::vector<float>& verts, std::vector<int32_t>& faces, float frac) {
    const int V = (int)verts.size() / 3;
    const size_t F = faces.size() / 3;
    std::vector<int> par((size_t)V);
    for (int i = 0; i < V; ++i) par[i] = i;
    auto find = [&](int x) { while (par[x] != x) { par[x] = par[par[x]]; x = par[x]; } return x; };
    auto uni  = [&](int a, int b) { int ra = find(a), rb = find(b); if (ra != rb) par[ra] = rb; };
    for (size_t f = 0; f < F; ++f) { uni(faces[3*f], faces[3*f+1]); uni(faces[3*f+1], faces[3*f+2]); }
    std::unordered_map<int,int> fc;
    for (size_t f = 0; f < F; ++f) fc[find(faces[3*f])]++;
    int maxfc = 0;
    for (auto& kv : fc) maxfc = std::max(maxfc, kv.second);
    const int thresh = (int)(frac * maxfc);
    int dropped = 0;
    for (auto& kv : fc) if (kv.second < thresh) ++dropped;
    if (dropped == 0) return 0;
    std::vector<int32_t> kf; kf.reserve(faces.size());
    for (size_t f = 0; f < F; ++f)
        if (fc[find(faces[3*f])] >= thresh)
            for (int k = 0; k < 3; ++k) kf.push_back(faces[3*f + k]);
    std::vector<int> remap((size_t)V, -1);
    std::vector<float> nv; nv.reserve(verts.size());
    for (size_t i = 0; i < kf.size(); ++i) {
        int v = kf[i];
        if (remap[v] < 0) { remap[v] = (int)(nv.size() / 3); nv.insert(nv.end(), &verts[3*v], &verts[3*v] + 3); }
        kf[i] = remap[v];
    }
    verts.swap(nv); faces.swap(kf);
    return dropped;
}

// Reference: weld_vertices before cosmic.cpp (unordered_map<cell, vector<kept vertex>>).
int weld_reference(std::vector<float>& verts, std::vector<int32_t>& faces, float step) {
    const int V = (int)verts.size() / 3;
    const float eps2 = step * step;
    auto ckey = [](int64_t x, int64_t y, int64_t z) {
        return ((uint64_t)(uint32_t)(int32_t)x << 42) ^ ((uint64_t)(uint32_t)(int32_t)y << 21) ^ (uint64_t)(uint32_t)(int32_t)z;
    };
    std::unordered_map<uint64_t, std::vector<int>> cells;
    cells.reserve((size_t)V * 2);
    std::vector<int> remap((size_t)V);
    std::vector<float> nv; nv.reserve(verts.size());
    int keep = 0;
    for (int v = 0; v < V; ++v) {
        const float* p = &verts[3*v];
        const int64_t cx = (int64_t)std::floor(p[0] / step),
                      cy = (int64_t)std::floor(p[1] / step),
                      cz = (int64_t)std::floor(p[2] / step);
        int found = -1;
        for (int dz = -1; dz <= 1 && found < 0; ++dz)
        for (int dy = -1; dy <= 1 && found < 0; ++dy)
        for (int dx = -1; dx <= 1 && found < 0; ++dx) {
            auto it = cells.find(ckey(cx+dx, cy+dy, cz+dz));
            if (it == cells.end()) continue;
            for (int c : it->second) {
                const float ex = nv[3*c]-p[0], ey = nv[3*c+1]-p[1], ez = nv[3*c+2]-p[2];
                if (ex*ex + ey*ey + ez*ez <= eps2) { found = c; break; }
            }
        }
        if (found < 0) {
            found = keep++;
            nv.insert(nv.end(), p, p + 3);
            cells[ckey(cx, cy, cz)].push_back(found);
        }
        remap[v] = found;
    }
    for (auto& f : faces) f = remap[f];
    verts.swap(nv);
    return V - keep;
}

bool load_post(const char* path, std::vector<float>& verts, std::vector<int32_t>& faces) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    int V = 0, F = 0, Mv = 0, res = 0;
    size_t header = fread(&V, 4, 1, f);
    header += fread(&F, 4, 1, f);
    header += fread(&Mv, 4, 1, f);
    header += fread(&res, 4, 1, f);
    bool ok = header == 4 && V > 0 && F > 0;
    if (ok) {
        verts.resize((size_t)V * 3);
        faces.resize((size_t)F * 3);
        ok = fread(verts.data(), 4, verts.size(), f) == verts.size() &&
             fread(faces.data(), 4, faces.size(), f) == faces.size();
    }
    fclose(f);
    return ok;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) { fprintf(stderr, "usage: cosmic-test-mesh-sort <dump.post>\n"); return 2; }
    std::vector<float> verts;
    std::vector<int32_t> faces;
    if (!load_post(argv[1], verts, faces)) { fprintf(stderr, "cannot load %s\n", argv[1]); return 2; }
    printf("mesh: V=%zu F=%zu\n", verts.size() / 3, faces.size() / 3);
    int failures = 0;

    {   // weld at the CLI's step for a res-1024 decode
        const float step = 1.0f / (1024.0f * 8.0f);
        std::vector<float> wv_ref = verts, wv_new = verts;
        std::vector<int32_t> wf_ref = faces, wf_new = faces;
        double tw = now();
        const int wr = weld_reference(wv_ref, wf_ref, step);
        const double w_ref = now() - tw; tw = now();
        const int wn = trellis::weld_vertices(wv_new, wf_new, nullptr, step);
        const double w_new = now() - tw;
        const bool same_weld = wr == wn && wv_ref == wv_new && wf_ref == wf_new;
        printf("weld_vertices: reference %.1fs (%d), new %.1fs (%d) -> %s\n",
               w_ref, wr, w_new, wn, same_weld ? "IDENTICAL" : "DIFFERENT");
        failures += !same_weld;
    }

    std::vector<int32_t> ref = faces, got = faces;
    double t = now();
    clean_mesh_reference(ref);
    const double t_ref = now() - t; t = now();
    trellis::clean_mesh((int)(verts.size() / 3), got);
    const double t_new = now() - t;
    const bool same_clean = ref == got;
    printf("clean_mesh: reference %.1fs, new %.1fs -> %s\n", t_ref, t_new, same_clean ? "IDENTICAL" : "DIFFERENT");
    failures += !same_clean;

    std::vector<float> vr = verts, vg = verts;
    std::vector<int32_t> fr = ref, fg = ref;
    t = now();
    const int dr = drop_reference(vr, fr, 0.02f);
    const double d_ref = now() - t; t = now();
    const int dg = trellis::drop_small_components(vg, fg, 0.02f);
    const double d_new = now() - t;
    const bool same_drop = dr == dg && fr == fg && vr == vg;
    printf("drop_small_components: reference %.1fs (%d), new %.1fs (%d) -> %s\n",
           d_ref, dr, d_new, dg, same_drop ? "IDENTICAL" : "DIFFERENT");
    failures += !same_drop;

    {   // batched BVH closest point (GPU when compiled in) vs the scalar CPU query
        const trellis::TriBvh bvh = trellis::TriBvh::build(verts.data(), (int64_t)verts.size() / 3,
                                                           faces.data(), (int64_t)faces.size() / 3);
        const int64_t nq = 1 << 20;
        std::vector<float> q((size_t)nq * 3);
        uint64_t rng = 0x243F6A8885A308D3ull;
        auto unit = [&rng]() { rng = rng * 6364136223846793005ull + 1442695040888963407ull;
                               return (float)((rng >> 40) & 0xFFFFFF) / 16777216.0f - 0.5f; };
        const int64_t V = (int64_t)verts.size() / 3;
        for (int64_t i = 0; i < nq; ++i) {      // near the surface, like the remesh/bake queries
            const int64_t v = (int64_t)(((uint64_t)i * 2654435761ull) % (uint64_t)V);   // spread over the mesh
            for (int k = 0; k < 3; ++k) q[3*i+k] = verts[3*v+k] + unit() * 0.01f;
        }
        const float max_dist = 0.006f;
        std::vector<float> d2((size_t)nq), pt((size_t)nq * 3);
        std::vector<int32_t> fc((size_t)nq);
        double tb = now();
        bvh.closest_batch(q.data(), nq, max_dist, d2.data(), fc.data(), pt.data());
        const double t_batch = now() - tb; tb = now();
        int64_t mismatch = 0;
        for (int64_t i = 0; i < nq; ++i) {
            const trellis::TriBvh::Hit h = bvh.closest(&q[3*i], max_dist);
            if (h.face != fc[i] || h.dist2 != d2[i] ||
                h.point[0] != pt[3*i] || h.point[1] != pt[3*i+1] || h.point[2] != pt[3*i+2]) ++mismatch;
        }
        const double t_scalar = now() - tb;
        printf("bvh closest_batch: %lld queries, batch %.2fs, scalar single-thread %.2fs, mismatches %lld -> %s\n",
               (long long)nq, t_batch, t_scalar, (long long)mismatch, mismatch ? "DIFFERENT" : "IDENTICAL");
        failures += mismatch != 0;
    }

    printf(failures ? "FAILED\n" : "all checks passed\n");
    return failures ? 1 : 0;
}
