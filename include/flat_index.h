// cosmic.cpp: flat open-addressing map from a 64-bit key to a non-negative int32 index.
// Drop-in for the std::unordered_map<uint64_t, int> lookups of the post-process (voxel and grid
// corner maps with millions of entries): one allocation, no per-node heap traffic, linear probing
// with Fibonacci hashing (the high bits of key * golden ratio mix every coordinate field).
#pragma once
#include <cstdint>
#include <vector>

namespace trellis {

class FlatIndex {
public:
    explicit FlatIndex(size_t expected) {
        size_t cap = 16;
        while (cap < expected * 2) { cap <<= 1; ++bits_; }
        keys_.resize(cap);
        values_.assign(cap, -1);
    }

    // Inserts key -> value if absent; returns {stored value, inserted?} like unordered_map::emplace.
    std::pair<int32_t, bool> emplace(uint64_t key, int32_t value) {
        size_t i = slot(key);
        while (values_[i] >= 0) {
            if (keys_[i] == key) return {values_[i], false};
            i = (i + 1) & (keys_.size() - 1);
        }
        keys_[i] = key;
        values_[i] = value;
        return {value, true};
    }

    // Inserts or overwrites key -> value (std::unordered_map operator[] assignment semantics).
    void assign(uint64_t key, int32_t value) {
        size_t i = slot(key);
        while (values_[i] >= 0 && keys_[i] != key) i = (i + 1) & (keys_.size() - 1);
        keys_[i] = key;
        values_[i] = value;
    }

    // Stored value, or -1 when the key is absent.
    int32_t find(uint64_t key) const {
        size_t i = slot(key);
        while (values_[i] >= 0) {
            if (keys_[i] == key) return values_[i];
            i = (i + 1) & (keys_.size() - 1);
        }
        return -1;
    }

private:
    size_t slot(uint64_t key) const { return (size_t)((key * 0x9E3779B97F4A7C15ull) >> (64 - bits_)); }

    int bits_ = 4;
    std::vector<uint64_t> keys_;
    std::vector<int32_t> values_;
};

}  // namespace trellis
