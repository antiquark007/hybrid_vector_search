#pragma once
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <random>
#include <cmath>
#include <limits>
#include <mutex>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <omp.h>

namespace hvs {

struct Node {
    int id;
    std::vector<float> embedding;
    std::vector<std::vector<int>> neighbors; // neighbors[layer] = list of neighbor ids
};

struct SearchResult {
    int id;
    float distance;
    bool operator<(const SearchResult& o) const { return distance < o.distance; }
    bool operator>(const SearchResult& o) const { return distance > o.distance; }
};

class HNSW {
public:
    HNSW(int M = 16, int ef_construction = 200, int max_elements = 1000000, int dim = 768)
        : M_(M), M0_(2 * M), ef_construction_(ef_construction),
          max_elements_(max_elements), dim_(dim),
          level_mult_(1.0 / std::log(static_cast<double>(M))),
          rng_(std::random_device{}()), dist_(0.0, 1.0),
          enter_point_(-1), max_level_(-1) {}

    void insert(int id, const std::vector<float>& vec);
    std::vector<SearchResult> search(const std::vector<float>& query, int k, int ef = 50) const;
    void save(const std::string& path) const;
    void load(const std::string& path);
    int size() const { return static_cast<int>(nodes_.size()); }
    bool has(int id) const { return nodes_.count(id) > 0; }

private:
    int M_, M0_, ef_construction_, max_elements_, dim_;
    double level_mult_;
    mutable std::mutex mtx_;
    std::mt19937 rng_;
    std::uniform_real_distribution<double> dist_;
    int enter_point_, max_level_;
    std::unordered_map<int, Node> nodes_;

    int random_level();
    float cosine_distance(const std::vector<float>& a, const std::vector<float>& b) const;
    float l2_distance(const std::vector<float>& a, const std::vector<float>& b) const;

    std::vector<SearchResult> search_layer(
        const std::vector<float>& query, int entry, int ef, int layer) const;
    std::vector<int> select_neighbors(
        const std::vector<SearchResult>& candidates, int M) const;
    void connect_neighbors(int id, const std::vector<int>& neighbors, int layer);
};

// ─── Implementation ───────────────────────────────────────────────────────────

inline float HNSW::cosine_distance(const std::vector<float>& a,
                                    const std::vector<float>& b) const {
    float dot = 0.f, na = 0.f, nb = 0.f;
    const int n = static_cast<int>(a.size());
    #pragma omp simd reduction(+:dot,na,nb)
    for (int i = 0; i < n; ++i) {
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }
    float denom = std::sqrt(na) * std::sqrt(nb);
    return denom < 1e-10f ? 1.f : 1.f - dot / denom;
}

inline float HNSW::l2_distance(const std::vector<float>& a,
                                 const std::vector<float>& b) const {
    float sum = 0.f;
    for (int i = 0; i < static_cast<int>(a.size()); ++i)
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    return std::sqrt(sum);
}

inline int HNSW::random_level() {
    return static_cast<int>(-std::log(dist_(rng_)) * level_mult_);
}

inline std::vector<SearchResult> HNSW::search_layer(
    const std::vector<float>& query, int entry, int ef, int layer) const {
    std::unordered_set<int> visited;
    // min-heap (closest first for result), max-heap for candidates to explore
    using MinH = std::priority_queue<SearchResult, std::vector<SearchResult>, std::greater<SearchResult>>;
    using MaxH = std::priority_queue<SearchResult>;
    MinH candidates;
    MaxH result;

    auto it = nodes_.find(entry);
    if (it == nodes_.end()) return {};
    float d = cosine_distance(query, it->second.embedding);
    candidates.push({entry, d});
    result.push({entry, d});
    visited.insert(entry);

    while (!candidates.empty()) {
        auto cur = candidates.top(); candidates.pop();
        if (cur.distance > result.top().distance) break;
        const Node& node = nodes_.at(cur.id);
        if (layer >= static_cast<int>(node.neighbors.size())) continue;
        for (int nb_id : node.neighbors[layer]) {
            if (visited.count(nb_id)) continue;
            visited.insert(nb_id);
            auto nb_it = nodes_.find(nb_id);
            if (nb_it == nodes_.end()) continue;
            float nd = cosine_distance(query, nb_it->second.embedding);
            if (nd < result.top().distance || static_cast<int>(result.size()) < ef) {
                candidates.push({nb_id, nd});
                result.push({nb_id, nd});
                if (static_cast<int>(result.size()) > ef) result.pop();
            }
        }
    }
    std::vector<SearchResult> res;
    while (!result.empty()) { res.push_back(result.top()); result.pop(); }
    std::sort(res.begin(), res.end());
    return res;
}

inline std::vector<int> HNSW::select_neighbors(
    const std::vector<SearchResult>& candidates, int M) const {
    std::vector<int> out;
    for (auto& c : candidates) {
        if (static_cast<int>(out.size()) >= M) break;
        out.push_back(c.id);
    }
    return out;
}

inline void HNSW::connect_neighbors(int id, const std::vector<int>& neighbors, int layer) {
    Node& node = nodes_.at(id);
    while (static_cast<int>(node.neighbors.size()) <= layer)
        node.neighbors.emplace_back();
    node.neighbors[layer] = neighbors;

    int M_max = (layer == 0) ? M0_ : M_;
    for (int nb_id : neighbors) {
        Node& nb = nodes_.at(nb_id);
        while (static_cast<int>(nb.neighbors.size()) <= layer)
            nb.neighbors.emplace_back();
        nb.neighbors[layer].push_back(id);
        if (static_cast<int>(nb.neighbors[layer].size()) > M_max) {
            // Prune to M_max using distance-based selection
            std::vector<SearchResult> cands;
            for (int c : nb.neighbors[layer])
                cands.push_back({c, cosine_distance(nb.embedding, nodes_.at(c).embedding)});
            std::sort(cands.begin(), cands.end());
            cands.resize(M_max);
            nb.neighbors[layer].clear();
            for (auto& c : cands) nb.neighbors[layer].push_back(c.id);
        }
    }
}

inline void HNSW::insert(int id, const std::vector<float>& vec) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (nodes_.count(id)) return;
    Node node;
    node.id = id;
    node.embedding = vec;
    int level = random_level();
    node.neighbors.resize(level + 1);
    nodes_[id] = std::move(node);

    if (enter_point_ == -1) {
        enter_point_ = id;
        max_level_ = level;
        return;
    }

    int cur_ep = enter_point_;
    for (int l = max_level_; l > level; --l) {
        auto res = search_layer(vec, cur_ep, 1, l);
        if (!res.empty()) cur_ep = res[0].id;
    }

    for (int l = std::min(level, max_level_); l >= 0; --l) {
        int ef = std::max(ef_construction_, M_);
        auto candidates = search_layer(vec, cur_ep, ef, l);
        int M_layer = (l == 0) ? M0_ : M_;
        auto selected = select_neighbors(candidates, M_layer);
        connect_neighbors(id, selected, l);
        if (!candidates.empty()) cur_ep = candidates[0].id;
    }

    if (level > max_level_) {
        max_level_ = level;
        enter_point_ = id;
    }
}

inline std::vector<SearchResult> HNSW::search(
    const std::vector<float>& query, int k, int ef) const {
    if (enter_point_ == -1) return {};
    int cur_ep = enter_point_;
    for (int l = max_level_; l > 0; --l) {
        auto res = search_layer(query, cur_ep, 1, l);
        if (!res.empty()) cur_ep = res[0].id;
    }
    int ef_actual = std::max(ef, k);
    auto res = search_layer(query, cur_ep, ef_actual, 0);
    if (static_cast<int>(res.size()) > k) res.resize(k);
    return res;
}

inline void HNSW::save(const std::string& path) const {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open file for writing: " + path);
    int sz = static_cast<int>(nodes_.size());
    f.write(reinterpret_cast<const char*>(&sz), sizeof(sz));
    f.write(reinterpret_cast<const char*>(&enter_point_), sizeof(enter_point_));
    f.write(reinterpret_cast<const char*>(&max_level_), sizeof(max_level_));
    f.write(reinterpret_cast<const char*>(&dim_), sizeof(dim_));
    f.write(reinterpret_cast<const char*>(&M_), sizeof(M_));
    f.write(reinterpret_cast<const char*>(&M0_), sizeof(M0_));
    for (auto& [id, node] : nodes_) {
        f.write(reinterpret_cast<const char*>(&id), sizeof(id));
        int vsz = static_cast<int>(node.embedding.size());
        f.write(reinterpret_cast<const char*>(&vsz), sizeof(vsz));
        f.write(reinterpret_cast<const char*>(node.embedding.data()), vsz * sizeof(float));
        int nlayers = static_cast<int>(node.neighbors.size());
        f.write(reinterpret_cast<const char*>(&nlayers), sizeof(nlayers));
        for (auto& layer : node.neighbors) {
            int lsz = static_cast<int>(layer.size());
            f.write(reinterpret_cast<const char*>(&lsz), sizeof(lsz));
            f.write(reinterpret_cast<const char*>(layer.data()), lsz * sizeof(int));
        }
    }
}

inline void HNSW::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open file for reading: " + path);
    int sz;
    f.read(reinterpret_cast<char*>(&sz), sizeof(sz));
    f.read(reinterpret_cast<char*>(&enter_point_), sizeof(enter_point_));
    f.read(reinterpret_cast<char*>(&max_level_), sizeof(max_level_));
    f.read(reinterpret_cast<char*>(&dim_), sizeof(dim_));
    f.read(reinterpret_cast<char*>(&M_), sizeof(M_));
    f.read(reinterpret_cast<char*>(&M0_), sizeof(M0_));
    nodes_.clear();
    for (int i = 0; i < sz; ++i) {
        Node node;
        f.read(reinterpret_cast<char*>(&node.id), sizeof(node.id));
        int vsz;
        f.read(reinterpret_cast<char*>(&vsz), sizeof(vsz));
        node.embedding.resize(vsz);
        f.read(reinterpret_cast<char*>(node.embedding.data()), vsz * sizeof(float));
        int nlayers;
        f.read(reinterpret_cast<char*>(&nlayers), sizeof(nlayers));
        node.neighbors.resize(nlayers);
        for (int l = 0; l < nlayers; ++l) {
            int lsz;
            f.read(reinterpret_cast<char*>(&lsz), sizeof(lsz));
            node.neighbors[l].resize(lsz);
            f.read(reinterpret_cast<char*>(node.neighbors[l].data()), lsz * sizeof(int));
        }
        nodes_[node.id] = std::move(node);
    }
}

} // namespace hvs
