#pragma once
#include <vector>
#include <algorithm>
#include <cmath>
#include "hnsw.hpp"

namespace hvs {

class BruteForce {
public:
    explicit BruteForce(int dim) : dim_(dim) {}

    void insert(int id, const std::vector<float>& vec) {
        ids_.push_back(id);
        data_.push_back(vec);
    }

    std::vector<SearchResult> search(const std::vector<float>& query, int k) const {
        std::vector<SearchResult> results;
        results.reserve(ids_.size());

        #pragma omp parallel for schedule(dynamic, 64)
        for (int i = 0; i < static_cast<int>(ids_.size()); ++i) {
            float dot = 0.f, na = 0.f, nb = 0.f;
            for (int j = 0; j < dim_; ++j) {
                dot += query[j] * data_[i][j];
                na  += query[j] * query[j];
                nb  += data_[i][j] * data_[i][j];
            }
            float denom = std::sqrt(na) * std::sqrt(nb);
            float dist  = denom < 1e-10f ? 1.f : 1.f - dot / denom;
            #pragma omp critical
            results.push_back({ids_[i], dist});
        }

        std::partial_sort(results.begin(),
                          results.begin() + std::min(k, (int)results.size()),
                          results.end());
        if (static_cast<int>(results.size()) > k) results.resize(k);
        return results;
    }

    int size() const { return static_cast<int>(ids_.size()); }

private:
    int dim_;
    std::vector<int> ids_;
    std::vector<std::vector<float>> data_;
};

} // namespace hvs
