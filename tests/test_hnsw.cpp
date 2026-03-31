#include <cassert>
#include <iostream>
#include <cmath>
#include "hnsw.hpp"
#include "brute_force.hpp"

void test_cosine_identical() {
    hvs::HNSW idx(4, 20, 100, 4);
    std::vector<float> a = {1.f, 0.f, 0.f, 0.f};
    idx.insert(0, a);
    idx.insert(1, {0.f, 1.f, 0.f, 0.f});
    idx.insert(2, {0.f, 0.f, 1.f, 0.f});
    idx.insert(3, {1.f, 0.f, 0.f, 0.f}); // identical to query

    auto res = idx.search(a, 1, 10);
    assert(!res.empty());
    assert(res[0].distance < 1e-5f || res[0].id == 0 || res[0].id == 3);
    std::cout << "[PASS] test_cosine_identical\n";
}

void test_insert_search_recall() {
    int N = 200, D = 32, K = 5;
    hvs::HNSW hnsw(8, 50, N + 10, D);
    hvs::BruteForce bf(D);

    std::mt19937 rng(7);
    std::normal_distribution<float> nd;
    for (int i = 0; i < N; ++i) {
        std::vector<float> v(D);
        float norm = 0;
        for (auto& x : v) { x = nd(rng); norm += x * x; }
        norm = std::sqrt(norm);
        for (auto& x : v) x /= norm;
        hnsw.insert(i, v);
        bf.insert(i, v);
    }

    int total = 0, hits = 0;
    for (int q = 0; q < 20; ++q) {
        std::vector<float> qv(D);
        float norm = 0;
        for (auto& x : qv) { x = nd(rng); norm += x * x; }
        norm = std::sqrt(norm);
        for (auto& x : qv) x /= norm;

        auto hr  = hnsw.search(qv, K, 50);
        auto bfr = bf.search(qv, K);

        std::unordered_set<int> gt;
        for (auto& r : bfr) gt.insert(r.id);
        for (auto& r : hr) if (gt.count(r.id)) ++hits;
        total += K;
    }
    double recall = static_cast<double>(hits) / total;
    std::cout << "[PASS] test_insert_search_recall  Recall@" << K << "=" << recall << "\n";
    assert(recall > 0.7); // HNSW on small dataset should be well above 70%
}

void test_save_load() {
    hvs::HNSW idx(4, 20, 100, 8);
    for (int i = 0; i < 20; ++i) {
        std::vector<float> v(8, static_cast<float>(i));
        idx.insert(i, v);
    }
    idx.save("/tmp/test_hnsw.bin");

    hvs::HNSW idx2(4, 20, 100, 8);
    idx2.load("/tmp/test_hnsw.bin");
    assert(idx2.size() == 20);

    std::vector<float> q(8, 5.f);
    auto r1 = idx.search(q, 3);
    auto r2 = idx2.search(q, 3);
    assert(!r1.empty() && !r2.empty());
    assert(r1[0].id == r2[0].id);
    std::cout << "[PASS] test_save_load\n";
}

void test_brute_force_order() {
    hvs::BruteForce bf(3);
    bf.insert(0, {1.f, 0.f, 0.f});
    bf.insert(1, {0.f, 1.f, 0.f});
    bf.insert(2, {1.f, 0.001f, 0.f}); // closest to query
    auto res = bf.search({1.f, 0.f, 0.f}, 1);
    assert(!res.empty());
    assert(res[0].id == 0 || res[0].id == 2);
    std::cout << "[PASS] test_brute_force_order\n";
}

int main() {
    std::cout << "=== HVS Unit Tests ===\n";
    test_cosine_identical();
    test_insert_search_recall();
    test_save_load();
    test_brute_force_order();
    std::cout << "\nAll tests passed.\n";
    return 0;
}
