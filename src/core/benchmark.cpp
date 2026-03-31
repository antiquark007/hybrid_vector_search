#include <iostream>
#include <chrono>
#include <random>
#include <iomanip>
#include "hnsw.hpp"
#include "brute_force.hpp"

static std::vector<float> rand_vec(int dim, std::mt19937& rng) {
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> v(dim);
    float norm = 0.f;
    for (auto& x : v) { x = nd(rng); norm += x * x; }
    norm = std::sqrt(norm);
    for (auto& x : v) x /= norm;
    return v;
}

double recall_at_k(const std::vector<hvs::SearchResult>& hnsw_res,
                   const std::vector<hvs::SearchResult>& bf_res, int k) {
    std::unordered_set<int> ground(bf_res.size());
    for (auto& r : bf_res) ground.insert(r.id);
    int hits = 0;
    for (int i = 0; i < std::min(k, (int)hnsw_res.size()); ++i)
        if (ground.count(hnsw_res[i].id)) ++hits;
    return static_cast<double>(hits) / std::min(k, (int)bf_res.size());
}

int main(int argc, char** argv) {
    int N   = (argc > 1) ? std::stoi(argv[1]) : 10000;
    int DIM = (argc > 2) ? std::stoi(argv[2]) : 128;
    int K   = 10;
    int Q   = 100;

    std::mt19937 rng(42);
    std::cout << "Hybrid Vector Search – Benchmark\n";
    std::cout << "N=" << N << " DIM=" << DIM << " K=" << K << " Queries=" << Q << "\n\n";

    // Build index
    hvs::HNSW hnsw(16, 200, N + 1000, DIM);
    hvs::BruteForce bf(DIM);

    std::vector<std::vector<float>> db(N);
    for (int i = 0; i < N; ++i) db[i] = rand_vec(DIM, rng);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) hnsw.insert(i, db[i]);
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) bf.insert(i, db[i]);

    double build_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "HNSW build time: " << build_ms << " ms  ("
              << N / (build_ms / 1000.0) << " inserts/sec)\n\n";

    // Search benchmark
    double total_hnsw_ms = 0, total_bf_ms = 0, total_recall = 0;
    for (int q = 0; q < Q; ++q) {
        auto qv = rand_vec(DIM, rng);

        auto ta = std::chrono::high_resolution_clock::now();
        auto hr  = hnsw.search(qv, K, 100);
        auto tb = std::chrono::high_resolution_clock::now();
        auto bfr = bf.search(qv, K);
        auto tc = std::chrono::high_resolution_clock::now();

        total_hnsw_ms += std::chrono::duration<double, std::milli>(tb - ta).count();
        total_bf_ms   += std::chrono::duration<double, std::milli>(tc - tb).count();
        total_recall  += recall_at_k(hr, bfr, K);
    }

    std::cout << "HNSW  avg latency : " << total_hnsw_ms / Q << " ms/query\n";
    std::cout << "BF    avg latency : " << total_bf_ms   / Q << " ms/query\n";
    std::cout << "Speedup           : " << total_bf_ms / total_hnsw_ms << "x\n";
    std::cout << "Recall@" << K << "         : " << total_recall / Q * 100.0 << "%\n";

    // Save/Load test
    hnsw.save("/tmp/hnsw_bench.bin");
    hvs::HNSW hnsw2(16, 200, N + 1000, DIM);
    hnsw2.load("/tmp/hnsw_bench.bin");
    std::cout << "\nIndex persisted and reloaded: " << hnsw2.size() << " nodes\n";
    return 0;
}
