#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "hnsw.hpp"
#include "brute_force.hpp"

namespace py = pybind11;
using namespace hvs;

PYBIND11_MODULE(hvs_core, m) {
    m.doc() = "Hybrid Vector Search – C++ core (HNSW + BruteForce)";

    py::class_<SearchResult>(m, "SearchResult")
        .def_readonly("id",       &SearchResult::id)
        .def_readonly("distance", &SearchResult::distance)
        .def("__repr__", [](const SearchResult& r) {
            return "<SearchResult id=" + std::to_string(r.id) +
                   " dist=" + std::to_string(r.distance) + ">";
        });

    py::class_<HNSW>(m, "HNSW")
        .def(py::init<int, int, int, int>(),
             py::arg("M") = 16,
             py::arg("ef_construction") = 200,
             py::arg("max_elements") = 1000000,
             py::arg("dim") = 768)
        .def("insert", [](HNSW& self, int id, py::array_t<float> arr) {
            auto buf = arr.unchecked<1>();
            std::vector<float> vec(buf.data(0), buf.data(0) + buf.shape(0));
            self.insert(id, vec);
        }, py::arg("id"), py::arg("embedding"))
        .def("search", [](const HNSW& self, py::array_t<float> arr, int k, int ef) {
            auto buf = arr.unchecked<1>();
            std::vector<float> vec(buf.data(0), buf.data(0) + buf.shape(0));
            return self.search(vec, k, ef);
        }, py::arg("query"), py::arg("k") = 10, py::arg("ef") = 50)
        .def("batch_insert", [](HNSW& self, py::array_t<int> ids, py::array_t<float> vecs) {
            auto ib = ids.unchecked<1>();
            auto vb = vecs.unchecked<2>();
            int n = ib.shape(0);
            int d = vb.shape(1);
            #pragma omp parallel for schedule(dynamic)
            for (int i = 0; i < n; ++i) {
                std::vector<float> vec(d);
                for (int j = 0; j < d; ++j) vec[j] = vb(i, j);
                self.insert(ib(i), vec);
            }
        }, py::arg("ids"), py::arg("embeddings"))
        .def("save", &HNSW::save, py::arg("path"))
        .def("load", &HNSW::load, py::arg("path"))
        .def("__len__", &HNSW::size)
        .def("__contains__", &HNSW::has);

    py::class_<BruteForce>(m, "BruteForce")
        .def(py::init<int>(), py::arg("dim") = 768)
        .def("insert", [](BruteForce& self, int id, py::array_t<float> arr) {
            auto buf = arr.unchecked<1>();
            std::vector<float> vec(buf.data(0), buf.data(0) + buf.shape(0));
            self.insert(id, vec);
        })
        .def("search", [](const BruteForce& self, py::array_t<float> arr, int k) {
            auto buf = arr.unchecked<1>();
            std::vector<float> vec(buf.data(0), buf.data(0) + buf.shape(0));
            return self.search(vec, k);
        }, py::arg("query"), py::arg("k") = 10)
        .def("__len__", &BruteForce::size);
}
