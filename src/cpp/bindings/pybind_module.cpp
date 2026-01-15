#include <pybind11/pybind11.h>
#include "src/cpp/lsm/lsm_model.h"

namespace py = pybind11;

PYBIND11_MODULE(lsm_cpp, m) {
    m.doc() = "C++ LSM Bermudan put pricer (Eigen + pybind11)";

    py::enum_<lsm::BasisType>(m, "BasisType")
        .value("Monomial", lsm::BasisType::Monomial)
        .value("Laguerre",  lsm::BasisType::Laguerre)
        .export_values();

    py::class_<lsm::LSMConfig>(m, "LSMConfig")
        .def(py::init<>())
        .def_readwrite("steps", lsm::LSMConfig::steps)
        .def_readwrite("paths", lsm::LSMConfig::paths)
        .def_readwrite("train_fraction", lsm::LSMConfig::train_fraction)
        .def_readwrite("seed", lsm::LSMConfig::seed)
        .def_readwrite("basis_degree", lsm::LSMConfig::basis_degree)
        .def_readwrite("basis", lsm::LSMConfig::basis)
        .def_readwrite("antithetic", lsm::LSMConfig::antithetic)
        .def_readwrite("ridge", lsm::LSMConfig::ridge)
        .def_readwrite("use_control_variate", lsm::LSMConfig::use_control_variate);

    py::class_<lsm::LSMPriceResult>(m, "LSMPriceResult")
        .def(py::init<>())
        .def_readwrite("price", lsm::LSMPriceResult::price)
        .def_readwrite("mc_stderr", lsm::LSMPriceResult::mc_stderr)
        .def("__repr__", [](const lsm::LSMPriceResult& r) {
            char buf[128];
            std::snprintf(buf, sizeof(buf), "<LSMPriceResult price=%.10f stderr=%.3e>", r.price, r.mc_stderr);
            return std::string(buf);
        });

    py::class_<lsm::LSMPriceDeltaResult>(m, "LSMPriceDeltaResult")
        .def(py::init<>())
        .def_readwrite("price", lsm::LSMPriceDeltaResult::price)
        .def_readwrite("delta", lsm::LSMPriceDeltaResult::delta)
        .def_readwrite("price_stderr", lsm::LSMPriceDeltaResult::price_stderr)
        .def_readwrite("delta_stderr", lsm::LSMPriceDeltaResult::delta_stderr)
        .def("__repr__", [](const lsm::LSMPriceDeltaResult& r) {
            char buf[160];
            std::snprintf(buf, sizeof(buf),
                          "<LSMPriceDeltaResult price=%.10f delta=%.10f p_se=%.3e d_se=%.3e>",
                          r.price, r.delta, r.price_stderr, r.delta_stderr);
            return std::string(buf);
        });

    m.def(
        "price_bermudan_put_lsm",
        [](double S0, double K, double r, double q, double sigma, double T, const lsm::LSMConfig& cfg) {
            py::gil_scoped_release release;
            return lsm::price_bermudan_put_lsm(S0, K, r, q, sigma, T, cfg);
        },
        py::arg("S0"), py::arg("K"), py::arg("r"), py::arg("q"), py::arg("sigma"), py::arg("T"), py::arg("cfg")
    );

    m.def(
        "price_and_delta_bermudan_put_lsm",
        [](double S0, double K, double r, double q, double sigma, double T, double eps_rel, const lsm::LSMConfig& cfg) {
            py::gil_scoped_release release;
            return lsm::price_and_delta_bermudan_put_lsm(S0, K, r, q, sigma, T, eps_rel, cfg);
        },
        py::arg("S0"), py::arg("K"), py::arg("r"), py::arg("q"), py::arg("sigma"), py::arg("T"),
        py::arg("eps_rel") = 1e-4,
        py::arg("cfg")
    );
}
