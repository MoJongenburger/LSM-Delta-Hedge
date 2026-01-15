#include <pybind11/pybind11.h>
#include "src/cpp/lsm/lsm_model.h"

namespace py = pybind11;

PYBIND11_MODULE(lsm_cpp, m) {
    m.doc() = "C++ LSM Bermudan put pricer (Eigen + pybind11)";

    // ---- Enums ----
    py::enum_<lsm::BasisType>(m, "BasisType")
        .value("Monomial", lsm::BasisType::Monomial)
        .value("Laguerre",  lsm::BasisType::Laguerre)
        .export_values();

    // ---- Config struct ----
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

    // ---- Result struct ----
    py::class_<lsm::LSMPriceResult>(m, "LSMPriceResult")
        .def(py::init<>())
        .def_readwrite("price", lsm::LSMPriceResult::price)
        .def_readwrite("mc_stderr", lsm::LSMPriceResult::mc_stderr)
        .def("__repr__", [](const lsm::LSMPriceResult& r) {
            char buf[128];
            std::snprintf(buf, sizeof(buf),
                          "<LSMPriceResult price=%.10f stderr=%.3e>",
                          r.price, r.mc_stderr);
            return std::string(buf);
        });

    // ---- Functions (GIL released) ----
    m.def(
        "price_bermudan_put_lsm",
        [](double S0, double K, double r, double q, double sigma, double T, const lsm::LSMConfig& cfg) {
            py::gil_scoped_release release;
            return lsm::price_bermudan_put_lsm(S0, K, r, q, sigma, T, cfg);
        },
        py::arg("S0"),
        py::arg("K"),
        py::arg("r"),
        py::arg("q"),
        py::arg("sigma"),
        py::arg("T"),
        py::arg("cfg"),
        R"pbdoc(
Price a Bermudan PUT using Longstaff–Schwartz (LSM) under GBM in risk-neutral measure.

Returns
-------
LSMPriceResult with fields:
- price
- mc_stderr
)pbdoc"
    );
}
