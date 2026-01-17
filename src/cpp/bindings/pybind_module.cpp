#include <pybind11/pybind11.h>
#include <stdexcept>
#include <cstdio>
#include <string>
#include <memory>

#include "src/cpp/lsm/lsm_model.h"
#include "src/cpp/lsm/lsm_model_object.h"

namespace py = pybind11;

PYBIND11_MODULE(lsm_cpp, m) {
    m.doc() = "C++ LSM Bermudan put pricer (Eigen + pybind11)";

    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) std::rethrow_exception(p);
        } catch (const std::invalid_argument& e) {
            PyErr_SetString(PyExc_ValueError, e.what());
        } catch (const std::runtime_error& e) {
            PyErr_SetString(PyExc_RuntimeError, e.what());
        }
    });

    py::enum_<lsm::BasisType>(m, "BasisType")
        .value("Monomial", lsm::BasisType::Monomial)
        .value("Laguerre",  lsm::BasisType::Laguerre)
        .export_values();

    py::class_<lsm::LSMConfig>(m, "LSMConfig")
        .def(py::init<>())
        .def_readwrite("steps", &lsm::LSMConfig::steps)
        .def_readwrite("paths", &lsm::LSMConfig::paths)
        .def_readwrite("train_fraction", &lsm::LSMConfig::train_fraction)
        .def_readwrite("seed", &lsm::LSMConfig::seed)
        .def_readwrite("basis_degree", &lsm::LSMConfig::basis_degree)
        .def_readwrite("basis", &lsm::LSMConfig::basis)
        .def_readwrite("antithetic", &lsm::LSMConfig::antithetic)
        .def_readwrite("ridge", &lsm::LSMConfig::ridge)
        .def_readwrite("use_control_variate", &lsm::LSMConfig::use_control_variate);

    py::class_<lsm::LSMPriceResult>(m, "LSMPriceResult")
        .def(py::init<>())
        .def_readwrite("price", &lsm::LSMPriceResult::price)
        .def_readwrite("mc_stderr", &lsm::LSMPriceResult::mc_stderr);

    py::class_<lsm::LSMPriceDeltaResult>(m, "LSMPriceDeltaResult")
        .def(py::init<>())
        .def_readwrite("price", &lsm::LSMPriceDeltaResult::price)
        .def_readwrite("delta", &lsm::LSMPriceDeltaResult::delta)
        .def_readwrite("price_stderr", &lsm::LSMPriceDeltaResult::price_stderr)
        .def_readwrite("delta_stderr", &lsm::LSMPriceDeltaResult::delta_stderr);

    // Stateless APIs (existing)
    m.def(
        "price_bermudan_put_lsm",
        [](double S0, double K, double r, double q, double sigma, double T, const lsm::LSMConfig& cfg) {
            py::gil_scoped_release release;
            return lsm::price_bermudan_put_lsm(S0, K, r, q, sigma, T, cfg);
        },
        py::arg("S0"), py::arg("K"), py::arg("r"), py::arg("q"),
        py::arg("sigma"), py::arg("T"), py::arg("cfg")
    );

    m.def(
        "price_and_delta_bermudan_put_lsm",
        [](double S0, double K, double r, double q, double sigma, double T, double eps_rel, const lsm::LSMConfig& cfg) {
            py::gil_scoped_release release;
            return lsm::price_and_delta_bermudan_put_lsm(S0, K, r, q, sigma, T, eps_rel, cfg);
        },
        py::arg("S0"), py::arg("K"), py::arg("r"), py::arg("q"),
        py::arg("sigma"), py::arg("T"),
        py::arg("eps_rel") = 1e-4,
        py::arg("cfg")
    );

    // Stateful desk-style model
    py::class_<lsm::LSMModel, std::shared_ptr<lsm::LSMModel>>(m, "LSMModel")
        .def(py::init<double,double,double,double,double,const lsm::LSMConfig&>(),
             py::arg("K"), py::arg("r"), py::arg("q"), py::arg("sigma"), py::arg("T"), py::arg("cfg"))
        .def("quote",
             [](const lsm::LSMModel& self, double S0, int start_step, double eps_rel) {
                 py::gil_scoped_release release;
                 return self.quote(S0, start_step, eps_rel);
             },
             py::arg("S0"),
             py::arg("start_step") = 0,
             py::arg("eps_rel") = 1e-4
        )
        .def_property_readonly("steps", &lsm::LSMModel::steps)
        .def_property_readonly("dt", &lsm::LSMModel::dt)
        .def_property_readonly("K", &lsm::LSMModel::K)
        .def_property_readonly("r", &lsm::LSMModel::r)
        .def_property_readonly("q", &lsm::LSMModel::q)
        .def_property_readonly("sigma", &lsm::LSMModel::sigma)
        .def_property_readonly("T", &lsm::LSMModel::T);
}
