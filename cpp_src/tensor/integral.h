#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
namespace py = pybind11;

std::tuple<py::array_t<double>, py::array_t<double>> compress_h1e_h2e(
    const py::array_t<double> &h1e, const py::array_t<double> &h2e,
    const int sorb);

std::tuple<py::array_t<double>, py::array_t<double>> decompress_h1e_h2e(
    const py::array_t<double> &h1e, const py::array_t<double> &h2e,
    const int sorb);