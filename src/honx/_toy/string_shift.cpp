//
// Created by honglu on 8/3/23.
//
#include "pybind11_kernel_helpers.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstdint>
#include <string>


void stringShift(void *out, const void** in) {
    const std::int64_t size = *reinterpret_cast<const std::int64_t *>(in[0]);
    const char *data = reinterpret_cast<const char *>(in[1]);
    char *result = reinterpret_cast<char *>(out);
    for (std::int64_t n = 0; n < size; n++) {
        result[n] = data[(n + 1) % size];
    }

}

pybind11::dict Registrations() {
    pybind11::dict dict;
    dict["string_shift"] = helper::EncapsulateFunction(stringShift);
    return dict;
}

PYBIND11_MODULE(_toy, m) {
m.def("get_registrations", &Registrations, "Get registrations");
}