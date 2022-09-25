#include <pybind11/pybind11.h>
#include "core.h"

namespace py = pybind11;

using namespace simsense;

PYBIND11_MODULE(pysimsense, m) {
    py::class_<DepthSensorEngine>(m, "DepthSensorEngine")
        .def(py::init<
                uint32_t, uint32_t, float, float, float, float, bool,
                uint8_t, uint8_t, uint32_t, uint8_t, uint8_t, uint8_t, uint8_t,
                uint8_t, uint8_t, uint8_t, py::array_t<float>, py::array_t<float>,
                py::array_t<float>, py::array_t<float>
            >())
        .def(py::init<
                uint32_t, uint32_t, uint32_t, uint32_t, float, float, float,
                float, bool, uint8_t, uint8_t, uint32_t, uint8_t, uint8_t,
                uint8_t, uint8_t, uint8_t, uint8_t, uint8_t, py::array_t<float>,
                py::array_t<float>, py::array_t<float>, py::array_t<float>,
                py::array_t<float>, py::array_t<float>, py::array_t<float>,
                float, float, float, bool
            >())
        .def("compute", &DepthSensorEngine::compute);
}
