#include <stdint.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <simsense/core.h>

namespace py = pybind11;

PYBIND11_MODULE(pysimsense, m) {
    auto PySimsense = py::class_<simsense::DepthSensorEngine>(m, "DepthSensorEngine");

    PySimsense.def(
        py::init<
            uint32_t, uint32_t, float, float, float, float, uint64_t,
            float, float, float, float, bool, uint8_t, uint8_t, uint32_t,
            uint8_t, uint8_t, uint8_t, uint8_t, uint8_t, uint8_t, uint8_t,
            py::array_t<float>, py::array_t<float>, py::array_t<float>, py::array_t<float>
        >()
    );
    PySimsense.def(
        py::init<
            uint32_t, uint32_t, uint32_t, uint32_t, float, float, float,
            float, uint64_t, float, float, float, float,  bool, uint8_t,
            uint8_t, uint32_t, uint8_t, uint8_t, uint8_t, uint8_t, uint8_t,
            uint8_t, uint8_t, py::array_t<float>, py::array_t<float>,
            py::array_t<float>, py::array_t<float>, py::array_t<float>,
            py::array_t<float>, py::array_t<float>, float, float, float, bool
        >()
    );
    PySimsense.def("compute", &simsense::DepthSensorEngine::compute);
    PySimsense.def("_set_ir_noise_parameters", &simsense::DepthSensorEngine::setInfraredNoiseParameters);
    PySimsense.def("_set_census_window_size", &simsense::DepthSensorEngine::setCensusWindowSize);
    PySimsense.def("_set_matching_block_size", &simsense::DepthSensorEngine::setMatchingBlockSize);
    PySimsense.def("_set_penalties", &simsense::DepthSensorEngine::setPenalties);
    PySimsense.def("_set_uniqueness_ratio", &simsense::DepthSensorEngine::setUniquenessRatio);
    PySimsense.def("_set_lr_max_diff", &simsense::DepthSensorEngine::setLrMaxDiff);
}
