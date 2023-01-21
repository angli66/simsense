#ifndef SIMSENSE_CORE_H
#define SIMSENSE_CORE_H

#include <stdint.h>
#include <iostream>
#include <driver_types.h>
#include <curand.h>
#include <curand_kernel.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <simsense/config.h>

namespace py = pybind11;

namespace simsense {

// Helper classes and functions
template <class T>
class Mat2d {
public:
    Mat2d(size_t rows, size_t cols, T *data) {
        m_rows = rows;
        m_cols = cols;
        m_data = data;
    }
    T *data() { return m_data; }
    size_t rows() const { return m_rows; }
    size_t cols() const { return m_cols; }
private:
    size_t m_rows, m_cols;
    T *m_data;
};

template <class T>
Mat2d<T> ndarray2Mat2d(py::array_t<T> arr) {
    py::buffer_info buf = arr.request();
    auto ptr = static_cast<T *>(buf.ptr);
    Mat2d<T> new_arr(buf.shape[0], buf.shape[1], ptr);
    return new_arr;
}

template <class T>
py::array_t<T> Mat2d2ndarray(Mat2d<T> arr) {
    // py::str NO_COPY; // Magic to let pybind create array without copying
    // py::array_t<T> new_arr = py::array({arr.rows(), arr.cols()}, arr.data(), NO_COPY);
    py::array_t<T> new_arr = py::array({arr.rows(), arr.cols()}, arr.data());
    return new_arr;
}

// Main sensor class
class DepthSensorEngine {
public:
    __attribute__((visibility("default")))
    DepthSensorEngine(
        uint32_t _rows, uint32_t _cols, float _focalLen, float _baselineLen, float _minDepth, float _maxDepth, uint64_t infraredNoiseSeed,
        float _speckleShape, float _speckleScale, float _gaussianMu, float _gaussianSigma, bool _rectified, uint8_t _censusWidth, uint8_t _censusHeight,
        uint32_t _maxDisp, uint8_t _bfWidth, uint8_t _bfHeight, uint8_t _p1, uint8_t _p2, uint8_t _uniqRatio, uint8_t _lrMaxDiff,
        uint8_t _mfSize, py::array_t<float> map_lx, py::array_t<float> map_ly, py::array_t<float> map_rx, py::array_t<float> map_ry
    );

    __attribute__((visibility("default")))
    DepthSensorEngine(
        uint32_t _rows, uint32_t _cols, uint32_t _rgbRows, uint32_t _rgbCols, float _focalLen, float _baselineLen, float _minDepth,
        float _maxDepth, uint64_t infraredNoiseSeed, float _speckleShape, float _speckleScale, float _gaussianMu, float _gaussianSigma,
        bool _rectified, uint8_t _censusWidth, uint8_t _censusHeight, uint32_t _maxDisp, uint8_t _bfWidth, uint8_t _bfHeight, uint8_t _p1,
        uint8_t _p2, uint8_t _uniqRatio, uint8_t _lrMaxDiff, uint8_t _mfSize, py::array_t<float> map_lx, py::array_t<float> map_ly,
        py::array_t<float> map_rx, py::array_t<float> map_ry, py::array_t<float> _a1, py::array_t<float> _a2, py::array_t<float> _a3,
        float _b1, float _b2, float _b3, bool _dilation
    );

    __attribute__((visibility("default")))
    py::array_t<float> compute(py::array_t<uint8_t> left_ndarray, py::array_t<uint8_t> right_ndarray);

    void setInfraredNoiseParameters(float _speckleShape, float _speckleScale, float _gaussianMu, float _gaussianSigma);
    void setPenalties(uint8_t _p1, uint8_t _p2);
    void setCensusWindowSize(uint8_t _censusWidth, uint8_t _censusHeight);
    void setMatchingBlockSize(uint8_t _bfWidth, uint8_t _bfHeight);
    void setUniquenessRatio(uint8_t _uniqRatio);
    void setLrMaxDiff(uint8_t _lrMaxDiff);

    ~DepthSensorEngine();

protected:
    cudaStream_t stream1, stream2, stream3;
    curandState_t *d_irNoiseStates0, *d_irNoiseStates1;
    float *d_mapLx, *d_mapLy, *d_mapRx, *d_mapRy, *d_a1, *d_a2, *d_a3;
    uint8_t *d_rawim0, *d_rawim1, *d_noisyim0, *d_noisyim1, *d_recim0, *d_recim1;
    uint32_t *d_census0, *d_census1;
    cost_t *d_rawcost, *d_hsum, *d_cost, *d_L0, *d_L1, *d_L2, *d_LAll;
    float *d_leftDisp, *d_filteredDisp, *d_depth, *d_rgbDepth, *h_disp, *h_depth;
    uint16_t *d_rightDisp;
    float speckleShape, speckleScale, gaussianMu, gaussianSigma;
    uint8_t censusWidth, censusHeight, bfWidth, bfHeight, p1, p2, uniqRatio, lrMaxDiff, mfSize;
    uint32_t rows, cols, size, maxDisp, rgbRows, rgbCols, rgbSize;
    float focalLen, baselineLen, minDepth, maxDepth, b1, b2, b3;
    bool rectified, registration, dilation;
};

}

#endif
