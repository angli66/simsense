#ifndef SIMSENSE_UTIL_H
#define SIMSENSE_UTIL_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <dlpack/dlpack.h>
#include <simsense/config.h>

namespace py = pybind11;

namespace simsense {

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

template <class T>
void DLMTensorDeleter(DLManagedTensor *self) {
    delete[] self->dl_tensor.shape;
    delete static_cast<std::shared_ptr<T> *>(self->manager_ctx);
    delete self;
}

void DLCapsuleDeleter(PyObject *data) {
    DLManagedTensor *tensor = (DLManagedTensor *)PyCapsule_GetPointer(data, "dltensor");
    if (tensor) {
        tensor->deleter(const_cast<DLManagedTensor *>(tensor));
    } else {
        PyErr_Clear();
    }
}

__global__
void processDLTensor(const float *src, uint8_t *dst, const int rows, const int cols) {
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= rows * cols) { return; }

    int temp = (int)(src[4*pos] * 255); // RGBA image, step size is 4
    uint8_t val;
    if (temp < 0) {
        val = 0;
    } else if (temp > 255) {
        val = 255;
    } else {
        val = temp;
    }

    dst[pos] = val;
}

}

#endif
