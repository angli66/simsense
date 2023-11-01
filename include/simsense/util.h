#ifndef SIMSENSE_UTIL_H
#define SIMSENSE_UTIL_H

// #include <dlpack/dlpack.h>
#include "config.h"
#include <memory>

namespace simsense {

template <class T> class Mat2d {
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

// template <class T> void DLMTensorDeleter(DLManagedTensor *self) {
//   delete[] self->dl_tensor.shape;
//   delete static_cast<std::shared_ptr<T> *>(self->manager_ctx);
//   delete self;
// }

} // namespace simsense

#endif
