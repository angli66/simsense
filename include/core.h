#ifndef CORE_H
#define CORE_H

#include <stdint.h>
#include <iostream>
#include "config.h"
#include "camera.h"
#include "csct.h"
#include "cost.h"
#include "aggr.h"
#include "wta.h"
#include "lrcheck.h"
#include "filter.h"

#define gpuErrCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) { exit(code); }
   }
}

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

void coreInitWithReg(uint32_t _rows, uint32_t _cols, float _focalLen, float _baselineLen, float _minDepth, float _maxDepth, bool _rectified,
                        uint8_t _censusWidth, uint8_t _censusHeight, uint32_t _maxDisp, uint8_t _bfWidth, uint8_t _bfHeight, uint8_t _p1, uint8_t _p2,
                        uint8_t _uniqRatio, uint8_t _lrMaxDiff, uint8_t _mfSize, Mat2d<float> mapLx, Mat2d<float> mapLy, Mat2d<float> mapRx, Mat2d<float> mapRy,
                        uint32_t _rgbRows, uint32_t _rgbCols, Mat2d<float> a1, Mat2d<float> a2, Mat2d<float> a3, float _b1, float _b2, float _b3, bool _dilation);

void coreInitWithoutReg(uint32_t _rows, uint32_t _cols, float _focalLen, float _baselineLen, float _minDepth, float _maxDepth, bool _rectified,
                        uint8_t _censusWidth, uint8_t _censusHeight, uint32_t _maxDisp, uint8_t _bfWidth, uint8_t _bfHeight, uint8_t _p1, uint8_t _p2,
                        uint8_t _uniqRatio, uint8_t lrMaxDiff, uint8_t _mfSize, Mat2d<float> mapLx, Mat2d<float> mapLy, Mat2d<float> mapRx, Mat2d<float> mapRy);

Mat2d<float> coreCompute(Mat2d<uint8_t> left, Mat2d<uint8_t> right);

void coreClose();

static void freeMemory();

}

#endif
