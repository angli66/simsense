#ifndef SIMSENSE_CORE_H
#define SIMSENSE_CORE_H

#include <stdint.h>
#include <driver_types.h>
#include "util.h"
#include "config.h"
#include <cstdio>

#define gpuErrCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) { exit(code); }
   }
}

// namespace py = pybind11;

namespace simsense {

void cudaFreeChecked(void *ptr) { gpuErrCheck(cudaFree(ptr)); }

// Main sensor class
class DepthSensorEngine {
public:
    __attribute__((visibility("default")))
    DepthSensorEngine(
        uint32_t _rows, uint32_t _cols, float _focalLen, float _baselineLen, float _minDepth, float _maxDepth, uint64_t infraredNoiseSeed,
        float _speckleShape, float _speckleScale, float _gaussianMu, float _gaussianSigma, bool _rectified, uint8_t _censusWidth, uint8_t _censusHeight,
        uint32_t _maxDisp, uint8_t _bfWidth, uint8_t _bfHeight, uint8_t _p1, uint8_t _p2, uint8_t _uniqRatio, uint8_t _lrMaxDiff,
        uint8_t _mfSize, Mat2d<float> mapLx, Mat2d<float> mapLy, Mat2d<float> mapRx, Mat2d<float> mapRy,
        float _mainFx, float _mainFy, float _mainSkew, float _mainCx, float _mainCy
    );

    __attribute__((visibility("default")))
    DepthSensorEngine(
        uint32_t _rows, uint32_t _cols, uint32_t _rgbRows, uint32_t _rgbCols, float _focalLen, float _baselineLen, float _minDepth,
        float _maxDepth, uint64_t infraredNoiseSeed, float _speckleShape, float _speckleScale, float _gaussianMu, float _gaussianSigma,
        bool _rectified, uint8_t _censusWidth, uint8_t _censusHeight, uint32_t _maxDisp, uint8_t _bfWidth, uint8_t _bfHeight, uint8_t _p1,
        uint8_t _p2, uint8_t _uniqRatio, uint8_t _lrMaxDiff, uint8_t _mfSize, Mat2d<float> mapLx, Mat2d<float> mapLy,
        Mat2d<float> mapRx, Mat2d<float> mapRy, Mat2d<float> a1, Mat2d<float> a2, Mat2d<float> a3,
        float _b1, float _b2, float _b3, bool _dilation, float _mainFx, float _mainFy, float _mainSkew, float _mainCx, float _mainCy
    );

    __attribute__((visibility("default")))
    void compute(Mat2d<uint8_t> left, Mat2d<uint8_t> right);
    
    __attribute__((visibility("default")))
    void compute(DLManagedTensor *leftDLMTensor, DLManagedTensor *rightDLMTensor);

    __attribute__((visibility("default")))
    Mat2d<float> getMat2d();

    __attribute__((visibility("default")))
    DLManagedTensor* getDLTensor();

    __attribute__((visibility("default")))
    Mat2d<float> getPointCloudMat2d();

    __attribute__((visibility("default")))
    DLManagedTensor* getPointCloudDLTensor();

    __attribute__((visibility("default")))
    Mat2d<float> getRgbPointCloudMat2d(DLManagedTensor *rgbaDLMTensor);

    __attribute__((visibility("default")))
    DLManagedTensor* getRgbPointCloudDLTensor(DLManagedTensor *rgbaDLMTensor);

    void setInfraredNoiseParameters(float _speckleShape, float _speckleScale, float _gaussianMu, float _gaussianSigma);
    void setPenalties(uint8_t _p1, uint8_t _p2);
    void setCensusWindowSize(uint8_t _censusWidth, uint8_t _censusHeight);
    void setMatchingBlockSize(uint8_t _bfWidth, uint8_t _bfHeight);
    void setUniquenessRatio(uint8_t _uniqRatio);
    void setLrMaxDiff(uint8_t _lrMaxDiff);

    ~DepthSensorEngine();

protected:
    cudaStream_t stream1, stream2, stream3;
    void *d_irNoiseStates0, *d_irNoiseStates1;
    float *d_mapLx, *d_mapLy, *d_mapRx, *d_mapRy, *d_a1, *d_a2, *d_a3;
    uint8_t *d_rawim0, *d_rawim1, *d_noisyim0, *d_noisyim1, *d_recim0, *d_recim1;
    uint32_t *d_census0, *d_census1;
    cost_t *d_rawcost, *d_hsum, *d_cost, *d_L0, *d_L1, *d_L2, *d_LAll;
    float *d_leftDisp, *d_filteredDisp, *d_depth, *d_rgbDepth, *h_depth;
    float *d_pc, *d_rgbPc, *h_pc, *h_rgbPc;
    uint16_t *d_rightDisp;
    float speckleShape, speckleScale, gaussianMu, gaussianSigma;
    uint8_t censusWidth, censusHeight, bfWidth, bfHeight, p1, p2, uniqRatio, lrMaxDiff, mfSize;
    uint32_t rows, cols, size, maxDisp, rgbRows, rgbCols, rgbSize;
    float focalLen, baselineLen, minDepth, maxDepth, b1, b2, b3;
    float mainFx, mainFy, mainSkew, mainCx, mainCy;
    bool rectified, registration, dilation, computed;
    // std::shared_ptr<cudaPtrContainer<float>> depthContainer, pcContainer, rgbPcContainer;

    std::shared_ptr<float> depthContainer, pcContainer, rgbPcContainer;

    void computeDepth(uint8_t *d_rawim0, uint8_t *d_rawim1);
};

}

#endif
