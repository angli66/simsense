#ifndef SIMSENSE_CORE_H
#define SIMSENSE_CORE_H

#include "config.h"
#include "util.h"
#include <cstdio>
#include <driver_types.h>
#include <stdint.h>
#include <vector>

namespace simsense {

// Main sensor class
class DepthSensorEngine {
public:
  DepthSensorEngine(uint32_t _rows, uint32_t _cols, float _focalLen, float _baselineLen,
                    float _minDepth, float _maxDepth, uint64_t infraredNoiseSeed,
                    float _speckleShape, float _speckleScale, float _gaussianMu,
                    float _gaussianSigma, bool _rectified, uint8_t _censusWidth,
                    uint8_t _censusHeight, uint32_t _maxDisp, uint8_t _bfWidth, uint8_t _bfHeight,
                    uint8_t _p1, uint8_t _p2, uint8_t _uniqRatio, uint8_t _lrMaxDiff,
                    uint8_t _mfSize, Mat2d<float> mapLx, Mat2d<float> mapLy, Mat2d<float> mapRx,
                    Mat2d<float> mapRy, float _mainFx, float _mainFy, float _mainSkew,
                    float _mainCx, float _mainCy);

  DepthSensorEngine(uint32_t _rows, uint32_t _cols, uint32_t _rgbRows, uint32_t _rgbCols,
                    float _focalLen, float _baselineLen, float _minDepth, float _maxDepth,
                    uint64_t infraredNoiseSeed, float _speckleShape, float _speckleScale,
                    float _gaussianMu, float _gaussianSigma, bool _rectified, uint8_t _censusWidth,
                    uint8_t _censusHeight, uint32_t _maxDisp, uint8_t _bfWidth, uint8_t _bfHeight,
                    uint8_t _p1, uint8_t _p2, uint8_t _uniqRatio, uint8_t _lrMaxDiff,
                    uint8_t _mfSize, Mat2d<float> mapLx, Mat2d<float> mapLy, Mat2d<float> mapRx,
                    Mat2d<float> mapRy, Mat2d<float> a1, Mat2d<float> a2, Mat2d<float> a3,
                    float _b1, float _b2, float _b3, bool _dilation, float _mainFx, float _mainFy,
                    float _mainSkew, float _mainCx, float _mainCy);

  void compute(Mat2d<uint8_t> left, Mat2d<uint8_t> right);

  // void compute(DLManagedTensor *leftDLMTensor, DLManagedTensor *rightDLMTensor);

  void compute(void *leftCuda, void *rightCuda);

  Mat2d<float> getMat2d();

  // DLManagedTensor *getDLTensor();

  Mat2d<float> getPointCloudMat2d();

  // DLManagedTensor *getPointCloudDLTensor();

  // Mat2d<float> getRgbPointCloudMat2d(DLManagedTensor *rgbaDLMTensor);

  // DLManagedTensor *getRgbPointCloudDLTensor(DLManagedTensor *rgbaDLMTensor);

  int getCudaId();
  void *getCudaPtr();
  void *getPointCloudCudaPtr();
  Mat2d<float> getRgbPointCloudMat2d(void *rgbaCuda);
  void *getRgbPointCloudCudaPtr(void *rgbaCuda);
  uint32_t getInputRows() { return rows; }
  uint32_t getInputCols() { return cols; }
  uint32_t getOutputRows() { return (registration) ? rgbRows : rows; }
  uint32_t getOutputCols() { return (registration) ? rgbCols : cols; }
  void setInfraredNoiseParameters(float _speckleShape, float _speckleScale, float _gaussianMu,
                                  float _gaussianSigma);
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
  std::shared_ptr<float> depthContainer, pcContainer, rgbPcContainer;

  void computeDepth(uint8_t *d_rawim0, uint8_t *d_rawim1);
};

} // namespace simsense

#endif
