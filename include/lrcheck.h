#ifndef LRCHECK_H
#define LRCHECK_H

#include "config.h"
#include <cmath>

namespace simsense {

__global__
void lrConsistencyCheck(float *d_leftDisp, const uint16_t *d_rightDisp, const int rows, const int cols, const int lrMaxDiff);

}

#endif
