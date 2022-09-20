#ifndef WTA_H
#define WTA_H

#include "config.h"

namespace simsense {

__global__
void winnerTakesAll(cost_t *d_LAll, float *d_leftDisp, uint16_t *d_rightDisp, const int cols, const int maxDisp, const int uniqRatio);

}

#endif
