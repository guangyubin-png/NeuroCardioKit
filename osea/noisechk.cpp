/*****************************************************************************
 *  noisechk.cpp - Noise level estimation
 *  Original: Patrick S. Hamilton, E.P. Limited
 *
 *  Estimates low-frequency noise in the isoelectric region between beats.
 *****************************************************************************/

#include <stdlib.h>
#include "qrsdet.h"

#define NB_LENGTH  MS1500
#define NS_LENGTH  MS50

static int NoiseBuffer[NB_LENGTH], NBPtr = 0;
static int NoiseEstimate;

int GetNoiseEstimate(void)
{
    return NoiseEstimate;
}


int NoiseCheck(int datum, int delay, int RR, int beatBegin, int beatEnd)
{
    int ptr, i;
    int ncStart, ncEnd, ncMax, ncMin;
    double noiseIndex;

    NoiseBuffer[NBPtr] = datum;
    if (++NBPtr == NB_LENGTH)
        NBPtr = 0;

    /* Check noise in region between end of previous beat and beginning of this beat */
    ncStart = delay + RR - beatEnd;
    ncEnd = delay + beatBegin;
    if (ncStart > ncEnd + MS250)
        ncStart = ncEnd + MS250;

    if ((delay != 0) && (ncStart < NB_LENGTH) && (ncStart > ncEnd)) {
        ptr = NBPtr - ncStart;
        if (ptr < 0)
            ptr += NB_LENGTH;

        ncMax = ncMin = NoiseBuffer[ptr];
        for (i = 0; i < ncStart - ncEnd; ++i) {
            if (NoiseBuffer[ptr] > ncMax)
                ncMax = NoiseBuffer[ptr];
            else if (NoiseBuffer[ptr] < ncMin)
                ncMin = NoiseBuffer[ptr];
            if (++ptr == NB_LENGTH)
                ptr = 0;
        }

        noiseIndex = (ncMax - ncMin);
        noiseIndex /= (ncStart - ncEnd);
        NoiseEstimate = (int)(noiseIndex * 10);
    }
    else
        NoiseEstimate = 0;

    return NoiseEstimate;
}
