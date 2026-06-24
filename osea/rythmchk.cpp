/*****************************************************************************
 *  rythmchk.cpp - Rhythm classification (RR interval analysis)
 *  Original: Patrick S. Hamilton, E.P. Limited
 *
 *  Classifies RR intervals as NORMAL, PVC, or UNKNOWN.
 *****************************************************************************/

#include "qrsdet.h"
#include "ecgcodes.h"
#include <stdlib.h>

/* RR interval types */
#define QQ  0  /* Unknown-Unknown */
#define NN  1  /* Normal-Normal */
#define NV  2  /* Normal-PVC */
#define VN  3  /* PVC-Normal */
#define VV  4  /* PVC-PVC */

#define RBB_LENGTH  8
#define LEARNING    0
#define READY       1
#define BRADY_LIMIT MS1500

/* Local prototypes */
static int RRMatch(int rr0, int rr1);
static int RRShort(int rr0, int rr1);
static int RRShort2(int *rrIntervals, int *rrTypes);

/* Global variables */
static int RRBuffer[RBB_LENGTH], RRTypes[RBB_LENGTH], BeatCount = 0;
static int ClassifyState = LEARNING;
static int BigeminyFlag;

void ResetRhythmChk(void)
{
    BeatCount = 0;
    ClassifyState = LEARNING;
}


int RhythmChk(int rr)
{
    int i, regular = 1;
    int NNEst, NVEst;

    BigeminyFlag = 0;

    if (BeatCount < 4) {
        if (++BeatCount == 4)
            ClassifyState = READY;
    }

    /* Shift RR interval buffer */
    for (i = RBB_LENGTH - 1; i > 0; --i) {
        RRBuffer[i] = RRBuffer[i - 1];
        RRTypes[i] = RRTypes[i - 1];
    }
    RRBuffer[0] = rr;

    if (ClassifyState == LEARNING) {
        RRTypes[0] = QQ;
        return UNKNOWN;
    }

    /* If last interval was unknown */
    if (RRTypes[1] == QQ) {
        for (i = 0, regular = 1; i < 3; ++i)
            if (RRMatch(RRBuffer[i], RRBuffer[i + 1]) == 0)
                regular = 0;

        if (regular == 1) {
            RRTypes[0] = NN;
            return NORMAL;
        }

        /* Check for bigeminy */
        for (i = 0, regular = 1; i < 6; ++i)
            if (RRMatch(RRBuffer[i], RRBuffer[i + 2]) == 0)
                regular = 0;
        for (i = 0; i < 6; ++i)
            if (RRMatch(RRBuffer[i], RRBuffer[i + 1]) != 0)
                regular = 0;

        if (regular == 1) {
            BigeminyFlag = 1;
            if (RRBuffer[0] < RRBuffer[1]) {
                RRTypes[0] = NV;
                RRTypes[1] = VN;
                return PVC;
            }
            else {
                RRTypes[0] = VN;
                RRTypes[1] = NV;
                return NORMAL;
            }
        }

        /* Check for NNVNNNV pattern */
        if (RRShort(RRBuffer[0], RRBuffer[1]) && RRMatch(RRBuffer[1], RRBuffer[2])
            && RRMatch(RRBuffer[2] * 2, RRBuffer[3] + RRBuffer[4]) &&
            RRMatch(RRBuffer[4], RRBuffer[0]) && RRMatch(RRBuffer[5], RRBuffer[2])) {
            RRTypes[0] = NV;
            RRTypes[1] = NN;
            return PVC;
        }
        else {
            RRTypes[0] = QQ;
            return UNKNOWN;
        }
    }

    /* Previous two beats were normal */
    else if (RRTypes[1] == NN) {
        if (RRShort2(RRBuffer, RRTypes)) {
            if (RRBuffer[1] < BRADY_LIMIT) {
                RRTypes[0] = NV;
                return PVC;
            }
            else {
                RRTypes[0] = QQ;
                return UNKNOWN;
            }
        }

        if (RRMatch(RRBuffer[0], RRBuffer[1])) {
            RRTypes[0] = NN;
            return NORMAL;
        }

        if (RRShort(RRBuffer[0], RRBuffer[1])) {
            if (RRMatch(RRBuffer[0], RRBuffer[2]) && (RRTypes[2] == NN)) {
                RRTypes[0] = NN;
                return NORMAL;
            }
            else if (RRBuffer[1] < BRADY_LIMIT) {
                RRTypes[0] = NV;
                return PVC;
            }
            else {
                RRTypes[0] = QQ;
                return UNKNOWN;
            }
        }
        else {
            RRTypes[0] = QQ;
            return NORMAL;
        }
    }

    /* Previous beat was a PVC */
    else if (RRTypes[1] == NV) {
        if (RRShort2(&RRBuffer[1], &RRTypes[1])) {
            if (RRMatch(RRBuffer[0], RRBuffer[1])) {
                RRTypes[0] = NN;
                RRTypes[1] = NN;
                return NORMAL;
            }
            else if (RRBuffer[0] > RRBuffer[1]) {
                RRTypes[0] = VN;
                return NORMAL;
            }
            else {
                RRTypes[0] = QQ;
                return UNKNOWN;
            }
        }

        if (RRMatch(RRBuffer[0], RRBuffer[1])) {
            RRTypes[0] = VV;
            return PVC;
        }

        if (RRBuffer[0] > RRBuffer[1]) {
            RRTypes[0] = VN;
            return NORMAL;
        }
        else {
            RRTypes[0] = QQ;
            return UNKNOWN;
        }
    }

    /* Previous beat followed a PVC or couplet */
    else if (RRTypes[1] == VN) {
        for (i = 2; (RRTypes[i] != NN) && (i < RBB_LENGTH); ++i);

        if (i != RBB_LENGTH) {
            NNEst = RRBuffer[i];
            if (RRMatch(RRBuffer[0], NNEst)) {
                RRTypes[0] = NN;
                return NORMAL;
            }
        }
        else NNEst = 0;

        for (i = 2; (RRTypes[i] != NV) && (i < RBB_LENGTH); ++i);
        if (i != RBB_LENGTH)
            NVEst = RRBuffer[i];
        else NVEst = 0;

        if ((NNEst == 0) && (NVEst != 0))
            NNEst = (RRBuffer[1] + NVEst) >> 1;

        if ((NVEst != 0) &&
            (abs(NNEst - RRBuffer[0]) < abs(NVEst - RRBuffer[0])) &&
            RRMatch(NNEst, RRBuffer[0])) {
            RRTypes[0] = NN;
            return NORMAL;
        }
        else if ((NVEst != 0) &&
            (abs(NNEst - RRBuffer[0]) > abs(NVEst - RRBuffer[0])) &&
            RRMatch(NVEst, RRBuffer[0])) {
            RRTypes[0] = NV;
            return PVC;
        }
        else {
            RRTypes[0] = QQ;
            return UNKNOWN;
        }
    }

    /* Otherwise VV */
    else {
        if (RRMatch(RRBuffer[0], RRBuffer[1])) {
            RRTypes[0] = VV;
            return PVC;
        }
        else {
            if (RRShort(RRBuffer[0], RRBuffer[1])) {
                RRTypes[0] = QQ;
                return UNKNOWN;
            }
            else {
                RRTypes[0] = VN;
                return NORMAL;
            }
        }
    }
}


static int RRMatch(int rr0, int rr1)
{
    if (abs(rr0 - rr1) < ((rr0 + rr1) >> 3))
        return 1;
    return 0;
}


static int RRShort(int rr0, int rr1)
{
    if (rr0 < rr1 - (rr1 >> 2))
        return 1;
    return 0;
}


int IsBigeminy(void)
{
    return BigeminyFlag;
}


static int RRShort2(int *rrIntervals, int *rrTypes)
{
    int rrMean = 0, i, nnCount;

    for (i = 1, nnCount = 0; (i < 7) && (nnCount < 4); ++i)
        if (rrTypes[i] == NN) {
            ++nnCount;
            rrMean += rrIntervals[i];
        }

    if (nnCount != 4)
        return 0;
    rrMean >>= 2;

    for (i = 1; (i < 7) && (nnCount < 4); ++i)
        if (rrTypes[i] == NN)
            if (abs(rrMean - rrIntervals[i]) > (rrMean >> 4))
                return 0;

    if (rrIntervals[0] < (rrMean - (rrMean >> 3)))
        return 1;
    return 0;
}
