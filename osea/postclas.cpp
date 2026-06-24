/*****************************************************************************
 *  postclas.cpp - Post classifier
 *  Original: Patrick S. Hamilton, E.P. Limited
 *
 *  Classifies beats based on preceding and following beats and RR intervals.
 *****************************************************************************/

#include "bdac.h"
#include "ecgcodes.h"

/* External prototypes */
double DomCompare(int newType, int domType);
int GetBeatTypeCount(int type);

/* Records of post classifications */
static int PostClass[MAXTYPES][8], PCInitCount = 0;
static int PCRhythm[MAXTYPES][8];

void ResetPostClassify(void)
{
    int i, j;
    for (i = 0; i < MAXTYPES; ++i)
        for (j = 0; j < 8; ++j) {
            PostClass[i][j] = 0;
            PCRhythm[i][j] = 0;
        }
    PCInitCount = 0;
}


void PostClassify(int *recentTypes, int domType, int *recentRRs,
    int width, double mi2, int rhythmClass)
{
    static int lastRC, lastWidth;
    static double lastMI2;
    int i, regCount, pvcCount, normRR;
    double mi3;

    /* If preceding and following beats are same type and regular,
       consider them dominant */
    if ((recentTypes[0] == recentTypes[2]) && (recentTypes[0] != domType)
        && (recentTypes[0] != recentTypes[1])) {
        mi3 = DomCompare(recentTypes[0], domType);
        for (i = regCount = 0; i < 8; ++i)
            if (PCRhythm[recentTypes[0]][i] == NORMAL)
                ++regCount;
        if ((mi3 < 2.0) && (regCount > 6))
            domType = recentTypes[0];
    }

    if (PCInitCount < 3) {
        ++PCInitCount;
        lastWidth = width;
        lastMI2 = 0;
        lastRC = 0;
        return;
    }

    if (recentTypes[1] < MAXTYPES) {
        /* Find first NN interval */
        for (i = 2; (i < 7) && (recentTypes[i] != recentTypes[i + 1]); ++i);
        if (i == 7) normRR = 0;
        else normRR = recentRRs[i];

        for (i = pvcCount = 0; i < 8; ++i)
            if (PostClass[recentTypes[1]][i] == PVC)
                ++pvcCount;

        for (i = 7; i > 0; --i) {
            PostClass[recentTypes[1]][i] = PostClass[recentTypes[1]][i - 1];
            PCRhythm[recentTypes[1]][i] = PCRhythm[recentTypes[1]][i - 1];
        }

        /* Premature + compensitory pause → PVC */
        if (((normRR - (normRR >> 3)) >= recentRRs[1]) &&
            ((recentRRs[0] - (recentRRs[0] >> 3)) >= normRR) &&
            (recentTypes[0] == domType) && (recentTypes[2] == domType) &&
            (recentTypes[1] != domType))
            PostClass[recentTypes[1]][0] = PVC;

        /* Previous two PVCs and this is slightly premature */
        else if (((normRR - (normRR >> 4)) > recentRRs[1]) &&
            ((normRR + (normRR >> 4)) < recentRRs[0]) &&
            (((PostClass[recentTypes[1]][1] == PVC) &&
              (PostClass[recentTypes[1]][2] == PVC)) || (pvcCount >= 6)) &&
            (recentTypes[0] == domType) && (recentTypes[2] == domType) &&
            (recentTypes[1] != domType))
            PostClass[recentTypes[1]][0] = PVC;

        /* Significantly different from dominant */
        else if ((recentTypes[0] == domType) && (recentTypes[2] == domType) &&
            (lastMI2 > 2.5))
            PostClass[recentTypes[1]][0] = PVC;
        else
            PostClass[recentTypes[1]][0] = UNKNOWN;

        /* Classify the rhythm */
        if (((normRR - (normRR >> 3)) > recentRRs[1]) &&
            ((recentRRs[0] - (recentRRs[0] >> 3)) > normRR))
            PCRhythm[recentTypes[1]][0] = PVC;
        else
            PCRhythm[recentTypes[1]][0] = lastRC;
    }

    lastWidth = width;
    lastMI2 = mi2;
    lastRC = rhythmClass;
}


int CheckPostClass(int type)
{
    int i, pvcs4 = 0, pvcs8;

    if (type == MAXTYPES)
        return UNKNOWN;

    for (i = 0; i < 4; ++i)
        if (PostClass[type][i] == PVC)
            ++pvcs4;
    for (pvcs8 = pvcs4; i < 8; ++i)
        if (PostClass[type][i] == PVC)
            ++pvcs8;

    if ((pvcs4 >= 3) || (pvcs8 >= 6))
        return PVC;
    return UNKNOWN;
}


int CheckPCRhythm(int type)
{
    int i, normCount, n;

    if (type == MAXTYPES)
        return UNKNOWN;

    if (GetBeatTypeCount(type) < 9)
        n = GetBeatTypeCount(type) - 1;
    else n = 8;

    for (i = normCount = 0; i < n; ++i)
        if (PCRhythm[type][i] == NORMAL)
            ++normCount;

    if (normCount >= 7)
        return NORMAL;
    if (((normCount == 0) && (n < 4)) ||
        ((normCount <= 1) && (n >= 4) && (n < 7)) ||
        ((normCount <= 2) && (n >= 7)))
        return PVC;
    return UNKNOWN;
}
