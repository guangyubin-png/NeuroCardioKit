/*****************************************************************************
 *  analbeat.cpp - Beat analysis (QRS onset/offset, isoelectric level, etc.)
 *  Original: Patrick S. Hamilton, E.P. Limited
 *
 *  Analyzes a beat template (100 Hz) to determine QRS onset, offset,
 *  beat boundaries, amplitude, and isoelectric level.
 *****************************************************************************/

#include "bdac.h"
#include <stdlib.h>

#define ISO_LENGTH1  BEAT_MS50
#define ISO_LENGTH2  BEAT_MS80
#define ISO_LIMIT    20

/* Local prototypes */
static int IsoCheck(int *data, int isoLength);
static void AnalyzeBeat(int *beat, int *onset, int *offset,
    int *isoLevel, int *beatBegin, int *beatEnd, int *amp);


static int IsoCheck(int *data, int isoLength)
{
    int i, max, min;

    for (i = 1, max = min = data[0]; i < isoLength; ++i) {
        if (data[i] > max)
            max = data[i];
        else if (data[i] < min)
            min = data[i];
    }

    if (max - min < ISO_LIMIT)
        return 1;
    return 0;
}


#define INF_CHK_N  BEAT_MS40

static void AnalyzeBeat(int *beat, int *onset, int *offset, int *isoLevel,
    int *beatBegin, int *beatEnd, int *amp)
{
    int maxSlope = 0, maxSlopeI, minSlope = 0, minSlopeI;
    int maxV, minV;
    int isoStart, isoEnd;
    int slope, i;

    /* Search back for isoelectric region preceding QRS */
    for (i = FIDMARK - ISO_LENGTH2;
         (i > 0) && (IsoCheck(&beat[i], ISO_LENGTH2) == 0); --i);

    if (i == 0) {
        for (i = FIDMARK - ISO_LENGTH1;
             (i > 0) && (IsoCheck(&beat[i], ISO_LENGTH1) == 0); --i);
        isoStart = i + (ISO_LENGTH1 - 1);
    }
    else isoStart = i + (ISO_LENGTH2 - 1);

    /* Search forward for isoelectric region following QRS */
    for (i = FIDMARK; (i < BEATLGTH) && (IsoCheck(&beat[i], ISO_LENGTH1) == 0); ++i);
    isoEnd = i;

    /* Find max and min slopes on QRS */
    i = FIDMARK - BEAT_MS150;
    maxSlope = beat[i] - beat[i - 1];
    maxSlopeI = minSlopeI = i;

    for (; i < FIDMARK + BEAT_MS150; ++i) {
        slope = beat[i] - beat[i - 1];
        if (slope > maxSlope) {
            maxSlope = slope;
            maxSlopeI = i;
        }
        else if (slope < minSlope) {
            minSlope = slope;
            minSlopeI = i;
        }
    }

    /* Use the smallest of max or min slope */
    if (maxSlope > -minSlope)
        maxSlope = -minSlope;
    else minSlope = -maxSlope;

    if (maxSlopeI < minSlopeI) {
        /* Search back from max slope for QRS onset */
        for (i = maxSlopeI;
             (i > 0) && ((beat[i] - beat[i - 1]) > (maxSlope >> 2)); --i);
        *onset = i - 1;

        /* Check for brief inflection */
        for (; (i > *onset - INF_CHK_N) && ((beat[i] - beat[i - 1]) <= (maxSlope >> 2)); --i);
        if (i > *onset - INF_CHK_N) {
            for (; (i > 0) && ((beat[i] - beat[i - 1]) > (maxSlope >> 2)); --i);
            *onset = i - 1;
        }
        i = *onset + 1;

        /* Check for large negative slope after inflection */
        for (; (i > *onset - INF_CHK_N) && ((beat[i - 1] - beat[i]) < (maxSlope >> 2)); --i);
        if (i > *onset - INF_CHK_N) {
            for (; (i > 0) && ((beat[i - 1] - beat[i]) > (maxSlope >> 2)); --i);
            *onset = i - 1;
        }

        /* Search forward from min slope for QRS offset */
        for (i = minSlopeI;
             (i < BEATLGTH) && ((beat[i] - beat[i - 1]) < (minSlope >> 2)); ++i);
        *offset = i;

        /* Check for inflection */
        for (; (i < *offset + INF_CHK_N) && ((beat[i] - beat[i - 1]) >= (minSlope >> 2)); ++i);
        if (i < *offset + INF_CHK_N) {
            for (; (i < BEATLGTH) && ((beat[i] - beat[i - 1]) < (minSlope >> 2)); ++i);
            *offset = i;
        }
        i = *offset;

        /* Check for significant upslope after downslope */
        for (; (i < *offset + BEAT_MS40) && ((beat[i - 1] - beat[i]) > (minSlope >> 2)); ++i);
        if (i < *offset + BEAT_MS40) {
            for (; (i < BEATLGTH) && ((beat[i - 1] - beat[i]) < (minSlope >> 2)); ++i);
            *offset = i;

            for (; (i < *offset + BEAT_MS60) && (beat[i] - beat[i - 1] > (minSlope >> 2)); ++i);
            if (i < *offset + BEAT_MS60) {
                for (; (i < BEATLGTH) && (beat[i] - beat[i - 1] < (minSlope >> 2)); ++i);
                *offset = i;
            }
        }
    }
    else {
        /* Search back from min slope for QRS onset */
        for (i = minSlopeI;
             (i > 0) && ((beat[i] - beat[i - 1]) < (minSlope >> 2)); --i);
        *onset = i - 1;

        for (; (i > *onset - INF_CHK_N) && ((beat[i] - beat[i - 1]) >= (minSlope >> 2)); --i);
        if (i > *onset - INF_CHK_N) {
            for (; (i > 0) && ((beat[i] - beat[i - 1]) < (minSlope >> 2)); --i);
            *onset = i - 1;
        }
        i = *onset + 1;

        for (; (i > *onset - INF_CHK_N) && ((beat[i - 1] - beat[i]) > (minSlope >> 2)); --i);
        if (i > *onset - INF_CHK_N) {
            for (; (i > 0) && ((beat[i - 1] - beat[i]) < (minSlope >> 2)); --i);
            *onset = i - 1;
        }

        /* Search forward from max slope for QRS offset */
        for (i = maxSlopeI;
             (i < BEATLGTH) && ((beat[i] - beat[i - 1]) > (maxSlope >> 2)); ++i);
        *offset = i;

        for (; (i < *offset + INF_CHK_N) && ((beat[i] - beat[i - 1]) <= (maxSlope >> 2)); ++i);
        if (i < *offset + INF_CHK_N) {
            for (; (i < BEATLGTH) && ((beat[i] - beat[i - 1]) > (maxSlope >> 2)); ++i);
            *offset = i;
        }
        i = *offset;

        for (; (i < *offset + BEAT_MS40) && ((beat[i - 1] - beat[i]) < (maxSlope >> 2)); ++i);
        if (i < *offset + BEAT_MS40) {
            for (; (i < BEATLGTH) && ((beat[i - 1] - beat[i]) > (maxSlope >> 2)); ++i);
            *offset = i;
        }
    }

    /* Adjust onset/offset based on isoelectric points */
    if ((isoStart == ISO_LENGTH1 - 1) && (*onset > isoStart))
        isoStart = *onset;
    else if (*onset - isoStart < BEAT_MS50)
        *onset = isoStart;

    if (isoEnd - *offset < BEAT_MS50)
        *offset = isoEnd;

    *isoLevel = beat[isoStart];

    /* Find max and min values in QRS */
    for (i = *onset, maxV = minV = beat[*onset]; i < *offset; ++i)
        if (beat[i] > maxV)
            maxV = beat[i];
        else if (beat[i] < minV)
            minV = beat[i];

    /* Extend offset if necessary */
    if ((beat[*onset] - beat[*offset] > ((maxV - minV) >> 2) + ((maxV - minV) >> 3))) {
        for (i = maxSlopeI = *offset, maxSlope = beat[*offset] - beat[*offset - 1];
             (i < *offset + BEAT_MS100) && (i < BEATLGTH); ++i) {
            slope = beat[i] - beat[i - 1];
            if (slope > maxSlope) {
                maxSlope = slope;
                maxSlopeI = i;
            }
        }

        if (maxSlope > 0) {
            for (i = maxSlopeI;
                 (i < BEATLGTH) && (beat[i] - beat[i - 1] > (maxSlope >> 1)); ++i);
            *offset = i;
        }
    }

    /* Determine beat beginning (P-wave onset) */
    for (i = FIDMARK - BEAT_MS250;
         (i >= BEAT_MS80) && (IsoCheck(&beat[i - BEAT_MS80], BEAT_MS80) == 0); --i);
    *beatBegin = i;

    if (*beatBegin == FIDMARK - BEAT_MS250) {
        for (; (i < *onset - BEAT_MS50) &&
             (IsoCheck(&beat[i - BEAT_MS80], BEAT_MS80) == 1); ++i);
        *beatBegin = i - 1;
    }
    else if (*beatBegin == BEAT_MS80 - 1) {
        for (; (i < *onset) && (IsoCheck(&beat[i - BEAT_MS80], BEAT_MS80) == 0); ++i);
        if (i < *onset) {
            for (; (i < *onset) && (IsoCheck(&beat[i - BEAT_MS80], BEAT_MS80) == 1); ++i);
            if (i < *onset)
                *beatBegin = i - 1;
        }
    }

    /* Determine beat end (T-wave offset) */
    for (i = FIDMARK + BEAT_MS300;
         (i < BEATLGTH) && (IsoCheck(&beat[i], BEAT_MS80) == 0); ++i);
    *beatEnd = i;

    /* Calculate beat amplitude */
    maxV = minV = beat[*onset];
    for (i = *onset; i < *offset; ++i)
        if (beat[i] > maxV)
            maxV = beat[i];
        else if (beat[i] < minV)
            minV = beat[i];
    *amp = maxV - minV;
}
