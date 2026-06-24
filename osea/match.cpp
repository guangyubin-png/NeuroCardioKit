/*****************************************************************************
 *  match.cpp - Beat template matching and type management
 *  Original: Patrick S. Hamilton, E.P. Limited
 *
 *  Manages beat templates, matching new beats to existing types,
 *  and beat feature access.
 *****************************************************************************/

#include <stdlib.h>
#include "ecgcodes.h"
#include "bdac.h"
#include "match.h"
#include "analbeat.h"

#define MATCH_LENGTH      BEAT_MS300
#define MATCH_LIMIT       1.2
#define COMBINE_LIMIT     0.8
#define MATCH_START       (FIDMARK-(MATCH_LENGTH/2))
#define MATCH_END         (FIDMARK+(MATCH_LENGTH/2))
#define MAXPREV           8
#define MAX_SHIFT         BEAT_MS40
#define WIDE_VAR_LIMIT    0.50

/* Local prototypes */
static double CompareBeats(int *beat1, int *beat2, int *shiftAdj);
static double CompareBeats2(int *beat1, int *beat2, int *shiftAdj);
static void UpdateBeat(int *aveBeat, int *newBeat, int shift);
static void BeatCopy(int srcBeat, int destBeat);

/* External data from postclas.cpp */
extern int PostClass[MAXTYPES][8];
extern int PCRhythm[MAXTYPES][8];

/* Global variables */
static int BeatTemplates[MAXTYPES][BEATLGTH];
static int BeatCounts[MAXTYPES];
static int BeatWidths[MAXTYPES];
static int BeatClassifications[MAXTYPES];
static int BeatBegins[MAXTYPES];
static int BeatEnds[MAXTYPES];
static int BeatsSinceLastMatch[MAXTYPES];
static int BeatAmps[MAXTYPES];
static int BeatCenters[MAXTYPES];
static double MIs[MAXTYPES][8];
static int TypeCount = 0;

void ResetMatch(void)
{
    int i, j;
    TypeCount = 0;
    for (i = 0; i < MAXTYPES; ++i) {
        BeatCounts[i] = 0;
        BeatClassifications[i] = UNKNOWN;
        for (j = 0; j < 8; ++j)
            MIs[i][j] = 0;
    }
}


static double CompareBeats(int *beat1, int *beat2, int *shiftAdj)
{
    int i, max, min, magSum, shift;
    long beatDiff, meanDiff, minDiff, minShift;
    double metric, scaleFactor, tempD;

    max = min = beat1[MATCH_START];
    for (i = MATCH_START + 1; i < MATCH_END; ++i)
        if (beat1[i] > max) max = beat1[i];
        else if (beat1[i] < min) min = beat1[i];
    magSum = max - min;

    i = MATCH_START;
    max = min = beat2[i];
    for (i = MATCH_START + 1; i < MATCH_END; ++i)
        if (beat2[i] > max) max = beat2[i];
        else if (beat2[i] < min) min = beat2[i];

    scaleFactor = magSum;
    scaleFactor /= max - min;
    magSum *= 2;

    for (shift = -MAX_SHIFT; shift <= MAX_SHIFT; ++shift) {
        for (i = FIDMARK - (MATCH_LENGTH >> 1), meanDiff = 0;
             i < FIDMARK + (MATCH_LENGTH >> 1); ++i) {
            tempD = beat2[i + shift];
            tempD *= scaleFactor;
            meanDiff += (long)(beat1[i] - tempD);
        }
        meanDiff /= MATCH_LENGTH;

        for (i = FIDMARK - (MATCH_LENGTH >> 1), beatDiff = 0;
             i < FIDMARK + (MATCH_LENGTH >> 1); ++i) {
            tempD = beat2[i + shift];
            tempD *= scaleFactor;
            beatDiff += abs((long)(beat1[i] - meanDiff - tempD));
        }

        if (shift == -MAX_SHIFT) {
            minDiff = beatDiff;
            minShift = -MAX_SHIFT;
        }
        else if (beatDiff < minDiff) {
            minDiff = beatDiff;
            minShift = shift;
        }
    }

    metric = (double)minDiff;
    *shiftAdj = minShift;
    metric /= magSum;
    metric *= 30;
    metric /= MATCH_LENGTH;
    return metric;
}


static double CompareBeats2(int *beat1, int *beat2, int *shiftAdj)
{
    int i, max, min, shift;
    int mag1, mag2;
    long beatDiff, meanDiff, minDiff, minShift;
    double metric;

    max = min = beat1[MATCH_START];
    for (i = MATCH_START + 1; i < MATCH_END; ++i)
        if (beat1[i] > max) max = beat1[i];
        else if (beat1[i] < min) min = beat1[i];
    mag1 = max - min;

    i = MATCH_START;
    max = min = beat2[i];
    for (i = MATCH_START + 1; i < MATCH_END; ++i)
        if (beat2[i] > max) max = beat2[i];
        else if (beat2[i] < min) min = beat2[i];
    mag2 = max - min;

    for (shift = -MAX_SHIFT; shift <= MAX_SHIFT; ++shift) {
        for (i = FIDMARK - (MATCH_LENGTH >> 1), meanDiff = 0;
             i < FIDMARK + (MATCH_LENGTH >> 1); ++i)
            meanDiff += beat1[i] - beat2[i + shift];
        meanDiff /= MATCH_LENGTH;

        for (i = FIDMARK - (MATCH_LENGTH >> 1), beatDiff = 0;
             i < FIDMARK + (MATCH_LENGTH >> 1); ++i)
            beatDiff += abs(beat1[i] - (int)meanDiff - beat2[i + shift]);

        if (shift == -MAX_SHIFT) {
            minDiff = beatDiff;
            minShift = -MAX_SHIFT;
        }
        else if (beatDiff < minDiff) {
            minDiff = beatDiff;
            minShift = shift;
        }
    }

    metric = (double)minDiff;
    *shiftAdj = minShift;
    metric /= (mag1 + mag2);
    metric *= 30;
    metric /= MATCH_LENGTH;
    return metric;
}


static void UpdateBeat(int *aveBeat, int *newBeat, int shift)
{
    int i;
    long tempLong;

    for (i = 0; i < BEATLGTH; ++i) {
        if ((i + shift >= 0) && (i + shift < BEATLGTH)) {
            tempLong = aveBeat[i];
            tempLong *= 7;
            tempLong += newBeat[i + shift];
            tempLong >>= 3;
            aveBeat[i] = (int)tempLong;
        }
    }
}


int GetTypesCount(void) { return TypeCount; }
int GetBeatTypeCount(int type) { return BeatCounts[type]; }
int GetBeatWidth(int type) { return BeatWidths[type]; }
int GetBeatCenter(int type) { return BeatCenters[type]; }

int GetBeatClass(int type)
{
    if (type == MAXTYPES) return UNKNOWN;
    return BeatClassifications[type];
}

void SetBeatClass(int type, int beatClass) { BeatClassifications[type] = beatClass; }


int NewBeatType(int *newBeat)
{
    int i, onset, offset, isoLevel, beatBegin, beatEnd;
    int mcType, amp;

    for (i = 0; i < TypeCount; ++i)
        ++BeatsSinceLastMatch[i];

    if (TypeCount < MAXTYPES) {
        for (i = 0; i < BEATLGTH; ++i)
            BeatTemplates[TypeCount][i] = newBeat[i];

        BeatCounts[TypeCount] = 1;
        BeatClassifications[TypeCount] = UNKNOWN;
        AnalyzeBeat(BeatTemplates[TypeCount], &onset, &offset, &isoLevel,
            &beatBegin, &beatEnd, &amp);
        BeatWidths[TypeCount] = offset - onset;
        BeatCenters[TypeCount] = (offset + onset) / 2;
        BeatBegins[TypeCount] = beatBegin;
        BeatEnds[TypeCount] = beatEnd;
        BeatAmps[TypeCount] = amp;
        BeatsSinceLastMatch[TypeCount] = 0;
        ++TypeCount;
        return TypeCount - 1;
    }
    else {
        mcType = 0;
        for (i = 1; i < MAXTYPES; ++i)
            if (BeatCounts[i] < BeatCounts[mcType])
                mcType = i;
            else if (BeatCounts[i] == BeatCounts[mcType])
                if (BeatsSinceLastMatch[i] > BeatsSinceLastMatch[mcType])
                    mcType = i;

        /* AdjustDomData called externally via classify.cpp */
        for (i = 0; i < BEATLGTH; ++i)
            BeatTemplates[mcType][i] = newBeat[i];

        BeatCounts[mcType] = 1;
        BeatClassifications[mcType] = UNKNOWN;
        AnalyzeBeat(BeatTemplates[mcType], &onset, &offset, &isoLevel,
            &beatBegin, &beatEnd, &amp);
        BeatWidths[mcType] = offset - onset;
        BeatCenters[mcType] = (offset + onset) / 2;
        BeatBegins[mcType] = beatBegin;
        BeatEnds[mcType] = beatEnd;
        BeatsSinceLastMatch[mcType] = 0;
        BeatAmps[mcType] = amp;
        return mcType;
    }
}


void BestMorphMatch(int *newBeat, int *matchType, double *matchIndex,
    double *mi2, int *shiftAdj)
{
    int type, i, bestMatch, nextBest, minShift, shift, temp;
    int bestShift2, nextShift2;
    double bestDiff2, nextDiff2;
    double beatDiff, minDiff, nextDiff = 10000;

    if (TypeCount == 0) {
        *matchType = 0;
        *matchIndex = 1000;
        *shiftAdj = 0;
        return;
    }

    for (type = 0; type < TypeCount; ++type) {
        beatDiff = CompareBeats(BeatTemplates[type], newBeat, &shift);
        if (type == 0) {
            bestMatch = 0;
            minDiff = beatDiff;
            minShift = shift;
        }
        else if (beatDiff < minDiff) {
            nextBest = bestMatch;
            nextDiff = minDiff;
            bestMatch = type;
            minDiff = beatDiff;
            minShift = shift;
        }
        else if ((TypeCount > 1) && (type == 1)) {
            nextBest = type;
            nextDiff = beatDiff;
        }
        else if (beatDiff < nextDiff) {
            nextBest = type;
            nextDiff = beatDiff;
        }
    }

    if ((minDiff < MATCH_LIMIT) && (nextDiff < MATCH_LIMIT) && (TypeCount > 1)) {
        bestDiff2 = CompareBeats2(BeatTemplates[bestMatch], newBeat, &bestShift2);
        nextDiff2 = CompareBeats2(BeatTemplates[nextBest], newBeat, &nextShift2);
        if (nextDiff2 < bestDiff2) {
            temp = bestMatch; bestMatch = nextBest; nextBest = temp;
            temp = (int)minDiff; minDiff = nextDiff; nextDiff = temp;
            minShift = nextShift2;
            *mi2 = bestDiff2;
        }
        else *mi2 = nextDiff2;

        beatDiff = CompareBeats(BeatTemplates[bestMatch], BeatTemplates[nextBest], &shift);

        if ((beatDiff < COMBINE_LIMIT) &&
            ((*mi2 < 1.0) || (!MinimumBeatVariation(nextBest)))) {
            if (bestMatch < nextBest) {
                for (i = 0; i < BEATLGTH; ++i)
                    if ((i + shift > 0) && (i + shift < BEATLGTH)) {
                        BeatTemplates[bestMatch][i] += BeatTemplates[nextBest][i + shift];
                        BeatTemplates[bestMatch][i] >>= 1;
                    }

                if ((BeatClassifications[bestMatch] == NORMAL) ||
                    (BeatClassifications[nextBest] == NORMAL))
                    BeatClassifications[bestMatch] = NORMAL;
                else if ((BeatClassifications[bestMatch] == PVC) ||
                    (BeatClassifications[nextBest] == PVC))
                    BeatClassifications[bestMatch] = PVC;

                BeatCounts[bestMatch] += BeatCounts[nextBest];
                for (type = nextBest; type < TypeCount - 1; ++type)
                    BeatCopy(type + 1, type);
            }
            else {
                for (i = 0; i < BEATLGTH; ++i) {
                    BeatTemplates[nextBest][i] += BeatTemplates[bestMatch][i];
                    BeatTemplates[nextBest][i] >>= 1;
                }

                if ((BeatClassifications[bestMatch] == NORMAL) ||
                    (BeatClassifications[nextBest] == NORMAL))
                    BeatClassifications[nextBest] = NORMAL;
                else if ((BeatClassifications[bestMatch] == PVC) ||
                    (BeatClassifications[nextBest] == PVC))
                    BeatClassifications[nextBest] = PVC;

                BeatCounts[nextBest] += BeatCounts[bestMatch];
                for (type = bestMatch; type < TypeCount - 1; ++type)
                    BeatCopy(type + 1, type);
                bestMatch = nextBest;
            }
            --TypeCount;
            BeatClassifications[TypeCount] = UNKNOWN;
        }
    }

    *mi2 = CompareBeats2(BeatTemplates[bestMatch], newBeat, &bestShift2);
    *matchType = bestMatch;
    *matchIndex = minDiff;
    *shiftAdj = minShift;
}


void UpdateBeatType(int matchType, int *newBeat, double mi2, int shiftAdj)
{
    int i, onset, offset, isoLevel, beatBegin, beatEnd;
    int amp;

    for (i = 0; i < TypeCount; ++i) {
        if (i != matchType)
            ++BeatsSinceLastMatch[i];
        else BeatsSinceLastMatch[i] = 0;
    }

    if (BeatCounts[matchType] == 1)
        for (i = 0; i < BEATLGTH; ++i)
            if ((i + shiftAdj >= 0) && (i + shiftAdj < BEATLGTH))
                BeatTemplates[matchType][i] =
                    (BeatTemplates[matchType][i] + newBeat[i + shiftAdj]) >> 1;
    else
        UpdateBeat(BeatTemplates[matchType], newBeat, shiftAdj);

    AnalyzeBeat(BeatTemplates[matchType], &onset, &offset, &isoLevel,
        &beatBegin, &beatEnd, &amp);

    BeatWidths[matchType] = offset - onset;
    BeatCenters[matchType] = (offset + onset) / 2;
    BeatBegins[matchType] = beatBegin;
    BeatEnds[matchType] = beatEnd;
    BeatAmps[matchType] = amp;
    ++BeatCounts[matchType];

    for (i = MAXPREV - 1; i > 0; --i)
        MIs[matchType][i] = MIs[matchType][i - 1];
    MIs[matchType][0] = mi2;
}


int GetDominantType(void)
{
    int maxCount = 0, maxType = -1;
    int type, totalCount;

    for (type = 0; type < MAXTYPES; ++type)
        if ((BeatClassifications[type] == NORMAL) && (BeatCounts[type] > maxCount)) {
            maxType = type;
            maxCount = BeatCounts[type];
        }

    if (maxType == -1) {
        for (type = 0, totalCount = 0; type < TypeCount; ++type)
            totalCount += BeatCounts[type];
        if (totalCount > 300)
            for (type = 0; type < TypeCount; ++type)
                if (BeatCounts[type] > maxCount) {
                    maxType = type;
                    maxCount = BeatCounts[type];
                }
    }
    return maxType;
}


void ClearLastNewType(void) { if (TypeCount != 0) --TypeCount; }
int GetBeatBegin(int type) { return BeatBegins[type]; }
int GetBeatEnd(int type) { return BeatEnds[type]; }
int GetBeatAmp(int type) { return BeatAmps[type]; }


double DomCompare2(int *newBeat, int domType)
{
    int shift;
    return CompareBeats2(BeatTemplates[domType], newBeat, &shift);
}

double DomCompare(int newType, int domType)
{
    int shift;
    return CompareBeats2(BeatTemplates[domType], BeatTemplates[newType], &shift);
}


static void BeatCopy(int srcBeat, int destBeat)
{
    int i;

    for (i = 0; i < BEATLGTH; ++i)
        BeatTemplates[destBeat][i] = BeatTemplates[srcBeat][i];

    BeatCounts[destBeat] = BeatCounts[srcBeat];
    BeatWidths[destBeat] = BeatWidths[srcBeat];
    BeatCenters[destBeat] = BeatCenters[srcBeat];
    for (i = 0; i < MAXPREV; ++i) {
        PostClass[destBeat][i] = PostClass[srcBeat][i];
        PCRhythm[destBeat][i] = PCRhythm[srcBeat][i];
    }
    BeatClassifications[destBeat] = BeatClassifications[srcBeat];
    BeatBegins[destBeat] = BeatBegins[srcBeat];
    BeatEnds[destBeat] = BeatBegins[srcBeat];
    BeatsSinceLastMatch[destBeat] = BeatsSinceLastMatch[srcBeat];
    BeatAmps[destBeat] = BeatAmps[srcBeat];
}


int MinimumBeatVariation(int type)
{
    int i;
    for (i = 0; i < MAXTYPES; ++i)
        if (MIs[type][i] > 0.5)
            return 0;
    return 1;
}


int WideBeatVariation(int type)
{
    int i, n;
    double aveMI;

    n = BeatCounts[type];
    if (n > 8) n = 8;

    for (i = 0, aveMI = 0; i < n; ++i)
        aveMI += MIs[type][i];

    aveMI /= n;
    if (aveMI > WIDE_VAR_LIMIT) return 1;
    return 0;
}
