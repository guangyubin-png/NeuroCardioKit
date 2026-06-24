/*****************************************************************************
 *  classify.cpp - Beat classification
 *  Original: Patrick S. Hamilton, E.P. Limited
 *
 *  Classifies beats based on template matching, rhythm, width, and noise.
 *  The only external entry point is Classify().
 *****************************************************************************/

#include "ecgcodes.h"
#include <stdlib.h>
#include "qrsdet.h"
#include "bdac.h"
#include "match.h"
#include "rythmchk.h"
#include "analbeat.h"
#include "postclas.h"

/* Detection Rule Parameters */
#define CLASSIFY_MATCH_LIMIT         1.3
#define MATCH_WITH_AMP_LIMIT         2.5
#define PVC_MATCH_WITH_AMP_LIMIT     0.9
#define BL_SHIFT_LIMIT               100
#define NEW_TYPE_NOISE_THRESHOLD     18
#define NEW_TYPE_HF_NOISE_LIMIT      75
#define MATCH_NOISE_THRESHOLD        0.7

/* TempClass classification rule parameters */
#define R2_DI_THRESHOLD       1.0
#define R3_WIDTH_THRESHOLD    BEAT_MS90
#define R7_DI_THRESHOLD       1.2
#define R8_DI_THRESHOLD       1.5
#define R9_DI_THRESHOLD       2.0
#define R10_BC_LIM            3
#define R10_DI_THRESHOLD      2.5
#define R11_MIN_WIDTH         BEAT_MS110
#define R11_WIDTH_BREAK       BEAT_MS140
#define R11_WIDTH_DIFF1       BEAT_MS40
#define R11_WIDTH_DIFF2       BEAT_MS60
#define R11_HF_THRESHOLD      45
#define R11_MA_THRESHOLD      14
#define R11_BC_LIM            1
#define R15_DI_THRESHOLD      3.5
#define R15_WIDTH_THRESHOLD   BEAT_MS100
#define R16_WIDTH_THRESHOLD   BEAT_MS100
#define R17_WIDTH_DELTA       BEAT_MS20
#define R18_DI_THRESHOLD      1.5
#define R19_HF_THRESHOLD      75

/* Dominant monitor constants */
#define DM_BUFFER_LENGTH      180
#define IRREG_RR_LIMIT        60

/* Local prototypes */
static int HFNoiseCheck(int *beat);
static int TempClass(int rhythmClass, int morphType, int beatWidth, int domWidth,
    int domType, int hfNoise, int noiseLevel, int blShift, double domIndex);
static int DomMonitor(int morphType, int rhythmClass, int beatWidth, int rr, int reset);
static int GetDomRhythm(void);
static int GetRunCount(void);

/* External prototypes from match.cpp */
void AdjustDomData(int oldType, int newType);
void CombineDomData(int oldType, int newType);
int GetDominantType(void);

/* Local Global variables */
static int DomType;
static int RecentRRs[8], RecentTypes[8];


int Classify(int *newBeat, int rr, int noiseLevel, int *beatMatch, int *fidAdj,
    int init)
{
    int rhythmClass, beatClass, i, beatWidth, blShift;
    static int morphType, runCount = 0;
    double matchIndex, domIndex, mi2;
    int shiftAdj;
    int domType, domWidth, onset, offset, amp;
    int beatBegin, beatEnd, tempClass;
    int hfNoise, isoLevel;
    static int lastIsoLevel = 0, lastRhythmClass = UNKNOWN, lastBeatWasNew = 0;

    if (init) {
        ResetRhythmChk();
        ResetMatch();
        ResetPostClassify();
        runCount = 0;
        DomMonitor(0, 0, 0, 0, 1);
        return 0;
    }

    hfNoise = HFNoiseCheck(newBeat);
    rhythmClass = RhythmChk(rr);

    AnalyzeBeat(newBeat, &onset, &offset, &isoLevel, &beatBegin, &beatEnd, &amp);

    blShift = abs(lastIsoLevel - isoLevel);
    lastIsoLevel = isoLevel;

    /* Make isoelectric level 0 */
    for (i = 0; i < BEATLGTH; ++i)
        newBeat[i] -= isoLevel;

    /* If baseline shift + last beat was new, delete the last new type */
    if ((blShift > BL_SHIFT_LIMIT) && (lastBeatWasNew == 1) &&
        (lastRhythmClass == NORMAL) && (rhythmClass == NORMAL))
        ClearLastNewType();
    lastBeatWasNew = 0;

    /* Find best matching template */
    BestMorphMatch(newBeat, &morphType, &matchIndex, &mi2, &shiftAdj);

    /* Disregard noise if match is good */
    if (matchIndex < MATCH_NOISE_THRESHOLD)
        hfNoise = noiseLevel = blShift = 0;

    /* Stricter match for premature beats */
    if ((matchIndex < CLASSIFY_MATCH_LIMIT) && (rhythmClass == PVC) &&
        MinimumBeatVariation(morphType) && (mi2 > PVC_MATCH_WITH_AMP_LIMIT)) {
        morphType = NewBeatType(newBeat);
        lastBeatWasNew = 1;
    }
    /* Match if within standard limits */
    else if ((matchIndex < CLASSIFY_MATCH_LIMIT) && (mi2 <= MATCH_WITH_AMP_LIMIT))
        UpdateBeatType(morphType, newBeat, mi2, shiftAdj);
    /* Not noisy but doesn't match → new beat type */
    else if ((blShift < BL_SHIFT_LIMIT) && (noiseLevel < NEW_TYPE_NOISE_THRESHOLD)
        && (hfNoise < NEW_TYPE_HF_NOISE_LIMIT)) {
        morphType = NewBeatType(newBeat);
        lastBeatWasNew = 1;
    }
    /* Noisy but irregular → new beat type */
    else if ((lastRhythmClass != NORMAL) || (rhythmClass != NORMAL)) {
        morphType = NewBeatType(newBeat);
        lastBeatWasNew = 1;
    }
    /* Noisy and regular → don't waste a template */
    else morphType = MAXTYPES;

    /* Update recent RR and type arrays */
    for (i = 7; i > 0; --i) {
        RecentRRs[i] = RecentRRs[i - 1];
        RecentTypes[i] = RecentTypes[i - 1];
    }
    RecentRRs[0] = rr;
    RecentTypes[0] = morphType;

    lastRhythmClass = rhythmClass;
    lastIsoLevel = isoLevel;

    /* Fetch beat features */
    if (morphType != MAXTYPES) {
        beatClass = GetBeatClass(morphType);
        beatWidth = GetBeatWidth(morphType);
        *fidAdj = GetBeatCenter(morphType) - FIDMARK;

        if ((beatWidth > offset - onset) && (GetBeatTypeCount(morphType) <= 4)) {
            beatWidth = offset - onset;
            *fidAdj = ((offset + onset) / 2) - FIDMARK;
        }
    }
    else {
        beatWidth = offset - onset;
        beatClass = UNKNOWN;
        *fidAdj = ((offset + onset) / 2) - FIDMARK;
    }

    /* Fetch dominant type features */
    DomType = domType = DomMonitor(morphType, rhythmClass, beatWidth, rr, 0);
    domWidth = GetBeatWidth(domType);

    /* Compare beat to dominant */
    if ((morphType != domType) && (morphType != 8))
        domIndex = DomCompare(morphType, domType);
    else if (morphType == 8)
        domIndex = DomCompare2(newBeat, domType);
    else domIndex = matchIndex;

    /* Update post classification of previous beat */
    PostClassify(RecentTypes, domType, RecentRRs, beatWidth, domIndex, rhythmClass);

    /* Temporary classification */
    tempClass = TempClass(rhythmClass, morphType, beatWidth, domWidth,
        domType, hfNoise, noiseLevel, blShift, domIndex);

    /* If morphology not classified yet, try to classify it */
    if ((beatClass == UNKNOWN) && (morphType < MAXTYPES)) {
        runCount = GetRunCount();

        if ((runCount >= 3) && (domType != -1) && (beatWidth < domWidth + BEAT_MS20))
            SetBeatClass(morphType, NORMAL);
        else if ((runCount >= 6) && (domType == -1))
            SetBeatClass(morphType, NORMAL);
        else if (IsBigeminy() == 1) {
            if ((rhythmClass == PVC) && (beatWidth > BEAT_MS100))
                SetBeatClass(morphType, PVC);
            else if (rhythmClass == NORMAL)
                SetBeatClass(morphType, NORMAL);
        }
    }

    *beatMatch = morphType;
    beatClass = GetBeatClass(morphType);

    if (beatClass != UNKNOWN)
        return beatClass;
    if (CheckPostClass(morphType) == PVC)
        return PVC;
    return tempClass;
}


/* HFNoiseCheck - gauges high frequency (muscle) noise */
#define AVELENGTH  BEAT_MS50

static int HFNoiseCheck(int *beat)
{
    int maxNoiseAve = 0, i;
    int sum = 0, aveBuff[AVELENGTH], avePtr = 0;
    int qrsMax = 0, qrsMin = 0;

    for (i = FIDMARK - BEAT_MS70; i < FIDMARK + BEAT_MS80; ++i)
        if (beat[i] > qrsMax) qrsMax = beat[i];
        else if (beat[i] < qrsMin) qrsMin = beat[i];

    for (i = 0; i < AVELENGTH; ++i)
        aveBuff[i] = 0;

    for (i = FIDMARK - BEAT_MS280; i < FIDMARK + BEAT_MS280; ++i) {
        sum -= aveBuff[avePtr];
        aveBuff[avePtr] = abs(beat[i] - (beat[i - BEAT_MS10] << 1) + beat[i - 2 * BEAT_MS10]);
        sum += aveBuff[avePtr];
        if (++avePtr == AVELENGTH)
            avePtr = 0;
        if ((i < (FIDMARK - BEAT_MS50)) || (i > (FIDMARK + BEAT_MS110)))
            if (sum > maxNoiseAve)
                maxNoiseAve = sum;
    }
    if ((qrsMax - qrsMin) >= 4)
        return ((maxNoiseAve * (50 / AVELENGTH)) / ((qrsMax - qrsMin) >> 2));
    return 0;
}


/* TempClass - classify beat based on features relative to dominant beat */
static int TempClass(int rhythmClass, int morphType,
    int beatWidth, int domWidth, int domType,
    int hfNoise, int noiseLevel, int blShift, double domIndex)
{
    (void)blShift;

    /* Rule 1: no dominant type → UNKNOWN */
    if (domType < 0)
        return UNKNOWN;

    /* Rule 2: premature + different from dominant + regular → PVC */
    if (MinimumBeatVariation(domType) && (rhythmClass == PVC)
        && (domIndex > R2_DI_THRESHOLD) && (GetDomRhythm() == 1))
        return PVC;

    /* Rule 3: narrow beat → NORMAL */
    if (beatWidth < R3_WIDTH_THRESHOLD)
        return NORMAL;

    /* Rule 5: unmatched + not premature → NORMAL (probably noisy) */
    if ((morphType == MAXTYPES) && (rhythmClass != PVC))
        return NORMAL;

    /* Rule 6: template full + new type + irregular → NORMAL */
    if ((GetTypesCount() == MAXTYPES) && (GetBeatTypeCount(morphType) == 1)
        && (rhythmClass == UNKNOWN))
        return NORMAL;

    /* Rule 7: similar to dominant + regular rhythm → NORMAL */
    if ((domIndex < R7_DI_THRESHOLD) && (rhythmClass == NORMAL))
        return NORMAL;

    /* Rule 8: close to dominant + post rhythm normal → NORMAL */
    if ((domIndex < R8_DI_THRESHOLD) && (CheckPCRhythm(morphType) == NORMAL))
        return NORMAL;

    /* Rule 9: not premature + similar to variable dominant → NORMAL */
    if ((domIndex < R9_DI_THRESHOLD) && (rhythmClass != PVC) && WideBeatVariation(domType))
        return NORMAL;

    /* Rule 10: significantly different + PVC post rhythm → PVC */
    if ((domIndex > R10_DI_THRESHOLD) && (GetBeatTypeCount(morphType) >= R10_BC_LIM) &&
        (CheckPCRhythm(morphType) == PVC) && (GetDomRhythm() == 1))
        return PVC;

    /* Rule 11: wide beat + wider than dominant → PVC */
    if ((beatWidth >= R11_MIN_WIDTH) &&
        (((beatWidth - domWidth >= R11_WIDTH_DIFF1) && (domWidth < R11_WIDTH_BREAK)) ||
        (beatWidth - domWidth >= R11_WIDTH_DIFF2)) &&
        (hfNoise < R11_HF_THRESHOLD) && (noiseLevel < R11_MA_THRESHOLD) &&
        (morphType < MAXTYPES) && (GetBeatTypeCount(morphType) > R11_BC_LIM))
        return PVC;

    /* Rule 12: regular + premature → PVC */
    if ((rhythmClass == PVC) && (GetDomRhythm() == 1))
        return PVC;

    /* Rule 14: regular rhythm → NORMAL */
    if ((rhythmClass == NORMAL) && (GetDomRhythm() == 1))
        return NORMAL;

    /* Rule 15: wider than normal + different from dominant → PVC */
    if ((beatWidth > domWidth) && (domIndex > R15_DI_THRESHOLD) &&
        (beatWidth >= R15_WIDTH_THRESHOLD))
        return PVC;

    /* Rule 16: narrow beat → NORMAL */
    if (beatWidth < R16_WIDTH_THRESHOLD)
        return NORMAL;

    /* Rule 17: not much wider than dominant → NORMAL */
    if (beatWidth < domWidth + R17_WIDTH_DELTA)
        return NORMAL;

    /* Rule 18: similar to dominant → NORMAL */
    if (domIndex < R18_DI_THRESHOLD)
        return NORMAL;

    /* Rule 19: noisy → NORMAL */
    if (hfNoise > R19_HF_THRESHOLD)
        return NORMAL;

    /* Rule 20: default → PVC */
    return PVC;
}


/* DomMonitor - monitors which beat morphology is dominant */
static int NewDom, DomRhythm;
static int DMBeatTypes[DM_BUFFER_LENGTH], DMBeatClasses[DM_BUFFER_LENGTH];
static int DMBeatRhythms[DM_BUFFER_LENGTH];
static int DMNormCounts[8], DMBeatCounts[8], DMIrregCount = 0;

void AdjustDomData(int oldType, int newType)
{
    int i;

    for (i = 0; i < DM_BUFFER_LENGTH; ++i)
        if (DMBeatTypes[i] == oldType)
            DMBeatTypes[i] = newType;

    if (newType != MAXTYPES) {
        DMNormCounts[newType] = DMNormCounts[oldType];
        DMBeatCounts[newType] = DMBeatCounts[oldType];
    }
    DMNormCounts[oldType] = DMBeatCounts[oldType] = 0;
}


void CombineDomData(int oldType, int newType)
{
    int i;

    for (i = 0; i < DM_BUFFER_LENGTH; ++i)
        if (DMBeatTypes[i] == oldType)
            DMBeatTypes[i] = newType;

    if (newType != MAXTYPES) {
        DMNormCounts[newType] += DMNormCounts[oldType];
        DMBeatCounts[newType] += DMBeatCounts[oldType];
    }
    DMNormCounts[oldType] = DMBeatCounts[oldType] = 0;
}


static int DomMonitor(int morphType, int rhythmClass, int beatWidth, int rr, int reset)
{
    static int brIndex = 0;
    int i, oldType, runCount, dom, max;

    i = brIndex - 2;
    if (i < 0) i += DM_BUFFER_LENGTH;
    oldType = DMBeatTypes[i];

    if (reset != 0) {
        for (i = 0; i < DM_BUFFER_LENGTH; ++i) {
            DMBeatTypes[i] = -1;
            DMBeatClasses[i] = 0;
        }
        for (i = 0; i < 8; ++i) {
            DMNormCounts[i] = 0;
            DMBeatCounts[i] = 0;
        }
        DMIrregCount = 0;
        return 0;
    }

    if ((DMBeatTypes[brIndex] != -1) && (DMBeatTypes[brIndex] != MAXTYPES)) {
        --DMBeatCounts[DMBeatTypes[brIndex]];
        DMNormCounts[DMBeatTypes[brIndex]] -= DMBeatClasses[brIndex];
        if (DMBeatRhythms[brIndex] == UNKNOWN)
            --DMIrregCount;
    }

    if (morphType != 8) {
        DMBeatTypes[brIndex] = morphType;
        ++DMBeatCounts[morphType];
        DMBeatRhythms[brIndex] = rhythmClass;

        if (rhythmClass == UNKNOWN)
            ++DMIrregCount;

        i = brIndex - 1;
        if (i < 0) i += DM_BUFFER_LENGTH;
        for (runCount = 0; (DMBeatTypes[i] == morphType) && (runCount < 6); ++runCount)
            if (--i < 0) i += DM_BUFFER_LENGTH;

        if ((rhythmClass == NORMAL) && (beatWidth < BEAT_MS130) && (runCount >= 1)) {
            DMBeatClasses[brIndex] = 1;
            ++DMNormCounts[morphType];
        }
        else if (rr < ((FIDMARK - GetBeatBegin(morphType)) * SAMPLE_RATE / BEAT_SAMPLE_RATE)
            && (oldType == morphType)) {
            DMBeatClasses[brIndex] = 1;
            ++DMNormCounts[morphType];
        }
        else DMBeatClasses[brIndex] = 0;
    }
    else {
        DMBeatClasses[brIndex] = 0;
        DMBeatTypes[brIndex] = -1;
    }

    if (++brIndex == DM_BUFFER_LENGTH)
        brIndex = 0;

    dom = 0;
    for (i = 1; i < 8; ++i)
        if (DMNormCounts[i] > DMNormCounts[dom])
            dom = i;

    max = 0;
    for (i = 1; i < 8; ++i)
        if (DMBeatCounts[i] > DMBeatCounts[max])
            max = i;

    if ((DMNormCounts[dom] == 0) || (DMBeatCounts[max] / DMBeatCounts[dom] >= 2))
        dom = GetDominantType();
    else if (DMBeatCounts[dom] / DMNormCounts[dom] >= 2)
        dom = GetDominantType();

    for (i = 0; i < 8; ++i)
        if ((DMBeatCounts[i] > 10) && (DMNormCounts[i] == 0) && (i != dom)
            && (GetBeatClass(i) == NORMAL))
            SetBeatClass(i, UNKNOWN);

    NewDom = dom;
    return dom;
}


int GetNewDominantType(void) { return NewDom; }

static int GetDomRhythm(void)
{
    if (DMIrregCount > IRREG_RR_LIMIT) return 0;
    return 1;
}


static int GetRunCount(void)
{
    int i;
    for (i = 1; (i < 8) && (RecentTypes[0] == RecentTypes[i]); ++i);
    return i;
}
