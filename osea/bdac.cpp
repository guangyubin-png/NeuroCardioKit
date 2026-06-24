/*****************************************************************************
 *  bdac.cpp - Beat Detection And Classification (main entry point)
 *  Original: Patrick S. Hamilton, E.P. Limited
 *
 *  Sample-by-sample beat detection and classification for 200 Hz ECG.
 *
 *  Public API:
 *    ResetBDAC() - reset/initialize all internal state
 *    BeatDetectAndClassify(ecgSample, &beatType, &beatMatch) - returns delay
 *****************************************************************************/

#include "qrsdet.h"
#include "bdac.h"

#define ECG_BUFFER_LENGTH  1000
#define BEAT_QUE_LENGTH    10

/* Local prototypes */
static void DownSampleBeat(int *beatOut, int *beatIn);

/* External function prototypes */
int QRSDet(int datum, int init);
int NoiseCheck(int datum, int delay, int RR, int beatBegin, int beatEnd);
int Classify(int *newBeat, int rr, int noiseLevel, int *beatMatch, int *fidAdj, int init);
int GetDominantType(void);
int GetBeatEnd(int type);
int GetBeatBegin(int type);

/* Global Variables */
static int ECGBuffer[ECG_BUFFER_LENGTH], ECGBufferIndex = 0;
static int BeatBuffer[BEATLGTH];
static int BeatQue[BEAT_QUE_LENGTH], BeatQueCount = 0;
static int RRCount = 0;
static int InitBeatFlag = 1;


void ResetBDAC(void)
{
    int dummy;
    QRSDet(0, 1);  /* Reset QRS detector */
    RRCount = 0;
    Classify(BeatBuffer, 0, 0, &dummy, &dummy, 1);
    InitBeatFlag = 1;
    BeatQueCount = 0;
}


int BeatDetectAndClassify(int ecgSample, int *beatType, int *beatMatch)
{
    int detectDelay, rr = 0, i, j;
    int noiseEst = 0, beatBegin = 0, beatEnd = 0;
    int domType;
    int fidAdj;
    int tempBeat[(SAMPLE_RATE / BEAT_SAMPLE_RATE) * BEATLGTH];

    /* Store new sample in circular buffer */
    ECGBuffer[ECGBufferIndex] = ecgSample;
    if (++ECGBufferIndex == ECG_BUFFER_LENGTH)
        ECGBufferIndex = 0;

    /* Increment RR interval count */
    ++RRCount;

    /* Increment detection delays for queued beats */
    for (i = 0; i < BeatQueCount; ++i)
        ++BeatQue[i];

    /* Run sample through QRS detector */
    detectDelay = QRSDet(ecgSample, 0);
    if (detectDelay != 0) {
        BeatQue[BeatQueCount] = detectDelay;
        ++BeatQueCount;
    }

    /* Return if no beat ready for classification */
    if ((BeatQue[0] < (BEATLGTH - FIDMARK) * (SAMPLE_RATE / BEAT_SAMPLE_RATE))
        || (BeatQueCount == 0)) {
        NoiseCheck(ecgSample, 0, rr, beatBegin, beatEnd);
        return 0;
    }

    /* Classify the beat at the head of the queue */
    rr = RRCount - BeatQue[0];
    detectDelay = RRCount = BeatQue[0];

    /* Estimate low frequency noise */
    domType = GetDominantType();
    if (domType == -1) {
        beatBegin = MS250;
        beatEnd = MS300;
    }
    else {
        beatBegin = (SAMPLE_RATE / BEAT_SAMPLE_RATE) * (FIDMARK - GetBeatBegin(domType));
        beatEnd = (SAMPLE_RATE / BEAT_SAMPLE_RATE) * (GetBeatEnd(domType) - FIDMARK);
    }
    noiseEst = NoiseCheck(ecgSample, detectDelay, rr, beatBegin, beatEnd);

    /* Copy beat from circular buffer and downsample */
    j = ECGBufferIndex - detectDelay - (SAMPLE_RATE / BEAT_SAMPLE_RATE) * FIDMARK;
    if (j < 0) j += ECG_BUFFER_LENGTH;

    for (i = 0; i < (SAMPLE_RATE / BEAT_SAMPLE_RATE) * BEATLGTH; ++i) {
        tempBeat[i] = ECGBuffer[j];
        if (++j == ECG_BUFFER_LENGTH)
            j = 0;
    }

    DownSampleBeat(BeatBuffer, tempBeat);

    /* Update the queue */
    for (i = 0; i < BeatQueCount - 1; ++i)
        BeatQue[i] = BeatQue[i + 1];
    --BeatQueCount;

    /* Skip the first beat */
    if (InitBeatFlag) {
        InitBeatFlag = 0;
        *beatType = 13;  /* UNKNOWN */
        *beatMatch = 0;
        fidAdj = 0;
    }
    else {
        *beatType = Classify(BeatBuffer, rr, noiseEst, beatMatch, &fidAdj, 0);
        fidAdj *= SAMPLE_RATE / BEAT_SAMPLE_RATE;
    }

    /* Ignore trailing edge of PVC */
    if (*beatType == 100) {
        RRCount += rr;
        return 0;
    }

    /* Limit fiducial mark adjustment */
    if (fidAdj > MS80)
        fidAdj = MS80;
    else if (fidAdj < -MS80)
        fidAdj = -MS80;

    return detectDelay - fidAdj;
}


static void DownSampleBeat(int *beatOut, int *beatIn)
{
    int i;
    for (i = 0; i < BEATLGTH; ++i)
        beatOut[i] = (beatIn[i << 1] + beatIn[(i << 1) + 1]) >> 1;
}
