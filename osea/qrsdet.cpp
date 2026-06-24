/*****************************************************************************
 *  qrsdet.cpp - QRS detector (Hamilton-Tompkins algorithm)
 *  Original: Patrick S. Hamilton, E.P. Limited
 *
 *  Sample-by-sample QRS detection for 200 Hz ECG data.
 *  QRSDet() returns detection delay when a QRS is found, or 0.
 *****************************************************************************/

#include <string.h>  /* for memmove */
#include <stdlib.h>  /* for abs */
#include "qrsdet.h"

/* External prototypes */
int QRSFilter(int datum, int init);
static int deriv1(int x0, int init);

/* Local prototypes */
static int Peak(int datum, int init);
static int median(int *array, int datnum);
static int thresh(int qmedian, int nmedian);
static int BLSCheck(int *dBuf, int dbPtr, int *maxder);
/* deriv1() is defined in qrsfilt.cpp (included in single-TU build) */

static double TH = 0.475;
static int DDBuffer[DER_DELAY], DDPtr;
static int Dly = 0;
static const int MEMMOVELEN = 7 * sizeof(int);

int QRSDet(int datum, int init)
{
    static int det_thresh, qpkcnt = 0;
    static int qrsbuf[8], noise[8], rrbuf[8];
    static int rsetBuff[8], rsetCount = 0;
    static int nmedian, qmedian, rrmedian;
    static int count, sbpeak = 0, sbloc, sbcount = MS1500;
    static int maxder, lastmax;
    static int initBlank, initMax;
    static int preBlankCnt, tempPeak;

    int fdatum, QrsDelay = 0;
    int i, newPeak, aPeak;

    if (init) {
        for (i = 0; i < 8; ++i) {
            noise[i] = 0;
            rrbuf[i] = MS1000;
        }
        qpkcnt = maxder = lastmax = count = sbpeak = 0;
        initBlank = initMax = preBlankCnt = DDPtr = 0;
        sbcount = MS1500;
        QRSFilter(0, 1);
        Peak(0, 1);
    }

    fdatum = QRSFilter(datum, 0);
    aPeak = Peak(fdatum, 0);

    /* Hold any peak for 200 ms in case a bigger one comes along */
    newPeak = 0;
    if (aPeak && !preBlankCnt) {
        tempPeak = aPeak;
        preBlankCnt = PRE_BLANK;
    }
    else if (!aPeak && preBlankCnt) {
        if (--preBlankCnt == 0)
            newPeak = tempPeak;
    }
    else if (aPeak) {
        if (aPeak > tempPeak) {
            tempPeak = aPeak;
            preBlankCnt = PRE_BLANK;
        }
        else if (--preBlankCnt == 0)
            newPeak = tempPeak;
    }

    /* Save derivative for T-wave and baseline shift discrimination */
    DDBuffer[DDPtr] = deriv1(datum, 0);
    if (++DDPtr == DER_DELAY)
        DDPtr = 0;

    /* Initialize qrs peak buffer with first eight local max peaks */
    if (qpkcnt < 8) {
        ++count;
        if (newPeak > 0) count = WINDOW_WIDTH;
        if (++initBlank == MS1000) {
            initBlank = 0;
            qrsbuf[qpkcnt] = initMax;
            initMax = 0;
            ++qpkcnt;
            if (qpkcnt == 8) {
                qmedian = median(qrsbuf, 8);
                nmedian = 0;
                rrmedian = MS1000;
                sbcount = MS1500 + MS150;
                det_thresh = thresh(qmedian, nmedian);
            }
        }
        if (newPeak > initMax)
            initMax = newPeak;
    }
    else {
        ++count;
        if (newPeak > 0) {
            if (!BLSCheck(DDBuffer, DDPtr, &maxder)) {
                if (newPeak > det_thresh) {
                    memmove(&qrsbuf[1], qrsbuf, MEMMOVELEN);
                    qrsbuf[0] = newPeak;
                    qmedian = median(qrsbuf, 8);
                    det_thresh = thresh(qmedian, nmedian);
                    memmove(&rrbuf[1], rrbuf, MEMMOVELEN);
                    rrbuf[0] = count - WINDOW_WIDTH;
                    rrmedian = median(rrbuf, 8);
                    sbcount = rrmedian + (rrmedian >> 1) + WINDOW_WIDTH;
                    count = WINDOW_WIDTH;
                    sbpeak = 0;
                    lastmax = maxder;
                    maxder = 0;
                    QrsDelay = WINDOW_WIDTH + FILTER_DELAY;
                    initBlank = initMax = rsetCount = 0;
                }
                else {
                    memmove(&noise[1], noise, MEMMOVELEN);
                    noise[0] = newPeak;
                    nmedian = median(noise, 8);
                    det_thresh = thresh(qmedian, nmedian);

                    if ((newPeak > sbpeak) && ((count - WINDOW_WIDTH) >= MS360)) {
                        sbpeak = newPeak;
                        sbloc = count - WINDOW_WIDTH;
                    }
                }
            }
        }

        /* Search back condition */
        if ((count > sbcount) && (sbpeak > (det_thresh >> 1))) {
            memmove(&qrsbuf[1], qrsbuf, MEMMOVELEN);
            qrsbuf[0] = sbpeak;
            qmedian = median(qrsbuf, 8);
            det_thresh = thresh(qmedian, nmedian);
            memmove(&rrbuf[1], rrbuf, MEMMOVELEN);
            rrbuf[0] = sbloc;
            rrmedian = median(rrbuf, 8);
            sbcount = rrmedian + (rrmedian >> 1) + WINDOW_WIDTH;
            QrsDelay = count = count - sbloc;
            QrsDelay += FILTER_DELAY;
            sbpeak = 0;
            lastmax = maxder;
            maxder = 0;
            initBlank = initMax = rsetCount = 0;
        }
    }

    /* Background threshold replacement if 8 sec without detection */
    if (qpkcnt == 8) {
        if (++initBlank == MS1000) {
            initBlank = 0;
            rsetBuff[rsetCount] = initMax;
            initMax = 0;
            ++rsetCount;

            if (rsetCount == 8) {
                for (i = 0; i < 8; ++i) {
                    qrsbuf[i] = rsetBuff[i];
                    noise[i] = 0;
                }
                qmedian = median(rsetBuff, 8);
                nmedian = 0;
                rrmedian = MS1000;
                sbcount = MS1500 + MS150;
                det_thresh = thresh(qmedian, nmedian);
                initBlank = initMax = rsetCount = 0;
                sbpeak = 0;
            }
        }
        if (newPeak > initMax)
            initMax = newPeak;
    }

    return QrsDelay;
}


/* Peak() - returns peak height when signal returns to half its peak height */
static int Peak(int datum, int init)
{
    static int max = 0, timeSinceMax = 0, lastDatum;
    int pk = 0;

    if (init)
        max = timeSinceMax = 0;

    if (timeSinceMax > 0)
        ++timeSinceMax;

    if ((datum > lastDatum) && (datum > max)) {
        max = datum;
        if (max > 2)
            timeSinceMax = 1;
    }
    else if (datum < (max >> 1)) {
        pk = max;
        max = 0;
        timeSinceMax = 0;
        Dly = 0;
    }
    else if (timeSinceMax > MS95) {
        pk = max;
        max = 0;
        timeSinceMax = 0;
        Dly = 3;
    }
    lastDatum = datum;
    return pk;
}


/* median() - returns median of a small integer array */
static int median(int *array, int datnum)
{
    int i, j, k, temp, sort[20];
    for (i = 0; i < datnum; ++i)
        sort[i] = array[i];
    for (i = 0; i < datnum; ++i) {
        temp = sort[i];
        for (j = 0; (temp < sort[j]) && (j < i); ++j);
        for (k = i - 1; k >= j; --k)
            sort[k + 1] = sort[k];
        sort[j] = temp;
    }
    return sort[datnum >> 1];
}


/* thresh() - calculates detection threshold from QRS and noise medians */
static int thresh(int qmedian, int nmedian)
{
    int thrsh, dmed;
    double temp;
    dmed = qmedian - nmedian;
    temp = dmed;
    temp *= TH;
    dmed = (int)temp;
    thrsh = nmedian + dmed;
    return thrsh;
}


/* BLSCheck() - baseline shift check (currently disabled, returns 0) */
static int BLSCheck(int *dBuf, int dbPtr, int *maxder)
{
    (void)dBuf;
    (void)dbPtr;
    (void)maxder;
    return 0;
}


/* deriv1() is provided by qrsfilt.cpp */
