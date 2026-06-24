/*****************************************************************************
 *  osea.h  -  Open Source ECG Analysis public API
 *
 *  C-compatible wrapper around the Hamilton-Tompkins QRS detector and
 *  beat classifier from E.P. Limited (Patrick S. Hamilton).
 *
 *  Input: ECG sampled at 200 Hz, normalized to 200 ADC units per mV
 *         (i.e. 5 uV/LSB, or 200 units = 1 mV).
 *
 *  Usage:
 *    1. Call osea_reset() once at the start.
 *    2. Feed samples one-by-one via osea_detect(sample, &type, &match).
 *    3. When return value > 0, a beat was detected.
 *       - Return value = detection delay (samples since R-wave).
 *       - type = beat annotation type (NORMAL=1, PVC=5, etc.).
 *       - match = template index (0-7, for debugging).
 *
 *  Annotation codes (from MIT/BIH standard):
 *    NORMAL=1, LBBB=2, RBBB=3, ABERR=4, PVC=5, FUSION=6,
 *    NPC=7, APC=8, SVPB=9, VESC=10, NESC=11, PACE=12, UNKNOWN=13
 *****************************************************************************/

#ifndef OSEA_H
#define OSEA_H

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
  #ifdef OSEA_BUILD_DLL
    #define OSEA_API __declspec(dllexport)
  #else
    #define OSEA_API __declspec(dllimport)
  #endif
#else
  #define OSEA_API
#endif

/**
 * Reset the beat detection and classification state.
 * Must be called once before processing a new ECG signal.
 */
OSEA_API void osea_reset(void);

/**
 * Process one ECG sample (200 Hz, 200 ADC units/mV).
 *
 * @param ecgSample  Raw ECG sample value (baseline ~0).
 * @param beatType   [out] Annotation type of detected beat (NORMAL=1, PVC=5, etc.).
 * @param beatMatch  [out] Template index that the beat matched (0-7, debug use).
 * @return           Detection delay (>0 if beat detected, 0 if no beat).
 *                   The delay is the number of samples since the R-wave peak.
 */
OSEA_API int osea_detect(int ecgSample, int *beatType, int *beatMatch);

/**
 * Low-level QRS detection only (no classification).
 * Same as osea_detect but only runs the QRS detector.
 *
 * @param datum  ECG sample.
 * @param init   Non-zero to reset internal state.
 * @return       Detection delay (>0 if QRS detected, 0 otherwise).
 */
OSEA_API int osea_qrs_det(int datum, int init);

#ifdef __cplusplus
}
#endif

#endif /* OSEA_H */
