/*****************************************************************************
 *  osea_bridge.cpp - DLL entry point
 *
 *  Single-compilation-unit wrapper that bundles all OSEA sources and
 *  exposes a flat C API via extern "C".
 *
 *  Compile (MSVC):
 *    cl /LD /O2 /DOSEA_BUILD_DLL osea_bridge.cpp /Feosea.dll
 *
 *  Compile (MinGW):
 *    g++ -shared -O2 -DOSEA_BUILD_DLL -o osea.dll osea_bridge.cpp
 *****************************************************************************/

#define OSEA_BUILD_DLL
#include "osea.h"

/* Include all implementation files as a single translation unit.
   All internal functions have been made static where possible. */

#include "qrsfilt.cpp"
#include "qrsdet.cpp"
#include "analbeat.cpp"
#include "noisechk.cpp"
#include "rythmchk.cpp"
#include "postclas.cpp"
#include "match.cpp"
#include "classify.cpp"
#include "bdac.cpp"

/* ------ Public C API (extern "C" is provided by osea.h) ------ */

void osea_reset(void)
{
    ResetBDAC();
}

int osea_detect(int ecgSample, int *beatType, int *beatMatch)
{
    return BeatDetectAndClassify(ecgSample, beatType, beatMatch);
}

int osea_qrs_det(int datum, int init)
{
    return QRSDet(datum, init);
}
