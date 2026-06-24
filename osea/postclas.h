/*****************************************************************************
 *  postclas.h - Post classifier prototype definitions
 *****************************************************************************/
#ifndef OSEA_POSTCLAS_H
#define OSEA_POSTCLAS_H

void ResetPostClassify(void);
void PostClassify(int *recentTypes, int domType, int *recentRRs,
    int width, double mi2, int rhythmClass);
int CheckPostClass(int type);
int CheckPCRhythm(int type);

#endif
