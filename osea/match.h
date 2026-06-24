/*****************************************************************************
 *  match.h - Beat matching prototype definitions
 *****************************************************************************/
#ifndef OSEA_MATCH_H
#define OSEA_MATCH_H

void ResetMatch(void);
void BestMorphMatch(int *newBeat, int *matchType, double *matchIndex, double *mi2, int *shiftAdj);
void UpdateBeatType(int matchType, int *newBeat, double mi2, int shiftAdj);
int NewBeatType(int *beat);
int GetTypesCount(void);
int GetBeatTypeCount(int type);
int GetBeatWidth(int type);
int GetBeatClass(int type);
void SetBeatClass(int type, int beatClass);
int GetDominantType(void);
void ClearLastNewType(void);
int GetBeatBegin(int type);
int GetBeatEnd(int type);
int GetBeatAmp(int type);
int GetBeatCenter(int type);
int MinimumBeatVariation(int type);
int WideBeatVariation(int type);
double DomCompare2(int *newBeat, int domType);
double DomCompare(int newType, int domType);

#endif
