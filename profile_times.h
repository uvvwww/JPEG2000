#ifndef PROFILE_TIMES_H
#define PROFILE_TIMES_H

#include <stdio.h>

extern double opj_clock(void);

struct TimingData {
    double dc_shift_time;
    double mct_time;
    double dwt_time;
    double t1_time;
    double rate_time;
    double t2_time;
};

extern struct TimingData global_timing;

#define PROF_START(var) double var##_start = opj_clock()
#define PROF_STOP(var, field) global_timing.field += (opj_clock() - var##_start)

#endif
