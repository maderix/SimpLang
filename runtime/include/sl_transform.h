#ifndef SL_TRANSFORM_H
#define SL_TRANSFORM_H

#include "kernel.h"

// Only apply to SimpleLang source files
#ifndef __cplusplus

// Rename main to sl_main in SimpleLang source files
#ifdef TEST_SIMD
#define main(sse, avx) sl_main(sse, avx)
#else
#define main sl_main
#endif

#endif // !__cplusplus

#endif // SL_TRANSFORM_H