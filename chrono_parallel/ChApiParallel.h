#ifndef CHAPIPARALLEL_H
#define CHAPIPARALLEL_H

//////////////////////////////////////////////////
//
//   ChApiParallel.h
//
//   Base header for all headers that have symbols
//   that can be exported.
//
//   HEADER file for CHRONO,
//   Multibody dynamics engine
//
// ------------------------------------------------
//   Copyright:Alessandro Tasora / DeltaKnowledge
//             www.deltaknowledge.com
// ------------------------------------------------
///////////////////////////////////////////////////

#include "core/ChPlatform.h"

// Chrono::Engine unit GPU, version
//
// This is an integer, as 0xaabbccdd where
// for example version 1.2.0 is 0x00010200

#define CH_VERSION_UNIT_GPU 0x00010200

// When compiling this library, remember to define CH_API_COMPILE_PARALLEL
// (so that the symbols with 'CH_PARALLEL_API' in front of them will be
// marked as exported). Otherwise, just do not define it if you
// link the library to your code, and the symbols will be imported.

#if defined(CH_API_COMPILE_PARALLEL)
#define CH_PARALLEL_API ChApiEXPORT
#else
#define CH_PARALLEL_API ChApiIMPORT
#endif


// Macros for specifying type alignment
#if (defined __GNUC__) || (defined __INTEL_COMPILER)
	#define CHRONO_ALIGN_16 __attribute__ ((aligned(16)))
#elif defined _MSC_VER
	#define CHRONO_ALIGN_16 __declspec(align(16))
#else
	#define CHRONO_ALIGN_16
#endif


#if defined _MSC_VER
#define fmax max
#define fmin min
#endif



#endif  // END of header


