#ifndef OPJ_CONFIG_PRIVATE_H_INCLUDED
#define OPJ_CONFIG_PRIVATE_H_INCLUDED

/* Minimal private config header for standalone Makefile builds.
 * This repository normally generates this file via CMake.
 */

#define OPJ_PACKAGE_VERSION "0"

/* Leave feature-detection macros undefined by default.
 * The library has fallbacks when these are not set.
 */

/* #define OPJ_BIG_ENDIAN */

#endif
