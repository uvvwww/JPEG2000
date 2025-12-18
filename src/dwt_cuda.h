/*
 * CUDA-accelerated Discrete Wavelet Transform (DWT) for JPEG2000
 * 
 * This header provides GPU-accelerated versions of DWT functions
 * for both reversible (5-3) and irreversible (9-7) transforms.
 */

#ifndef OPJ_DWT_CUDA_H
#define OPJ_DWT_CUDA_H

#include "opj_includes.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Forward 5-3 wavelet transform in 2-D (CUDA version).
 * Apply a reversible DWT transform to a component of an image on GPU.
 * @param p_tcd TCD handle
 * @param tilec Tile component information (current tile)
 * @return OPJ_TRUE on success, OPJ_FALSE on failure
 */
OPJ_BOOL opj_dwt_encode_cuda(opj_tcd_t *p_tcd,
                             opj_tcd_tilecomp_t * tilec);

/**
 * Inverse 5-3 wavelet transform in 2-D (CUDA version).
 * Apply a reversible inverse DWT transform to a component of an image on GPU.
 * @param p_tcd TCD handle
 * @param tilec Tile component information (current tile)
 * @param numres Number of resolution levels to decode
 * @return OPJ_TRUE on success, OPJ_FALSE on failure
 */
OPJ_BOOL opj_dwt_decode_cuda(opj_tcd_t *p_tcd,
                             opj_tcd_tilecomp_t* tilec,
                             OPJ_UINT32 numres);

/**
 * Forward 9-7 wavelet transform in 2-D (CUDA version).
 * Apply an irreversible DWT transform to a component of an image on GPU.
 * @param p_tcd TCD handle
 * @param tilec Tile component information (current tile)
 * @return OPJ_TRUE on success, OPJ_FALSE on failure
 */
OPJ_BOOL opj_dwt_encode_real_cuda(opj_tcd_t *p_tcd,
                                  opj_tcd_tilecomp_t * tilec);

/**
 * Inverse 9-7 wavelet transform in 2-D (CUDA version).
 * Apply an irreversible inverse DWT transform to a component of an image on GPU.
 * @param p_tcd TCD handle
 * @param tilec Tile component information (current tile)
 * @param numres Number of resolution levels to decode
 * @return OPJ_TRUE on success, OPJ_FALSE on failure
 */
OPJ_BOOL opj_dwt_decode_real_cuda(opj_tcd_t *p_tcd,
                                  opj_tcd_tilecomp_t* tilec,
                                  OPJ_UINT32 numres);

/**
 * Initialize CUDA device and check availability.
 * @return OPJ_TRUE if CUDA device is available, OPJ_FALSE otherwise
 */
OPJ_BOOL opj_dwt_cuda_init(void);

/**
 * Cleanup CUDA resources.
 */
void opj_dwt_cuda_cleanup(void);

#ifdef __cplusplus
}
#endif

#endif /* OPJ_DWT_CUDA_H */
