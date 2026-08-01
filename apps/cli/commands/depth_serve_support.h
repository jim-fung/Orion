// apps/cli/commands/depth_serve_support.h
//
// Portable validation and CPU final-RMSNorm primitives for depth-serve.

#ifndef ORION_DEPTH_SERVE_SUPPORT_H
#define ORION_DEPTH_SERVE_SUPPORT_H

#import <Foundation/Foundation.h>

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    ORION_DEPTH_SERVE_SUPPORT_OK = 0,
    ORION_DEPTH_SERVE_SUPPORT_INVALID_ARGUMENT,
    ORION_DEPTH_SERVE_SUPPORT_BLOB_LENGTH,
    ORION_DEPTH_SERVE_SUPPORT_BLOB_HEADER,
    ORION_DEPTH_SERVE_SUPPORT_BLOB_PAYLOAD,
    ORION_DEPTH_SERVE_SUPPORT_BLOB_SCALE,
    ORION_DEPTH_SERVE_SUPPORT_RMS_EPSILON,
    ORION_DEPTH_SERVE_SUPPORT_RMS_INPUT,
    ORION_DEPTH_SERVE_SUPPORT_RMS_OUTPUT,
    ORION_DEPTH_SERVE_SUPPORT_RMS_ALLOCATION,
} OrionDepthServeSupportStatus;

// Validates a single-chunk BLOBFILE final-norm blob and returns a pointer to
// its fp16 scale payload. `blob` must remain alive while `out_scale` is used.
OrionDepthServeSupportStatus orion_depth_serve_final_norm_scale(
    NSData *blob, size_t channels, const _Float16 **out_scale);

// Applies RMSNorm to channel-major surface data at `slot`:
// out[c] = surface[c * seq_len + slot] / sqrt(mean(square) + epsilon) * scale[c].
// All inputs and epsilon are checked before `out` is modified.
OrionDepthServeSupportStatus orion_depth_serve_host_rmsnorm_checked(
    const _Float16 *surface, size_t channels, size_t seq_len, size_t slot,
    const _Float16 *scale, float epsilon, float *out);

const char *orion_depth_serve_support_status_string(
    OrionDepthServeSupportStatus status);

#ifdef __cplusplus
}
#endif

#endif  // ORION_DEPTH_SERVE_SUPPORT_H
