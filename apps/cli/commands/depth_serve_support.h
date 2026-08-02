// apps/cli/commands/depth_serve_support.h
//
// Portable validation and CPU final-RMSNorm primitives for depth-serve.

#ifndef ORION_DEPTH_SERVE_SUPPORT_H
#define ORION_DEPTH_SERVE_SUPPORT_H

#import <Foundation/Foundation.h>

#include "compiler/binding_plan.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

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

// Strict lowercase SHA-256 text helpers used by the v2 transport boundary.
bool orion_depth_serve_is_lower_hex_sha256(NSString *value);
NSString *orion_depth_serve_sha256_data(NSData *data);

// Canonical v2 protocol hashes. All string inputs must be valid protocol
// strings; digest inputs must be strict lowercase SHA-256 text.
NSString *orion_depth_serve_semantic_request_sha256(
    NSString *worker_session_id, NSString *stream_session_id,
    uint64_t logical_frame_id, NSString *request_id, uint64_t request_sequence,
    NSString *launch_challenge, NSString *challenge_nonce, uint64_t c0,
    double gamma, NSString *hidden_cond_sha256, NSString *hidden_uncond_sha256,
    NSString *program_set_digest);

// Deterministic compiler-only fixture for the typed transcript specification.
// This does not require ANE runtime or model assets.
NSString *orion_depth_serve_protocol_v2_fixture_sha256(void);
NSString *orion_depth_serve_plan_weights_sha256(const OrionBindingPlan *plan);
NSString *orion_depth_serve_program_set_sha256(
    NSString *mode, NSArray<NSDictionary *> *layers,
    NSString *final_norm_placement, NSString *final_norm_program_sha256,
    NSString *final_norm_sha256,
    NSString *audio_embeddings_sha256, NSString *audio_head_sha256,
    NSString *audio_head_placement, NSString *cfg_placement,
    NSString *sampler_placement);
NSString *orion_depth_serve_receipt_sha256(NSDictionary *receipt);

// Diagnostic companion to orion_depth_serve_receipt_sha256. Returns NULL when
// the receipt hashes successfully, otherwise a heap C string naming the first
// field or guard that rejected the receipt (for example "ok" or
// "private_invocations_actual"). Caller must free() the returned string.
// Does not weaken validation: orion_depth_serve_receipt_sha256 is unchanged.
char *orion_depth_serve_receipt_sha256_failure_field(NSDictionary *receipt);

#ifdef __cplusplus
}
#endif

#endif  // ORION_DEPTH_SERVE_SUPPORT_H
