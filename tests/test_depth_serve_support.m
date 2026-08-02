// tests/test_depth_serve_support.m
//
// Portable tests for depth-serve final-norm blob validation and CPU RMSNorm.

#import <Foundation/Foundation.h>

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "apps/cli/commands/depth_serve_support.h"

static int g_passed = 0;
static int g_failed = 0;

#define TEST(name) do { \
    printf("  %-48s ", #name); \
    if (test_##name()) { printf("PASS\n"); g_passed++; } \
    else { printf("FAIL\n"); g_failed++; } \
} while (0)

static void write_le32(uint8_t *dst, uint32_t value) {
    dst[0] = (uint8_t)value;
    dst[1] = (uint8_t)(value >> 8);
    dst[2] = (uint8_t)(value >> 16);
    dst[3] = (uint8_t)(value >> 24);
}

static NSMutableData *valid_final_norm_blob(size_t channels) {
    NSMutableData *blob = [NSMutableData dataWithLength:128 + channels * sizeof(_Float16)];
    uint8_t *bytes = blob.mutableBytes;
    bytes[0] = 1;
    bytes[4] = 2;
    bytes[64] = 0xef;
    bytes[65] = 0xbe;
    bytes[66] = 0xad;
    bytes[67] = 0xde;
    bytes[68] = 1;
    write_le32(bytes + 72, (uint32_t)(channels * sizeof(_Float16)));
    write_le32(bytes + 80, 128);
    _Float16 *scale = (_Float16 *)(bytes + 128);
    for (size_t c = 0; c < channels; c++) scale[c] = (_Float16)(1.0f + c * 0.25f);
    return blob;
}

static bool test_final_norm_blob_accepts_valid_scale(void) {
    NSMutableData *blob = valid_final_norm_blob(4);
    const _Float16 *scale = NULL;
    OrionDepthServeSupportStatus status =
        orion_depth_serve_final_norm_scale(blob, 4, &scale);
    return status == ORION_DEPTH_SERVE_SUPPORT_OK && scale &&
        fabsf((float)scale[0] - 1.0f) < 1e-4f &&
        fabsf((float)scale[3] - 1.75f) < 1e-4f;
}

static bool test_final_norm_blob_rejects_bad_length(void) {
    NSMutableData *blob = valid_final_norm_blob(4);
    [blob setLength:blob.length - 1];
    const _Float16 *scale = (_Float16 *)(uintptr_t)1;
    return orion_depth_serve_final_norm_scale(blob, 4, &scale) ==
               ORION_DEPTH_SERVE_SUPPORT_BLOB_LENGTH &&
           scale == NULL;
}

static bool test_final_norm_blob_rejects_bad_header(void) {
    NSMutableData *blob = valid_final_norm_blob(4);
    ((uint8_t *)blob.mutableBytes)[64] = 0;
    const _Float16 *scale = NULL;
    return orion_depth_serve_final_norm_scale(blob, 4, &scale) ==
        ORION_DEPTH_SERVE_SUPPORT_BLOB_HEADER;
}

static bool test_final_norm_blob_rejects_bad_payload_descriptor(void) {
    NSMutableData *blob = valid_final_norm_blob(4);
    write_le32((uint8_t *)blob.mutableBytes + 80, 64);
    const _Float16 *scale = NULL;
    return orion_depth_serve_final_norm_scale(blob, 4, &scale) ==
        ORION_DEPTH_SERVE_SUPPORT_BLOB_PAYLOAD;
}

static bool test_final_norm_blob_rejects_nonfinite_scales(void) {
    NSMutableData *nan_blob = valid_final_norm_blob(2);
    ((_Float16 *)((uint8_t *)nan_blob.mutableBytes + 128))[0] = (_Float16)NAN;
    const _Float16 *scale = (_Float16 *)(uintptr_t)1;
    if (orion_depth_serve_final_norm_scale(nan_blob, 2, &scale) !=
            ORION_DEPTH_SERVE_SUPPORT_BLOB_SCALE ||
        scale != NULL) {
        return false;
    }

    NSMutableData *infinity_blob = valid_final_norm_blob(2);
    ((_Float16 *)((uint8_t *)infinity_blob.mutableBytes + 128))[1] =
        (_Float16)INFINITY;
    scale = (_Float16 *)(uintptr_t)1;
    return orion_depth_serve_final_norm_scale(infinity_blob, 2, &scale) ==
               ORION_DEPTH_SERVE_SUPPORT_BLOB_SCALE &&
           scale == NULL;
}

static bool test_host_rmsnorm_reads_channel_major_slot(void) {
    const _Float16 surface[] = {
        99.0, 3.0, 99.0,
        99.0, 4.0, 99.0,
    };
    const _Float16 scale[] = {2.0, 0.5};
    float out[] = {0.0f, 0.0f};
    OrionDepthServeSupportStatus status = orion_depth_serve_host_rmsnorm_checked(
        surface, 2, 3, 1, scale, 1e-5f, out);
    const float denom = sqrtf((9.0f + 16.0f) / 2.0f + 1e-5f);
    return status == ORION_DEPTH_SERVE_SUPPORT_OK &&
        fabsf(out[0] - 3.0f / denom * 2.0f) < 1e-5f &&
        fabsf(out[1] - 4.0f / denom * 0.5f) < 1e-5f;
}

static bool test_host_rmsnorm_handles_maximum_finite_scale(void) {
    const _Float16 surface[] = {65504.0, 65504.0};
    const _Float16 scale[] = {65504.0, 65504.0};
    float out[] = {0.0f, 0.0f};
    OrionDepthServeSupportStatus status = orion_depth_serve_host_rmsnorm_checked(
        surface, 2, 1, 0, scale, 1e-5f, out);
    return status == ORION_DEPTH_SERVE_SUPPORT_OK &&
        isfinite(out[0]) && isfinite(out[1]) &&
        fabsf(out[0] - 65504.0f) < 1.0f &&
        fabsf(out[1] - 65504.0f) < 1.0f;
}

static bool test_host_rmsnorm_rejects_invalid_args_without_write(void) {
    const _Float16 surface[] = {1.0, 2.0};
    const _Float16 scale[] = {1.0, 1.0};
    float out[] = {123.0f, 456.0f};
    OrionDepthServeSupportStatus status = orion_depth_serve_host_rmsnorm_checked(
        surface, 2, 1, 1, scale, 1e-5f, out);
    return status == ORION_DEPTH_SERVE_SUPPORT_INVALID_ARGUMENT &&
        out[0] == 123.0f && out[1] == 456.0f;
}

static bool test_host_rmsnorm_rejects_nonfinite_inputs_without_write(void) {
    const _Float16 surface[] = {NAN, 2.0};
    const _Float16 scale[] = {1.0, 1.0};
    float out[] = {123.0f, 456.0f};
    OrionDepthServeSupportStatus status = orion_depth_serve_host_rmsnorm_checked(
        surface, 2, 1, 0, scale, 1e-5f, out);
    return status == ORION_DEPTH_SERVE_SUPPORT_RMS_INPUT &&
        out[0] == 123.0f && out[1] == 456.0f;
}

static bool test_host_rmsnorm_rejects_invalid_epsilon_without_write(void) {
    const _Float16 surface[] = {1.0, 2.0};
    const _Float16 scale[] = {1.0, 1.0};
    float out[] = {123.0f, 456.0f};
    OrionDepthServeSupportStatus status = orion_depth_serve_host_rmsnorm_checked(
        surface, 2, 1, 0, scale, 0.0f, out);
    return status == ORION_DEPTH_SERVE_SUPPORT_RMS_EPSILON &&
        out[0] == 123.0f && out[1] == 456.0f;
}

// C _Bool has a distinct NSNumber boxing from CFBoolean and numeric NSNumber.
// Verify that the typed _Bool boxing hashes identically to the CFBoolean form.
static NSMutableDictionary *receipt_with_ok_boxing(id ok_value) {
    return [@{
        @"receipt_schema": @"voxtream.orion.depth-operational-receipt/2",
        @"ok": ok_value,
        @"worker_session_id": @"550e8400-e29b-41d4-a716-446655440000",
        @"stream_session_id": @"stream-alpha",
        @"logical_frame_id": @42,
        @"request_id": @"request-000042",
        @"request_sequence": @7,
        @"challenge_nonce": @"abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
        @"hidden_cond_sha256": @"1111111111111111111111111111111111111111111111111111111111111111",
        @"hidden_uncond_sha256": @"2222222222222222222222222222222222222222222222222222222222222222",
        @"semantic_request_sha256": @"53808229003275463352dc3f86f60f27e50b220b39e06ba42933b4acc787fc3a",
        @"program_set_digest": @"3333333333333333333333333333333333333333333333333333333333333333",
        @"previous_receipt_sha256": @"0000000000000000000000000000000000000000000000000000000000000000",
        @"private_invocation_known": @YES,
        @"private_invocations_expected": @60,
        @"private_invocations_actual": @60,
        @"private_invocations_successful": @60,
        @"private_invocations_failed": @0,
        @"private_invocations_by_program": @{
            @"layer0": @{@"expected": @15, @"actual": @15, @"successful": @15, @"failed": @0},
            @"layer1": @{@"expected": @15, @"actual": @15, @"successful": @15, @"failed": @0},
            @"layer2": @{@"expected": @15, @"actual": @15, @"successful": @15, @"failed": @0},
            @"layer3": @{@"expected": @15, @"actual": @15, @"successful": @15, @"failed": @0},
            @"final_norm": @{@"expected": @0, @"actual": @0, @"successful": @0, @"failed": @0},
        },
        @"compile_count_before": @4,
        @"compile_count_after": @4,
        @"compile_count_delta": @0,
        @"native_fallback_used": @NO,
        @"frame_start_monotonic_ns": @1000000,
        @"frame_end_monotonic_ns": @2000000,
        @"frame_duration_ns": @1000000,
        @"output_placement": @"host-memory+atomic-tokens-bin",
        @"tokens": @[@100, @101, @102, @103, @104, @105, @106, @107, @108, @109,
                     @110, @111, @112, @113, @114],
        @"token_sha256": @"904db4399aa10326d1155440f63b38ca363af85c3490684f51847084f9d11c3d",
        @"tokens_bin_sha256": @"904db4399aa10326d1155440f63b38ca363af85c3490684f51847084f9d11c3d",
    } mutableCopy];
}

static bool test_receipt_accepts_c_bool_ok_boxing(void) {
    _Bool c_bool = 1;
    id bool_expr = @(c_bool);
    NSMutableDictionary *r = receipt_with_ok_boxing(bool_expr);
    NSString *sha = orion_depth_serve_receipt_sha256(r);
    // Must hash, and must equal the @YES form (semantically identical).
    NSMutableDictionary *r_yes = receipt_with_ok_boxing(@YES);
    NSString *sha_yes = orion_depth_serve_receipt_sha256(r_yes);
    return sha != nil && sha_yes != nil &&
        [sha isEqualToString:sha_yes] &&
        [sha isEqualToString:
            @"e60f295cc2515e969fbc60f47ac87616ad3ff3f8c4325055b61f969d33b74d76"];
}

static bool test_receipt_rejects_numeric_one_in_ok_slot(void) {
    // Numeric NSNumber values must not be silently accepted as Booleans, even
    // when their value is 1. Only CFBoolean and boxed C _Bool are valid.
    NSMutableDictionary *r = receipt_with_ok_boxing(@1);
    return orion_depth_serve_receipt_sha256(r) == nil;
}

static bool test_receipt_diagnostic_names_the_failing_field(void) {
    NSMutableDictionary *r = receipt_with_ok_boxing(@YES);
    [r removeObjectForKey:@"private_invocations_actual"];
    char *field = orion_depth_serve_receipt_sha256_failure_field(r);
    bool ok = field != NULL && strcmp(field, "private_invocations_actual") == 0;
    if (field) free(field);
    return ok;
}

int main(void) {
    @autoreleasepool {
        printf("test_depth_serve_support:\n");
        TEST(final_norm_blob_accepts_valid_scale);
        TEST(final_norm_blob_rejects_bad_length);
        TEST(final_norm_blob_rejects_bad_header);
        TEST(final_norm_blob_rejects_bad_payload_descriptor);
        TEST(final_norm_blob_rejects_nonfinite_scales);
        TEST(host_rmsnorm_reads_channel_major_slot);
        TEST(host_rmsnorm_handles_maximum_finite_scale);
        TEST(host_rmsnorm_rejects_invalid_args_without_write);
        TEST(host_rmsnorm_rejects_nonfinite_inputs_without_write);
        TEST(host_rmsnorm_rejects_invalid_epsilon_without_write);
        TEST(receipt_accepts_c_bool_ok_boxing);
        TEST(receipt_rejects_numeric_one_in_ok_slot);
        TEST(receipt_diagnostic_names_the_failing_field);
        printf("\n%d/%d passed\n", g_passed, g_passed + g_failed);
        return g_failed ? 1 : 0;
    }
}
