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
        printf("\n%d/%d passed\n", g_passed, g_passed + g_failed);
        return g_failed ? 1 : 0;
    }
}
