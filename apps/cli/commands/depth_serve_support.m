// apps/cli/commands/depth_serve_support.m

#import "apps/cli/commands/depth_serve_support.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define ORION_DEPTH_SERVE_BLOB_HEADER_BYTES 128u
#define ORION_DEPTH_SERVE_CHUNK_HEADER_OFFSET 64u
#define ORION_DEPTH_SERVE_CHUNK_MAGIC 0xdeadbeefu
#define ORION_DEPTH_SERVE_CHUNK_PAYLOAD_BYTES_OFFSET 72u
#define ORION_DEPTH_SERVE_CHUNK_PAYLOAD_OFFSET_OFFSET 80u

static uint32_t read_le32(const uint8_t *src) {
    return (uint32_t)src[0] | ((uint32_t)src[1] << 8) |
           ((uint32_t)src[2] << 16) | ((uint32_t)src[3] << 24);
}

OrionDepthServeSupportStatus orion_depth_serve_final_norm_scale(
    NSData *blob, size_t channels, const _Float16 **out_scale) {
    if (!out_scale || channels == 0 ||
        channels > UINT32_MAX / sizeof(_Float16)) {
        return ORION_DEPTH_SERVE_SUPPORT_INVALID_ARGUMENT;
    }
    *out_scale = NULL;
    if (!blob) return ORION_DEPTH_SERVE_SUPPORT_INVALID_ARGUMENT;

    const size_t payload_bytes = channels * sizeof(_Float16);
    if (blob.length != ORION_DEPTH_SERVE_BLOB_HEADER_BYTES + payload_bytes) {
        return ORION_DEPTH_SERVE_SUPPORT_BLOB_LENGTH;
    }

    const uint8_t *bytes = blob.bytes;
    if (!bytes || bytes[0] != 1 || bytes[4] != 2 ||
        read_le32(bytes + ORION_DEPTH_SERVE_CHUNK_HEADER_OFFSET) !=
            ORION_DEPTH_SERVE_CHUNK_MAGIC ||
        bytes[ORION_DEPTH_SERVE_CHUNK_HEADER_OFFSET + 4] != 1) {
        return ORION_DEPTH_SERVE_SUPPORT_BLOB_HEADER;
    }
    if (read_le32(bytes + ORION_DEPTH_SERVE_CHUNK_PAYLOAD_BYTES_OFFSET) != payload_bytes ||
        read_le32(bytes + ORION_DEPTH_SERVE_CHUNK_PAYLOAD_OFFSET_OFFSET) !=
            ORION_DEPTH_SERVE_BLOB_HEADER_BYTES) {
        return ORION_DEPTH_SERVE_SUPPORT_BLOB_PAYLOAD;
    }

    const _Float16 *scale =
        (const _Float16 *)(bytes + ORION_DEPTH_SERVE_BLOB_HEADER_BYTES);
    for (size_t c = 0; c < channels; c++) {
        if (!isfinite((float)scale[c])) {
            return ORION_DEPTH_SERVE_SUPPORT_BLOB_SCALE;
        }
    }

    *out_scale = scale;
    return ORION_DEPTH_SERVE_SUPPORT_OK;
}

OrionDepthServeSupportStatus orion_depth_serve_host_rmsnorm_checked(
    const _Float16 *surface, size_t channels, size_t seq_len, size_t slot,
    const _Float16 *scale, float epsilon, float *out) {
    if (!surface || !scale || !out || channels == 0 || seq_len == 0 ||
        slot >= seq_len || channels > SIZE_MAX / seq_len ||
        channels > SIZE_MAX / sizeof(float)) {
        return ORION_DEPTH_SERVE_SUPPORT_INVALID_ARGUMENT;
    }
    if (!isfinite(epsilon) || epsilon <= 0.0f) {
        return ORION_DEPTH_SERVE_SUPPORT_RMS_EPSILON;
    }

    double squared_sum = 0.0;
    for (size_t c = 0; c < channels; c++) {
        float value = (float)surface[c * seq_len + slot];
        float weight = (float)scale[c];
        if (!isfinite(value) || !isfinite(weight)) {
            return ORION_DEPTH_SERVE_SUPPORT_RMS_INPUT;
        }
        squared_sum += (double)value * value;
    }
    const double mean_square = squared_sum / (double)channels;
    const double denominator = sqrt(mean_square + (double)epsilon);
    if (!isfinite(denominator) || denominator <= 0.0) {
        return ORION_DEPTH_SERVE_SUPPORT_RMS_INPUT;
    }

    const double inverse = 1.0 / denominator;
    float *normalized = malloc(channels * sizeof(*normalized));
    if (!normalized) {
        return ORION_DEPTH_SERVE_SUPPORT_RMS_ALLOCATION;
    }
    for (size_t c = 0; c < channels; c++) {
        normalized[c] = (float)((double)surface[c * seq_len + slot] * inverse *
                                (double)scale[c]);
        if (!isfinite(normalized[c])) {
            free(normalized);
            return ORION_DEPTH_SERVE_SUPPORT_RMS_OUTPUT;
        }
    }
    memcpy(out, normalized, channels * sizeof(*out));
    free(normalized);
    return ORION_DEPTH_SERVE_SUPPORT_OK;
}

const char *orion_depth_serve_support_status_string(
    OrionDepthServeSupportStatus status) {
    switch (status) {
        case ORION_DEPTH_SERVE_SUPPORT_OK: return "ok";
        case ORION_DEPTH_SERVE_SUPPORT_INVALID_ARGUMENT: return "invalid-argument";
        case ORION_DEPTH_SERVE_SUPPORT_BLOB_LENGTH: return "blob-length";
        case ORION_DEPTH_SERVE_SUPPORT_BLOB_HEADER: return "blob-header";
        case ORION_DEPTH_SERVE_SUPPORT_BLOB_PAYLOAD: return "blob-payload";
        case ORION_DEPTH_SERVE_SUPPORT_BLOB_SCALE: return "blob-scale";
        case ORION_DEPTH_SERVE_SUPPORT_RMS_EPSILON: return "rms-epsilon";
        case ORION_DEPTH_SERVE_SUPPORT_RMS_INPUT: return "rms-input";
        case ORION_DEPTH_SERVE_SUPPORT_RMS_OUTPUT: return "rms-output";
        case ORION_DEPTH_SERVE_SUPPORT_RMS_ALLOCATION: return "rms-allocation";
    }
    return "unknown";
}
