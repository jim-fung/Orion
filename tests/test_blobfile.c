// tests/test_blobfile.c — Portable shared BLOBFILE validator regressions.

#include <CommonCrypto/CommonDigest.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "core/blobfile.h"

static int g_passed = 0, g_failed = 0;

#define TEST(name) do { \
    printf("  %-50s ", #name); \
    if (test_##name()) { printf("PASS\n"); g_passed++; } \
    else { printf("FAIL\n"); g_failed++; } \
} while (0)

static void write_le32(unsigned char *bytes, uint32_t value) {
    bytes[0] = (unsigned char)value;
    bytes[1] = (unsigned char)(value >> 8);
    bytes[2] = (unsigned char)(value >> 16);
    bytes[3] = (unsigned char)(value >> 24);
}

static void digest_hex(const unsigned char *bytes, size_t length, char out[65]) {
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(bytes, (CC_LONG)length, digest);
    for (int i = 0; i < CC_SHA256_DIGEST_LENGTH; i++)
        snprintf(out + i * 2, 3, "%02x", digest[i]);
}

static void make_chunk(unsigned char *blob, size_t length, uint64_t header,
                       uint32_t payload, uint32_t payload_bytes) {
    memset(blob, 0, length);
    blob[0] = 1;
    blob[4] = 2;
    blob[header + 0] = 0xef;
    blob[header + 1] = 0xbe;
    blob[header + 2] = 0xad;
    blob[header + 3] = 0xde;
    blob[header + 4] = 1;
    write_le32(blob + header + 8, payload_bytes);
    write_le32(blob + header + 16, payload);
    for (uint32_t i = 0; i < payload_bytes; i++) blob[payload + i] = (unsigned char)(i + 1);
}

static OrionBlobfileStatus validate(const unsigned char *blob, size_t length,
                                    uint64_t header, uint64_t expected_bytes,
                                    const char *expected_sha,
                                    OrionBlobfileChunk *chunk) {
    OrionBlobfileValidationOptions options = {
        .chunk_offset = header,
        .expected_payload_size = expected_bytes,
        .expected_sha256 = expected_sha,
        .compute_sha256 = expected_sha == NULL,
    };
    return orion_blobfile_validate_memory(blob, length, &options, chunk);
}

static int test_valid_legacy_and_aggregate_offsets(void) {
    unsigned char legacy[136];
    unsigned char aggregate[208];
    OrionBlobfileChunk chunk = {0};
    make_chunk(legacy, sizeof(legacy), 64, 128, 8);
    make_chunk(aggregate, sizeof(aggregate), 128, 200, 8);
    return validate(legacy, sizeof(legacy), 64, 8, NULL, &chunk) == ORION_BLOBFILE_OK &&
           chunk.payload_offset == 128 && chunk.payload_end == 136 && chunk.has_sha256 &&
           validate(aggregate, sizeof(aggregate), 128, 8, NULL, &chunk) == ORION_BLOBFILE_OK &&
           chunk.payload_offset == 200 && chunk.payload_end == 208;
}

static int test_malformed_magic_and_version(void) {
    unsigned char blob[136];
    OrionBlobfileChunk chunk;
    make_chunk(blob, sizeof(blob), 64, 128, 8);
    blob[64] = 0;
    int bad_magic = validate(blob, sizeof(blob), 64, 8, NULL, &chunk) ==
        ORION_BLOBFILE_HEADER_MAGIC_INVALID;
    make_chunk(blob, sizeof(blob), 64, 128, 8);
    blob[68] = 2;
    return bad_magic && validate(blob, sizeof(blob), 64, 8, NULL, &chunk) ==
        ORION_BLOBFILE_HEADER_VERSION_INVALID;
}

static int test_truncated_header_and_payload(void) {
    unsigned char blob[136];
    OrionBlobfileChunk chunk;
    make_chunk(blob, sizeof(blob), 64, 128, 8);
    int header = validate(blob, 127, 64, 8, NULL, &chunk) ==
        ORION_BLOBFILE_HEADER_TRUNCATED;
    return header && validate(blob, 135, 64, 8, NULL, &chunk) ==
        ORION_BLOBFILE_PAYLOAD_TRUNCATED;
}

static int test_length_and_payload_offset_validation(void) {
    unsigned char blob[136];
    OrionBlobfileChunk chunk;
    make_chunk(blob, sizeof(blob), 64, 128, 8);
    int length = validate(blob, sizeof(blob), 64, 7, NULL, &chunk) ==
        ORION_BLOBFILE_PAYLOAD_LENGTH_MISMATCH;
    write_le32(blob + 80, 127);
    int before_header = validate(blob, sizeof(blob), 64, 8, NULL, &chunk) ==
        ORION_BLOBFILE_PAYLOAD_OFFSET_INVALID;
    write_le32(blob + 80, 132);
    return length && before_header && validate(blob, sizeof(blob), 64, 8, NULL, &chunk) ==
        ORION_BLOBFILE_PAYLOAD_TRUNCATED;
}

static int test_range_overflow(void) {
    unsigned char blob[5] = {1, 0, 0, 0, 2};
    OrionBlobfileChunk chunk;
    return validate(blob, sizeof(blob), UINT64_MAX - 31, 8, NULL, &chunk) ==
        ORION_BLOBFILE_RANGE_OVERFLOW;
}

static int test_sha256_validation(void) {
    unsigned char blob[136];
    OrionBlobfileChunk chunk;
    char digest[65];
    make_chunk(blob, sizeof(blob), 64, 128, 8);
    digest_hex(blob + 128, 8, digest);
    int good = validate(blob, sizeof(blob), 64, 8, digest, &chunk) == ORION_BLOBFILE_OK;
    digest[0] = digest[0] == '0' ? '1' : '0';
    int mismatch = validate(blob, sizeof(blob), 64, 8, digest, &chunk) ==
        ORION_BLOBFILE_SHA256_MISMATCH;
    return good && mismatch &&
        validate(blob, sizeof(blob), 64, 8, "bad", &chunk) == ORION_BLOBFILE_SHA256_INVALID;
}

static int test_file_backed_validation(void) {
    unsigned char blob[136];
    char path[] = "/tmp/orion-blobfile-XXXXXX";
    OrionBlobfileChunk chunk;
    make_chunk(blob, sizeof(blob), 64, 128, 8);
    int fd = mkstemp(path);
    if (fd < 0) return 0;
    ssize_t written = write(fd, blob, sizeof(blob));
    OrionBlobfileValidationOptions options = {
        .chunk_offset = 64,
        .expected_payload_size = 8,
        .expected_sha256 = NULL,
        .compute_sha256 = true,
    };
    int ok = written == (ssize_t)sizeof(blob) &&
        orion_blobfile_validate_fd(fd, &options, &chunk) == ORION_BLOBFILE_OK &&
        chunk.payload_offset == 128 && chunk.has_sha256;
    close(fd);
    unlink(path);
    return ok;
}

static int test_status_strings_are_stable(void) {
    return strcmp(orion_blobfile_status_string(ORION_BLOBFILE_OK), "ok") == 0 &&
           strcmp(orion_blobfile_status_string(ORION_BLOBFILE_RANGE_OVERFLOW),
                  "range-overflow") == 0 &&
           strcmp(orion_blobfile_status_string(ORION_BLOBFILE_SHA256_MISMATCH),
                  "sha256-mismatch") == 0;
}

int main(void) {
    printf("test_blobfile:\n");
    TEST(valid_legacy_and_aggregate_offsets);
    TEST(malformed_magic_and_version);
    TEST(truncated_header_and_payload);
    TEST(length_and_payload_offset_validation);
    TEST(range_overflow);
    TEST(sha256_validation);
    TEST(file_backed_validation);
    TEST(status_strings_are_stable);
    printf("\n%d/%d passed\n", g_passed, g_passed + g_failed);
    return g_failed ? 1 : 0;
}
