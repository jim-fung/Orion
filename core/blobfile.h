// core/blobfile.h — Shared checked BLOBFILE chunk validation.

#ifndef ORION_BLOBFILE_H
#define ORION_BLOBFILE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ORION_BLOBFILE_CHUNK_HEADER_BYTES 64u
#define ORION_BLOBFILE_SHA256_HEX_LENGTH 64u
#define ORION_BLOBFILE_PAYLOAD_SIZE_ANY UINT64_MAX

typedef enum {
    ORION_BLOBFILE_OK = 0,
    ORION_BLOBFILE_INVALID_ARGUMENT = 1,
    ORION_BLOBFILE_SOURCE_ERROR = 2,
    ORION_BLOBFILE_PREAMBLE_TRUNCATED = 3,
    ORION_BLOBFILE_PREAMBLE_INVALID = 4,
    ORION_BLOBFILE_RANGE_OVERFLOW = 5,
    ORION_BLOBFILE_HEADER_TRUNCATED = 6,
    ORION_BLOBFILE_HEADER_MAGIC_INVALID = 7,
    ORION_BLOBFILE_HEADER_VERSION_INVALID = 8,
    ORION_BLOBFILE_PAYLOAD_LENGTH_MISMATCH = 9,
    ORION_BLOBFILE_PAYLOAD_OFFSET_INVALID = 10,
    ORION_BLOBFILE_PAYLOAD_TRUNCATED = 11,
    ORION_BLOBFILE_SHA256_INVALID = 12,
    ORION_BLOBFILE_SHA256_READ_ERROR = 13,
    ORION_BLOBFILE_SHA256_MISMATCH = 14,
} OrionBlobfileStatus;

typedef bool (*OrionBlobfileLengthFn)(void *context, uint64_t *out_length);
typedef bool (*OrionBlobfileReadFn)(void *context, uint64_t offset,
                                    void *destination, size_t length);

typedef struct {
    void *context;
    OrionBlobfileLengthFn length;
    OrionBlobfileReadFn read;
} OrionBlobfileSource;

typedef struct {
    uint64_t chunk_offset;
    // Use ORION_BLOBFILE_PAYLOAD_SIZE_ANY to accept the header's declaration.
    uint64_t expected_payload_size;
    // Optional 64-character hexadecimal digest. Comparison is case-insensitive.
    const char *expected_sha256;
    // When true, calculate sha256 even when no expected digest is supplied.
    bool compute_sha256;
} OrionBlobfileValidationOptions;

typedef struct {
    uint64_t chunk_offset;
    uint64_t header_end;
    uint64_t payload_offset;
    uint64_t payload_size;
    uint64_t payload_end;
    bool has_sha256;
    char sha256[ORION_BLOBFILE_SHA256_HEX_LENGTH + 1];
} OrionBlobfileChunk;

// Validate one chunk from an arbitrary bounded random-access source.
OrionBlobfileStatus orion_blobfile_validate_chunk(
    const OrionBlobfileSource *source,
    const OrionBlobfileValidationOptions *options,
    OrionBlobfileChunk *out_chunk);

// Convenience entry points for the two production storage forms.
OrionBlobfileStatus orion_blobfile_validate_memory(
    const void *bytes, size_t length,
    const OrionBlobfileValidationOptions *options,
    OrionBlobfileChunk *out_chunk);
OrionBlobfileStatus orion_blobfile_validate_fd(
    int fd, const OrionBlobfileValidationOptions *options,
    OrionBlobfileChunk *out_chunk);

// Stable diagnostic spelling for every OrionBlobfileStatus value.
const char *orion_blobfile_status_string(OrionBlobfileStatus status);

#ifdef __cplusplus
}
#endif

#endif  // ORION_BLOBFILE_H
