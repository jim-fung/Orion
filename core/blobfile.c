// core/blobfile.c — Shared checked BLOBFILE chunk validation.

#include "blobfile.h"

#include <CommonCrypto/CommonDigest.h>
#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define ORION_BLOBFILE_PREAMBLE_BYTES 5u
#define ORION_BLOBFILE_HASH_BUFFER_BYTES 16384u

static bool checked_add_u64(uint64_t left, uint64_t right, uint64_t *out) {
    if (!out || left > UINT64_MAX - right) return false;
    *out = left + right;
    return true;
}

static uint32_t read_le32(const unsigned char *bytes) {
    return (uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8) |
           ((uint32_t)bytes[2] << 16) | ((uint32_t)bytes[3] << 24);
}

static int hex_value(unsigned char byte) {
    if (byte >= '0' && byte <= '9') return (int)(byte - '0');
    byte = (unsigned char)tolower(byte);
    if (byte >= 'a' && byte <= 'f') return (int)(byte - 'a' + 10);
    return -1;
}

static bool parse_sha256(const char *text, unsigned char digest[CC_SHA256_DIGEST_LENGTH]) {
    if (!text || !digest || strnlen(text, ORION_BLOBFILE_SHA256_HEX_LENGTH + 1) !=
                                   ORION_BLOBFILE_SHA256_HEX_LENGTH) {
        return false;
    }
    for (size_t i = 0; i < CC_SHA256_DIGEST_LENGTH; i++) {
        int high = hex_value((unsigned char)text[i * 2]);
        int low = hex_value((unsigned char)text[i * 2 + 1]);
        if (high < 0 || low < 0) return false;
        digest[i] = (unsigned char)((high << 4) | low);
    }
    return true;
}

static void format_sha256(const unsigned char digest[CC_SHA256_DIGEST_LENGTH],
                          char out[ORION_BLOBFILE_SHA256_HEX_LENGTH + 1]) {
    static const char hex[] = "0123456789abcdef";
    for (size_t i = 0; i < CC_SHA256_DIGEST_LENGTH; i++) {
        out[i * 2] = hex[digest[i] >> 4];
        out[i * 2 + 1] = hex[digest[i] & 0x0f];
    }
    out[ORION_BLOBFILE_SHA256_HEX_LENGTH] = '\0';
}

static OrionBlobfileStatus hash_payload(
    const OrionBlobfileSource *source, uint64_t payload_offset,
    uint64_t payload_size, const unsigned char *expected_digest,
    OrionBlobfileChunk *chunk) {
    CC_SHA256_CTX context;
    CC_SHA256_Init(&context);
    unsigned char buffer[ORION_BLOBFILE_HASH_BUFFER_BYTES];
    uint64_t remaining = payload_size;
    uint64_t position = payload_offset;

    while (remaining > 0) {
        size_t count = remaining < sizeof(buffer) ? (size_t)remaining : sizeof(buffer);
        if (!source->read(source->context, position, buffer, count)) {
            return ORION_BLOBFILE_SHA256_READ_ERROR;
        }
        CC_SHA256_Update(&context, buffer, (CC_LONG)count);
        remaining -= count;
        position += count;
    }

    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256_Final(digest, &context);
    format_sha256(digest, chunk->sha256);
    chunk->has_sha256 = true;
    if (expected_digest &&
        memcmp(digest, expected_digest, CC_SHA256_DIGEST_LENGTH) != 0) {
        return ORION_BLOBFILE_SHA256_MISMATCH;
    }
    return ORION_BLOBFILE_OK;
}

OrionBlobfileStatus orion_blobfile_validate_chunk(
    const OrionBlobfileSource *source,
    const OrionBlobfileValidationOptions *options,
    OrionBlobfileChunk *out_chunk) {
    if (out_chunk) memset(out_chunk, 0, sizeof(*out_chunk));
    if (!source || !source->length || !source->read || !options || !out_chunk) {
        return ORION_BLOBFILE_INVALID_ARGUMENT;
    }

    unsigned char expected_digest[CC_SHA256_DIGEST_LENGTH];
    const unsigned char *expected = NULL;
    if (options->expected_sha256) {
        if (!parse_sha256(options->expected_sha256, expected_digest)) {
            return ORION_BLOBFILE_SHA256_INVALID;
        }
        expected = expected_digest;
    }

    uint64_t source_length = 0;
    if (!source->length(source->context, &source_length)) {
        return ORION_BLOBFILE_SOURCE_ERROR;
    }
    if (source_length < ORION_BLOBFILE_PREAMBLE_BYTES) {
        return ORION_BLOBFILE_PREAMBLE_TRUNCATED;
    }

    unsigned char preamble[ORION_BLOBFILE_PREAMBLE_BYTES];
    if (!source->read(source->context, 0, preamble, sizeof(preamble))) {
        return ORION_BLOBFILE_SOURCE_ERROR;
    }
    if (preamble[0] != 0x01 || preamble[4] != 0x02) {
        return ORION_BLOBFILE_PREAMBLE_INVALID;
    }

    uint64_t header_end = 0;
    if (!checked_add_u64(options->chunk_offset,
                         ORION_BLOBFILE_CHUNK_HEADER_BYTES, &header_end)) {
        return ORION_BLOBFILE_RANGE_OVERFLOW;
    }
    if (header_end > source_length) return ORION_BLOBFILE_HEADER_TRUNCATED;

    unsigned char header[ORION_BLOBFILE_CHUNK_HEADER_BYTES];
    if (!source->read(source->context, options->chunk_offset,
                      header, sizeof(header))) {
        return ORION_BLOBFILE_SOURCE_ERROR;
    }
    if (header[0] != 0xef || header[1] != 0xbe ||
        header[2] != 0xad || header[3] != 0xde) {
        return ORION_BLOBFILE_HEADER_MAGIC_INVALID;
    }
    if (header[4] != 0x01) return ORION_BLOBFILE_HEADER_VERSION_INVALID;

    uint64_t payload_size = read_le32(header + 8);
    uint64_t payload_offset = read_le32(header + 16);
    if (options->expected_payload_size != ORION_BLOBFILE_PAYLOAD_SIZE_ANY &&
        payload_size != options->expected_payload_size) {
        return ORION_BLOBFILE_PAYLOAD_LENGTH_MISMATCH;
    }
    if (payload_offset < header_end) {
        return ORION_BLOBFILE_PAYLOAD_OFFSET_INVALID;
    }

    uint64_t payload_end = 0;
    if (!checked_add_u64(payload_offset, payload_size, &payload_end)) {
        return ORION_BLOBFILE_RANGE_OVERFLOW;
    }
    if (payload_end > source_length) return ORION_BLOBFILE_PAYLOAD_TRUNCATED;

    out_chunk->chunk_offset = options->chunk_offset;
    out_chunk->header_end = header_end;
    out_chunk->payload_offset = payload_offset;
    out_chunk->payload_size = payload_size;
    out_chunk->payload_end = payload_end;

    if (options->compute_sha256 || expected) {
        return hash_payload(source, payload_offset, payload_size, expected, out_chunk);
    }
    return ORION_BLOBFILE_OK;
}

typedef struct {
    const unsigned char *bytes;
    uint64_t length;
} MemorySource;

static bool memory_length(void *context, uint64_t *out_length) {
    MemorySource *memory = context;
    if (!memory || !out_length) return false;
    *out_length = memory->length;
    return true;
}

static bool memory_read(void *context, uint64_t offset,
                        void *destination, size_t length) {
    MemorySource *memory = context;
    if (!memory || !destination || (length && !memory->bytes) ||
        offset > memory->length || length > memory->length - offset ||
        offset > SIZE_MAX) {
        return false;
    }
    if (length) memcpy(destination, memory->bytes + (size_t)offset, length);
    return true;
}

OrionBlobfileStatus orion_blobfile_validate_memory(
    const void *bytes, size_t length,
    const OrionBlobfileValidationOptions *options,
    OrionBlobfileChunk *out_chunk) {
    if (length && !bytes) {
        if (out_chunk) memset(out_chunk, 0, sizeof(*out_chunk));
        return ORION_BLOBFILE_INVALID_ARGUMENT;
    }
    MemorySource memory = { .bytes = bytes, .length = length };
    OrionBlobfileSource source = {
        .context = &memory,
        .length = memory_length,
        .read = memory_read,
    };
    return orion_blobfile_validate_chunk(&source, options, out_chunk);
}

typedef struct {
    int fd;
} FileSource;

static bool fd_length(void *context, uint64_t *out_length) {
    FileSource *file = context;
    struct stat statbuf;
    if (!file || file->fd < 0 || !out_length ||
        fstat(file->fd, &statbuf) != 0 || !S_ISREG(statbuf.st_mode) ||
        statbuf.st_size < 0) {
        return false;
    }
    *out_length = (uint64_t)statbuf.st_size;
    return true;
}

static bool fd_read(void *context, uint64_t offset,
                    void *destination, size_t length) {
    FileSource *file = context;
    if (!file || file->fd < 0 || !destination) return false;
    unsigned char *output = destination;
    size_t total = 0;
    while (total < length) {
        uint64_t position = offset + total;
        off_t file_offset = (off_t)position;
        if (file_offset < 0 || (uint64_t)file_offset != position) return false;
        ssize_t count = pread(file->fd, output + total, length - total, file_offset);
        if (count > 0) {
            total += (size_t)count;
            continue;
        }
        if (count < 0 && errno == EINTR) continue;
        return false;
    }
    return true;
}

OrionBlobfileStatus orion_blobfile_validate_fd(
    int fd, const OrionBlobfileValidationOptions *options,
    OrionBlobfileChunk *out_chunk) {
    if (fd < 0) {
        if (out_chunk) memset(out_chunk, 0, sizeof(*out_chunk));
        return ORION_BLOBFILE_INVALID_ARGUMENT;
    }
    FileSource file = { .fd = fd };
    OrionBlobfileSource source = {
        .context = &file,
        .length = fd_length,
        .read = fd_read,
    };
    return orion_blobfile_validate_chunk(&source, options, out_chunk);
}

const char *orion_blobfile_status_string(OrionBlobfileStatus status) {
    switch (status) {
        case ORION_BLOBFILE_OK: return "ok";
        case ORION_BLOBFILE_INVALID_ARGUMENT: return "invalid-argument";
        case ORION_BLOBFILE_SOURCE_ERROR: return "source-error";
        case ORION_BLOBFILE_PREAMBLE_TRUNCATED: return "preamble-truncated";
        case ORION_BLOBFILE_PREAMBLE_INVALID: return "preamble-invalid";
        case ORION_BLOBFILE_RANGE_OVERFLOW: return "range-overflow";
        case ORION_BLOBFILE_HEADER_TRUNCATED: return "header-truncated";
        case ORION_BLOBFILE_HEADER_MAGIC_INVALID: return "header-magic-invalid";
        case ORION_BLOBFILE_HEADER_VERSION_INVALID: return "header-version-invalid";
        case ORION_BLOBFILE_PAYLOAD_LENGTH_MISMATCH: return "payload-length-mismatch";
        case ORION_BLOBFILE_PAYLOAD_OFFSET_INVALID: return "payload-offset-invalid";
        case ORION_BLOBFILE_PAYLOAD_TRUNCATED: return "payload-truncated";
        case ORION_BLOBFILE_SHA256_INVALID: return "sha256-invalid";
        case ORION_BLOBFILE_SHA256_READ_ERROR: return "sha256-read-error";
        case ORION_BLOBFILE_SHA256_MISMATCH: return "sha256-mismatch";
        default: return "unknown";
    }
}
