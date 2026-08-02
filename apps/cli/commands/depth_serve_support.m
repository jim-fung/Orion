// apps/cli/commands/depth_serve_support.m

#import "apps/cli/commands/depth_serve_support.h"

#import <CommonCrypto/CommonDigest.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define ORION_DEPTH_SERVE_BLOB_HEADER_BYTES 128u
#define ORION_DEPTH_SERVE_CHUNK_HEADER_OFFSET 64u
#define ORION_DEPTH_SERVE_CHUNK_MAGIC 0xdeadbeefu
#define ORION_DEPTH_SERVE_CHUNK_PAYLOAD_BYTES_OFFSET 72u
#define ORION_DEPTH_SERVE_CHUNK_PAYLOAD_OFFSET_OFFSET 80u
#define ORION_DEPTH_PROTOCOL_VERSION 2u
#define ORION_DEPTH_TYPED_STRING 0x01u
#define ORION_DEPTH_TYPED_UNSIGNED 0x02u
#define ORION_DEPTH_TYPED_FLOAT64 0x03u
#define ORION_DEPTH_TYPED_DIGEST 0x04u
#define ORION_DEPTH_TYPED_BOOL 0x05u

static void typed_update(CC_SHA256_CTX *context, const void *bytes, size_t length) {
    if (length) CC_SHA256_Update(context, bytes, (CC_LONG)length);
}

static void typed_u32(CC_SHA256_CTX *context, uint32_t value) {
    uint8_t bytes[4];
    for (int i = 0; i < 4; i++) bytes[i] = (uint8_t)(value >> (8 * i));
    typed_update(context, bytes, sizeof(bytes));
}

static void typed_u64(CC_SHA256_CTX *context, uint64_t value) {
    uint8_t bytes[8];
    for (int i = 0; i < 8; i++) bytes[i] = (uint8_t)(value >> (8 * i));
    typed_update(context, bytes, sizeof(bytes));
}

static bool typed_name(CC_SHA256_CTX *context, const char *name, uint8_t type) {
    if (!context || !name || strlen(name) > UINT32_MAX) return false;
    typed_u32(context, (uint32_t)strlen(name));
    typed_update(context, name, strlen(name));
    typed_update(context, &type, 1);
    return true;
}

static bool typed_string(CC_SHA256_CTX *context, const char *name, NSString *value) {
    if (![value isKindOfClass:[NSString class]] || [value rangeOfString:@"\0"].location != NSNotFound)
        return false;
    NSData *utf8 = [value dataUsingEncoding:NSUTF8StringEncoding allowLossyConversion:NO];
    if (!utf8 || !typed_name(context, name, ORION_DEPTH_TYPED_STRING)) return false;
    typed_u64(context, utf8.length);
    typed_update(context, utf8.bytes, utf8.length);
    return true;
}

static bool typed_unsigned(CC_SHA256_CTX *context, const char *name, uint64_t value) {
    if (!typed_name(context, name, ORION_DEPTH_TYPED_UNSIGNED)) return false;
    typed_u64(context, value);
    return true;
}

static bool typed_float64(CC_SHA256_CTX *context, const char *name, double value) {
    if (!isfinite(value) || !typed_name(context, name, ORION_DEPTH_TYPED_FLOAT64)) return false;
    if (value == 0.0) value = 0.0;
    uint64_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    typed_u64(context, bits);
    return true;
}

static bool typed_bool(CC_SHA256_CTX *context, const char *name, bool value) {
    if (!typed_name(context, name, ORION_DEPTH_TYPED_BOOL)) return false;
    uint8_t byte = value ? 1 : 0;
    typed_update(context, &byte, 1);
    return true;
}

bool orion_depth_serve_is_lower_hex_sha256(NSString *value) {
    if (![value isKindOfClass:[NSString class]] || value.length != 64) return false;
    NSCharacterSet *bad = [[NSCharacterSet characterSetWithCharactersInString:
        @"0123456789abcdef"] invertedSet];
    return [value rangeOfCharacterFromSet:bad].location == NSNotFound;
}

static bool typed_digest(CC_SHA256_CTX *context, const char *name, NSString *value) {
    if (!orion_depth_serve_is_lower_hex_sha256(value) ||
        !typed_name(context, name, ORION_DEPTH_TYPED_DIGEST)) return false;
    uint8_t digest[CC_SHA256_DIGEST_LENGTH];
    const char *hex = value.UTF8String;
    for (int i = 0; i < CC_SHA256_DIGEST_LENGTH; i++) {
        unsigned hi = (unsigned)(hex[i * 2] <= '9' ? hex[i * 2] - '0' : hex[i * 2] - 'a' + 10);
        unsigned lo = (unsigned)(hex[i * 2 + 1] <= '9' ? hex[i * 2 + 1] - '0' : hex[i * 2 + 1] - 'a' + 10);
        digest[i] = (uint8_t)((hi << 4) | lo);
    }
    typed_update(context, digest, sizeof(digest));
    return true;
}

static NSString *typed_final(CC_SHA256_CTX *context) {
    uint8_t digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256_Final(digest, context);
    NSMutableString *hex = [NSMutableString stringWithCapacity:64];
    for (int i = 0; i < CC_SHA256_DIGEST_LENGTH; i++) [hex appendFormat:@"%02x", digest[i]];
    return hex;
}

NSString *orion_depth_serve_sha256_data(NSData *data) {
    if (!data) return nil;
    uint8_t digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(data.bytes, (CC_LONG)data.length, digest);
    NSMutableString *hex = [NSMutableString stringWithCapacity:64];
    for (int i = 0; i < CC_SHA256_DIGEST_LENGTH; i++) [hex appendFormat:@"%02x", digest[i]];
    return hex;
}

NSString *orion_depth_serve_semantic_request_sha256(
    NSString *worker_session_id, NSString *stream_session_id,
    uint64_t logical_frame_id, NSString *request_id, uint64_t request_sequence,
    NSString *launch_challenge, NSString *challenge_nonce, uint64_t c0,
    double gamma, NSString *hidden_cond_sha256, NSString *hidden_uncond_sha256,
    NSString *program_set_digest) {
    CC_SHA256_CTX context;
    CC_SHA256_Init(&context);
    if (!typed_string(&context, "domain", @"voxtream.orion.depth-semantic-request/2") ||
        !typed_unsigned(&context, "protocol_version", ORION_DEPTH_PROTOCOL_VERSION) ||
        !typed_string(&context, "worker_session_id", worker_session_id) ||
        !typed_string(&context, "stream_session_id", stream_session_id) ||
        !typed_unsigned(&context, "logical_frame_id", logical_frame_id) ||
        !typed_string(&context, "request_id", request_id) ||
        !typed_unsigned(&context, "request_sequence", request_sequence) ||
        !typed_digest(&context, "launch_challenge", launch_challenge) ||
        !typed_digest(&context, "challenge_nonce", challenge_nonce) ||
        !typed_unsigned(&context, "c0", c0) ||
        !typed_float64(&context, "gamma", gamma) ||
        !typed_digest(&context, "hidden_cond_sha256", hidden_cond_sha256) ||
        !typed_digest(&context, "hidden_uncond_sha256", hidden_uncond_sha256) ||
        !typed_digest(&context, "program_set_digest", program_set_digest)) return nil;
    return typed_final(&context);
}

NSString *orion_depth_serve_protocol_v2_fixture_sha256(void) {
    return orion_depth_serve_semantic_request_sha256(
        @"550e8400-e29b-41d4-a716-446655440000", @"stream-alpha", 42,
        @"request-000042", 7,
        @"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        @"abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
        2049, 3.0,
        @"1111111111111111111111111111111111111111111111111111111111111111",
        @"2222222222222222222222222222222222222222222222222222222222222222",
        @"3333333333333333333333333333333333333333333333333333333333333333");
}

NSString *orion_depth_serve_plan_weights_sha256(const OrionBindingPlan *plan) {
    if (!plan || plan->n_weights < 0 || plan->n_weights > ORION_MAX_PLAN_WEIGHTS) return nil;
    CC_SHA256_CTX context;
    CC_SHA256_Init(&context);
    if (!typed_string(&context, "domain", @"voxtream.orion.depth-plan-weights/2")) return nil;
    for (int i = 0; i < plan->n_weights; i++) {
        const OrionBindingPlanWeight *weight = &plan->weights[i];
        if (!typed_string(&context, "path", @(weight->path)) ||
            !typed_unsigned(&context, "offset", weight->offset) ||
            !typed_unsigned(&context, "logical_bytes", weight->logical_bytes) ||
            !typed_digest(&context, "sha256", @(weight->sha256))) return nil;
    }
    return typed_final(&context);
}

NSString *orion_depth_serve_program_set_sha256(
    NSString *mode, NSArray<NSDictionary *> *layers,
    NSString *final_norm_placement, NSString *final_norm_program_sha256,
    NSString *final_norm_sha256,
    NSString *audio_embeddings_sha256, NSString *audio_head_sha256,
    NSString *audio_head_placement, NSString *cfg_placement,
    NSString *sampler_placement) {
    if (![layers isKindOfClass:[NSArray class]] || layers.count != 4) return nil;
    CC_SHA256_CTX context;
    CC_SHA256_Init(&context);
    if (!typed_string(&context, "domain", @"voxtream.orion.depth-program-set/2") ||
        !typed_string(&context, "mode", mode)) return nil;
    for (NSUInteger i = 0; i < layers.count; i++) {
        NSDictionary *layer = layers[i];
        NSString *expected = [NSString stringWithFormat:@"layer%lu", (unsigned long)i];
        if (![layer isKindOfClass:[NSDictionary class]] ||
            ![layer[@"name"] isEqualToString:expected] ||
            !typed_string(&context, "component", layer[@"name"]) ||
            !typed_digest(&context, "mil_sha256", layer[@"mil_sha256"]) ||
            !typed_digest(&context, "weights_sha256", layer[@"weights_sha256"])) return nil;
    }
    if (!typed_string(&context, "component", @"final_norm") ||
        !typed_string(&context, "final_norm_placement", final_norm_placement) ||
        !typed_digest(&context, "final_norm_program_sha256", final_norm_program_sha256) ||
        !typed_digest(&context, "final_norm_sha256", final_norm_sha256) ||
        !typed_string(&context, "component", @"audio_embeddings") ||
        !typed_digest(&context, "audio_embeddings_sha256", audio_embeddings_sha256) ||
        !typed_string(&context, "component", @"audio_head") ||
        !typed_digest(&context, "audio_head_sha256", audio_head_sha256) ||
        !typed_string(&context, "audio_head_placement", audio_head_placement) ||
        !typed_string(&context, "cfg_placement", cfg_placement) ||
        !typed_string(&context, "sampler_placement", sampler_placement)) return nil;
    return typed_final(&context);
}

static bool receipt_unsigned_value(NSNumber *value, uint64_t *out) {
    if (![value isKindOfClass:[NSNumber class]] ||
        CFGetTypeID((__bridge CFTypeRef)value) == CFBooleanGetTypeID()) return false;
    const char *type = value.objCType;
    if (strcmp(type, @encode(float)) == 0 || strcmp(type, @encode(double)) == 0 ||
        strcmp(type, @encode(long double)) == 0) return false;
    if ([value compare:@0] == NSOrderedAscending) return false;
    unsigned long long converted = value.unsignedLongLongValue;
    if ([value compare:@(converted)] != NSOrderedSame) return false;
    if (out) *out = converted;
    return true;
}

static bool typed_receipt_unsigned(CC_SHA256_CTX *context, NSDictionary *receipt,
                                   const char *name, uint64_t *out) {
    uint64_t value = 0;
    if (!receipt_unsigned_value(receipt[@(name)], &value)) return false;
    if (out) *out = value;
    return typed_unsigned(context, name, value);
}

static bool receipt_has_exact_keys(NSDictionary *value, NSSet<NSString *> *keys) {
    if (![value isKindOfClass:[NSDictionary class]] || value.count != keys.count) return false;
    for (id key in value) {
        if (![key isKindOfClass:[NSString class]] || ![keys containsObject:key]) return false;
    }
    return true;
}

// Receipt Boolean fields accept CFBoolean and the distinct NSNumber representation
// produced by boxing a C _Bool. Numeric NSNumber values, including @0 and @1,
// are rejected rather than coerced into a Boolean.
static bool receipt_boolean(NSDictionary *receipt, const char *name, bool *out) {
    NSNumber *value = receipt[@(name)];
    if (![value isKindOfClass:[NSNumber class]]) return false;
    if (CFGetTypeID((__bridge CFTypeRef)value) == CFBooleanGetTypeID()) {
        if (out) *out = value.boolValue;
        return true;
    }
    if (strcmp(value.objCType, @encode(_Bool)) != 0) return false;
    if (out) *out = value.boolValue;
    return true;
}

// Single source of truth for receipt validation. Updates `context` for every
// accepted field. When `failure_field` is non-NULL and the receipt is rejected,
// sets `*failure_field` to a freshly strdup'd C string naming the first
// rejecting field (caller frees). String-literal names are strdup'd uniformly
// so the caller can always free() regardless of which field failed.
static bool receipt_sha256_step(CC_SHA256_CTX *context, NSDictionary *receipt,
                                char **failure_field) {
#define FAIL(name) do { \
        if (failure_field) *failure_field = strdup(name); \
        return false; \
    } while (0)
#define CHECK(expr, name) do { if (!(expr)) FAIL(name); } while (0)
    static NSSet<NSString *> *success_keys;
    static NSSet<NSString *> *failure_keys;
    static NSSet<NSString *> *program_keys;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        success_keys = [NSSet setWithArray:@[
            @"receipt_schema", @"ok", @"worker_session_id", @"stream_session_id",
            @"logical_frame_id", @"request_id", @"request_sequence", @"challenge_nonce",
            @"hidden_cond_sha256", @"hidden_uncond_sha256", @"semantic_request_sha256",
            @"program_set_digest", @"previous_receipt_sha256", @"private_invocation_known",
            @"private_invocations_expected", @"private_invocations_actual",
            @"private_invocations_successful", @"private_invocations_failed",
            @"private_invocations_by_program", @"compile_count_before",
            @"compile_count_after", @"compile_count_delta", @"native_fallback_used",
            @"frame_start_monotonic_ns", @"frame_end_monotonic_ns", @"frame_duration_ns",
            @"tokens", @"token_sha256", @"tokens_bin_sha256", @"output_placement",
        ]];
        failure_keys = [success_keys mutableCopy];
        NSMutableSet *mutable_failure_keys = [failure_keys mutableCopy];
        [mutable_failure_keys removeObject:@"tokens"];
        [mutable_failure_keys removeObject:@"token_sha256"];
        [mutable_failure_keys removeObject:@"tokens_bin_sha256"];
        [mutable_failure_keys addObject:@"error"];
        failure_keys = [mutable_failure_keys copy];
        program_keys = [NSSet setWithArray:@[@"expected", @"actual", @"successful", @"failed"]];
    });

    bool ok = false;
    CHECK(receipt_boolean(receipt, "ok", &ok), "ok");
    NSSet<NSString *> *allowed_keys = ok ? success_keys : failure_keys;
    CHECK(typed_bool(context, "ok", ok), "ok");
    if (!ok) CHECK(typed_string(context, "error", receipt[@"error"]), "error");
    CHECK([(NSString *)receipt[@"receipt_schema"] isEqualToString:
              @"voxtream.orion.depth-operational-receipt/2"], "receipt_schema");
    CHECK(typed_string(context, "receipt_schema", receipt[@"receipt_schema"]), "receipt_schema");
    CHECK(typed_string(context, "worker_session_id", receipt[@"worker_session_id"]),
          "worker_session_id");
    CHECK(typed_string(context, "stream_session_id", receipt[@"stream_session_id"]),
          "stream_session_id");
    uint64_t logical_frame_id, request_sequence, expected, actual, successful, failed;
    uint64_t compile_before, compile_after, compile_delta, frame_start, frame_end, frame_duration;
    CHECK(typed_receipt_unsigned(context, receipt, "logical_frame_id", &logical_frame_id),
          "logical_frame_id");
    CHECK(typed_string(context, "request_id", receipt[@"request_id"]), "request_id");
    CHECK(typed_receipt_unsigned(context, receipt, "request_sequence", &request_sequence),
          "request_sequence");
    CHECK(typed_digest(context, "challenge_nonce", receipt[@"challenge_nonce"]), "challenge_nonce");
    CHECK(typed_digest(context, "hidden_cond_sha256", receipt[@"hidden_cond_sha256"]),
          "hidden_cond_sha256");
    CHECK(typed_digest(context, "hidden_uncond_sha256", receipt[@"hidden_uncond_sha256"]),
          "hidden_uncond_sha256");
    CHECK(typed_digest(context, "semantic_request_sha256", receipt[@"semantic_request_sha256"]),
          "semantic_request_sha256");
    CHECK(typed_digest(context, "program_set_digest", receipt[@"program_set_digest"]),
          "program_set_digest");
    CHECK(typed_digest(context, "previous_receipt_sha256", receipt[@"previous_receipt_sha256"]),
          "previous_receipt_sha256");
    bool invocation_known = false;
    CHECK(receipt_boolean(receipt, "private_invocation_known", &invocation_known),
          "private_invocation_known");
    if (!invocation_known) FAIL("private_invocation_known");
    CHECK(typed_bool(context, "private_invocation_known", invocation_known),
          "private_invocation_known");
    CHECK(typed_receipt_unsigned(context, receipt, "private_invocations_expected", &expected),
          "private_invocations_expected");
    CHECK(typed_receipt_unsigned(context, receipt, "private_invocations_actual", &actual),
          "private_invocations_actual");
    CHECK(typed_receipt_unsigned(context, receipt, "private_invocations_successful", &successful),
          "private_invocations_successful");
    CHECK(typed_receipt_unsigned(context, receipt, "private_invocations_failed", &failed),
          "private_invocations_failed");
    NSDictionary *programs = receipt[@"private_invocations_by_program"];
    if (![programs isKindOfClass:[NSDictionary class]] || programs.count != 5) FAIL("private_invocations_by_program");
    bool seqpack_expected_shape = true;
    bool separate_expected_shape = true;
    uint64_t summed_expected = 0, summed_actual = 0, summed_successful = 0, summed_failed = 0;
    for (NSString *program in @[@"layer0", @"layer1", @"layer2", @"layer3", @"final_norm"]) {
        NSDictionary *counts = programs[program];
        if (![counts isKindOfClass:[NSDictionary class]] || programs.count != 5) FAIL("private_invocations_by_program");
        if (!receipt_has_exact_keys(counts, program_keys)) FAIL("private_invocations_by_program");
        uint64_t program_expected, program_actual, program_successful, program_failed;
        CHECK(typed_string(context, "program", program), "program");
        CHECK(typed_receipt_unsigned(context, counts, "expected", &program_expected), "expected");
        CHECK(typed_receipt_unsigned(context, counts, "actual", &program_actual), "actual");
        CHECK(typed_receipt_unsigned(context, counts, "successful", &program_successful), "successful");
        CHECK(typed_receipt_unsigned(context, counts, "failed", &program_failed), "failed");
        bool is_final_norm = [program isEqualToString:@"final_norm"];
        if (program_expected != (is_final_norm ? 0 : 15)) seqpack_expected_shape = false;
        if (program_expected != 30) separate_expected_shape = false;
        if (program_actual > program_expected || program_successful > program_actual ||
            program_failed > program_actual - program_successful ||
            program_successful + program_failed != program_actual ||
            UINT64_MAX - summed_expected < program_expected ||
            UINT64_MAX - summed_actual < program_actual ||
            UINT64_MAX - summed_successful < program_successful ||
            UINT64_MAX - summed_failed < program_failed) FAIL("private_invocations_by_program");
        summed_expected += program_expected;
        summed_actual += program_actual;
        summed_successful += program_successful;
        summed_failed += program_failed;
    }
    if (expected != summed_expected || actual != summed_actual || successful != summed_successful ||
        failed != summed_failed || actual > expected || successful + failed != actual) FAIL("private_invocations");
    if (ok && (!seqpack_expected_shape && !separate_expected_shape))
        FAIL("private_invocations_by_program");
    if (ok && (actual != expected || successful != expected || failed != 0))
        FAIL("private_invocations");
    CHECK(typed_receipt_unsigned(context, receipt, "compile_count_before", &compile_before),
          "compile_count_before");
    CHECK(typed_receipt_unsigned(context, receipt, "compile_count_after", &compile_after),
          "compile_count_after");
    CHECK(typed_receipt_unsigned(context, receipt, "compile_count_delta", &compile_delta),
          "compile_count_delta");
    if (compile_after < compile_before || compile_delta != compile_after - compile_before)
        FAIL("compile_count_delta");
    if (ok && compile_delta != 0) FAIL("compile_count_delta");
    bool native_fallback = false;
    CHECK(receipt_boolean(receipt, "native_fallback_used", &native_fallback),
          "native_fallback_used");
    CHECK(typed_bool(context, "native_fallback_used", native_fallback),
          "native_fallback_used");
    // v2 never permits fallback for either successful or chained failure receipts.
    if (native_fallback) FAIL("native_fallback_used");
    CHECK(typed_receipt_unsigned(context, receipt, "frame_start_monotonic_ns", &frame_start),
          "frame_start_monotonic_ns");
    CHECK(typed_receipt_unsigned(context, receipt, "frame_end_monotonic_ns", &frame_end),
          "frame_end_monotonic_ns");
    CHECK(typed_receipt_unsigned(context, receipt, "frame_duration_ns", &frame_duration),
          "frame_duration_ns");
    if (frame_end < frame_start || frame_duration != frame_end - frame_start) FAIL("frame_duration_ns");
    if (ok) {
        NSArray *tokens = receipt[@"tokens"];
        uint8_t token_bytes[15 * sizeof(uint32_t)];
        if (![tokens isKindOfClass:[NSArray class]] || tokens.count != 15) FAIL("tokens");
        CHECK(typed_unsigned(context, "token_count", tokens.count), "token_count");
        for (NSUInteger i = 0; i < tokens.count; i++) {
            char name[16];
            uint64_t token;
            snprintf(name, sizeof(name), "token%lu", (unsigned long)i);
            CHECK(receipt_unsigned_value(tokens[i], &token) && token <= UINT32_MAX, "tokens");
            CHECK(typed_unsigned(context, name, token), "tokens");
            for (int byte = 0; byte < 4; byte++)
                token_bytes[i * 4 + byte] = (uint8_t)(token >> (byte * 8));
        }
        NSString *token_digest = orion_depth_serve_sha256_data(
            [NSData dataWithBytes:token_bytes length:sizeof(token_bytes)]);
        CHECK(typed_digest(context, "token_sha256", receipt[@"token_sha256"]),
              "token_sha256");
        if (![receipt[@"token_sha256"] isEqualToString:token_digest]) FAIL("token_sha256");
        CHECK(typed_digest(context, "tokens_bin_sha256", receipt[@"tokens_bin_sha256"]),
              "tokens_bin_sha256");
        if (![receipt[@"tokens_bin_sha256"] isEqualToString:token_digest])
            FAIL("tokens_bin_sha256");
    }
    CHECK(typed_string(context, "output_placement", receipt[@"output_placement"]),
          "output_placement");
    // All chained v2 receipts identify the one permitted output location.
    if (![receipt[@"output_placement"] isEqualToString:@"host-memory+atomic-tokens-bin"])
        FAIL("output_placement");
    for (NSString *key in allowed_keys) {
        if (receipt[key] == nil) FAIL(key.UTF8String);
    }
    CHECK(receipt_has_exact_keys(receipt, allowed_keys), "receipt-schema");
#undef CHECK
#undef FAIL
    return true;
}

NSString *orion_depth_serve_receipt_sha256(NSDictionary *receipt) {
    if (![receipt isKindOfClass:[NSDictionary class]]) return nil;
    CC_SHA256_CTX context;
    CC_SHA256_Init(&context);
    CC_SHA256_CTX *ctx = &context;
    // Domain is hashed first, unconditionally, before any field validation.
    if (!typed_string(ctx, "domain", @"voxtream.orion.depth-operational-receipt/2"))
        return nil;
    // Production hash does not need the failing-field name; pass NULL so no
    // allocation is performed and no free is required.
    if (!receipt_sha256_step(ctx, receipt, NULL)) return nil;
    return typed_final(ctx);
}

char *orion_depth_serve_receipt_sha256_failure_field(NSDictionary *receipt) {
    if (![receipt isKindOfClass:[NSDictionary class]]) return strdup("receipt");
    CC_SHA256_CTX context;
    CC_SHA256_Init(&context);
    CC_SHA256_CTX *ctx = &context;
    if (!typed_string(ctx, "domain", @"voxtream.orion.depth-operational-receipt/2"))
        return strdup("domain");
    char *failure_field = NULL;
    if (!receipt_sha256_step(ctx, receipt, &failure_field)) {
        return failure_field ? failure_field : strdup("unknown");
    }
    return NULL;
}

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
