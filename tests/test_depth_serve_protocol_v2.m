// tests/test_depth_serve_protocol_v2.m
// Deterministic compiler-only known-answer vectors for protocol v2 transcripts.

#import <Foundation/Foundation.h>

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "apps/cli/commands/depth_serve_support.h"

static int expect_digest(NSString *actual, const char *expected, const char *name) {
    if (!actual || strcmp(actual.UTF8String, expected) != 0) {
        fprintf(stderr, "%s mismatch: expected %s, got %s\n", name, expected,
                actual.UTF8String ?: "(null)");
        return 1;
    }
    return 0;
}

static NSDictionary *invocation_counts(unsigned expected, unsigned actual,
                                       unsigned successful, unsigned failed) {
    return @{
        @"expected": @(expected), @"actual": @(actual),
        @"successful": @(successful), @"failed": @(failed),
    };
}

static bool expect_receipt_rejection(NSDictionary *receipt, const char *field) {
    if (orion_depth_serve_receipt_sha256(receipt) != nil) {
        fprintf(stderr, "receipt schema accepted invalid %s\n", field);
        return false;
    }
    char *actual = orion_depth_serve_receipt_sha256_failure_field(receipt);
    bool matched = actual != NULL && strcmp(actual, field) == 0;
    if (!matched) fprintf(stderr, "%s rejection reported %s\n", field, actual ?: "(null)");
    if (actual) free(actual);
    return matched;
}

static void make_separate_success(NSMutableDictionary *receipt) {
    receipt[@"private_invocations_expected"] = @150;
    receipt[@"private_invocations_actual"] = @150;
    receipt[@"private_invocations_successful"] = @150;
    receipt[@"private_invocations_failed"] = @0;
    NSMutableDictionary *programs = [receipt[@"private_invocations_by_program"] mutableCopy];
    for (NSString *program in @[@"layer0", @"layer1", @"layer2", @"layer3", @"final_norm"])
        programs[program] = invocation_counts(30, 30, 30, 0);
    receipt[@"private_invocations_by_program"] = programs;
}

static NSMutableDictionary *fixture_receipt(BOOL ok) {
    NSMutableArray *tokens = [NSMutableArray arrayWithCapacity:15];
    for (int i = 0; i < 15; i++) [tokens addObject:@(100 + i)];
    NSMutableDictionary *receipt = [@{
        @"receipt_schema": @"voxtream.orion.depth-operational-receipt/2",
        @"ok": @(ok),
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
        @"private_invocations_actual": @(ok ? 60 : 18),
        @"private_invocations_successful": @(ok ? 60 : 17),
        @"private_invocations_failed": @(ok ? 0 : 1),
        @"private_invocations_by_program": @{
            @"layer0": invocation_counts(15, 15, 15, 0),
            @"layer1": invocation_counts(15, ok ? 15 : 3, ok ? 15 : 2, ok ? 0 : 1),
            @"layer2": invocation_counts(15, ok ? 15 : 0, ok ? 15 : 0, 0),
            @"layer3": invocation_counts(15, ok ? 15 : 0, ok ? 15 : 0, 0),
            @"final_norm": invocation_counts(0, 0, 0, 0),
        },
        @"compile_count_before": @4,
        @"compile_count_after": @4,
        @"compile_count_delta": @0,
        @"native_fallback_used": @NO,
        @"frame_start_monotonic_ns": @1000000,
        @"frame_end_monotonic_ns": @2000000,
        @"frame_duration_ns": @1000000,
        @"output_placement": @"host-memory+atomic-tokens-bin",
    } mutableCopy];
    if (ok) {
        receipt[@"tokens"] = tokens;
        receipt[@"token_sha256"] = @"904db4399aa10326d1155440f63b38ca363af85c3490684f51847084f9d11c3d";
        receipt[@"tokens_bin_sha256"] = @"904db4399aa10326d1155440f63b38ca363af85c3490684f51847084f9d11c3d";
    } else {
        receipt[@"error"] = @"private-evaluation-failed";
    }
    return receipt;
}

int main(void) {
    @autoreleasepool {
        int failures = 0;
        NSString *semantic = orion_depth_serve_protocol_v2_fixture_sha256();
        failures += expect_digest(semantic,
            "53808229003275463352dc3f86f60f27e50b220b39e06ba42933b4acc787fc3a",
            "semantic request");

        NSMutableArray *layers = [NSMutableArray arrayWithCapacity:4];
        for (int i = 0; i < 4; i++) {
            NSString *mil = [@"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                stringByReplacingCharactersInRange:NSMakeRange(0, 1)
                withString:[NSString stringWithFormat:@"%x", i]];
            NSString *weights = [@"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
                stringByReplacingCharactersInRange:NSMakeRange(0, 1)
                withString:[NSString stringWithFormat:@"%x", i + 4]];
            [layers addObject:@{
                @"name": [NSString stringWithFormat:@"layer%d", i],
                @"mil_sha256": mil, @"weights_sha256": weights,
            }];
        }
        NSString *program_set = orion_depth_serve_program_set_sha256(
            @"seqpack", layers, @"cpu-checked-rmsnorm",
            @"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
            @"dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
            @"eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
            @"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
            @"cpu-accelerate", @"cpu-vdsp", @"cpu-argmax");
        failures += expect_digest(program_set,
            "75d6f7ef3614d74108a9e958f99a18238d5275c862b5064b01d82fde44ade0dd",
            "program set");
        failures += expect_digest(
            orion_depth_serve_receipt_sha256(fixture_receipt(YES)),
            "e60f295cc2515e969fbc60f47ac87616ad3ff3f8c4325055b61f969d33b74d76",
            "success receipt");
        failures += expect_digest(
            orion_depth_serve_receipt_sha256(fixture_receipt(NO)),
            "9c27e4c0fb313bf41366963750fff8cffe3d634c6c6b21cb80ecb817ee35405c",
            "failure receipt");

        // Exercise the separate-CFG (150-eval, 30-per-layer) receipt shape.
        NSMutableDictionary *separate = fixture_receipt(YES);
        make_separate_success(separate);
        NSString *separate_sha = orion_depth_serve_receipt_sha256(separate);
        if (separate_sha == nil) {
            fprintf(stderr, "separate-mode (150-eval) valid receipt was rejected\n");
            failures++;
        }

        if (!orion_depth_serve_is_lower_hex_sha256(semantic) ||
            orion_depth_serve_is_lower_hex_sha256([semantic uppercaseString])) {
            fprintf(stderr, "strict lowercase SHA-256 validation failed\n");
            failures++;
        }
        NSMutableDictionary *missing = fixture_receipt(YES);
        [missing removeObjectForKey:@"private_invocation_known"];
        if (orion_depth_serve_receipt_sha256(missing) != nil) {
            fprintf(stderr, "receipt schema accepted missing private_invocation_known\n");
            failures++;
        }
        NSMutableDictionary *old_key = fixture_receipt(YES);
        old_key[@"private_evals_attempted"] = @60;
        [old_key removeObjectForKey:@"private_invocations_actual"];
        if (orion_depth_serve_receipt_sha256(old_key) != nil) {
            fprintf(stderr, "receipt schema accepted legacy counter key\n");
            failures++;
        }
        NSMutableArray<NSDictionary *> *invalid_receipts = [NSMutableArray array];
        NSMutableDictionary *invalid = fixture_receipt(YES);
        invalid[@"unexpected"] = @1;
        [invalid_receipts addObject:invalid];

        invalid = fixture_receipt(YES);
        invalid[@"ok"] = @1;
        [invalid_receipts addObject:invalid];

        invalid = fixture_receipt(YES);
        invalid[@"private_invocations_actual"] = @59;
        [invalid_receipts addObject:invalid];

        invalid = fixture_receipt(YES);
        invalid[@"frame_duration_ns"] = @999999;
        [invalid_receipts addObject:invalid];

        invalid = fixture_receipt(NO);
        invalid[@"tokens"] = @[];
        [invalid_receipts addObject:invalid];

        invalid = fixture_receipt(YES);
        invalid[@"logical_frame_id"] = @(-1);
        [invalid_receipts addObject:invalid];

        invalid = fixture_receipt(YES);
        invalid[@"request_sequence"] = @7.5;
        [invalid_receipts addObject:invalid];
        for (NSDictionary *invalid in invalid_receipts) {
            if (orion_depth_serve_receipt_sha256(invalid) != nil) {
                fprintf(stderr, "receipt schema accepted an invalid receipt\n");
                failures++;
            }
        }
        if (failures) return 1;
        printf("test_depth_serve_protocol_v2: PASS\n");
        return 0;
    }
}
