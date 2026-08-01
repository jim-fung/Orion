// tests/test_aggregate_plan_generation.m — Portable aggregate BLOBFILE plan regression.
//
// This is intentionally lower-level than the VoXtream ANE integration tests:
// it creates a production-shaped aggregate BLOBFILE on disk, generates the
// binding plan from graph blob references, and then applies the runtime
// dictionary gate to the same bytes without requiring ANE availability.

#import <Foundation/Foundation.h>

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "core/ane_runtime.h"
#include "compiler/binding_plan.h"
#include "compiler/builder.h"
#include "compiler/graph.h"

static void write_le32(uint8_t *bytes, uint32_t value) {
    bytes[0] = (uint8_t)value;
    bytes[1] = (uint8_t)(value >> 8);
    bytes[2] = (uint8_t)(value >> 16);
    bytes[3] = (uint8_t)(value >> 24);
}

static OrionGraph *aggregate_graph(void) {
    OrionGraph *graph = orion_graph_create();
    int shape[4] = {1, 2, 1, 2};
    int input = orion_gb_input(graph, "input", ORION_DTYPE_FP16, shape);
    int first = orion_gb_const_weight(graph, "first", ORION_DTYPE_FP16, shape,
                                      "@model_path/weights/aggregate.blob", 64);
    int second = orion_gb_const_weight(graph, "second", ORION_DTYPE_FP16, shape,
                                       "@model_path/weights/aggregate.blob", 136);
    int with_first = orion_gb_add(graph, input, first, "with_first");
    int result = orion_gb_add(graph, with_first, second, "result");
    orion_gb_output(graph, result, "result");
    return graph;
}

static bool generate_plan(const char *root, OrionBindingPlan *plan) {
    OrionBindingPlanIdentity identity = {
        .orion_revision = "aggregate-plan-regression",
        .compiler_build = "aggregate-plan-regression",
        .weight_root = root,
    };
    char error[256] = {0};
    OrionGraph *graph = aggregate_graph();
    bool generated = graph && orion_binding_plan_generate_with_identity(
        graph, "portable-aggregate", &identity, plan, error, sizeof(error));
    orion_graph_free(graph);
    return generated;
}

static NSData *aggregate_blob(void) {
    // A BLOBFILE header is 64 bytes.  Each graph blob offset below addresses
    // its own header; neither payload is derived from the first chunk.
    NSMutableData *blob = [NSMutableData dataWithLength:272];
    uint8_t *bytes = blob.mutableBytes;
    const uint64_t first_header = 64, second_header = 136;
    const uint64_t first_payload = 256, second_payload = 264;

    bytes[0] = 1; bytes[4] = 2;  // Retain the normal BLOBFILE file preamble.
    for (int chunk = 0; chunk < 2; chunk++) {
        uint64_t header = chunk ? second_header : first_header;
        uint64_t payload = chunk ? second_payload : first_payload;
        bytes[header + 0] = 0xef;
        bytes[header + 1] = 0xbe;
        bytes[header + 2] = 0xad;
        bytes[header + 3] = 0xde;
        bytes[header + 4] = 1;
        write_le32(bytes + header + 8, 8);
        write_le32(bytes + header + 16, (uint32_t)payload);
        for (int i = 0; i < 8; i++) bytes[payload + i] = (uint8_t)(chunk * 16 + i);
    }
    return blob;
}

static bool test_aggregate_plan_generation_and_runtime_validation(void) {
    NSString *root = [NSTemporaryDirectory()
        stringByAppendingPathComponent:[NSString stringWithFormat:@"orion-aggregate-%@",
                                       NSUUID.UUID.UUIDString]];
    NSString *weights = [root stringByAppendingPathComponent:@"weights"];
    NSString *blob_path = [weights stringByAppendingPathComponent:@"aggregate.blob"];
    NSFileManager *manager = NSFileManager.defaultManager;
    NSData *blob = aggregate_blob();
    bool created = [manager createDirectoryAtPath:weights
                      withIntermediateDirectories:YES attributes:nil error:nil];
    bool written = created && [blob writeToFile:blob_path atomically:YES];
    if (!written) {
        [manager removeItemAtPath:root error:nil];
        return false;
    }

    OrionBindingPlan plan = {0};
    bool generated = generate_plan(root.fileSystemRepresentation, &plan);
    bool finalized = generated && orion_binding_plan_set_mil_identity(
        &plan, "program(1.3) { func main<ios18>() {} -> (); }");
    char validation_error[256] = {0};
    bool binding_valid = finalized && orion_binding_plan_validate_finalized(
        &plan, validation_error, sizeof(validation_error));
    NSDictionary *dictionary = @{
        @"@model_path/weights/aggregate.blob": @{ @"offset": @0, @"data": blob }
    };
    bool runtime_valid = binding_valid &&
        orion_validate_weight_dictionary_for_plan(&plan, dictionary);

    // A header corruption must fail in both trust boundaries: generation reads
    // the physical artifact and runtime validation reads its submitted bytes.
    NSMutableData *bad_magic = [blob mutableCopy];
    ((uint8_t *)bad_magic.mutableBytes)[136] ^= 1;
    bool binding_rejects_magic = false;
    if ([bad_magic writeToFile:blob_path atomically:YES]) {
        OrionBindingPlan rejected_plan = {0};
        binding_rejects_magic = !generate_plan(root.fileSystemRepresentation, &rejected_plan);
    }
    bool runtime_rejects_magic = !orion_validate_weight_dictionary_for_plan(
        &plan, @{ @"@model_path/weights/aggregate.blob":
                      @{ @"offset": @0, @"data": bad_magic } });

    // A malformed payload offset in the second header is likewise rejected.
    NSMutableData *bad_payload_offset = [blob mutableCopy];
    write_le32((uint8_t *)bad_payload_offset.mutableBytes + 136 + 16, 270);
    bool binding_rejects_payload_offset = false;
    if ([bad_payload_offset writeToFile:blob_path atomically:YES]) {
        OrionBindingPlan rejected_plan = {0};
        binding_rejects_payload_offset =
            !generate_plan(root.fileSystemRepresentation, &rejected_plan);
    }
    bool runtime_rejects_payload_offset = !orion_validate_weight_dictionary_for_plan(
        &plan, @{ @"@model_path/weights/aggregate.blob":
                      @{ @"offset": @0, @"data": bad_payload_offset } });

    [manager removeItemAtPath:root error:nil];
    return generated && plan.n_weights == 2 && finalized && binding_valid && runtime_valid &&
        binding_rejects_magic && runtime_rejects_magic &&
        binding_rejects_payload_offset && runtime_rejects_payload_offset;
}

int main(void) {
    @autoreleasepool {
        printf("test_aggregate_plan_generation:\n");
        bool passed = test_aggregate_plan_generation_and_runtime_validation();
        printf("  aggregate_plan_generation_and_runtime_validation %s\n",
               passed ? "PASS" : "FAIL");
        return passed ? 0 : 1;
    }
}
