// tests/test_ane_passes.m — T133: ANE-specific pass tests

#import <Foundation/Foundation.h>
#include "compiler/graph.h"
#include "compiler/builder.h"
#include "compiler/pass_sram.h"
#include "compiler/pass_uniform_outputs.h"
#include "compiler/pass_ane_validate.h"
#include "compiler/binding_plan.h"
#include "compiler/kernel_adapter.h"
#include "compiler/validate.h"

static int g_passed = 0, g_failed = 0;

#define TEST(name) do { \
    printf("  %-50s ", #name); \
    if (test_##name()) { printf("PASS\n"); g_passed++; } \
    else { printf("FAIL\n"); g_failed++; } \
} while(0)

static void write_le32(uint8_t* bytes, uint32_t value) {
    bytes[0] = (uint8_t)value;
    bytes[1] = (uint8_t)(value >> 8);
    bytes[2] = (uint8_t)(value >> 16);
    bytes[3] = (uint8_t)(value >> 24);
}

// Test: SRAM estimation for known dimensions
static bool test_sram_normal(void) {
    OrionGraph* g = orion_graph_create();
    int shape[4] = {1, 768, 1, 256}; // 768*256*2 = 393KB < 32MB
    int x = orion_gb_input(g, "x", ORION_DTYPE_FP16, shape);
    int y = orion_gb_add(g, x, x, "y");
    orion_gb_output(g, y, "y");

    bool spill = orion_pass_sram(g);
    orion_graph_free(g);
    return !spill; // Should NOT have spill risk
}

// Test: SRAM estimation flags large tensors
static bool test_sram_large_tensor(void) {
    OrionGraph* g = orion_graph_create();
    // 32768 * 1024 * 2 = 64MB > 32MB
    int shape[4] = {1, 32768, 1, 1024};
    int x = orion_gb_input(g, "x", ORION_DTYPE_FP16, shape);
    int y = orion_gb_add(g, x, x, "y");
    orion_gb_output(g, y, "y");

    bool spill = orion_pass_sram(g);
    orion_graph_free(g);
    return spill; // Should flag spill risk
}

// Test: tensor bytes calculation
static bool test_tensor_bytes(void) {
    OrionNode n;
    memset(&n, 0, sizeof(n));
    n.dtype = ORION_DTYPE_FP16;
    n.shape[0] = 1; n.shape[1] = 768; n.shape[2] = 1; n.shape[3] = 64;

    int64_t bytes = orion_node_tensor_bytes(&n);
    return bytes == 1 * 768 * 1 * 64 * 2; // fp16 = 2 bytes
}

// Test: uniform outputs detects mixed sizes
static bool test_uniform_outputs_mixed(void) {
    OrionGraph* g = orion_graph_create();
    int shape_d[4] = {1, 768, 1, 256};
    int shape_h[4] = {1, 2048, 1, 256};
    int x = orion_gb_input(g, "x", ORION_DTYPE_FP16, shape_d);
    int y = orion_gb_input(g, "y", ORION_DTYPE_FP16, shape_h);

    orion_gb_output(g, x, "out_d");
    orion_gb_output(g, y, "out_h");

    bool needs_pad = orion_pass_uniform_outputs(g);
    int max_ch = orion_outputs_max_channels(g);

    orion_graph_free(g);
    return needs_pad && max_ch == 2048;
}

// Test: uniform outputs passes for same sizes
static bool test_uniform_outputs_same(void) {
    OrionGraph* g = orion_graph_create();
    int shape[4] = {1, 768, 1, 256};
    int x = orion_gb_input(g, "x", ORION_DTYPE_FP16, shape);
    int y = orion_gb_add(g, x, x, "y");

    orion_gb_output(g, x, "out1");
    orion_gb_output(g, y, "out2");

    bool needs_pad = orion_pass_uniform_outputs(g);
    orion_graph_free(g);
    return !needs_pad; // Same size, no padding needed
}

// Test: ANE validation catches concat
static bool test_ane_validate_concat(void) {
    OrionGraph* g = orion_graph_create();
    int shape[4] = {1, 768, 1, 64};
    int x = orion_gb_input(g, "x", ORION_DTYPE_FP16, shape);

    OrionNode n;
    memset(&n, 0, sizeof(n));
    n.op = ORION_OP_CONCAT_BANNED;
    snprintf(n.name, ORION_MAX_NAME, "bad");
    n.dtype = ORION_DTYPE_FP16;
    memcpy(n.shape, shape, sizeof(n.shape));
    n.inputs[0] = x;
    n.n_inputs = 1;
    n.is_live = true;
    int bad = orion_graph_add_node(g, &n);
    orion_gb_output(g, bad, "bad");

    OrionANEValidationResult r = orion_pass_ane_validate(g);
    orion_graph_free(g);
    return !r.valid && r.n_errors > 0 && r.errors[0].constraint_id == 1;
}

// Test: ANE validation passes valid graph
static bool test_ane_validate_valid(void) {
    OrionGraph* g = orion_graph_create();
    int shape[4] = {1, 768, 1, 64}; // 768*64*2 = 98KB > 49KB minimum
    int x = orion_gb_input(g, "x", ORION_DTYPE_FP16, shape);
    int y = orion_gb_add(g, x, x, "y");
    orion_gb_output(g, y, "y");

    OrionANEValidationResult r = orion_pass_ane_validate(g);
    orion_graph_free(g);
    return r.valid;
}

static OrionGraph* invalid_concat_frontend(int layer __attribute__((unused)),
                                           int bucket __attribute__((unused)),
                                           const OrionModelConfig* cfg __attribute__((unused))) {
    OrionGraph* g = orion_graph_create();
    int shape[4] = {1, 32, 1, 16};
    int x = orion_gb_input(g, "x", ORION_DTYPE_FP16, shape);
    OrionNode n = {0};
    n.op = ORION_OP_CONCAT_BANNED;
    snprintf(n.name, sizeof(n.name), "bad_concat");
    n.dtype = ORION_DTYPE_FP16;
    memcpy(n.shape, shape, sizeof(shape));
    n.inputs[0] = x;
    n.n_inputs = 1;
    n.is_live = true;
    int bad = orion_graph_add_node(g, &n);
    orion_gb_output(g, bad, "bad_concat");
    return g;
}

static bool test_adapter_rejects_invalid_graph(void) {
    NSString* mil = orion_kernel_adapter_generate_mil(
        invalid_concat_frontend, 0, 16, NULL);
    return mil == nil;
}

static bool test_ane_validate_bad_blob_and_bias(void) {
    OrionGraph* g = orion_graph_create();
    int input_shape[4] = {1, 4, 1, 16};
    int weight_shape[4] = {4, 4, 1, 1};
    int bias_shape[4] = {1, 4, 1, 1};
    int x = orion_gb_input(g, "x", ORION_DTYPE_FP16, input_shape);
    int w = orion_gb_const_weight(g, "w", ORION_DTYPE_FP16, weight_shape,
                                  "../unsafe.blob", 128);
    int b = orion_gb_const_weight(g, "b", ORION_DTYPE_FP16, bias_shape,
                                  "@model_path/b.blob", 64);
    int y = orion_gb_conv1x1(g, x, w, b, "y", 4, 16);
    orion_gb_output(g, y, "y");
    OrionANEValidationResult r = orion_pass_ane_validate(g);
    orion_graph_free(g);
    return !r.valid && r.n_errors >= 3;
}

static bool test_ane_validate_rejects_unimplemented_ops(void) {
    OrionGraph* g = orion_graph_create();
    int shape[4] = {1, 32, 1, 16};
    int x = orion_gb_input(g, "x", ORION_DTYPE_FP16, shape);
    OrionNode split = {0};
    split.op = ORION_OP_SPLIT;
    snprintf(split.name, sizeof(split.name), "unsupported_split");
    split.dtype = ORION_DTYPE_FP16;
    memcpy(split.shape, shape, sizeof(shape));
    split.inputs[0] = x;
    split.n_inputs = 1;
    split.is_live = true;
    int node = orion_graph_add_node(g, &split);
    orion_gb_output(g, node, "unsupported_split");
    OrionANEValidationResult result = orion_pass_ane_validate(g);
    orion_graph_free(g);
    return !result.valid && result.n_errors == 1 && result.errors[0].constraint_id == 15;
}

// Test: generated plans pad small external tensors to the minimum allocation.
static bool test_binding_plan_pads_small_tensor(void) {
    OrionGraph* g = orion_graph_create();
    int shape[4] = {1, 32, 1, 1}; // 32*1*2 = 64 bytes << 49KB
    int x = orion_gb_input(g, "x", ORION_DTYPE_FP16, shape);
    int y = orion_gb_add(g, x, x, "y");
    orion_gb_output(g, y, "y");

    OrionBindingPlan plan;
    char error[256] = {0};
    bool ok = orion_binding_plan_generate(g, "small", &plan, error, sizeof(error));
    orion_graph_free(g);
    return ok && plan.inputs[0].logical_bytes == 64 &&
           plan.inputs[0].allocation_bytes == ORION_ANE_MIN_ALLOCATION_BYTES &&
           plan.outputs[0].allocation_bytes == ORION_ANE_MIN_ALLOCATION_BYTES;
}

// Test: plans sort multi-input names and use one allocation size per group.
static bool test_binding_plan_sorts_and_pads(void) {
    OrionGraph* g = orion_graph_create();
    int small[4] = {1, 32, 1, 16};
    int large[4] = {1, 1024, 1, 16};
    int z = orion_gb_input(g, "z", ORION_DTYPE_FP16, small);
    int a = orion_gb_input(g, "a", ORION_DTYPE_FP16, large);
    int y = orion_gb_add(g, z, z, "y");
    orion_gb_output(g, y, "z_out");
    orion_gb_output(g, a, "a_out");

    OrionBindingPlan plan;
    char error[256] = {0};
    bool ok = orion_binding_plan_generate(g, "sorted", &plan, error, sizeof(error));
    orion_graph_free(g);
    return ok && strcmp(plan.inputs[0].name, "a") == 0 &&
           strcmp(plan.inputs[1].name, "z") == 0 &&
            strcmp(plan.outputs[0].name, "a") == 0 &&
            strcmp(plan.outputs[1].name, "y") == 0 &&
           plan.inputs[0].allocation_bytes == plan.inputs[1].allocation_bytes &&
           plan.outputs[0].allocation_bytes == plan.outputs[1].allocation_bytes;
}

static bool test_binding_plan_serializes_deterministically(void) {
    OrionGraph* g = orion_graph_create();
    int shape[4] = {1, 32, 1, 16};
    int x = orion_gb_input(g, "x", ORION_DTYPE_FP16, shape);
    int y = orion_gb_add(g, x, x, "y");
    orion_gb_output(g, y, "y");
    OrionBindingPlanIdentity identity = {
        .orion_revision = "0123456789abcdef0123456789abcdef01234567",
        .compiler_build = "test-clang",
        .weight_root = NULL,
    };
    OrionBindingPlan plan;
    char error[256] = {0};
    bool ok = orion_binding_plan_generate_with_identity(
        g, "serialize", &identity, &plan, error, sizeof(error));
    orion_graph_free(g);
    orion_binding_plan_set_mil_identity(&plan, "test-mil-program");
    NSString* base = NSTemporaryDirectory();
    NSString* first = [base stringByAppendingPathComponent:@"orion-plan-a.json"];
    NSString* second = [base stringByAppendingPathComponent:@"orion-plan-b.json"];
    NSString* report = [base stringByAppendingPathComponent:@"orion-plan.md"];
    ok = ok && orion_binding_plan_write_json(&plan, first.fileSystemRepresentation,
                                               error, sizeof(error)) &&
         orion_binding_plan_write_json(&plan, second.fileSystemRepresentation,
                                       error, sizeof(error)) &&
         orion_binding_plan_write_markdown(&plan, report.fileSystemRepresentation,
                                           error, sizeof(error));
    NSData* a = [NSData dataWithContentsOfFile:first];
    NSData* b = [NSData dataWithContentsOfFile:second];
    NSString* json_text = [[NSString alloc] initWithData:a encoding:NSUTF8StringEncoding];
    NSString* markdown = [NSString stringWithContentsOfFile:report
        encoding:NSUTF8StringEncoding error:nil];
    [[NSFileManager defaultManager] removeItemAtPath:first error:nil];
    [[NSFileManager defaultManager] removeItemAtPath:second error:nil];
    [[NSFileManager defaultManager] removeItemAtPath:report error:nil];
    return ok && [a isEqualToData:b] &&
        [json_text containsString:@"\"mil_identity\""] &&
        [markdown containsString:@"Graph identity"] &&
        [markdown containsString:@"MIL identity"] &&
        strlen(plan.mil_identity) == ORION_SHA256_HEX_LENGTH &&
        strlen(plan.graph_identity) == ORION_SHA256_HEX_LENGTH;
}

static bool test_binding_plan_identity_includes_io_abi(void) {
    OrionGraph* first = orion_graph_create();
    OrionGraph* second = orion_graph_create();
    int shape[4] = {1, 32, 1, 16};
    int first_x = orion_gb_input(first, "x", ORION_DTYPE_FP16, shape);
    int second_x = orion_gb_input(second, "x", ORION_DTYPE_FP16, shape);
    int first_y = orion_gb_add(first, first_x, first_x, "y");
    int second_y = orion_gb_add(second, second_x, second_x, "y");
    orion_gb_output(first, first_y, "result");
    orion_gb_output(second, second_y, "renamed_result");
    OrionBindingPlan first_plan, second_plan;
    char error[256] = {0};
    bool ok = orion_binding_plan_generate(first, "first", &first_plan, error, sizeof(error)) &&
        orion_binding_plan_generate(second, "second", &second_plan, error, sizeof(error));
    orion_graph_free(first);
    orion_graph_free(second);
    return ok && strcmp(first_plan.graph_identity, second_plan.graph_identity) != 0;
}

static bool test_binding_plan_hashes_and_sizes_weights(void) {
    NSString* root = [NSTemporaryDirectory() stringByAppendingPathComponent:
        [NSString stringWithFormat:@"orion-weights-%@", NSUUID.UUID.UUIDString]];
    [[NSFileManager defaultManager] createDirectoryAtPath:root
        withIntermediateDirectories:YES attributes:nil error:nil];
    NSString* path = [root stringByAppendingPathComponent:@"w.blob"];
    NSMutableData* data = [NSMutableData dataWithLength:136];
    uint8_t* bytes = data.mutableBytes;
    bytes[0] = 0x01;
    bytes[4] = 0x02;
    bytes[64] = 0xef; bytes[65] = 0xbe; bytes[66] = 0xad; bytes[67] = 0xde;
    bytes[68] = 0x01;
    write_le32(bytes + 72, 8);
    write_le32(bytes + 80, 128);
    [data writeToFile:path atomically:YES];

    OrionGraph* g = orion_graph_create();
    int input_shape[4] = {1, 2, 1, 16};
    int weight_shape[4] = {2, 2, 1, 1};
    int x = orion_gb_input(g, "x", ORION_DTYPE_FP16, input_shape);
    int w = orion_gb_const_weight(g, "w", ORION_DTYPE_FP16, weight_shape,
                                  "@model_path/w.blob", 64);
    int y = orion_gb_conv1x1(g, x, w, -1, "y", 2, 16);
    orion_gb_output(g, y, "y");
    OrionBindingPlanIdentity identity = {
        .orion_revision = "0123456789abcdef0123456789abcdef01234567",
        .compiler_build = "test-clang",
        .weight_root = root.fileSystemRepresentation,
    };
    OrionBindingPlan plan;
    char error[256] = {0};
    bool ok = orion_binding_plan_generate_with_identity(
        g, "weighted", &identity, &plan, error, sizeof(error));
    [data setLength:135];
    [data writeToFile:path atomically:YES];
    OrionBindingPlan truncated;
    bool rejects_truncated = !orion_binding_plan_generate_with_identity(
        g, "truncated", &identity, &truncated, error, sizeof(error));
    [data setLength:136];
    memset(data.mutableBytes, 0, data.length);
    [data writeToFile:path atomically:YES];
    OrionBindingPlan malformed;
    bool rejects_malformed = !orion_binding_plan_generate_with_identity(
        g, "malformed", &identity, &malformed, error, sizeof(error));
    orion_graph_free(g);
    [[NSFileManager defaultManager] removeItemAtPath:root error:nil];
    return ok && plan.n_weights == 1 && plan.weights[0].offset == 64 &&
        strlen(plan.weights[0].sha256) == ORION_SHA256_HEX_LENGTH &&
        rejects_truncated && rejects_malformed;
}

// G009: Verify the binding plan accepts a multi-chunk aggregate BLOBFILE
// where two weights share one physical file at different chunk offsets.
// Layout matches the diagnostic probe: 64-byte chunk headers between payloads.
static bool test_binding_plan_accepts_multichunk_aggregate(void) {
    NSString* root = [NSTemporaryDirectory() stringByAppendingPathComponent:
        [NSString stringWithFormat:@"orion-aggregate-%@", NSUUID.UUID.UUIDString]];
    [[NSFileManager defaultManager] createDirectoryAtPath:root
        withIntermediateDirectories:YES attributes:nil error:nil];

    // Layout: preamble, header0@64/payload0@128, then header1@136/payload1@200.
    // Each header independently declares its payload size at +8 and absolute
    // payload offset at +16, matching the per-chunk parser.
    NSString* agg_path = [root stringByAppendingPathComponent:@"aggregate.blob"];
    NSMutableData* data = [NSMutableData dataWithLength:208];
    uint8_t* bytes = data.mutableBytes;
    bytes[0] = 0x01; bytes[4] = 0x02;
    bytes[64] = 0xef; bytes[65] = 0xbe; bytes[66] = 0xad; bytes[67] = 0xde;
    bytes[68] = 0x01;
    write_le32(bytes + 72, 8);
    write_le32(bytes + 80, 128);
    bytes[136] = 0xef; bytes[137] = 0xbe; bytes[138] = 0xad; bytes[139] = 0xde;
    bytes[140] = 0x01;
    write_le32(bytes + 144, 8);
    write_le32(bytes + 152, 200);
    for (int i = 0; i < 8; i++) bytes[128 + i] = (uint8_t)(i + 1);
    for (int i = 0; i < 8; i++) bytes[200 + i] = (uint8_t)(i + 10);
    [data writeToFile:agg_path atomically:YES];

    OrionGraph* g = orion_graph_create();
    int input_shape[4] = {1, 2, 1, 16};
    int weight_shape[4] = {2, 2, 1, 1};
    int x = orion_gb_input(g, "x", ORION_DTYPE_FP16, input_shape);
    int w0 = orion_gb_const_weight(g, "w0", ORION_DTYPE_FP16, weight_shape,
                                   "@model_path/aggregate.blob", 64);
    int w1 = orion_gb_const_weight(g, "w1", ORION_DTYPE_FP16, weight_shape,
                                   "@model_path/aggregate.blob", 136);
    int conv0 = orion_gb_conv1x1(g, x, w0, -1, "c0", 2, 16);
    int conv1 = orion_gb_conv1x1(g, conv0, w1, -1, "c1", 2, 16);
    orion_gb_output(g, conv1, "y");

    OrionBindingPlanIdentity identity = {
        .orion_revision = "0123456789abcdef0123456789abcdef01234567",
        .compiler_build = "test-clang",
        .weight_root = root.fileSystemRepresentation,
    };
    OrionBindingPlan plan;
    char error[256] = {0};
    bool ok = orion_binding_plan_generate_with_identity(
        g, "aggregate-test", &identity, &plan, error, sizeof(error));
    orion_graph_free(g);
    [[NSFileManager defaultManager] removeItemAtPath:root error:nil];
    return ok && plan.n_weights == 2 &&
           plan.weights[0].offset == 64 && plan.weights[1].offset == 136;
}

// Physical payload declarations, rather than nominal chunk positions, govern
// overlap. Distinct headers may maliciously point at the same payload.
static bool test_binding_plan_rejects_overlapping_chunks(void) {
    NSString* root = [NSTemporaryDirectory() stringByAppendingPathComponent:
        [NSString stringWithFormat:@"orion-overlap-%@", NSUUID.UUID.UUIDString]];
    [[NSFileManager defaultManager] createDirectoryAtPath:root
        withIntermediateDirectories:YES attributes:nil error:nil];

    NSString* agg_path = [root stringByAppendingPathComponent:@"aggregate.blob"];
    NSMutableData* data = [NSMutableData dataWithLength:208];
    uint8_t* bytes = data.mutableBytes;
    bytes[0] = 0x01; bytes[4] = 0x02;
    bytes[64] = 0xef; bytes[65] = 0xbe; bytes[66] = 0xad; bytes[67] = 0xde;
    bytes[68] = 0x01;
    write_le32(bytes + 72, 8);
    write_le32(bytes + 80, 200);
    bytes[136] = 0xef; bytes[137] = 0xbe; bytes[138] = 0xad; bytes[139] = 0xde;
    bytes[140] = 0x01;
    write_le32(bytes + 144, 8);
    write_le32(bytes + 152, 200);  // same declared physical payload as w0
    [data writeToFile:agg_path atomically:YES];

    OrionGraph* g = orion_graph_create();
    int input_shape[4] = {1, 2, 1, 16};
    int weight_shape[4] = {2, 2, 1, 1};
    int x = orion_gb_input(g, "x", ORION_DTYPE_FP16, input_shape);
    int w0 = orion_gb_const_weight(g, "w0", ORION_DTYPE_FP16, weight_shape,
                                   "@model_path/aggregate.blob", 64);
    int w1 = orion_gb_const_weight(g, "w1", ORION_DTYPE_FP16, weight_shape,
                                   "@model_path/aggregate.blob", 136);
    int conv0 = orion_gb_conv1x1(g, x, w0, -1, "c0", 2, 16);
    int conv1 = orion_gb_conv1x1(g, conv0, w1, -1, "c1", 2, 16);
    orion_gb_output(g, conv1, "y");

    OrionBindingPlanIdentity identity = {
        .orion_revision = "0123456789abcdef0123456789abcdef01234567",
        .compiler_build = "test-clang",
        .weight_root = root.fileSystemRepresentation,
    };
    OrionBindingPlan plan;
    char error[256] = {0};
    bool rejects = !orion_binding_plan_generate_with_identity(
        g, "overlap-test", &identity, &plan, error, sizeof(error));
    orion_graph_free(g);
    [[NSFileManager defaultManager] removeItemAtPath:root error:nil];
    return rejects;
}

static bool test_graph_validation_rejects_bad_operator_shapes(void) {
    int shape[4] = {1, 4, 1, 4};
    int shape_const_shape[4] = {4, 0, 0, 0};

    OrionGraph* reshape = orion_graph_create();
    int x = orion_gb_input(reshape, "x", ORION_DTYPE_FP16, shape);
    int bad_shape[4] = {1, 4, 1, 5};
    int shape_node = orion_gb_const_int32(
        reshape, "shape", shape_const_shape, bad_shape, 4);
    int y = orion_gb_reshape(reshape, x, shape_node, "bad_reshape", bad_shape);
    orion_gb_output(reshape, y, "bad_reshape");
    OrionValidationResult reshape_result = orion_graph_validate(reshape);
    orion_graph_free(reshape);

    OrionGraph* transpose = orion_graph_create();
    x = orion_gb_input(transpose, "x", ORION_DTYPE_FP16, shape);
    int perm_values[4] = {0, 1, 3, 2};
    int perm = orion_gb_const_int32(
        transpose, "perm", shape_const_shape, perm_values, 4);
    int wrong_transpose_shape[4] = {1, 4, 1, 4};
    y = orion_gb_transpose(
        transpose, x, perm, "bad_transpose", perm_values,
        wrong_transpose_shape);
    orion_gb_output(transpose, y, "bad_transpose");
    OrionValidationResult transpose_result = orion_graph_validate(transpose);
    orion_graph_free(transpose);

    OrionGraph* matmul = orion_graph_create();
    int a_shape[4] = {2, 4, 3, 5};
    int b_shape[4] = {2, 1, 6, 7};
    int a = orion_gb_input(matmul, "a", ORION_DTYPE_FP16, a_shape);
    int b = orion_gb_input(matmul, "b", ORION_DTYPE_FP16, b_shape);
    int output_shape[4] = {2, 4, 3, 7};
    y = orion_gb_matmul(matmul, a, b, false, false, "bad_matmul", output_shape);
    orion_gb_output(matmul, y, "bad_matmul");
    OrionValidationResult matmul_result = orion_graph_validate(matmul);
    orion_graph_free(matmul);

    OrionGraph* broadcast = orion_graph_create();
    int left_shape[4] = {2, 4, 3, 5};
    int right_shape[4] = {2, 3, 3, 5};
    a = orion_gb_input(broadcast, "a", ORION_DTYPE_FP16, left_shape);
    b = orion_gb_input(broadcast, "b", ORION_DTYPE_FP16, right_shape);
    y = orion_gb_add(broadcast, a, b, "bad_broadcast");
    orion_gb_output(broadcast, y, "bad_broadcast");
    OrionValidationResult broadcast_result = orion_graph_validate(broadcast);
    orion_graph_free(broadcast);

    return !reshape_result.valid && !transpose_result.valid &&
        !matmul_result.valid && !broadcast_result.valid;
}

static bool test_binding_plan_rejects_malformed_fixed_strings(void) {
    OrionGraph* g = orion_graph_create();
    int shape[4] = {1, 32, 1, 16};
    int x = orion_gb_input(g, "x", ORION_DTYPE_FP16, shape);
    orion_gb_output(g, x, "x");
    OrionBindingPlan plan;
    char error[256] = {0};
    bool ok = orion_binding_plan_generate(g, "safe", &plan, error, sizeof(error));
    orion_graph_free(g);
    OrionBindingPlan bad_sha = plan;
    memset(bad_sha.graph_identity, 'g', sizeof(bad_sha.graph_identity));
    OrionBindingPlan bad_name = plan;
    memset(bad_name.inputs[0].name, 'x', sizeof(bad_name.inputs[0].name));
    return ok && !orion_binding_plan_validate(&bad_sha, error, sizeof(error)) &&
        !orion_binding_plan_validate(&bad_name, error, sizeof(error));
}

static OrionGraph* adapter_weight_free_frontend(int layer __attribute__((unused)),
                                                int bucket __attribute__((unused)),
                                                const OrionModelConfig* cfg __attribute__((unused))) {
    OrionGraph* g = orion_graph_create();
    int shape[4] = {1, 32, 1, 16};
    int x = orion_gb_input(g, "x", ORION_DTYPE_FP16, shape);
    int y = orion_gb_add(g, x, x, "y");
    orion_gb_output(g, y, "y");
    return g;
}

static bool test_validate_finalized_requires_mil_identity(void) {
    OrionGraph* g = orion_graph_create();
    int shape[4] = {1, 32, 1, 16};
    int x = orion_gb_input(g, "x", ORION_DTYPE_FP16, shape);
    orion_gb_output(g, x, "x");
    OrionBindingPlan plan;
    char error[256] = {0};
    bool ok = orion_binding_plan_generate(g, "test", &plan, error, sizeof(error));
    orion_graph_free(g);
    bool rejects_empty = !orion_binding_plan_validate_finalized(&plan, error, sizeof(error));
    orion_binding_plan_set_mil_identity(&plan, "dummy-mil");
    bool accepts_set = orion_binding_plan_validate_finalized(&plan, error, sizeof(error));
    return ok && rejects_empty && accepts_set;
}

static bool test_artifact_pair_publication(void) {
    OrionBindingPlanIdentity identity = {
        .orion_revision = "0123456789abcdef0123456789abcdef01234567",
        .compiler_build = "test-clang",
        .weight_root = NULL,
    };
    NSString* dir = [NSTemporaryDirectory() stringByAppendingPathComponent:
        [NSString stringWithFormat:@"orion-art-%@", NSUUID.UUID.UUIDString]];
    [[NSFileManager defaultManager] createDirectoryAtPath:dir
        withIntermediateDirectories:YES attributes:nil error:nil];
    NSString* jsonPath = [dir stringByAppendingPathComponent:@"plan.json"];
    NSString* mdPath = [dir stringByAppendingPathComponent:@"plan.md"];

    OrionBindingPlan plan;
    char error[256] = {0};
    (void)error; // artifacts path; adapter does not yet surface diagnostics here
    NSString* mil = orion_kernel_adapter_generate_mil_with_artifacts(
        adapter_weight_free_frontend, 0, 16, NULL, &identity,
        jsonPath.fileSystemRepresentation, mdPath.fileSystemRepresentation, &plan);
    bool files_exist = [[NSFileManager defaultManager] fileExistsAtPath:jsonPath] &&
                       [[NSFileManager defaultManager] fileExistsAtPath:mdPath];
    NSData* jsonData = [NSData dataWithContentsOfFile:jsonPath];
    NSString* mdContent = [NSString stringWithContentsOfFile:mdPath
                                       encoding:NSUTF8StringEncoding error:nil];
    bool json_nonempty = jsonData.length > 0;
    bool md_nonempty = mdContent.length > 0;
    NSString* jsonText = [[NSString alloc] initWithData:jsonData encoding:NSUTF8StringEncoding];
    bool mil_identity_in_json = [jsonText containsString:@"\"mil_identity\""];

    OrionBindingPlan plan2;
    NSString* mil2 = orion_kernel_adapter_generate_mil_with_artifacts(
        adapter_weight_free_frontend, 0, 16, NULL, &identity,
        jsonPath.fileSystemRepresentation, mdPath.fileSystemRepresentation, &plan2);
    bool idempotent = mil2 != nil && plan2.n_inputs == plan.n_inputs;

    [[NSFileManager defaultManager] removeItemAtPath:dir error:nil];
    return mil != nil && files_exist && json_nonempty && md_nonempty &&
           mil_identity_in_json && idempotent &&
           strcmp(plan.mil_identity, plan2.mil_identity) == 0;
}

int main(int argc __attribute__((unused)), char* argv[] __attribute__((unused))) {
    @autoreleasepool {
        printf("test_ane_passes:\n");

        TEST(sram_normal);
        TEST(sram_large_tensor);
        TEST(tensor_bytes);
        TEST(uniform_outputs_mixed);
        TEST(uniform_outputs_same);
        TEST(ane_validate_concat);
        TEST(ane_validate_valid);
        TEST(adapter_rejects_invalid_graph);
        TEST(ane_validate_bad_blob_and_bias);
        TEST(ane_validate_rejects_unimplemented_ops);
        TEST(binding_plan_pads_small_tensor);
        TEST(binding_plan_sorts_and_pads);
        TEST(binding_plan_serializes_deterministically);
        TEST(binding_plan_identity_includes_io_abi);
        TEST(binding_plan_hashes_and_sizes_weights);
        TEST(binding_plan_accepts_multichunk_aggregate);
        TEST(binding_plan_rejects_overlapping_chunks);
        TEST(graph_validation_rejects_bad_operator_shapes);
        TEST(binding_plan_rejects_malformed_fixed_strings);
        TEST(validate_finalized_requires_mil_identity);
        TEST(artifact_pair_publication);

        printf("\n%d/%d passed\n", g_passed, g_passed + g_failed);
        return g_failed > 0 ? 1 : 0;
    }
}
