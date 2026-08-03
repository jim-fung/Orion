// tests/test_plan_weight_dict.m — Plan-verified BLOBFILE dictionary checks.

#import <Foundation/Foundation.h>
#import <CommonCrypto/CommonDigest.h>
#include "core/ane_runtime.h"
#include "core/kernel.h"
#include "core/iosurface_tensor.h"
#include "compiler/graph.h"
#include "compiler/builder.h"
#include "compiler/binding_plan.h"
#include "compiler/kernel_adapter.h"
#include "compiler/pipeline.h"
#include "orion_test.h"

static int g_passed = 0, g_failed = 0, g_skipped = 0;
// Per-case flag: a test function sets this before returning true to signal
// that its assertions did not actually run (ANE/mil/compile prerequisite
// missing). The TEST macro reads (and resets) it so SKIP is distinguished
// from PASS — without this, a skipped case on a non-ANE host would be counted
// as PASS and main() would report success despite zero assertions running.
static bool g_last_case_skipped = false;
#define TEST(name) do { \
    printf("  %-50s ", #name); \
    g_last_case_skipped = false; \
    if (test_##name()) { \
        if (g_last_case_skipped) { printf("SKIP\n"); g_skipped++; } \
        else { printf("PASS\n"); g_passed++; } \
    } else { printf("FAIL\n"); g_failed++; } \
} while (0)
// SKIP_CASE: mark this case as skipped (prerequisite unmet) and return success
// so the TEST macro routes to the SKIP branch rather than FAIL.
#define SKIP_CASE(reason) do { \
    printf("SKIP(%s) ", reason); \
    g_last_case_skipped = true; \
    return true; \
} while (0)

static void digest_hex(NSData* data, char out[65]) {
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(data.bytes, (CC_LONG)data.length, digest);
    for (int i = 0; i < CC_SHA256_DIGEST_LENGTH; i++) snprintf(out + i * 2, 3, "%02x", digest[i]);
}

static void write_le32(uint8_t* bytes, uint32_t value) {
    bytes[0] = (uint8_t)value;
    bytes[1] = (uint8_t)(value >> 8);
    bytes[2] = (uint8_t)(value >> 16);
    bytes[3] = (uint8_t)(value >> 24);
}

static void fill_entry(OrionBindingPlanEntry* entry, const char* name) {
    snprintf(entry->name, sizeof(entry->name), "%s", name);
    entry->dtype = ORION_DTYPE_FP16;
    entry->declared_shape[0] = 1; entry->declared_shape[1] = 2;
    entry->declared_shape[2] = 1; entry->declared_shape[3] = 2;
    entry->logical_bytes = 8;
    entry->packed_prefix_bytes = 8;
    entry->allocation_bytes = ORION_ANE_MIN_ALLOCATION_BYTES;
    snprintf(entry->layout, sizeof(entry->layout), "packed-NCHW-prefix");
    snprintf(entry->allocation_layout, sizeof(entry->allocation_layout), "flat-byte-buffer");
}

static NSDictionary* valid_dictionary(OrionBindingPlan* plan) {
    NSMutableData* data = [NSMutableData dataWithLength:136];
    uint8_t* bytes = data.mutableBytes;
    bytes[0] = 1; bytes[4] = 2;
    bytes[64] = 0xef; bytes[65] = 0xbe; bytes[66] = 0xad; bytes[67] = 0xde;
    bytes[68] = 1; write_le32(bytes + 72, 8); write_le32(bytes + 80, 128);
    for (int i = 0; i < 8; i++) bytes[128 + i] = (uint8_t)(i + 1);
    digest_hex([data subdataWithRange:NSMakeRange(128, 8)], plan->weights[0].sha256);
    return @{ @"@model_path/weights/w.bin": @{ @"offset": @0, @"data": data } };
}

static bool test_plan_weight_dictionary_is_exact(void) {
    OrionBindingPlan plan = {0};
    plan.version = ORION_BINDING_PLAN_VERSION;
    snprintf(plan.program_name, sizeof(plan.program_name), "weight-test");
    memset(plan.graph_identity, 'a', ORION_SHA256_HEX_LENGTH);
    memset(plan.mil_identity, 'b', ORION_SHA256_HEX_LENGTH);
    snprintf(plan.orion_revision, sizeof(plan.orion_revision), "test");
    snprintf(plan.compiler_build, sizeof(plan.compiler_build), "test");
    snprintf(plan.position_semantics, sizeof(plan.position_semantics), "fixed");
    snprintf(plan.mask_semantics, sizeof(plan.mask_semantics), "fixed");
    snprintf(plan.failure_semantics, sizeof(plan.failure_semantics), "fail-closed");
    plan.n_inputs = plan.n_outputs = 1;
    fill_entry(&plan.inputs[0], "hidden");
    fill_entry(&plan.outputs[0], "hidden_out");
    plan.n_weights = 1;
    snprintf(plan.weights[0].name, sizeof(plan.weights[0].name), "w");
    snprintf(plan.weights[0].path, sizeof(plan.weights[0].path), "@model_path/weights/w.bin");
    plan.weights[0].offset = 64;
    plan.weights[0].dtype = ORION_DTYPE_FP16;
    plan.weights[0].shape[0] = 2; plan.weights[0].shape[1] = 2;
    plan.weights[0].shape[2] = plan.weights[0].shape[3] = 1;
    plan.weights[0].logical_bytes = 8;
    NSDictionary* good = valid_dictionary(&plan);
    NSMutableDictionary* extra = [good mutableCopy];
    extra[@"@model_path/weights/extra.bin"] = good[@"@model_path/weights/w.bin"];
    NSMutableDictionary* wrong_offset = [good mutableCopy];
    wrong_offset[@"@model_path/weights/w.bin"] = @{ @"offset": @64, @"data": good[@"@model_path/weights/w.bin"][@"data"] };
    NSMutableData* tampered_data = [good[@"@model_path/weights/w.bin"][@"data"] mutableCopy];
    ((uint8_t*)tampered_data.mutableBytes)[128] ^= 1;
    NSDictionary* tampered = @{ @"@model_path/weights/w.bin": @{ @"offset": @0, @"data": tampered_data } };
    NSMutableData* bad_first_preamble = [good[@"@model_path/weights/w.bin"][@"data"] mutableCopy];
    ((uint8_t*)bad_first_preamble.mutableBytes)[0] = 0;
    NSMutableData* bad_second_preamble = [good[@"@model_path/weights/w.bin"][@"data"] mutableCopy];
    ((uint8_t*)bad_second_preamble.mutableBytes)[4] = 0;
    return orion_validate_weight_dictionary_for_plan(&plan, good) &&
        !orion_validate_weight_dictionary_for_plan(&plan, @{}) &&
        !orion_validate_weight_dictionary_for_plan(&plan, extra) &&
        !orion_validate_weight_dictionary_for_plan(&plan, wrong_offset) &&
        !orion_validate_weight_dictionary_for_plan(&plan, tampered) &&
        !orion_validate_weight_dictionary_for_plan(&plan,
            @{ @"@model_path/weights/w.bin": @{ @"offset": @0, @"data": bad_first_preamble } }) &&
        !orion_validate_weight_dictionary_for_plan(&plan,
            @{ @"@model_path/weights/w.bin": @{ @"offset": @0, @"data": bad_second_preamble } });
}

static bool test_aggregate_weight_dictionary_hashes_chunk_payloads(void) {
    OrionBindingPlan plan = {0};
    plan.version = ORION_BINDING_PLAN_VERSION;
    snprintf(plan.program_name, sizeof(plan.program_name), "aggregate-weight-test");
    memset(plan.graph_identity, 'a', ORION_SHA256_HEX_LENGTH);
    memset(plan.mil_identity, 'b', ORION_SHA256_HEX_LENGTH);
    snprintf(plan.orion_revision, sizeof(plan.orion_revision), "test");
    snprintf(plan.compiler_build, sizeof(plan.compiler_build), "test");
    snprintf(plan.position_semantics, sizeof(plan.position_semantics), "fixed");
    snprintf(plan.mask_semantics, sizeof(plan.mask_semantics), "fixed");
    snprintf(plan.failure_semantics, sizeof(plan.failure_semantics), "fail-closed");
    plan.n_inputs = plan.n_outputs = 1;
    fill_entry(&plan.inputs[0], "hidden");
    fill_entry(&plan.outputs[0], "hidden_out");
    plan.n_weights = 2;
    for (int i = 0; i < 2; i++) {
        snprintf(plan.weights[i].name, sizeof(plan.weights[i].name), "w%d", i);
        snprintf(plan.weights[i].path, sizeof(plan.weights[i].path),
                 "@model_path/weights/aggregate.bin");
        plan.weights[i].offset = i ? 136 : 64;
        plan.weights[i].dtype = ORION_DTYPE_FP16;
        plan.weights[i].shape[0] = 2; plan.weights[i].shape[1] = 2;
        plan.weights[i].shape[2] = plan.weights[i].shape[3] = 1;
        plan.weights[i].logical_bytes = 8;
    }
    NSMutableData* data = [NSMutableData dataWithLength:208];
    uint8_t* bytes = data.mutableBytes;
    // Production aggregate layout: each plan offset names its own chunk header.
    // In particular the first chunk declares only its eight-byte payload.
    bytes[0] = 1; bytes[4] = 2;
    bytes[64] = 0xef; bytes[65] = 0xbe; bytes[66] = 0xad; bytes[67] = 0xde;
    bytes[68] = 1; write_le32(bytes + 72, 8); write_le32(bytes + 80, 128);
    bytes[136] = 0xef; bytes[137] = 0xbe; bytes[138] = 0xad; bytes[139] = 0xde;
    bytes[140] = 1; write_le32(bytes + 144, 8); write_le32(bytes + 152, 200);
    for (int i = 0; i < 8; i++) {
        bytes[128 + i] = (uint8_t)(i + 1);
        bytes[200 + i] = (uint8_t)(i + 9);
    }
    digest_hex([data subdataWithRange:NSMakeRange(128, 8)], plan.weights[0].sha256);
    digest_hex([data subdataWithRange:NSMakeRange(200, 8)], plan.weights[1].sha256);
    NSDictionary* good = @{ @"@model_path/weights/aggregate.bin": @{ @"offset": @0, @"data": data } };
    NSMutableData* tampered_data = [data mutableCopy];
    ((uint8_t*)tampered_data.mutableBytes)[200] ^= 1;
    NSDictionary* tampered = @{ @"@model_path/weights/aggregate.bin": @{ @"offset": @0, @"data": tampered_data } };
    NSMutableData* bad_header_data = [data mutableCopy];
    ((uint8_t*)bad_header_data.mutableBytes)[136] ^= 1;
    NSDictionary* bad_header = @{ @"@model_path/weights/aggregate.bin": @{ @"offset": @0, @"data": bad_header_data } };
    NSMutableData* bad_range_data = [data mutableCopy];
    write_le32((uint8_t*)bad_range_data.mutableBytes + 152, 204);
    NSDictionary* bad_range = @{ @"@model_path/weights/aggregate.bin": @{ @"offset": @0, @"data": bad_range_data } };
    NSMutableData* overlapping_data = [data mutableCopy];
    write_le32((uint8_t*)overlapping_data.mutableBytes + 152, 128);
    OrionBindingPlan dynamically_overlapping = plan;
    digest_hex([overlapping_data subdataWithRange:NSMakeRange(128, 8)],
               dynamically_overlapping.weights[1].sha256);
    NSDictionary* dynamically_overlapping_dict = @{
        @"@model_path/weights/aggregate.bin": @{ @"offset": @0, @"data": overlapping_data }
    };
    return orion_validate_weight_dictionary_for_plan(&plan, good) &&
        !orion_validate_weight_dictionary_for_plan(&plan, tampered) &&
        !orion_validate_weight_dictionary_for_plan(&plan, bad_header) &&
        !orion_validate_weight_dictionary_for_plan(&plan, bad_range) &&
        !orion_validate_weight_dictionary_for_plan(&dynamically_overlapping,
                                                   dynamically_overlapping_dict);
}

static bool test_compile_with_plan_rejects_bad_dict_with_error(void) {
    OrionBindingPlan plan = {0};
    plan.version = ORION_BINDING_PLAN_VERSION;
    snprintf(plan.program_name, sizeof(plan.program_name), "weight-test");
    memset(plan.graph_identity, 'a', ORION_SHA256_HEX_LENGTH);
    memset(plan.mil_identity, 'b', ORION_SHA256_HEX_LENGTH);
    snprintf(plan.orion_revision, sizeof(plan.orion_revision), "test");
    snprintf(plan.compiler_build, sizeof(plan.compiler_build), "test");
    snprintf(plan.position_semantics, sizeof(plan.position_semantics), "fixed");
    snprintf(plan.mask_semantics, sizeof(plan.mask_semantics), "fixed");
    snprintf(plan.failure_semantics, sizeof(plan.failure_semantics), "fail-closed");
    plan.n_inputs = plan.n_outputs = 1;
    fill_entry(&plan.inputs[0], "hidden");
    fill_entry(&plan.outputs[0], "hidden_out");
    plan.n_weights = 1;
    snprintf(plan.weights[0].name, sizeof(plan.weights[0].name), "w");
    snprintf(plan.weights[0].path, sizeof(plan.weights[0].path), "@model_path/weights/w.bin");
    plan.weights[0].offset = 64;
    plan.weights[0].dtype = ORION_DTYPE_FP16;
    plan.weights[0].shape[0] = 2; plan.weights[0].shape[1] = 2;
    plan.weights[0].shape[2] = plan.weights[0].shape[3] = 1;
    plan.weights[0].logical_bytes = 8;
    (void)valid_dictionary(&plan);
    char error[256] = {0};
    OrionProgram* prog = orion_compile_mil_with_plan(
        "program(1.3) { func main<ios18>() {} -> (); }",
        &plan, @{}, "reject-test", error, sizeof(error));
    bool rejected = prog == NULL && strlen(error) > 0;
    if (prog) orion_release_program(prog);
    return rejected;
}

static OrionGraph* weight_free_frontend(int layer __attribute__((unused)),
                                        int bucket __attribute__((unused)),
                                        const OrionModelConfig* cfg __attribute__((unused))) {
    OrionGraph* g = orion_graph_create();
    int shape[4] = {1, 32, 1, 16};
    int x = orion_gb_input(g, "x", ORION_DTYPE_FP16, shape);
    int y = orion_gb_add(g, x, x, "y");
    orion_gb_output(g, y, "y");
    return g;
}

// Distinct graph (mul vs add) so MIL differs from weight_free_frontend.
static OrionGraph* distinct_weight_free_frontend(int layer __attribute__((unused)),
                                                 int bucket __attribute__((unused)),
                                                 const OrionModelConfig* cfg __attribute__((unused))) {
    OrionGraph* g = orion_graph_create();
    int shape[4] = {1, 32, 1, 16};
    int x = orion_gb_input(g, "x", ORION_DTYPE_FP16, shape);
    int y = orion_gb_mul(g, x, x, "y");
    orion_gb_output(g, y, "y");
    return g;
}

static bool test_program_plan_accessor(void) {
    if (!orion_ane_init()) {
        SKIP_CASE("ane-unavailable");
    }
    NSString* mil = orion_kernel_adapter_generate_mil(weight_free_frontend, 0, 16, NULL);
    if (!mil) { SKIP_CASE("mil-gen"); }
    OrionProgram* prog = orion_compile_mil(mil.UTF8String, @{}, "plan-accessor-test");
    if (!prog) { SKIP_CASE("ane-compile"); }
    const OrionBindingPlan* cached = orion_program_plan(prog);
    bool ok = cached == NULL;
    if (prog) orion_release_program(prog);
    return ok;
}

static bool test_program_plan_caches_plan_from_compile_with_plan(void) {
    if (!orion_ane_init()) {
        SKIP_CASE("ane-unavailable");
    }
    OrionBindingPlan plan;
    NSString* mil = orion_kernel_adapter_generate_mil_with_plan_identity(
        weight_free_frontend, 0, 16, NULL, NULL, &plan);
    if (!mil) { SKIP_CASE("mil-gen"); }
    char error[256] = {0};
    OrionProgram* prog = orion_compile_mil_with_plan(
        mil.UTF8String, &plan, @{}, "plan-cache-test", error, sizeof(error));
    if (!prog) { SKIP_CASE("ane-compile"); }
    const OrionBindingPlan* cached = orion_program_plan(prog);
    bool ok = cached != NULL && cached->n_inputs == plan.n_inputs &&
        strcmp(cached->graph_identity, plan.graph_identity) == 0;
    if (prog) orion_release_program(prog);
    return ok;
}

static bool test_compile_with_plan_rejects_mil_mismatch(void) {
    OrionBindingPlan plan;
    NSString* mil_a = orion_kernel_adapter_generate_mil_with_plan_identity(
        weight_free_frontend, 0, 16, NULL, NULL, &plan);
    if (!mil_a) { SKIP_CASE("mil-gen"); }
    NSString* mil_b = orion_kernel_adapter_generate_mil(
        distinct_weight_free_frontend, 0, 16, NULL);
    if (!mil_b) { SKIP_CASE("mil-gen-b"); }
    char error[256] = {0};
    OrionProgram* prog = orion_compile_mil_with_plan(
        mil_b.UTF8String, &plan, @{}, "mil-mismatch-test", error, sizeof(error));
    bool rejected = prog == NULL && strlen(error) > 0;
    if (prog) orion_release_program(prog);
    return rejected;
}

static NSDictionary* weight_free_wdict(int layer __attribute__((unused)),
                                        int bucket __attribute__((unused)),
                                        const char* dir __attribute__((unused))) {
    return @{};
}

static bool test_adapted_kernel_compile_and_eval_path(void) {
    if (!orion_ane_init()) {
        SKIP_CASE("ane-unavailable");
    }
    OrionKernel kernel = orion_kernel_from_frontend(
        "test_weight_free", weight_free_frontend, weight_free_wdict, 1, 1);
    OrionModelConfig cfg = {0};
    OrionWeightsBinding wb = { .weights_id = "test-step-0", .bucket = 16 };

    OrionProgram* first = orion_kernel_compile(&kernel, 0, 16, &cfg, "/tmp", &wb);
    if (!first) { SKIP_CASE("ane-compile"); }

    const OrionBindingPlan* cached = orion_program_plan(first);
    if (!cached) return false;

    // Second compile with same weights binding must return the cached program
    // (verified by graph + MIL identity, not just by cache key).
    OrionProgram* second = orion_kernel_compile(&kernel, 0, 16, &cfg, "/tmp", &wb);
    bool cached_hit = second == first;

    // Eval via plan-bound path.
    OrionSurfaceStatus status = ORION_SURFACE_OK;
    IOSurfaceRef input = orion_tensor_create_for_binding(&cached->inputs[0], &status);
    IOSurfaceRef output = orion_tensor_create_for_binding(&cached->outputs[0], &status);
    bool eval_ok = false;
    bool foreign_plan_rejected = false;
    if (input && output && status == ORION_SURFACE_OK) {
        OrionBindingPlan foreign_plan = *cached;
        foreign_plan.graph_identity[0] = foreign_plan.graph_identity[0] == 'a' ? 'b' : 'a';
        foreign_plan_rejected = !orion_eval_with_plan(first, &foreign_plan,
                                                       &input, 1, &output, 1);
        eval_ok = orion_kernel_eval(&kernel, 0, 16, &cfg, "/tmp", &wb,
                                    &input, 1, &output, 1);
    }
    if (input) CFRelease(input);
    if (output) CFRelease(output);
    return cached_hit && foreign_plan_rejected && eval_ok &&
        strlen(cached->mil_identity) == ORION_SHA256_HEX_LENGTH;
}

static bool test_stale_cache_guard_forces_recompile(void) {
    if (!orion_ane_init()) {
        SKIP_CASE("ane-unavailable");
    }
    OrionKernel kernel_add = orion_kernel_from_frontend(
        "same-name", weight_free_frontend, weight_free_wdict, 1, 1);
    OrionKernel kernel_mul = orion_kernel_from_frontend(
        "same-name", distinct_weight_free_frontend, weight_free_wdict, 1, 1);
    OrionModelConfig cfg = {0};
    OrionWeightsBinding wb = { .weights_id = "stale-test", .bucket = 16 };

    OrionProgram* first = orion_kernel_compile(&kernel_add, 0, 16, &cfg, "/tmp", &wb);
    if (!first) { SKIP_CASE("ane-compile"); }

    // Same cache key ("same-name:0:stale-test:16") but different graph/mil identity.
    // The stale-cache guard must detect the identity drift and force a recompile.
    OrionProgram* second = orion_kernel_compile(&kernel_mul, 0, 16, &cfg, "/tmp", &wb);
    bool cache_miss = second != NULL && second != first;

    // Cache owns both programs; evicting releases the current one.
    // first was already released by the cache when second replaced it.
    orion_cache_evict("stale-test");
    return cache_miss;
}

int main(void) {
    @autoreleasepool {
        printf("test_plan_weight_dict:\n");
        TEST(plan_weight_dictionary_is_exact);
        TEST(aggregate_weight_dictionary_hashes_chunk_payloads);
        TEST(compile_with_plan_rejects_bad_dict_with_error);
        TEST(program_plan_accessor);
        TEST(program_plan_caches_plan_from_compile_with_plan);
        TEST(compile_with_plan_rejects_mil_mismatch);
        TEST(adapted_kernel_compile_and_eval_path);
        TEST(stale_cache_guard_forces_recompile);
        // FAIL beats SKIP beats PASS: a skipped case on a non-ANE host must not
        // be reported as a clean PASS, but a real failure still wins. Returning
        // ORION_TEST_SKIP (77) makes the runner print SKIP so the high-value
        // ANE-gated cases (cache identity drift, foreign-plan rejection) are
        // visibly unrun rather than silently green.
        ORION_TEST_SUMMARY(g_passed, g_failed, g_skipped);
        if (g_failed) return ORION_TEST_FAIL;
        if (g_skipped) return ORION_TEST_SKIP;
        return ORION_TEST_PASS;
    }
}
