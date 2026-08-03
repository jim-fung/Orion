// tests/test_voxtream_depth_chain_ane.m — Opt-in real-artifact decomposed
// VoXtream depth-layer ANE spike.
//
// The M5 Pro private compiler accepts at most one unique large BLOBFILE per
// program, so a complete layer executes as a 15-program chain. Stage donor
// programs are compiled once for layer 0 with strict plan-verified
// dictionaries; layers 1..3 derive per-layer programs from the donors via
// orion_program_patch_weights() (a new descriptor bakes the new blobs, so
// no recompile is needed and results are compared to a frozen fp16 reference.
// Constant-only stages reuse their layer-0 donors directly. The normal run
// therefore compiles 15 stage donors plus an optional final norm program.
//
// Set VOXTREAM_DEPTH_EXPORT to a validated depth export directory. The test
// deliberately skips when the private runtime or artifacts are unavailable.

#import <Foundation/Foundation.h>
#include "orion_test.h"
#import <IOSurface/IOSurface.h>
#import <CommonCrypto/CommonDigest.h>
#include <mach/mach_time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "compiler/binding_plan.h"
#include "compiler/frontends/voxtream_depth.h"
#include "compiler/frontends/voxtream_depth_chain.h"
#include "compiler/kernel_adapter.h"
#include "core/ane_runtime.h"
#include "core/iosurface_tensor.h"

#define MAX_TENSORS 20
#define MAX_CHANNELS ORION_VOXTREAM_DEPTH_HIDDEN_DIM

static uint64_t monotonic_now(void) {
    return mach_absolute_time();
}

static double monotonic_elapsed_ms(uint64_t start) {
    static mach_timebase_info_data_t timebase;
    if (timebase.denom == 0) mach_timebase_info(&timebase);
    uint64_t elapsed = mach_absolute_time() - start;
    return (double)elapsed * (double)timebase.numer /
           (double)timebase.denom / 1000000.0;
}

typedef struct {
    const char* name;       // canonical tensor name shared between stages
    int channels;
    _Float16* data;         // packed channel-major [channels, seq]
} ChainTensor;

typedef struct {
    int stage;
    const char* inputs[3];  // plan input entry name -> canonical tensor
    int n_inputs;
    const char* output;     // canonical tensor produced
    int out_channels;
} ChainLink;

static NSString* sha256_file(NSString* path) {
    NSData* data = [NSData dataWithContentsOfFile:path];
    if (!data) return nil;
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(data.bytes, (CC_LONG)data.length, digest);
    NSMutableString* value = [NSMutableString stringWithCapacity:64];
    for (int i = 0; i < CC_SHA256_DIGEST_LENGTH; i++)
        [value appendFormat:@"%02x", digest[i]];
    return value;
}

static NSString* qualified_reference(const char* root, NSString* name,
                                     int batch_index,
                                     NSString** export_identity) {
    NSString* manifest_path = [@(root) stringByAppendingPathComponent:
        @"depth_export.manifest.json"];
    NSData* data = [NSData dataWithContentsOfFile:manifest_path];
    NSDictionary* manifest = data ? [NSJSONSerialization JSONObjectWithData:data
        options:0 error:nil] : nil;
    if (![manifest[@"schema_version"] isEqual:@"voxtream.orion.depth-export/1.5"])
        return nil;
    NSArray* records = manifest[@"ane_references"];
    if (![records isKindOfClass:[NSArray class]]) return nil;
    NSString* expected_relative = [NSString stringWithFormat:
        @"ane_references/%@.batch%d.fp16.bin", name, batch_index];
    NSString* resolved_root = [[@(root) stringByStandardizingPath]
        stringByResolvingSymlinksInPath];
    NSString* root_prefix = [resolved_root stringByAppendingString:@"/"];
    NSArray* expected_shape = @[@1, @ORION_VOXTREAM_DEPTH_DIM, @1,
                                @ORION_VOXTREAM_DEPTH_ANE_SEQ];
    for (NSDictionary* record in records) {
        id batch = record[@"batch_index"];
        if (![record isKindOfClass:[NSDictionary class]] ||
            ![record[@"name"] isEqual:name] ||
            ![record[@"dtype"] isEqual:@"float16-little-endian"] ||
            ![batch isKindOfClass:[NSNumber class]] ||
            CFGetTypeID((__bridge CFTypeRef)batch) == CFBooleanGetTypeID() ||
            [batch intValue] != batch_index ||
            ![record[@"shape"] isEqual:expected_shape] ||
            ![record[@"path"] isKindOfClass:[NSString class]] ||
            ![record[@"path"] isEqual:expected_relative]) continue;
        NSString* path = [[[resolved_root
            stringByAppendingPathComponent:record[@"path"]]
            stringByStandardizingPath] stringByResolvingSymlinksInPath];
        if (![path hasPrefix:root_prefix]) return nil;
        if (![[sha256_file(path) lowercaseString]
                isEqual:[record[@"sha256"] lowercaseString]]) return nil;
        if (export_identity) *export_identity = manifest[@"identity_sha256"];
        return path;
    }
    return nil;
}

static int compare_double(const void* lhs, const void* rhs) {
    double a = *(const double*)lhs, b = *(const double*)rhs;
    return (a > b) - (a < b);
}

static double percentile(const double* values, int count, double fraction) {
    double* sorted = malloc((size_t)count * sizeof(*sorted));
    if (!sorted || count <= 0) {
        free(sorted);
        return NAN;
    }
    memcpy(sorted, values, (size_t)count * sizeof(*sorted));
    qsort(sorted, (size_t)count, sizeof(*sorted), compare_double);
    double rank = fraction * (count - 1);
    int lower = (int)floor(rank), upper = (int)ceil(rank);
    double result = sorted[lower] + (sorted[upper] - sorted[lower]) * (rank - lower);
    free(sorted);
    return result;
}

static bool injected_failure(const char* stem, const char* point) {
    const char* stage = getenv("VOXTREAM_DEPTH_CHAIN_FAIL_STAGE");
    const char* requested = getenv("VOXTREAM_DEPTH_CHAIN_FAIL_POINT");
    bool matches = stage && requested && strstr(stem, stage) &&
                   strcmp(requested, point) == 0;
    return matches;
}

static bool s_injected_failure_observed = false;

static bool inject_now(const char* stem, const char* point) {
    bool matches = injected_failure(stem, point);
    if (matches) s_injected_failure_observed = true;
    return matches;
}

static bool failure_injection_requested(void) {
    const char* stage = getenv("VOXTREAM_DEPTH_CHAIN_FAIL_STAGE");
    const char* point = getenv("VOXTREAM_DEPTH_CHAIN_FAIL_POINT");
    return stage && stage[0] && point && point[0];
}

static NSString* safe_evidence_component(const char* value) {
    NSMutableString* safe = [NSMutableString string];
    for (const unsigned char* p = (const unsigned char*)value; p && *p; p++) {
        if ((*p >= 'a' && *p <= 'z') || (*p >= 'A' && *p <= 'Z') ||
            (*p >= '0' && *p <= '9') || *p == '-' || *p == '_') {
            [safe appendFormat:@"%c", *p];
        } else {
            [safe appendString:@"_"];
        }
    }
    return safe;
}

static bool publish_failure_evidence(const char* root, const char* revision,
                                     const char* compiler_build,
                                     int compile_count_start,
                                     bool cleanup_completed) {
    if (!failure_injection_requested()) return true;
    const char* stage = getenv("VOXTREAM_DEPTH_CHAIN_FAIL_STAGE");
    const char* point = getenv("VOXTREAM_DEPTH_CHAIN_FAIL_POINT");
    NSDateFormatter* formatter = [[NSDateFormatter alloc] init];
    formatter.locale = [NSLocale localeWithLocaleIdentifier:@"en_US_POSIX"];
    formatter.timeZone = [NSTimeZone timeZoneForSecondsFromGMT:0];
    formatter.dateFormat = @"yyyy-MM-dd'T'HH:mm:ss.SSS'Z'";
    NSDictionary* evidence = @{
        @"schema_version": @"orion.voxtream.depth-chain-failure/1.0",
        @"status": s_injected_failure_observed ? @"pass" : @"fail",
        @"stage": @(stage),
        @"point": @(point),
        @"cleanup_completed": @(cleanup_completed),
        @"failure_kind": @"host-injected-boundary",
        @"operating_system": NSProcessInfo.processInfo.operatingSystemVersionString,
        @"orion_revision": @(revision && revision[0]
            ? revision : "diagnostic-unqualified"),
        @"compiler_build": @(compiler_build && compiler_build[0]
            ? compiler_build : "diagnostic-unqualified"),
        @"compile_count_delta": @(orion_compile_count() - compile_count_start),
        @"timestamp_utc": [formatter stringFromDate:[NSDate date]],
    };
    NSData* json = [NSJSONSerialization dataWithJSONObject:evidence
        options:NSJSONWritingPrettyPrinted error:nil];
    NSString* filename = [NSString stringWithFormat:
        @"depth_chain_failure.%@.%@.json", safe_evidence_component(stage),
        safe_evidence_component(point)];
    NSString* path = [@(root) stringByAppendingPathComponent:filename];
    if (!json || ![json writeToFile:path atomically:YES]) {
        fprintf(stderr, "FAIL: unable to publish failure evidence %s\n",
                path.UTF8String);
        return false;
    }
    return true;
}

static NSArray* sample_array(const double* samples, int count) {
    NSMutableArray* values = [NSMutableArray arrayWithCapacity:(NSUInteger)count];
    for (int i = 0; i < count; i++) [values addObject:@(samples[i])];
    return values;
}

static const ChainLink kChain[] = {
    {ORION_VDX_WQ,       {"hidden"}, 1, "q", 1024},
    {ORION_VDX_WK,       {"hidden"}, 1, "k", 256},
    {ORION_VDX_WV,       {"hidden"}, 1, "v", 256},
    {ORION_VDX_QSWAP,    {"q"}, 1, "q_swapped", 1024},
    {ORION_VDX_QCOS,     {"q"}, 1, "q_cos", 1024},
    {ORION_VDX_QSIN_ADD, {"q_cos", "q_swapped"}, 2, "q_rope", 1024},
    {ORION_VDX_KSWAP,    {"k"}, 1, "k_swapped", 256},
    {ORION_VDX_KCOS,     {"k"}, 1, "k_cos", 256},
    {ORION_VDX_KSIN_ADD, {"k_cos", "k_swapped"}, 2, "k_rope", 256},
    {ORION_VDX_ATTN_CORE, {"q_rope", "k_rope", "v"}, 3, "attn_ctx", 1024},
    {ORION_VDX_WO_RES,   {"attn_ctx", "hidden"}, 2, "attn_res", 1024},
    {ORION_VDX_W1,       {"attn_res"}, 1, "mlp_up", 8192},
    {ORION_VDX_W3,       {"attn_res"}, 1, "mlp_gate_in", 8192},
    {ORION_VDX_GATE,     {"mlp_up", "mlp_gate_in"}, 2, "gated", 8192},
    {ORION_VDX_W2_RES,   {"gated", "attn_res"}, 2, "hidden_out", 1024},
};

// Plan input entry names differ from canonical tensor names for stages whose
// graphs were written with explicit *_in input names. Map them here.
static const char* canonical_for_entry(const char* entry) {
    if (strcmp(entry, "q_in") == 0) return "q";
    if (strcmp(entry, "k_in") == 0) return "k";
    if (strcmp(entry, "v_in") == 0) return "v";
    if (strcmp(entry, "q_cos_in") == 0) return "q_cos";
    if (strcmp(entry, "q_swapped_in") == 0) return "q_swapped";
    if (strcmp(entry, "k_cos_in") == 0) return "k_cos";
    if (strcmp(entry, "k_swapped_in") == 0) return "k_swapped";
    if (strcmp(entry, "q_rope_in") == 0) return "q_rope";
    if (strcmp(entry, "k_rope_in") == 0) return "k_rope";
    if (strcmp(entry, "attn_ctx_in") == 0) return "attn_ctx";
    if (strcmp(entry, "hidden_in") == 0) return "hidden";
    if (strcmp(entry, "attn_res_in") == 0) return "attn_res";
    if (strcmp(entry, "mlp_up_in") == 0) return "mlp_up";
    if (strcmp(entry, "mlp_gate_in") == 0) return "mlp_gate_in";
    if (strcmp(entry, "gated_in") == 0) return "gated";
    return entry;
}

static ChainTensor* find_tensor(ChainTensor* tensors, int count,
                                const char* name) {
    for (int i = 0; i < count; i++) {
        if (strcmp(tensors[i].name, name) == 0) return &tensors[i];
    }
    return NULL;
}

static void cleanup_chain(OrionProgram* donors[16], ChainTensor* tensors,
                          int n_tensors) {
    for (int i = 0; i < 16; i++) {
        if (donors[i]) orion_release_program(donors[i]);
    }
    for (int i = 0; i < n_tensors; i++) free(tensors[i].data);
}

// Build a weight dict for `layer` using the donor plan's (layer-0) keys.
// Constant blobs are layer-independent; `weights/layer0/` paths are
// substituted with the requested layer when reading blob data. The keys
// stay at the donor paths so generated MIL and binding plans retain one stable
// ABI while patches load layer-specific BLOBFILE bytes.
static NSDictionary* remapped_dict_from_plan(const OrionBindingPlan* plan,
                                             const char* root, int layer) {
    NSMutableDictionary* dictionary = [NSMutableDictionary dictionary];
    for (int i = 0; i < plan->n_weights; i++) {
        const char* rel = plan->weights[i].path;
        const char* prefix = "@model_path/";
        size_t prefix_len = strlen(prefix);
        if (strncmp(rel, prefix, prefix_len) != 0) return nil;
        NSString* file_rel = @(rel + prefix_len);
        if (layer > 0) {
            NSString* donor_prefix = @"weights/layer0/";
            if ([file_rel hasPrefix:donor_prefix]) {
                file_rel = [NSString stringWithFormat:
                    @"weights/layer%d/%@", layer,
                    [file_rel substringFromIndex:donor_prefix.length]];
            }
        }
        NSString* file = [@(root) stringByAppendingPathComponent:file_rel];
        NSData* data = [NSData dataWithContentsOfFile:file];
        if (!data) return nil;
        dictionary[@(rel)] = @{@"offset": @0, @"data": data};
    }
    return dictionary;
}

static NSDictionary* dict_from_plan(const OrionBindingPlan* plan,
                                    const char* root) {
    return remapped_dict_from_plan(plan, root, 0);
}

// True when any plan weight lives under weights/layer0/ (i.e. the stage has
// layer-specific weights). Constant-only stages can reuse the donor program
// unchanged for every layer.
static bool plan_has_layer_weights(const OrionBindingPlan* plan) {
    for (int i = 0; i < plan->n_weights; i++) {
        if (strstr(plan->weights[i].path, "weights/layer0/")) return true;
    }
    return false;
}

// Final-norm graph has a 0-arg frontend; wrap it in the adapter signature.
static OrionGraph* final_norm_adapter(
    int layer __attribute__((unused)), int bucket,
    const OrionModelConfig* config) {
    if (!config || bucket != ORION_VOXTREAM_DEPTH_BUCKET ||
        config->d_model != ORION_VOXTREAM_DEPTH_DIM ||
        config->max_seq != ORION_VOXTREAM_DEPTH_BUCKET) return NULL;
    return orion_frontend_voxtream_depth_final_norm();
}

// Compile a stage donor program for layer 0, writing MIL/plan artifacts.
// Returns NULL on failure; on success *plan_out holds the donor plan and
// *mil_out retains the exact MIL text (needed for weight patching).
static OrionProgram* compile_donor(OrionFrontendFn frontend, int layer,
                                   const OrionModelConfig* config,
                                   const OrionBindingPlanIdentity* identity,
                                   const char* blob_dir, const char* stem,
                                   OrionBindingPlan* plan_out,
                                   int expected_inputs,
                                   double* compile_ms_out,
                                   NSString** mil_out) {
    NSString* root = @(blob_dir);
    NSString* mil = orion_kernel_adapter_generate_mil_with_artifacts(
        frontend, layer, ORION_VOXTREAM_DEPTH_BUCKET, config, identity,
        [root stringByAppendingPathComponent:
            [@(stem) stringByAppendingString:@".plan.json"]].UTF8String,
        [root stringByAppendingPathComponent:
            [@(stem) stringByAppendingString:@".plan.md"]].UTF8String,
        plan_out);
    if (!mil || ![mil writeToFile:
            [root stringByAppendingPathComponent:
                [@(stem) stringByAppendingString:@".mil"]]
            atomically:YES encoding:NSUTF8StringEncoding error:nil]) {
        fprintf(stderr, "FAIL: %s artifact generation failed\n", stem);
        return NULL;
    }
    if ((int)plan_out->n_inputs != expected_inputs) {
        fprintf(stderr, "FAIL: %s expected %d inputs, plan has %d\n",
                stem, expected_inputs, (int)plan_out->n_inputs);
        return NULL;
    }
    NSDictionary* wdict = dict_from_plan(plan_out, blob_dir);
    if (inject_now(stem, "before-compile")) return NULL;
    char error[256];
    uint64_t compile_start = monotonic_now();
    OrionProgram* program = orion_compile_mil_with_plan(
        mil.UTF8String, plan_out, wdict, stem, error, sizeof(error));
    *compile_ms_out = monotonic_elapsed_ms(compile_start);
    if (!program) {
        fprintf(stderr, "FAIL: %s compile failed (%.2f ms) %s\n",
                stem, *compile_ms_out, error);
        return NULL;
    }
    if (mil_out) *mil_out = mil;
    return program;
}

// Evaluate a (possibly weight-reloaded) donor program, moving plan inputs
// from host tensors and the plan output back into the canonical tensor.
static bool eval_donor(OrionProgram* program, const OrionBindingPlan* plan,
                       const char* stem,
                       ChainTensor* tensors, int* n_tensors,
                       const char* output_name, int out_channels, int seq,
                       double* prepare_ms_out, double* eval_ms_out,
                       double* readback_ms_out, double* total_ms_out) {
    uint64_t total_start = monotonic_now();
    IOSurfaceRef inputs[3] = {NULL, NULL, NULL};
    IOSurfaceRef output = NULL;
    OrionSurfaceStatus surface_status = ORION_SURFACE_OK;
    bool ok = true;
    for (int i = 0; i < (int)plan->n_inputs && ok; i++) {
        const char* tensor_name = canonical_for_entry(plan->inputs[i].name);
        ChainTensor* tensor = find_tensor(tensors, *n_tensors, tensor_name);
        if (!tensor) {
            fprintf(stderr, "FAIL: %s input %s missing\n", stem,
                    plan->inputs[i].name);
            ok = false;
            break;
        }
        inputs[i] = orion_tensor_create_for_binding(&plan->inputs[i],
                                                    &surface_status);
        ok = inputs[i] && surface_status == ORION_SURFACE_OK &&
             orion_tensor_write_for_binding(
                 inputs[i], &plan->inputs[i], tensor->data,
                 plan->inputs[i].packed_prefix_bytes) == ORION_SURFACE_OK;
    }
    if (ok && inject_now(stem, "after-input-write")) ok = false;
    if (ok) {
        output = orion_tensor_create_for_binding(&plan->outputs[0],
                                                 &surface_status);
        ok = output && surface_status == ORION_SURFACE_OK;
    }
    if (ok && inject_now(stem, "after-surface-create")) ok = false;
    *prepare_ms_out = monotonic_elapsed_ms(total_start);
    uint64_t eval_start = monotonic_now();
    if (ok) {
        if (inject_now(stem, "before-eval")) {
            ok = false;
        } else {
            ok = orion_eval_with_plan(program, plan, inputs,
                                      (int)plan->n_inputs, &output, 1);
        }
    }
    *eval_ms_out = monotonic_elapsed_ms(eval_start);

    uint64_t readback_start = monotonic_now();
    ChainTensor* out_tensor = NULL;
    if (ok && inject_now(stem, "after-eval")) ok = false;
    if (ok) {
        out_tensor = find_tensor(tensors, *n_tensors, output_name);
        if (!out_tensor) {
            if (*n_tensors >= MAX_TENSORS) {
                ok = false;
            } else {
                out_tensor = &tensors[(*n_tensors)++];
            }
            if (ok) {
                out_tensor->name = output_name;
                out_tensor->channels = out_channels;
                out_tensor->data = calloc((size_t)out_channels * seq,
                                           sizeof(_Float16));
            }
        }
        if (ok && inject_now(stem, "before-readback")) ok = false;
        ok = ok && out_tensor->data &&
             orion_tensor_read_for_binding(
                 output, &plan->outputs[0], out_tensor->data,
                 plan->outputs[0].packed_prefix_bytes) == ORION_SURFACE_OK;
    }
    if (ok) {
        size_t elements = (size_t)out_channels * seq;
        for (size_t i = 0; i < elements; i++) {
            if (!isfinite((float)out_tensor->data[i])) {
                ok = false;
                break;
            }
        }
    }
    for (int i = 0; i < (int)plan->n_inputs; i++) {
        if (inputs[i]) CFRelease(inputs[i]);
    }
    if (output) CFRelease(output);
    *readback_ms_out = monotonic_elapsed_ms(readback_start);
    *total_ms_out = monotonic_elapsed_ms(total_start);
    if (!ok) fprintf(stderr, "FAIL: %s eval/readback failed\n", stem);
    return ok;
}

int main(void) {
    @autoreleasepool {
        const char* blob_dir = getenv("VOXTREAM_DEPTH_EXPORT");
        if (!blob_dir || !blob_dir[0]) {
            printf("SKIP: set VOXTREAM_DEPTH_EXPORT to a validated export directory\n");
            return ORION_TEST_SKIP;
        }
        if (!orion_ane_init()) {
            printf("SKIP: ANE runtime unavailable\n");
            return ORION_TEST_SKIP;
        }

        OrionModelConfig config = {
            .n_layer = ORION_VOXTREAM_DEPTH_LAYERS,
            .n_head = ORION_VOXTREAM_DEPTH_HEADS,
            .d_model = ORION_VOXTREAM_DEPTH_DIM,
            .head_dim = ORION_VOXTREAM_DEPTH_HEAD_DIM,
            .hidden_dim = ORION_VOXTREAM_DEPTH_HIDDEN_DIM,
            .max_seq = ORION_VOXTREAM_DEPTH_BUCKET,
        };
        const char* revision = getenv("VOXTREAM_ORION_REVISION");
        const char* compiler_build = getenv("VOXTREAM_COMPILER_BUILD");
        OrionBindingPlanIdentity identity = {
            .orion_revision = revision ?: "diagnostic-unqualified",
            .compiler_build = compiler_build ?: "diagnostic-unqualified",
            .weight_root = blob_dir,
        };

        const int s = ORION_VOXTREAM_DEPTH_ANE_SEQ;
        const int n_links = (int)(sizeof(kChain) / sizeof(kChain[0]));
        if (n_links != orion_voxtream_depth_chain_stage_count()) {
            fprintf(stderr, "FAIL: chain table out of sync with stage count\n");
            return 1;
        }

        ChainTensor tensors[MAX_TENSORS];
        int n_tensors = 0;
        OrionProgram* donors[16] = {NULL};
        double total_compile_ms = 0.0;
        double total_patch_load_ms = 0.0;
        const int compile_count_start = orion_compile_count();
        const bool evidence_diagnostic =
            getenv("VOXTREAM_DEPTH_CHAIN_ONLY_STAGE") != NULL ||
            getenv("VOXTREAM_DEPTH_CHAIN_REPEAT") != NULL ||
            getenv("VOXTREAM_DEPTH_CHAIN_INPUT_BIN") != NULL ||
            failure_injection_requested();
        const char* batch_env = getenv("VOXTREAM_DEPTH_CHAIN_BATCH");
        const int batch_index = batch_env ? atoi(batch_env) : 0;
        if (batch_index < 0 || batch_index > 1) {
            fprintf(stderr, "FAIL: VOXTREAM_DEPTH_CHAIN_BATCH must be 0 or 1\n");
            return 1;
        }
        const char* samples_env = getenv("VOXTREAM_DEPTH_CHAIN_SAMPLES");
        const int sample_count = samples_env ? atoi(samples_env) : 30;
        const char* warmup_env = getenv("VOXTREAM_DEPTH_CHAIN_WARMUP");
        const int warmup_count = warmup_env ? atoi(warmup_env) : 2;
        if (sample_count < 1 || sample_count > 100 || warmup_count < 0 || warmup_count > 20) {
            fprintf(stderr, "FAIL: samples must be 1..100 and warmup 0..20\n");
            return 1;
        }
        if (!evidence_diagnostic && sample_count < 30) {
            fprintf(stderr, "FAIL: non-diagnostic evidence requires at least 30 samples\n");
            return 1;
        }
        double stage_sum_samples[100] = {0};
        double prepare_samples[100] = {0};
        double device_samples[100] = {0};
        double readback_samples[100] = {0};

        // Seed the fixture: batch 0 of prefix-8-guided-step0, logical
        // [S,D] = (token*D + channel + 7)/D, packed channel-major, with zero
        // padding in token slots [16, 32).
        ChainTensor* hidden = &tensors[n_tensors++];
        hidden->name = "hidden";
        hidden->channels = ORION_VOXTREAM_DEPTH_DIM;
        hidden->data = calloc((size_t)hidden->channels * s, sizeof(_Float16));
        NSString* input_reference = qualified_reference(
            blob_dir, @"packed_hidden", batch_index, NULL);
        const char* input_bin = getenv("VOXTREAM_DEPTH_CHAIN_INPUT_BIN");
        if (!input_bin && input_reference) input_bin = input_reference.UTF8String;
        if (input_bin && input_bin[0]) {
            // Seed hidden from a raw packed [channels, seq] fp16 file (used
            // for per-layer parity checks against exported references).
            NSData* seed = [NSData dataWithContentsOfFile:@(input_bin)];
            size_t expect = (size_t)hidden->channels * s * sizeof(_Float16);
            if (seed.length != expect) {
                fprintf(stderr, "FAIL: input bin size %lu != %lu\n",
                        (unsigned long)seed.length, (unsigned long)expect);
                cleanup_chain(donors, tensors, n_tensors);
                return 1;
            }
            memcpy(hidden->data, seed.bytes, expect);
        } else {
            for (int c = 0; c < hidden->channels; c++) {
                for (int t = 0; t < s; t++) {
                    hidden->data[(size_t)c * s + t] =
                        t < ORION_VOXTREAM_DEPTH_BUCKET
                            ? (_Float16)(((float)(t * hidden->channels + c + 7)) /
                                         hidden->channels)
                            : (_Float16)0;
                }
            }
        }

        const char* layer_env = getenv("VOXTREAM_DEPTH_CHAIN_LAYERS");
        int n_layers = layer_env ? atoi(layer_env) : 1;
        if (n_layers < 1 || n_layers > ORION_VOXTREAM_DEPTH_LAYERS) {
            fprintf(stderr, "FAIL: VOXTREAM_DEPTH_CHAIN_LAYERS must be 1..%d\n",
                    ORION_VOXTREAM_DEPTH_LAYERS);
            cleanup_chain(donors, tensors, n_tensors);
            return 1;
        }
        // Diagnostic: compile+eval a single (layer, stage) program only.
        const char* only_layer_env = getenv("VOXTREAM_DEPTH_CHAIN_ONLY_LAYER");
        const char* only_stage_env = getenv("VOXTREAM_DEPTH_CHAIN_ONLY_STAGE");

        // Diagnostic: compile one stage N times in-process to probe the
        // private compiler's per-process budget.
        const char* repeat_env = getenv("VOXTREAM_DEPTH_CHAIN_REPEAT");
        if (repeat_env && only_stage_env) {
            int repeats = atoi(repeat_env);
            int only_layer = only_layer_env ? atoi(only_layer_env) : 0;
            int stage = -1;
            for (int link = 0; link < n_links; link++) {
                if (strcmp(only_stage_env,
                           orion_voxtream_depth_chain_stage_name(
                               kChain[link].stage)) == 0) {
                    stage = kChain[link].stage;
                    break;
                }
            }
            if (stage < 0 || !orion_voxtream_depth_chain_set_stage(stage)) {
                fprintf(stderr, "FAIL: unknown repeat stage %s\n",
                        only_stage_env);
                return 1;
            }
            int failures = 0;
            for (int i = 1; i <= repeats; i++) {
                OrionBindingPlan plan = {0};
                NSString* mil =
                    orion_kernel_adapter_generate_mil_with_plan_identity(
                        orion_frontend_voxtream_depth_chain_adapter,
                        only_layer, ORION_VOXTREAM_DEPTH_BUCKET, &config,
                        &identity, &plan);
                if (!mil || !orion_binding_plan_validate_finalized(&plan, NULL, 0)) {
                    fprintf(stderr, "FAIL: budget probe graph generation failed\n");
                    return 1;
                }
                NSDictionary* wdict = dict_from_plan(&plan, blob_dir);
                char error[256];
                OrionProgram* program = orion_compile_mil_with_plan(
                    mil.UTF8String, &plan, wdict, "budget_probe",
                    error, sizeof(error));
                if (!program) {
                    printf("compile %d FAILED\n", i);
                    failures++;
                    continue;
                }
                orion_release_program(program);
            }
            printf("budget probe: %d/%d compiles failed (count=%d)\n",
                   failures, repeats, orion_compile_count());
            return failures ? 1 : 0;
        }

        // Donor programs are compiled lazily, one per stage slot (always
        // with layer-0 weight paths). Layer-specific stages patch into
        // isolated programs; constant-only stages reuse their donor. Slot
        // n_links is the final norm.
        OrionBindingPlan donor_plans[16];
        NSString* donor_mils[16] = {nil};

        // Diagnostic: compile every stage fresh per layer (old behavior)
        // instead of reusing layer-0 donors with reloaded weights.
        const bool no_reload = getenv("VOXTREAM_DEPTH_CHAIN_NO_RELOAD") != 0;

        for (int layer = 0; layer < n_layers; layer++) {
            // Layers beyond 0 consume the previous layer's hidden_out
            // (skipped layers leave the seeded input in place).
            if (layer > 0) {
                ChainTensor* prev =
                    find_tensor(tensors, n_tensors, "hidden_out");
                if (prev) {
                    memcpy(hidden->data, prev->data,
                           (size_t)hidden->channels * s * sizeof(_Float16));
                }
                // Each layer rebuilds intermediates with its own weights;
                // reset so stale tensors cannot leak across layers.
                for (int i = 1; i < n_tensors; i++) free(tensors[i].data);
                n_tensors = 1;
            }
            for (int link = 0; link < n_links; link++) {
                const ChainLink* step = &kChain[link];
                const char* stage_name =
                    orion_voxtream_depth_chain_stage_name(step->stage);
                if (only_layer_env && layer != atoi(only_layer_env)) break;
                if (only_stage_env &&
                    strcmp(only_stage_env, stage_name) != 0) continue;
                if (!orion_voxtream_depth_chain_set_stage(step->stage)) {
                    fprintf(stderr, "FAIL: invalid stage %d\n", step->stage);
                    cleanup_chain(donors, tensors, n_tensors);
                    return 1;
                }
                double compile_ms = 0.0, prepare_ms = 0.0, eval_ms = 0.0;
                double readback_ms = 0.0, stage_total_ms = 0.0;
                char stem[128];
                snprintf(stem, sizeof(stem), "layer%d_chain_%s_ane",
                         layer, stage_name);
                if (!donors[link] && (layer == 0 || !no_reload)) {
                    char donor_stem[128];
                    snprintf(donor_stem, sizeof(donor_stem),
                             "layer0_chain_%s_ane", stage_name);
                    NSString* donor_mil = nil;
                    donors[link] = compile_donor(
                        orion_frontend_voxtream_depth_chain_adapter,
                        0, &config, &identity, blob_dir, donor_stem,
                        &donor_plans[link], step->n_inputs,
                        &compile_ms, &donor_mil);
                    donor_mils[link] = donor_mil;
                    if (!donors[link]) {
                        cleanup_chain(donors, tensors, n_tensors);
                        if (s_injected_failure_observed) {
                            if (!publish_failure_evidence(
                                    blob_dir, revision, compiler_build,
                                    compile_count_start, true)) return 1;
                            printf("PASS: injected failure observed and resources cleaned up\n");
                            return 0;
                        }
                        return 1;
                    }
                    total_compile_ms += compile_ms;
                    printf("stage %-9s compile=%7.2fms\n",
                           stage_name, compile_ms);
                    const char* dbg = getenv("VOXTREAM_DEPTH_CHAIN_DEBUG_DIR");
                    if (dbg) {
                        printf("donor %s dir: %s\n", stage_name,
                               orion_program_model_dir(donors[link]));
                        fflush(stdout);
                        sleep((unsigned int)atoi(dbg));
                    }
                }
                OrionProgram* stage_program = donors[link];
                OrionBindingPlan* stage_plan = &donor_plans[link];
                bool patched = false;
                if (layer > 0 && !no_reload &&
                    plan_has_layer_weights(stage_plan)) {
                    // Patch a new model from the donor with this layer's
                    // weights: a new descriptor bakes the new blobs, so no
                    // recompile is needed and results match a fresh compile.
                    NSDictionary* patched_weights = remapped_dict_from_plan(
                        stage_plan, blob_dir, layer);
                    uint64_t patch_start = monotonic_now();
                    stage_program = orion_program_patch_weights(
                        donors[link], donor_mils[link].UTF8String,
                        patched_weights, stem);
                    total_patch_load_ms += monotonic_elapsed_ms(patch_start);
                    if (!stage_program) {
                        fprintf(stderr,
                                "FAIL: layer %d stage %s weight patch\n",
                                layer, stage_name);
                        cleanup_chain(donors, tensors, n_tensors);
                        return 1;
                    }
                    patched = true;
                    stage_plan = (OrionBindingPlan*)orion_program_plan(stage_program);
                    if (!stage_plan || strcmp(stage_plan->graph_identity,
                                               donor_plans[link].graph_identity) != 0 ||
                        strcmp(stage_plan->mil_identity,
                               donor_plans[link].mil_identity) == 0) {
                        fprintf(stderr, "FAIL: patched stage lost its binding plan\n");
                        orion_release_program(stage_program);
                        cleanup_chain(donors, tensors, n_tensors);
                        return 1;
                    }
                } else if (layer > 0 && no_reload) {
                    // Fresh per-layer compile (diagnostic; budget-limited).
                    static OrionBindingPlan fresh_plans[16];
                    stage_program = compile_donor(
                        orion_frontend_voxtream_depth_chain_adapter,
                        layer, &config, &identity, blob_dir, stem,
                        &fresh_plans[link], step->n_inputs,
                        &compile_ms, NULL);
                    if (!stage_program) {
                        cleanup_chain(donors, tensors, n_tensors);
                        return 1;
                    }
                    stage_plan = &fresh_plans[link];
                    patched = true;
                    total_compile_ms += compile_ms;
                }
                for (int sample = -warmup_count; sample < sample_count; sample++) {
                    if (!eval_donor(stage_program, stage_plan, stem,
                                    tensors, &n_tensors, step->output,
                                    step->out_channels, s, &prepare_ms, &eval_ms,
                                    &readback_ms, &stage_total_ms)) {
                        if (patched) orion_release_program(stage_program);
                        cleanup_chain(donors, tensors, n_tensors);
                        if (s_injected_failure_observed) {
                            if (!publish_failure_evidence(
                                    blob_dir, revision, compiler_build,
                                    compile_count_start, true)) return 1;
                            printf("PASS: injected failure observed and resources cleaned up\n");
                            return 0;
                        }
                        return 1;
                    }
                    if (sample >= 0) {
                        prepare_samples[sample] += prepare_ms;
                        device_samples[sample] += eval_ms;
                        readback_samples[sample] += readback_ms;
                        stage_sum_samples[sample] += stage_total_ms;
                    }
                }
                if (patched) orion_release_program(stage_program);
                printf("layer %d stage %-9s eval=%6.2fms\n",
                       layer, stage_name, eval_ms);
            }
        }

        if (getenv("VOXTREAM_DEPTH_CHAIN_FINAL_NORM")) {
            double compile_ms = 0.0, prepare_ms = 0.0, eval_ms = 0.0;
            double readback_ms = 0.0, stage_total_ms = 0.0;
            ChainTensor* prev = find_tensor(tensors, n_tensors, "hidden_out");
            if (!prev) {
                fprintf(stderr, "FAIL: final norm requires a complete layer output\n");
                cleanup_chain(donors, tensors, n_tensors);
                return 1;
            }
            memcpy(hidden->data, prev->data,
                   (size_t)hidden->channels * s * sizeof(_Float16));
            const int norm_slot = n_links;
            if (!donors[norm_slot]) {
                NSString* norm_mil = nil;
                donors[norm_slot] = compile_donor(
                    final_norm_adapter, 0, &config, &identity, blob_dir,
                    "chain_final_norm_ane", &donor_plans[norm_slot], 1,
                    &compile_ms, &norm_mil);
                donor_mils[norm_slot] = norm_mil;
                if (!donors[norm_slot]) {
                    cleanup_chain(donors, tensors, n_tensors);
                    if (s_injected_failure_observed) {
                        if (!publish_failure_evidence(
                                blob_dir, revision, compiler_build,
                                compile_count_start, true)) return 1;
                        printf("PASS: injected failure observed and resources cleaned up\n");
                        return 0;
                    }
                    return 1;
                }
                total_compile_ms += compile_ms;
            }
            for (int sample = -warmup_count; sample < sample_count; sample++) {
                if (!eval_donor(donors[norm_slot], &donor_plans[norm_slot],
                                "chain_final_norm_ane", tensors, &n_tensors,
                                "final_out", ORION_VOXTREAM_DEPTH_DIM, s,
                                &prepare_ms, &eval_ms, &readback_ms,
                                &stage_total_ms)) {
                    cleanup_chain(donors, tensors, n_tensors);
                    if (s_injected_failure_observed) {
                        if (!publish_failure_evidence(
                                blob_dir, revision, compiler_build,
                                compile_count_start, true)) return 1;
                        printf("PASS: injected failure observed and resources cleaned up\n");
                        return 0;
                    }
                    return 1;
                }
                if (sample >= 0) {
                    prepare_samples[sample] += prepare_ms;
                    device_samples[sample] += eval_ms;
                    readback_samples[sample] += readback_ms;
                    stage_sum_samples[sample] += stage_total_ms;
                }
            }
            printf("final_norm compile=%7.2fms eval=%6.2fms\n",
                   compile_ms, eval_ms);
        }

        const char* result_name = getenv("VOXTREAM_DEPTH_CHAIN_FINAL_NORM")
            ? "final_out" : "hidden_out";
        ChainTensor* result = find_tensor(tensors, n_tensors, result_name);
        if (!result && only_stage_env) {
            printf("PASS: diagnostic stage %s compiled and evaluated\n",
                   only_stage_env);
            cleanup_chain(donors, tensors, n_tensors);
            if (failure_injection_requested()) {
                if (!publish_failure_evidence(blob_dir, revision, compiler_build,
                                              compile_count_start, true)) return 1;
                return s_injected_failure_observed ? 0 : 1;
            }
            return 0;
        }
        if (!result) {
            fprintf(stderr, "FAIL: chain produced no %s\n", result_name);
            cleanup_chain(donors, tensors, n_tensors);
            return 1;
        }

        NSString* export_identity = nil;
        NSString* reference_name = getenv("VOXTREAM_DEPTH_CHAIN_FINAL_NORM")
            ? @"packed_final_norm_output"
            : [NSString stringWithFormat:@"packed_layer_%d_output", n_layers - 1];
        NSString* canonical_reference = qualified_reference(
            blob_dir, reference_name, batch_index, &export_identity);
        const char* reference_path = canonical_reference.UTF8String;
        double max_abs = 0.0;
        double sum_sq = 0.0;
        double ref_sq = 0.0;
        bool compared = false;
        const bool diagnostic = evidence_diagnostic;
        if (!reference_path || !reference_path[0]) {
            if (!diagnostic) {
                fprintf(stderr, "FAIL: validated export lacks canonical %s reference\n",
                        reference_name.UTF8String);
                cleanup_chain(donors, tensors, n_tensors);
                return 1;
            }
        } else if (!diagnostic && (!revision || !revision[0] ||
                                   !compiler_build || !compiler_build[0])) {
            fprintf(stderr, "FAIL: qualification requires VOXTREAM_ORION_REVISION "
                            "and VOXTREAM_COMPILER_BUILD\n");
            cleanup_chain(donors, tensors, n_tensors);
            return 1;
        } else {
            NSData* reference = [NSData dataWithContentsOfFile:
                @(reference_path)];
            size_t expect = (size_t)ORION_VOXTREAM_DEPTH_DIM * s *
                            sizeof(_Float16);
            if (reference.length != expect) {
                fprintf(stderr, "FAIL: reference size %lu != %lu\n",
                        (unsigned long)reference.length,
                        (unsigned long)expect);
                cleanup_chain(donors, tensors, n_tensors);
                return 1;
            }
            const _Float16* ref = (const _Float16*)reference.bytes;
            for (int c = 0; c < ORION_VOXTREAM_DEPTH_DIM; c++) {
                for (int t = 0; t < ORION_VOXTREAM_DEPTH_BUCKET; t++) {
                    size_t idx = (size_t)c * s + t;
                    if (!isfinite((float)ref[idx])) {
                        fprintf(stderr, "FAIL: reference contains non-finite logical output\n");
                        cleanup_chain(donors, tensors, n_tensors);
                        return 1;
                    }
                    double diff = fabs((double)result->data[idx] -
                                       (double)ref[idx]);
                    if (diff > max_abs) max_abs = diff;
                    sum_sq += diff * diff;
                    ref_sq += (double)ref[idx] * (double)ref[idx];
                }
            }
            if (ref_sq == 0.0) {
                fprintf(stderr, "FAIL: reference has zero logical norm\n");
                cleanup_chain(donors, tensors, n_tensors);
                return 1;
            }
            compared = true;
        }

        const char* dump_path = getenv("VOXTREAM_DEPTH_ANE_OUTPUT");
        if (dump_path && dump_path[0]) {
            size_t bytes = (size_t)ORION_VOXTREAM_DEPTH_DIM * s *
                           sizeof(_Float16);
            if (![[NSData dataWithBytes:result->data length:bytes]
                    writeToFile:@(dump_path) atomically:YES]) {
                fprintf(stderr, "FAIL: unable to write %s\n", dump_path);
                cleanup_chain(donors, tensors, n_tensors);
                return 1;
            }
        }

        // Diagnostic: dump any named intermediate tensor (packed fp16).
        const char* dump_tensor = getenv("VOXTREAM_DEPTH_CHAIN_DUMP_TENSOR");
        if (dump_tensor && dump_tensor[0] && dump_path && dump_path[0]) {
            ChainTensor* t = find_tensor(tensors, n_tensors, dump_tensor);
            if (!t) {
                fprintf(stderr, "FAIL: dump tensor %s not found\n",
                        dump_tensor);
                cleanup_chain(donors, tensors, n_tensors);
                return 1;
            }
            size_t bytes = (size_t)t->channels * s * sizeof(_Float16);
            if (![[NSData dataWithBytes:t->data length:bytes]
                    writeToFile:@(dump_path) atomically:YES]) {
                fprintf(stderr, "FAIL: unable to write %s\n", dump_path);
                cleanup_chain(donors, tensors, n_tensors);
                return 1;
            }
            printf("dumped tensor %s [%d x %d] -> %s\n",
                   t->name, t->channels, s, dump_path);
        }

        cleanup_chain(donors, tensors, n_tensors);

        if (getenv("VOXTREAM_DEPTH_CHAIN_FAIL_POINT") &&
            !s_injected_failure_observed) {
            fprintf(stderr, "FAIL: requested failure injection point was not reached\n");
            if (!publish_failure_evidence(blob_dir, revision, compiler_build,
                                          compile_count_start, true)) return 1;
            return 1;
        }

        double relative_l2 = compared ? sqrt(sum_sq / ref_sq) : 0.0;
        double stage_sum_p50 = percentile(stage_sum_samples, sample_count, 0.50);
        double stage_sum_p95 = percentile(stage_sum_samples, sample_count, 0.95);
        double stage_sum_p99 = percentile(stage_sum_samples, sample_count, 0.99);
        double prepare_p99 = percentile(prepare_samples, sample_count, 0.99);
        double device_p99 = percentile(device_samples, sample_count, 0.99);
        double readback_p99 = percentile(readback_samples, sample_count, 0.99);
        const bool complete_depth_measurement = n_layers == 4 &&
            getenv("VOXTREAM_DEPTH_CHAIN_FINAL_NORM") != NULL;
        // Stage-summed bound: stage_sum_p99 is the sum of all 4 layer stages,
        // so this is more complete than a one-layer projection, but it is still
        // a stage-summed NON-CONTIGUOUS bound, NOT a real dependent loop
        // measurement. The real dependent 15-step × 2-branch depth loop
        // measured p99 = 70.8 ms on M5 Pro.
        double projected_separate_cfg_stage_sum_p99_ms = complete_depth_measurement
            ? stage_sum_p99 * 15.0 * 2.0 : NAN;
        const double depth_budget_ms = 60.0;
        const int compile_count_used = orion_compile_count() - compile_count_start;
        const int expected_compiles = n_links +
            (getenv("VOXTREAM_DEPTH_CHAIN_FINAL_NORM") ? 1 : 0);
        if (!no_reload && !only_stage_env && compile_count_used != expected_compiles) {
            fprintf(stderr, "FAIL: expected %d compiles, observed %d\n",
                    expected_compiles, compile_count_used);
            return 1;
        }
        printf("chain: %d layer(s) x %d stages%s, compiles=%d compile=%.2fms "
               "patch_load=%.2fms stage_sum_sample0=%.2fms",
                n_layers, n_links,
                getenv("VOXTREAM_DEPTH_CHAIN_FINAL_NORM") ? " + final_norm" : "",
                compile_count_used, total_compile_ms, total_patch_load_ms,
                stage_sum_samples[0]);
        if (compared && !diagnostic) {
            printf(" max_abs=%.5f relative_l2=%.6f", max_abs, relative_l2);
        }
        printf(" stage_sum_p50_ms=%.2f stage_sum_p95_ms=%.2f "
               "stage_sum_p99_ms=%.2f prepare_stage_sum_p99_ms=%.2f "
               "device_eval_stage_sum_p99_ms=%.2f "
               "readback_stage_sum_p99_ms=%.2f",
               stage_sum_p50, stage_sum_p95, stage_sum_p99, prepare_p99,
               device_p99, readback_p99);
        if (complete_depth_measurement) {
            printf(" stage_summed_bound_15step_separate_cfg_p99_ms=%.2f",
                   projected_separate_cfg_stage_sum_p99_ms);
        }
        printf("\n");
        // Only the one-layer PRD parity gates are enforceable here. Four-layer
        // execution and stage-sum timing remain architectural diagnostics.
        const double max_abs_gate = 0.03;
        const double relative_l2_gate = 0.005;
        const bool one_layer_numerical_passed = !compared ||
            (max_abs <= max_abs_gate && relative_l2 <= relative_l2_gate);
        const bool enforce_numerical_gate = n_layers == 1;
        NSString* continuation_decision = complete_depth_measurement
            ? @"architectural-no-go-diagnostic"
            : @"pending-complete-depth-measurement";
        if (compared && !diagnostic) {
            NSMutableArray* plans = [NSMutableArray array];
            const int plan_count = n_links +
                (getenv("VOXTREAM_DEPTH_CHAIN_FINAL_NORM") ? 1 : 0);
            for (int i = 0; i < plan_count; i++) {
                [plans addObject:@{
                    @"graph_identity": @(donor_plans[i].graph_identity),
                    @"mil_identity": @(donor_plans[i].mil_identity),
                }];
            }
            NSDictionary* result_record = @{
                @"schema_version": @"orion.voxtream.depth-chain-result/1.1",
                @"status": @"diagnostic",
                @"qualification_status": @"diagnostic",
                @"continuation_decision": continuation_decision,
                @"export_identity": export_identity ?: @"",
                @"orion_revision": @(revision),
                @"compiler_build": @(compiler_build),
                @"operating_system": NSProcessInfo.processInfo.operatingSystemVersionString,
                @"layers": @(n_layers),
                @"batch_index": @(batch_index),
                @"final_norm": @(getenv("VOXTREAM_DEPTH_CHAIN_FINAL_NORM") != NULL),
                @"compile_count": @(compile_count_used),
                @"compile_ms": @(total_compile_ms),
                @"patch_load_ms": @(total_patch_load_ms),
                @"samples": @(sample_count),
                @"warmup": @(warmup_count),
                @"timing_method": @"sum-of-stage-samples-not-contiguous-chain",
                @"timing_qualified": @NO,
                @"stage_sum_samples_ms": sample_array(stage_sum_samples, sample_count),
                @"prepare_stage_sum_samples_ms": sample_array(prepare_samples, sample_count),
                @"device_eval_stage_sum_samples_ms": sample_array(device_samples, sample_count),
                @"readback_stage_sum_samples_ms": sample_array(readback_samples, sample_count),
                @"stage_sum_p50_ms": @(stage_sum_p50),
                @"stage_sum_p95_ms": @(stage_sum_p95),
                @"stage_sum_p99_ms": @(stage_sum_p99),
                @"prepare_stage_sum_p99_ms": @(prepare_p99),
                @"device_eval_stage_sum_p99_ms": @(device_p99),
                @"readback_stage_sum_p99_ms": @(readback_p99),
                @"stage_summed_bound_15step_separate_cfg_p99_ms":
                    complete_depth_measurement
                        ? @(projected_separate_cfg_stage_sum_p99_ms)
                        : [NSNull null],
                @"depth_budget_ms": @(depth_budget_ms),
                @"numerical_passed": enforce_numerical_gate
                    ? @(one_layer_numerical_passed) : [NSNull null],
                @"numerical_scope": enforce_numerical_gate
                    ? @"one-layer-prd-gate"
                    : @"stateless-four-layer-diagnostic",
                @"performance_passed": [NSNull null],
                @"max_abs": @(max_abs),
                @"max_abs_gate": enforce_numerical_gate
                    ? @(max_abs_gate) : [NSNull null],
                @"relative_l2": @(relative_l2),
                @"relative_l2_definition":
                    @"sqrt(sum(error^2)/sum(reference^2))",
                @"relative_l2_gate": enforce_numerical_gate
                    ? @(relative_l2_gate) : [NSNull null],
                @"reference_name": reference_name,
                @"reference_sha256": sha256_file(canonical_reference),
                @"plans": plans,
            };
            NSData* result_json = [NSJSONSerialization dataWithJSONObject:
                result_record options:NSJSONWritingPrettyPrinted error:nil];
            NSString* result_path = [@(blob_dir) stringByAppendingPathComponent:
                [NSString stringWithFormat:@"depth_chain_result.layers%d.final%d.batch%d.json",
                    n_layers, getenv("VOXTREAM_DEPTH_CHAIN_FINAL_NORM") != NULL,
                    batch_index]];
            if (!result_json || ![result_json writeToFile:result_path atomically:YES]) {
                fprintf(stderr, "FAIL: unable to publish qualification result\n");
                return 1;
            }
        }
        if (enforce_numerical_gate && !one_layer_numerical_passed) {
            fprintf(stderr,
                    "FAIL: one-layer parity gate failed (max_abs %.5f, gate %.5f; "
                    "relative_l2 %.6f, gate %.6f)\n", max_abs, max_abs_gate,
                    relative_l2, relative_l2_gate);
            return 1;
        }
        printf("PASS: VoXtream decomposed layer chain\n");
        return 0;
    }
}
