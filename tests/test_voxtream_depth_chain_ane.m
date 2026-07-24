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
#import <IOSurface/IOSurface.h>
#import <CommonCrypto/CommonDigest.h>
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
                                     NSString** export_identity) {
    NSString* manifest_path = [@(root) stringByAppendingPathComponent:
        @"depth_export.manifest.json"];
    NSData* data = [NSData dataWithContentsOfFile:manifest_path];
    NSDictionary* manifest = data ? [NSJSONSerialization JSONObjectWithData:data
        options:0 error:nil] : nil;
    if (![manifest[@"schema_version"] isEqual:@"voxtream.orion.depth-export/1.4"])
        return nil;
    NSArray* records = manifest[@"ane_references"];
    for (NSDictionary* record in records) {
        if (![record[@"name"] isEqual:name] ||
            ![record[@"dtype"] isEqual:@"float16-little-endian"] ||
            [record[@"batch_index"] intValue] != 0) continue;
        NSString* path = [@(root) stringByAppendingPathComponent:record[@"path"]];
        if (![[sha256_file(path) lowercaseString]
                isEqual:[record[@"sha256"] lowercaseString]]) return nil;
        if (export_identity) *export_identity = manifest[@"identity_sha256"];
        return path;
    }
    return nil;
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
    char error[256];
    CFAbsoluteTime compile_start = CFAbsoluteTimeGetCurrent();
    OrionProgram* program = orion_compile_mil_with_plan(
        mil.UTF8String, plan_out, wdict, stem, error, sizeof(error));
    *compile_ms_out = (CFAbsoluteTimeGetCurrent() - compile_start) * 1000.0;
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
                       double* eval_ms_out) {
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
    if (ok) {
        output = orion_tensor_create_for_binding(&plan->outputs[0],
                                                 &surface_status);
        ok = output && surface_status == ORION_SURFACE_OK;
    }
    CFAbsoluteTime eval_start = CFAbsoluteTimeGetCurrent();
    if (ok) {
        ok = orion_eval_with_plan(program, plan, inputs,
                                  (int)plan->n_inputs, &output, 1);
    }
    *eval_ms_out = (CFAbsoluteTimeGetCurrent() - eval_start) * 1000.0;

    ChainTensor* out_tensor = NULL;
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
        ok = out_tensor->data &&
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
    if (!ok) fprintf(stderr, "FAIL: %s eval/readback failed\n", stem);
    return ok;
}

int main(void) {
    @autoreleasepool {
        const char* blob_dir = getenv("VOXTREAM_DEPTH_EXPORT");
        if (!blob_dir || !blob_dir[0]) {
            printf("SKIP: set VOXTREAM_DEPTH_EXPORT to a validated export directory\n");
            return 0;
        }
        if (!orion_ane_init()) {
            printf("SKIP: ANE runtime unavailable\n");
            return 0;
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
        double total_eval_ms = 0.0;
        const int compile_count_start = orion_compile_count();

        // Seed the fixture: batch 0 of prefix-8-guided-step0, logical
        // [S,D] = (token*D + channel + 7)/D, packed channel-major, with zero
        // padding in token slots [16, 32).
        ChainTensor* hidden = &tensors[n_tensors++];
        hidden->name = "hidden";
        hidden->channels = ORION_VOXTREAM_DEPTH_DIM;
        hidden->data = calloc((size_t)hidden->channels * s, sizeof(_Float16));
        const char* input_bin = getenv("VOXTREAM_DEPTH_CHAIN_INPUT_BIN");
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
                double compile_ms = 0.0, eval_ms = 0.0;
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
                    CFAbsoluteTime patch_start = CFAbsoluteTimeGetCurrent();
                    stage_program = orion_program_patch_weights(
                        donors[link], donor_mils[link].UTF8String,
                        patched_weights, stem);
                    total_patch_load_ms +=
                        (CFAbsoluteTimeGetCurrent() - patch_start) * 1000.0;
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
                if (!eval_donor(stage_program, stage_plan, stem,
                                tensors, &n_tensors, step->output,
                                step->out_channels, s, &eval_ms)) {
                    if (patched) orion_release_program(stage_program);
                    cleanup_chain(donors, tensors, n_tensors);
                    return 1;
                }
                if (patched) orion_release_program(stage_program);
                total_eval_ms += eval_ms;
                printf("layer %d stage %-9s eval=%6.2fms\n",
                       layer, stage_name, eval_ms);
            }
        }

        if (getenv("VOXTREAM_DEPTH_CHAIN_FINAL_NORM")) {
            double compile_ms = 0.0, eval_ms = 0.0;
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
                    return 1;
                }
                total_compile_ms += compile_ms;
            }
            if (!eval_donor(donors[norm_slot], &donor_plans[norm_slot],
                            "chain_final_norm_ane", tensors, &n_tensors,
                            "final_out", ORION_VOXTREAM_DEPTH_DIM, s,
                            &eval_ms)) {
                cleanup_chain(donors, tensors, n_tensors);
                return 1;
            }
            total_eval_ms += eval_ms;
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
            blob_dir, reference_name, &export_identity);
        const char* reference_path = canonical_reference.UTF8String;
        double max_abs = 0.0;
        double sum_sq = 0.0;
        double ref_sq = 0.0;
        bool compared = false;
        const bool diagnostic = only_stage_env || repeat_env;
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

        double relative_l2 = compared ? sqrt(sum_sq / ref_sq) : 0.0;
        const int compile_count_used = orion_compile_count() - compile_count_start;
        const int expected_compiles = n_links +
            (getenv("VOXTREAM_DEPTH_CHAIN_FINAL_NORM") ? 1 : 0);
        if (!no_reload && !only_stage_env && compile_count_used != expected_compiles) {
            fprintf(stderr, "FAIL: expected %d compiles, observed %d\n",
                    expected_compiles, compile_count_used);
            return 1;
        }
        printf("chain: %d layer(s) x %d stages%s, compiles=%d compile=%.2fms patch_load=%.2fms eval=%.2fms",
                n_layers, n_links,
                getenv("VOXTREAM_DEPTH_CHAIN_FINAL_NORM") ? " + final_norm" : "",
                compile_count_used, total_compile_ms, total_patch_load_ms, total_eval_ms);
        if (compared && !diagnostic) {
            printf(" max_abs=%.5f relative_l2=%.6f", max_abs, relative_l2);
        }
        printf("\n");
        // Parity gates default to the PRD single-layer thresholds
        // (max_abs 0.03, nrmse 0.005) and may be overridden for documented
        // fp16-accumulation evidence runs.
        const char* max_abs_env = getenv("VOXTREAM_DEPTH_CHAIN_MAX_ABS");
        const char* relative_l2_env = getenv("VOXTREAM_DEPTH_CHAIN_RELATIVE_L2");
        double max_abs_gate = max_abs_env ? atof(max_abs_env) : 0.03;
        double relative_l2_gate = relative_l2_env ? atof(relative_l2_env) : 0.005;
        const bool parity_passed = !compared ||
            (max_abs <= max_abs_gate && relative_l2 <= relative_l2_gate);
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
                @"schema_version": @"orion.voxtream.depth-chain-result/1.0",
                @"status": parity_passed ? @"pass" : @"fail",
                @"export_identity": export_identity ?: @"",
                @"orion_revision": @(revision),
                @"compiler_build": @(compiler_build),
                @"operating_system": NSProcessInfo.processInfo.operatingSystemVersionString,
                @"layers": @(n_layers),
                @"final_norm": @(getenv("VOXTREAM_DEPTH_CHAIN_FINAL_NORM") != NULL),
                @"compile_count": @(compile_count_used),
                @"compile_ms": @(total_compile_ms),
                @"patch_load_ms": @(total_patch_load_ms),
                @"eval_ms": @(total_eval_ms),
                @"max_abs": @(max_abs),
                @"max_abs_gate": @(max_abs_gate),
                @"relative_l2": @(relative_l2),
                @"relative_l2_gate": @(relative_l2_gate),
                @"reference_name": reference_name,
                @"reference_sha256": sha256_file(canonical_reference),
                @"plans": plans,
            };
            NSData* result_json = [NSJSONSerialization dataWithJSONObject:
                result_record options:NSJSONWritingPrettyPrinted error:nil];
            NSString* result_path = [@(blob_dir) stringByAppendingPathComponent:
                @"depth_chain_result.json"];
            if (!result_json || ![result_json writeToFile:result_path atomically:YES]) {
                fprintf(stderr, "FAIL: unable to publish qualification result\n");
                return 1;
            }
        }
        if (!parity_passed) {
            fprintf(stderr,
                    "FAIL: parity gate failed (max_abs %.5f, gate %.5f; "
                     "relative_l2 %.6f, gate %.6f)\n", max_abs, max_abs_gate,
                      relative_l2, relative_l2_gate);
            return 1;
        }
        printf("PASS: VoXtream decomposed layer chain\n");
        return 0;
    }
}
