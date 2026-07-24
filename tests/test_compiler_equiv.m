// tests/test_compiler_equiv.m — T139: Compiler equivalence tests
// Verify that compiler-generated MIL structurally matches hand-written MIL.
// Full numerical equivalence requires ANE hardware and weight files —
// this test verifies structural equivalence (same ops, same weight refs).

#import <Foundation/Foundation.h>
#include <string.h>
#include "compiler/graph.h"
#include "compiler/builder.h"
#include "compiler/pipeline.h"
#include "compiler/validate.h"
#import "compiler/codegen.h"
#import "compiler/mil_diff.h"
#include "compiler/frontends/gpt2_prefill.h"
#include "compiler/frontends/gpt2_decode.h"
#include "compiler/frontends/gpt2_final.h"
#include "compiler/frontends/classifier_softmax.h"
#include "compiler/frontends/stories_train.h"
#include "compiler/frontends/voxtream_depth.h"
#include "compiler/model_config.h"
#include "compiler/pass_ane_validate.h"
#import "compiler/kernel_adapter.h"

static int g_passed = 0, g_failed = 0;

#define TEST(name) do { \
    printf("  %-50s ", #name); \
    if (test_##name()) { printf("PASS\n"); g_passed++; } \
    else { printf("FAIL\n"); g_failed++; } \
} while(0)

// GPT-2 config for testing
static const OrionModelConfig kTestGPT2 = {
    .n_layer = 12, .n_head = 12, .d_model = 768,
    .head_dim = 64, .hidden_dim = 3072, .vocab = 50257, .max_seq = 1024
};

// Stories config for testing
static const OrionModelConfig kTestStories = {
    .n_layer = 12, .n_head = 12, .d_model = 768,
    .head_dim = 64, .hidden_dim = 2048, .vocab = 32000, .max_seq = 256
};

// Helper: build graph -> validate -> optimize -> codegen
static NSString* compile_frontend(OrionGraph* (*frontend)(int, int, const OrionModelConfig*),
                                   int layer, int bucket, const OrionModelConfig* cfg) {
    OrionGraph* g = frontend(layer, bucket, cfg);
    if (!g) return nil;

    OrionValidationResult vr = orion_graph_validate(g);
    if (!vr.valid) {
        NSLog(@"Validation failed: %s", vr.message);
        orion_graph_free(g);
        return nil;
    }

    orion_pipeline_optimize(g);
    NSString* mil = orion_codegen_mil(g, "main");
    orion_graph_free(g);
    return mil;
}

// Helper for 2-arg frontends (decode doesn't take bucket)
static NSString* compile_frontend_2arg(OrionGraph* (*frontend)(int, const OrionModelConfig*),
                                        int layer, const OrionModelConfig* cfg) {
    OrionGraph* g = frontend(layer, cfg);
    if (!g) return nil;

    OrionValidationResult vr = orion_graph_validate(g);
    if (!vr.valid) {
        NSLog(@"Validation failed: %s", vr.message);
        orion_graph_free(g);
        return nil;
    }

    orion_pipeline_optimize(g);
    NSString* mil = orion_codegen_mil(g, "main");
    orion_graph_free(g);
    return mil;
}

// Test: GPT-2 prefill attention generates valid MIL
static bool test_gpt2_prefill_attn(void) {
    NSString* mil = compile_frontend(orion_frontend_gpt2_prefill_attn, 0, 64, &kTestGPT2);
    if (!mil) return false;

    // Structural checks
    bool has_conv = [mil containsString:@"conv("];
    bool has_matmul = [mil containsString:@"matmul("];
    bool has_softmax = [mil containsString:@"softmax("];
    bool has_outputs = [mil containsString:@"hidden"] && [mil containsString:@"k_cache"] && [mil containsString:@"v_cache"];
    bool has_weights = [mil containsString:@"layer0/wq.bin"];

    return has_conv && has_matmul && has_softmax && has_outputs && has_weights;
}

// Test: GPT-2 prefill FFN generates valid MIL
static bool test_gpt2_prefill_ffn(void) {
    NSString* mil = compile_frontend(orion_frontend_gpt2_prefill_ffn, 0, 64, &kTestGPT2);
    if (!mil) return false;

    bool has_conv = [mil containsString:@"conv("];
    bool has_tanh = [mil containsString:@"tanh("]; // GELU decomposition
    bool has_output = [mil containsString:@"hidden"];
    return has_conv && has_tanh && has_output;
}

// Test: GPT-2 decode proj generates valid MIL
static bool test_gpt2_decode_proj(void) {
    NSString* mil = compile_frontend_2arg(orion_frontend_gpt2_decode_proj, 0, &kTestGPT2);
    if (!mil) return false;

    bool has_q = [mil containsString:@"q32"];
    bool has_k = [mil containsString:@"k32"];
    bool has_v = [mil containsString:@"v32"];
    return has_q && has_k && has_v;
}

// Test: GPT-2 decode FFN generates valid MIL
static bool test_gpt2_decode_ffn(void) {
    NSString* mil = compile_frontend_2arg(orion_frontend_gpt2_decode_ffn, 0, &kTestGPT2);
    if (!mil) return false;
    return [mil containsString:@"hidden"];
}

// Test: Stories forward attention's compiler-required split generates valid MIL.
static bool test_stories_fwd_attn(void) {
    OrionGraph* g = orion_frontend_fwd_attn(0, &kTestStories);
    if (!g) return false;
    OrionValidationResult vr = orion_graph_validate(g);
    if (!vr.valid) { orion_graph_free(g); return false; }
    orion_pipeline_optimize(g);
    NSString* mil = orion_codegen_mil(g, "main");
    orion_graph_free(g);

    OrionGraph* core = orion_frontend_fwd_attn_core(0, &kTestStories);
    if (!core) return false;
    vr = orion_graph_validate(core);
    if (!vr.valid) { orion_graph_free(core); return false; }
    orion_pipeline_optimize(core);
    NSString* core_mil = orion_codegen_mil(core, "main");
    orion_graph_free(core);

    return mil != nil && [mil containsString:@"q_conv"] &&
           core_mil != nil && [core_mil containsString:@"matmul("] &&
           [core_mil containsString:@"softmax("];
}

// Test: Stories fwdFFN generates valid MIL
static bool test_stories_fwd_ffn(void) {
    OrionGraph* g = orion_frontend_fwd_ffn(0, &kTestStories);
    if (!g) return false;
    OrionValidationResult vr = orion_graph_validate(g);
    if (!vr.valid) { orion_graph_free(g); return false; }
    orion_pipeline_optimize(g);
    NSString* mil = orion_codegen_mil(g, "main");
    orion_graph_free(g);

    return mil != nil && [mil containsString:@"sigmoid("]; // SiLU decomposition
}

// Test: Stories ffnBwd generates valid MIL
static bool test_stories_ffn_bwd(void) {
    OrionGraph* g = orion_frontend_ffn_bwd(0, &kTestStories);
    if (!g) return false;
    OrionValidationResult vr = orion_graph_validate(g);
    if (!vr.valid) { orion_graph_free(g); return false; }
    orion_pipeline_optimize(g);
    NSString* mil = orion_codegen_mil(g, "main");
    orion_graph_free(g);

    return mil != nil && [mil containsString:@"dx"] && [mil containsString:@"slice_by_index("];
}

// Test: Stories sdpaBwd1 generates valid MIL
static bool test_stories_sdpa_bwd1(void) {
    OrionGraph* g = orion_frontend_sdpa_bwd1(0, &kTestStories);
    if (!g) return false;
    OrionValidationResult vr = orion_graph_validate(g);
    if (!vr.valid) { orion_graph_free(g); return false; }
    orion_pipeline_optimize(g);
    NSString* mil = orion_codegen_mil(g, "main");
    orion_graph_free(g);

    return mil != nil && [mil containsString:@"dvf"] && [mil containsString:@"softmax("];
}

// Test: Stories sdpaBwd2 generates valid MIL
static bool test_stories_sdpa_bwd2(void) {
    OrionGraph* g = orion_frontend_sdpa_bwd2(0, &kTestStories);
    if (!g) return false;
    OrionValidationResult vr = orion_graph_validate(g);
    if (!vr.valid) { orion_graph_free(g); return false; }
    orion_pipeline_optimize(g);
    NSString* mil = orion_codegen_mil(g, "main");
    orion_graph_free(g);

    return mil != nil && [mil containsString:@"dqf"] && [mil containsString:@"dkf"];
}

// Test: GPT-2 final LayerNorm generates valid MIL
static bool test_gpt2_final_ln(void) {
    // gpt2_final_ln takes (bucket, cfg) — not a standard 3-arg or 2-arg frontend
    OrionGraph* g = orion_frontend_gpt2_final_ln(64, &kTestGPT2);
    if (!g) return false;

    OrionValidationResult vr = orion_graph_validate(g);
    if (!vr.valid) { orion_graph_free(g); return false; }
    orion_pipeline_optimize(g);
    NSString* mil = orion_codegen_mil(g, "main");
    orion_graph_free(g);

    if (!mil) return false;
    bool has_output = [mil containsString:@"hidden"];
    bool has_weights = [mil containsString:@"ln_f_g.bin"];
    return has_output && has_weights;
}

// Test: Classifier forward generates valid MIL
static bool test_classifier_fwd(void) {
    // classifier_fwd takes (dim, vocab) — standalone signature
    OrionGraph* g = orion_frontend_classifier_fwd(768, 32000);
    if (!g) return false;

    OrionValidationResult vr = orion_graph_validate(g);
    if (!vr.valid) { orion_graph_free(g); return false; }
    orion_pipeline_optimize(g);
    NSString* mil = orion_codegen_mil(g, "main");
    orion_graph_free(g);

    if (!mil) return false;
    bool has_output = [mil containsString:@"output"];
    bool has_conv = [mil containsString:@"conv("];
    bool has_embed = [mil containsString:@"embed.bin"];
    return has_output && has_conv && has_embed;
}

// Test: Vocab softmax generates valid MIL
static bool test_vocab_softmax(void) {
    // vocab_softmax takes (vocab, seq_len) — standalone signature
    OrionGraph* g = orion_frontend_vocab_softmax(32000, 256);
    if (!g) return false;

    OrionValidationResult vr = orion_graph_validate(g);
    if (!vr.valid) { orion_graph_free(g); return false; }
    orion_pipeline_optimize(g);
    NSString* mil = orion_codegen_mil(g, "main");
    orion_graph_free(g);

    if (!mil) return false;
    bool has_output = [mil containsString:@"output"];
    bool has_softmax = [mil containsString:@"softmax("];
    return has_output && has_softmax;
}

// Test: Stories qkvBwd generates valid MIL
static bool test_stories_qkv_bwd(void) {
    OrionGraph* g = orion_frontend_qkv_bwd(0, &kTestStories);
    if (!g) return false;
    OrionValidationResult vr = orion_graph_validate(g);
    if (!vr.valid) { orion_graph_free(g); return false; }
    orion_pipeline_optimize(g);
    NSString* mil = orion_codegen_mil(g, "main");
    orion_graph_free(g);

    return mil != nil && [mil containsString:@"dx"];
}

static const OrionNode* live_node_named(const OrionGraph* g, const char* name) {
    for (int i = 0; i < g->n_nodes; i++) {
        if (g->nodes[i].is_live && strcmp(g->nodes[i].name, name) == 0) return &g->nodes[i];
    }
    return NULL;
}

static bool shape_is(const OrionNode* node, int a, int b, int c, int d) {
    return node && node->shape[0] == a && node->shape[1] == b &&
           node->shape[2] == c && node->shape[3] == d;
}

static bool test_voxtream_depth_layer(void) {
    OrionGraph* g = orion_frontend_voxtream_depth_layer(0);
    if (!g || orion_frontend_voxtream_depth_layer(-1) != NULL ||
        orion_frontend_voxtream_depth_layer(4) != NULL) {
        orion_graph_free(g);
        return false;
    }
    OrionValidationResult vr = orion_graph_validate(g);
    if (!vr.valid) { orion_graph_free(g); return false; }
    orion_pipeline_optimize(g);
    OrionANEValidationResult ar = orion_pass_ane_validate(g);
    if (!ar.valid || g->n_inputs != 1 || g->n_outputs != 1) {
        orion_graph_free(g);
        return false;
    }

    bool shapes =
        shape_is(live_node_named(g, "hidden"), 1, 1024, 1, 32) &&
        shape_is(live_node_named(g, "q_conv"), 1, 1024, 1, 32) &&
        shape_is(live_node_named(g, "k_conv"), 1, 256, 1, 32) &&
        shape_is(live_node_named(g, "v_conv"), 1, 256, 1, 32) &&
        shape_is(live_node_named(g, "q_rope_swapped"), 1, 1024, 1, 32) &&
        shape_is(live_node_named(g, "k_rope_swapped"), 1, 256, 1, 32) &&
        shape_is(live_node_named(g, "attention_scores"), 2, 4, 32, 32) &&
        shape_is(live_node_named(g, "attention_out"), 1, 1024, 1, 32) &&
        shape_is(live_node_named(g, "mlp_w1_conv"), 1, 8192, 1, 32) &&
        shape_is(live_node_named(g, "hidden_out"), 1, 1024, 1, 32);

    const char* expected_paths[] = {
        "@model_path/constants/causal_mask.bin",
        "@model_path/constants/rope_k_cos.bin",
        "@model_path/constants/rope_k_perm.bin",
        "@model_path/constants/rope_k_sin.bin",
        "@model_path/constants/rope_q_cos.bin",
        "@model_path/constants/rope_q_perm.bin",
        "@model_path/constants/rope_q_sin.bin",
        "@model_path/weights/layer0/rms1.bin",
        "@model_path/weights/layer0/rms2.bin",
        "@model_path/weights/layer0/w1.bin",
        "@model_path/weights/layer0/w2.bin",
        "@model_path/weights/layer0/w3.bin",
        "@model_path/weights/layer0/wk.bin",
        "@model_path/weights/layer0/wo.bin",
        "@model_path/weights/layer0/wq.bin",
        "@model_path/weights/layer0/wv.bin",
    };
    bool seen[16] = {false};
    int weight_count = 0;
    bool no_banned_ops = true;
    for (int i = 0; i < g->n_nodes; i++) {
        const OrionNode* node = &g->nodes[i];
        if (!node->is_live) continue;
        if (node->op == ORION_OP_CONCAT_BANNED || node->op == ORION_OP_SPLIT ||
            node->op == ORION_OP_PAD) no_banned_ops = false;
        if (node->op != ORION_OP_CONST || !node->attrs.blob_path[0]) continue;
        weight_count++;
        if (node->attrs.blob_offset != 64) no_banned_ops = false;
        for (int j = 0; j < 16; j++) {
            if (strcmp(node->attrs.blob_path, expected_paths[j]) == 0) seen[j] = true;
        }
    }
    bool all_weights = weight_count == 16;
    for (int i = 0; i < 16; i++) all_weights = all_weights && seen[i];

    NSString* mil = orion_codegen_mil(g, "voxtream_depth_layer0");
    bool mil_ok = mil != nil && [mil containsString:@"softmax("] &&
                  [mil containsString:@"sigmoid("] &&
                  ![mil containsString:@"concat("] &&
                  ![mil containsString:@"scaled_dot_product_attention"] &&
                  ![mil containsString:@"gelu("] &&
                  [mil containsString:@"-> (hidden_out)"];
    orion_graph_free(g);

    OrionGraph* final_norm = orion_frontend_voxtream_depth_final_norm();
    if (!final_norm) return false;
    OrionValidationResult final_vr = orion_graph_validate(final_norm);
    orion_pipeline_optimize(final_norm);
    OrionANEValidationResult final_ar = orion_pass_ane_validate(final_norm);
    const OrionNode* final_output = live_node_named(final_norm, "hidden_out");
    bool final_weight = false;
    for (int i = 0; i < final_norm->n_nodes; i++) {
        const OrionNode* node = &final_norm->nodes[i];
        if (node->is_live && node->op == ORION_OP_CONST &&
            strcmp(node->attrs.blob_path, "@model_path/weights/final_norm.bin") == 0 &&
            node->attrs.blob_offset == 64) final_weight = true;
    }
    NSString* final_mil = orion_codegen_mil(final_norm, "voxtream_depth_final_norm");
    bool final_ok = final_vr.valid && final_ar.valid &&
        shape_is(final_output, 1, 1024, 1, 32) && final_weight &&
        final_mil != nil && [final_mil containsString:@"-> (hidden_out)"];
    orion_graph_free(final_norm);
    return shapes && all_weights && no_banned_ops && mil_ok && final_ok;
}

static bool test_voxtream_depth_adapter_path(void) {
    OrionModelConfig config = {
        .n_layer = ORION_VOXTREAM_DEPTH_LAYERS,
        .n_head = ORION_VOXTREAM_DEPTH_HEADS,
        .d_model = ORION_VOXTREAM_DEPTH_DIM,
        .head_dim = ORION_VOXTREAM_DEPTH_HEAD_DIM,
        .hidden_dim = ORION_VOXTREAM_DEPTH_HIDDEN_DIM,
        .max_seq = ORION_VOXTREAM_DEPTH_BUCKET,
    };
    NSString* mil = orion_kernel_adapter_generate_mil(
        orion_frontend_voxtream_depth_layer_adapter, 0,
        ORION_VOXTREAM_DEPTH_BUCKET, &config);
    config.hidden_dim--;
    NSString* wrong_config = orion_kernel_adapter_generate_mil(
        orion_frontend_voxtream_depth_layer_adapter, 0,
        ORION_VOXTREAM_DEPTH_BUCKET, &config);
    NSString* wrong_bucket = orion_kernel_adapter_generate_mil(
        orion_frontend_voxtream_depth_layer_adapter, 0,
        ORION_VOXTREAM_DEPTH_BUCKET - 1, NULL);
    NSString* missing_config = orion_kernel_adapter_generate_mil(
        orion_frontend_voxtream_depth_layer_adapter, 0,
        ORION_VOXTREAM_DEPTH_BUCKET, NULL);
    return mil != nil && [mil containsString:@"-> (hidden_out)"] &&
        wrong_config == nil && wrong_bucket == nil && missing_config == nil;
}

static bool test_voxtream_depth_all_layers_have_distinct_paths(void) {
    for (int layer = 0; layer < ORION_VOXTREAM_DEPTH_LAYERS; layer++) {
        OrionGraph* g = orion_frontend_voxtream_depth_layer(layer);
        if (!g || !orion_graph_validate(g).valid) { orion_graph_free(g); return false; }
        char expected[256];
        snprintf(expected, sizeof(expected), "@model_path/weights/layer%d/wq.bin", layer);
        bool found = false;
        for (int i = 0; i < g->n_nodes; i++) {
            if (g->nodes[i].op == ORION_OP_CONST &&
                strcmp(g->nodes[i].attrs.blob_path, expected) == 0 &&
                g->nodes[i].attrs.blob_offset == 64) found = true;
        }
        orion_graph_free(g);
        if (!found) return false;
    }
    return true;
}

int main(int argc __attribute__((unused)), char* argv[] __attribute__((unused))) {
    @autoreleasepool {
        printf("test_compiler_equiv:\n");

        TEST(gpt2_prefill_attn);
        TEST(gpt2_prefill_ffn);
        TEST(gpt2_decode_proj);
        TEST(gpt2_decode_ffn);
        TEST(gpt2_final_ln);
        TEST(classifier_fwd);
        TEST(vocab_softmax);
        TEST(stories_fwd_attn);
        TEST(stories_fwd_ffn);
        TEST(stories_ffn_bwd);
        TEST(stories_sdpa_bwd1);
        TEST(stories_sdpa_bwd2);
        TEST(stories_qkv_bwd);
        TEST(voxtream_depth_layer);
        TEST(voxtream_depth_adapter_path);
        TEST(voxtream_depth_all_layers_have_distinct_paths);

        printf("\n%d/%d passed\n", g_passed, g_passed + g_failed);
        return g_failed > 0 ? 1 : 0;
    }
}
