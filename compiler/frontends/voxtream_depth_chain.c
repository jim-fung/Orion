// Decomposed VoXtream depth-transformer layer stage graphs.

#include "voxtream_depth_chain.h"
#include "voxtream_depth.h"
#include "../builder.h"
#include "../patterns.h"
#include <stdio.h>

static const char* kStageNames[ORION_VDX_STAGE_COUNT] = {
    "wq", "wk", "wv",
    "qswap", "qcos", "qsin_add",
    "kswap", "kcos", "ksin_add",
    "attn_core", "wo_res",
    "w1", "w3", "gate", "w2_res",
};

int orion_voxtream_depth_chain_stage_count(void) {
    return ORION_VDX_STAGE_COUNT;
}

const char* orion_voxtream_depth_chain_stage_name(int stage) {
    if (stage < 0 || stage >= ORION_VDX_STAGE_COUNT) return "invalid";
    return kStageNames[stage];
}

static bool valid_layer(int layer) {
    return layer >= 0 && layer < ORION_VOXTREAM_DEPTH_LAYERS;
}

static int add_input(OrionGraph* g, const char* name, int channels, int seq) {
    int shape[4] = {1, channels, 1, seq};
    return orion_gb_input(g, name, ORION_DTYPE_FP16, shape);
}

static int add_scale_weight(OrionGraph* g, const char* name, int layer,
                            const char* file, int channels) {
    int shape[4] = {1, channels, 1, 1};
    char path[256];
    snprintf(path, sizeof(path), "@model_path/weights/layer%d/%s", layer, file);
    return orion_gb_const_weight(g, name, ORION_DTYPE_FP16, shape, path, 64);
}

static void layer_path(char* out, size_t size, int layer, const char* file) {
    snprintf(out, size, "@model_path/weights/layer%d/%s", layer, file);
}

// rms projection stages: hidden -> rms1 -> {wq,wk,wv}
static OrionGraph* build_rms_proj(int layer, int out_dim, const char* weight,
                                  const char* out_name) {
    if (!valid_layer(layer)) return NULL;
    const int d = ORION_VOXTREAM_DEPTH_DIM;
    const int s = ORION_VOXTREAM_DEPTH_ANE_SEQ;
    OrionGraph* g = orion_graph_create();
    if (!g) return NULL;
    int hidden = add_input(g, "hidden", d, s);
    int rms_w = add_scale_weight(g, "rms1_weight", layer, "rms1.bin", d);
    int normalized = orion_gb_rmsnorm(g, hidden, rms_w,
                                      ORION_VOXTREAM_DEPTH_RMS_EPS,
                                      "rms1", d, s);
    char path[256];
    layer_path(path, sizeof(path), layer, weight);
    int projected = orion_gb_linear(g, normalized, out_name, d, out_dim, s,
                                    path, NULL);
    orion_gb_output(g, projected, out_name);
    return g;
}

// rope swap: x -> conv1x1(perm) -> x_swapped
static OrionGraph* build_rope_swap(int channels, const char* perm_path,
                                   const char* in_name, const char* out_name) {
    const int s = ORION_VOXTREAM_DEPTH_ANE_SEQ;
    OrionGraph* g = orion_graph_create();
    if (!g) return NULL;
    int input = add_input(g, in_name, channels, s);
    int perm_shape[4] = {channels, channels, 1, 1};
    int perm = orion_gb_const_weight(g, "rope_perm", ORION_DTYPE_FP16,
                                     perm_shape, perm_path, 64);
    int swapped = orion_gb_conv1x1(g, input, perm, -1, out_name, channels, s);
    orion_gb_output(g, swapped, out_name);
    return g;
}

// rope table multiply: x -> x * table -> out
static OrionGraph* build_rope_table_mul(int channels, const char* table_path,
                                        const char* table_name,
                                        const char* in_name,
                                        const char* out_name) {
    const int s = ORION_VOXTREAM_DEPTH_ANE_SEQ;
    OrionGraph* g = orion_graph_create();
    if (!g) return NULL;
    int input = add_input(g, in_name, channels, s);
    int table_shape[4] = {1, channels, 1, s};
    int table = orion_gb_const_weight(g, table_name, ORION_DTYPE_FP16,
                                      table_shape, table_path, 64);
    int out = orion_gb_mul(g, input, table, out_name);
    orion_gb_output(g, out, out_name);
    return g;
}

// rope combine: cos_part + swapped * sin_table -> rope_out (2 inputs)
static OrionGraph* build_rope_combine(int channels, const char* sin_path,
                                      const char* cos_in, const char* swap_in,
                                      const char* out_name) {
    const int s = ORION_VOXTREAM_DEPTH_ANE_SEQ;
    OrionGraph* g = orion_graph_create();
    if (!g) return NULL;
    int cos_part = add_input(g, cos_in, channels, s);
    int swapped = add_input(g, swap_in, channels, s);
    int table_shape[4] = {1, channels, 1, s};
    int sin_table = orion_gb_const_weight(g, "rope_sin", ORION_DTYPE_FP16,
                                          table_shape, sin_path, 64);
    int sin_scaled = orion_gb_mul(g, swapped, sin_table, "rope_sin_scaled");
    int out = orion_gb_add(g, cos_part, sin_scaled, out_name);
    orion_gb_output(g, out, out_name);
    return g;
}

// attention core: q_rope, k_rope, v -> GQA context (3 inputs, mask only)
static OrionGraph* build_attn_core(void) {
    const int d = ORION_VOXTREAM_DEPTH_DIM;
    const int kv = ORION_VOXTREAM_DEPTH_KV_HEADS * ORION_VOXTREAM_DEPTH_HEAD_DIM;
    const int s = ORION_VOXTREAM_DEPTH_ANE_SEQ;
    OrionGraph* g = orion_graph_create();
    if (!g) return NULL;
    int q = add_input(g, "q_rope_in", d, s);
    int k = add_input(g, "k_rope_in", kv, s);
    int v = add_input(g, "v_in", kv, s);
    int ctx = orion_pattern_gqa_attention(
        g, q, k, v, "@model_path/constants/causal_mask.bin",
        ORION_VOXTREAM_DEPTH_HEADS, ORION_VOXTREAM_DEPTH_KV_HEADS,
        ORION_VOXTREAM_DEPTH_HEAD_DIM, s, "attention");
    orion_gb_output(g, ctx, "attn_ctx");
    return g;
}

// wo projection plus attention residual: ctx, hidden -> wo(ctx) + hidden
static OrionGraph* build_wo_res(int layer) {
    if (!valid_layer(layer)) return NULL;
    const int d = ORION_VOXTREAM_DEPTH_DIM;
    const int s = ORION_VOXTREAM_DEPTH_ANE_SEQ;
    OrionGraph* g = orion_graph_create();
    if (!g) return NULL;
    int ctx = add_input(g, "attn_ctx_in", d, s);
    int hidden = add_input(g, "hidden_in", d, s);
    char path[256];
    layer_path(path, sizeof(path), layer, "wo.bin");
    int projected = orion_gb_linear(g, ctx, "wo", d, d, s, path, NULL);
    int out = orion_gb_add(g, hidden, projected, "attn_res");
    orion_gb_output(g, out, "attn_res");
    return g;
}

// mlp up stages: attn_res -> rms2 -> {w1,w3}
static OrionGraph* build_mlp_up(int layer, const char* weight,
                                const char* out_name) {
    if (!valid_layer(layer)) return NULL;
    const int d = ORION_VOXTREAM_DEPTH_DIM;
    const int h = ORION_VOXTREAM_DEPTH_HIDDEN_DIM;
    const int s = ORION_VOXTREAM_DEPTH_ANE_SEQ;
    OrionGraph* g = orion_graph_create();
    if (!g) return NULL;
    int hidden = add_input(g, "attn_res_in", d, s);
    int rms_w = add_scale_weight(g, "rms2_weight", layer, "rms2.bin", d);
    int normalized = orion_gb_rmsnorm(g, hidden, rms_w,
                                      ORION_VOXTREAM_DEPTH_RMS_EPS,
                                      "rms2", d, s);
    char path[256];
    layer_path(path, sizeof(path), layer, weight);
    int up = orion_gb_linear(g, normalized, out_name, d, h, s, path, NULL);
    orion_gb_output(g, up, out_name);
    return g;
}

// gate: silu(mlp_up) * mlp_gate_in (2 inputs, no weights)
static OrionGraph* build_gate(void) {
    const int h = ORION_VOXTREAM_DEPTH_HIDDEN_DIM;
    const int s = ORION_VOXTREAM_DEPTH_ANE_SEQ;
    OrionGraph* g = orion_graph_create();
    if (!g) return NULL;
    int up = add_input(g, "mlp_up_in", h, s);
    int gate_in = add_input(g, "mlp_gate_in", h, s);
    int activated = orion_gb_silu(g, up, "mlp_silu", h, s);
    int out = orion_gb_mul(g, activated, gate_in, "gated");
    orion_gb_output(g, out, "gated");
    return g;
}

// w2 down projection plus mlp residual: gated, attn_res -> w2(gated)+attn_res
static OrionGraph* build_w2_res(int layer) {
    if (!valid_layer(layer)) return NULL;
    const int d = ORION_VOXTREAM_DEPTH_DIM;
    const int h = ORION_VOXTREAM_DEPTH_HIDDEN_DIM;
    const int s = ORION_VOXTREAM_DEPTH_ANE_SEQ;
    OrionGraph* g = orion_graph_create();
    if (!g) return NULL;
    int gated = add_input(g, "gated_in", h, s);
    int attn_res = add_input(g, "attn_res_in", d, s);
    char path[256];
    layer_path(path, sizeof(path), layer, "w2.bin");
    int down = orion_gb_linear(g, gated, "w2", h, d, s, path, NULL);
    int out = orion_gb_add(g, attn_res, down, "hidden_out");
    orion_gb_output(g, out, "hidden_out");
    return g;
}

OrionGraph* orion_frontend_voxtream_depth_chain_stage_graph(int layer,
                                                            int stage) {
    const int d = ORION_VOXTREAM_DEPTH_DIM;
    const int kv = ORION_VOXTREAM_DEPTH_KV_HEADS * ORION_VOXTREAM_DEPTH_HEAD_DIM;
    switch (stage) {
    case ORION_VDX_WQ:
        return build_rms_proj(layer, d, "wq.bin", "q");
    case ORION_VDX_WK:
        return build_rms_proj(layer, kv, "wk.bin", "k");
    case ORION_VDX_WV:
        return build_rms_proj(layer, kv, "wv.bin", "v");
    case ORION_VDX_QSWAP:
        return build_rope_swap(d, "@model_path/constants/rope_q_perm.bin",
                               "q_in", "q_swapped");
    case ORION_VDX_QCOS:
        return build_rope_table_mul(d, "@model_path/constants/rope_q_cos.bin",
                                    "rope_q_cos", "q_in", "q_cos");
    case ORION_VDX_QSIN_ADD:
        return build_rope_combine(d, "@model_path/constants/rope_q_sin.bin",
                                  "q_cos_in", "q_swapped_in", "q_rope");
    case ORION_VDX_KSWAP:
        return build_rope_swap(kv, "@model_path/constants/rope_k_perm.bin",
                               "k_in", "k_swapped");
    case ORION_VDX_KCOS:
        return build_rope_table_mul(kv, "@model_path/constants/rope_k_cos.bin",
                                    "rope_k_cos", "k_in", "k_cos");
    case ORION_VDX_KSIN_ADD:
        return build_rope_combine(kv, "@model_path/constants/rope_k_sin.bin",
                                  "k_cos_in", "k_swapped_in", "k_rope");
    case ORION_VDX_ATTN_CORE:
        return build_attn_core();
    case ORION_VDX_WO_RES:
        return build_wo_res(layer);
    case ORION_VDX_W1:
        return build_mlp_up(layer, "w1.bin", "mlp_up");
    case ORION_VDX_W3:
        return build_mlp_up(layer, "w3.bin", "mlp_gate_in");
    case ORION_VDX_GATE:
        return build_gate();
    case ORION_VDX_W2_RES:
        return build_w2_res(layer);
    default:
        return NULL;
    }
}

static _Thread_local int s_selected_stage = -1;

bool orion_voxtream_depth_chain_set_stage(int stage) {
    if (stage < 0 || stage >= ORION_VDX_STAGE_COUNT) return false;
    s_selected_stage = stage;
    return true;
}

static bool fixed_config(int bucket, const OrionModelConfig* config) {
    return config && bucket == ORION_VOXTREAM_DEPTH_BUCKET &&
        config->n_layer == ORION_VOXTREAM_DEPTH_LAYERS &&
        config->n_head == ORION_VOXTREAM_DEPTH_HEADS &&
        config->d_model == ORION_VOXTREAM_DEPTH_DIM &&
        config->head_dim == ORION_VOXTREAM_DEPTH_HEAD_DIM &&
        config->hidden_dim == ORION_VOXTREAM_DEPTH_HIDDEN_DIM &&
        config->max_seq == ORION_VOXTREAM_DEPTH_BUCKET;
}

OrionGraph* orion_frontend_voxtream_depth_chain_adapter(
    int layer, int bucket, const OrionModelConfig* config) {
    if (!fixed_config(bucket, config) || s_selected_stage < 0) return NULL;
    return orion_frontend_voxtream_depth_chain_stage_graph(layer,
                                                           s_selected_stage);
}
