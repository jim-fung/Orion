// Fixed VoXtream depth-transformer layer frontend.

#include "voxtream_depth.h"
#include "../builder.h"
#include "../patterns.h"
#include <stdio.h>
#include <stdlib.h>

static bool fixed_config(int bucket, const OrionModelConfig* config) {
    return config && bucket == ORION_VOXTREAM_DEPTH_BUCKET &&
        config->n_layer == ORION_VOXTREAM_DEPTH_LAYERS &&
        config->n_head == ORION_VOXTREAM_DEPTH_HEADS &&
        config->d_model == ORION_VOXTREAM_DEPTH_DIM &&
        config->head_dim == ORION_VOXTREAM_DEPTH_HEAD_DIM &&
        config->hidden_dim == ORION_VOXTREAM_DEPTH_HIDDEN_DIM &&
        config->max_seq == ORION_VOXTREAM_DEPTH_BUCKET;
}

static OrionGraph* build_attention(int layer) {
    if (layer < 0 || layer >= ORION_VOXTREAM_DEPTH_LAYERS) return NULL;

    const int d = ORION_VOXTREAM_DEPTH_DIM;
    const int kv = ORION_VOXTREAM_DEPTH_KV_HEADS * ORION_VOXTREAM_DEPTH_HEAD_DIM;
    const int s = ORION_VOXTREAM_DEPTH_ANE_SEQ;
    OrionGraph* g = orion_graph_create();
    if (!g) return NULL;

    int hidden_shape[4] = {1, d, 1, s};
    int scale_shape[4] = {1, d, 1, 1};
    int hidden = orion_gb_input(g, "hidden", ORION_DTYPE_FP16, hidden_shape);
    char path[256];

    snprintf(path, sizeof(path), "@model_path/weights/layer%d/rms1.bin", layer);
    int rms1_weight = orion_gb_const_weight(g, "rms1_weight", ORION_DTYPE_FP16,
                                            scale_shape, path, 64);
    int normalized = orion_gb_rmsnorm(g, hidden, rms1_weight,
                                      ORION_VOXTREAM_DEPTH_RMS_EPS,
                                      "rms1", d, s);

    snprintf(path, sizeof(path), "@model_path/weights/layer%d/wq.bin", layer);
    int q = orion_gb_linear(g, normalized, "q", d, d, s, path, NULL);
    snprintf(path, sizeof(path), "@model_path/weights/layer%d/wk.bin", layer);
    int k = orion_gb_linear(g, normalized, "k", d, kv, s, path, NULL);
    snprintf(path, sizeof(path), "@model_path/weights/layer%d/wv.bin", layer);
    int v = orion_gb_linear(g, normalized, "v", d, kv, s, path, NULL);

    int q_rope = orion_pattern_rope_interleaved_elementwise(
        g, q, "@model_path/constants/rope_q_cos.bin",
        "@model_path/constants/rope_q_sin.bin",
        "@model_path/constants/rope_q_perm.bin",
        ORION_VOXTREAM_DEPTH_HEADS, ORION_VOXTREAM_DEPTH_HEAD_DIM, s, "q_rope");
    int k_rope = orion_pattern_rope_interleaved_elementwise(
        g, k, "@model_path/constants/rope_k_cos.bin",
        "@model_path/constants/rope_k_sin.bin",
        "@model_path/constants/rope_k_perm.bin",
        ORION_VOXTREAM_DEPTH_KV_HEADS, ORION_VOXTREAM_DEPTH_HEAD_DIM, s, "k_rope");

    int attention = orion_pattern_gqa_attention(
        g, q_rope, k_rope, v, "@model_path/constants/causal_mask.bin",
        ORION_VOXTREAM_DEPTH_HEADS, ORION_VOXTREAM_DEPTH_KV_HEADS,
        ORION_VOXTREAM_DEPTH_HEAD_DIM, s, "attention");
    snprintf(path, sizeof(path), "@model_path/weights/layer%d/wo.bin", layer);
    int projected = orion_gb_linear(g, attention, "wo", d, d, s, path, NULL);
    int attention_residual = orion_pattern_residual(g, hidden, projected,
                                                     "attention_residual");

    orion_gb_output(g, attention_residual, "hidden_out");
    return g;
}

static OrionGraph* build_mlp(int layer) {
    if (layer < 0 || layer >= ORION_VOXTREAM_DEPTH_LAYERS) return NULL;
    const int d = ORION_VOXTREAM_DEPTH_DIM;
    const int h = ORION_VOXTREAM_DEPTH_HIDDEN_DIM;
    const int s = ORION_VOXTREAM_DEPTH_ANE_SEQ;
    OrionGraph* g = orion_graph_create();
    if (!g) return NULL;
    int hidden_shape[4] = {1, d, 1, s};
    int scale_shape[4] = {1, d, 1, 1};
    int hidden = orion_gb_input(g, "hidden", ORION_DTYPE_FP16, hidden_shape);
    char path[256];

    snprintf(path, sizeof(path), "@model_path/weights/layer%d/rms2.bin", layer);
    int rms2_weight = orion_gb_const_weight(g, "rms2_weight", ORION_DTYPE_FP16,
                                            scale_shape, path, 64);
    int mlp_input = orion_gb_rmsnorm(g, hidden, rms2_weight,
                                     ORION_VOXTREAM_DEPTH_RMS_EPS,
                                     "rms2", d, s);
    char w1[256], w2[256], w3[256];
    snprintf(w1, sizeof(w1), "@model_path/weights/layer%d/w1.bin", layer);
    snprintf(w2, sizeof(w2), "@model_path/weights/layer%d/w2.bin", layer);
    snprintf(w3, sizeof(w3), "@model_path/weights/layer%d/w3.bin", layer);
    int mlp = orion_pattern_swiglu_ffn(g, mlp_input, "mlp", d, h, s,
                                       w1, w3, w2);
    int output = orion_pattern_residual(g, hidden, mlp, "hidden_out");
    orion_gb_output(g, output, "hidden_out");
    return g;
}

static OrionGraph* build_qproj(int layer) {
    if (layer < 0 || layer >= ORION_VOXTREAM_DEPTH_LAYERS) return NULL;
    const int d = ORION_VOXTREAM_DEPTH_DIM;
    const int s = ORION_VOXTREAM_DEPTH_ANE_SEQ;
    OrionGraph* g = orion_graph_create();
    if (!g) return NULL;
    int hidden_shape[4] = {1, d, 1, s};
    int scale_shape[4] = {1, d, 1, 1};
    int hidden = orion_gb_input(g, "hidden", ORION_DTYPE_FP16, hidden_shape);
    char path[256];
    snprintf(path, sizeof(path), "@model_path/weights/layer%d/rms1.bin", layer);
    int rms_weight = orion_gb_const_weight(g, "rms1_weight", ORION_DTYPE_FP16,
                                            scale_shape, path, 64);
    int normalized = orion_gb_rmsnorm(g, hidden, rms_weight,
                                      ORION_VOXTREAM_DEPTH_RMS_EPS, "rms1", d, s);
    snprintf(path, sizeof(path), "@model_path/weights/layer%d/wq.bin", layer);
    int q = orion_gb_linear(g, normalized, "q", d, d, s, path, NULL);
    orion_gb_output(g, q, "hidden_out");
    return g;
}

static OrionGraph* build_mlp_up(int layer) {
    if (layer < 0 || layer >= ORION_VOXTREAM_DEPTH_LAYERS) return NULL;
    const int d = ORION_VOXTREAM_DEPTH_DIM;
    const int h = ORION_VOXTREAM_DEPTH_HIDDEN_DIM;
    const int s = ORION_VOXTREAM_DEPTH_ANE_SEQ;
    OrionGraph* g = orion_graph_create();
    if (!g) return NULL;
    int hidden_shape[4] = {1, d, 1, s};
    int scale_shape[4] = {1, d, 1, 1};
    int hidden = orion_gb_input(g, "hidden", ORION_DTYPE_FP16, hidden_shape);
    char path[256];
    snprintf(path, sizeof(path), "@model_path/weights/layer%d/rms2.bin", layer);
    int rms_weight = orion_gb_const_weight(g, "rms2_weight", ORION_DTYPE_FP16,
                                            scale_shape, path, 64);
    int normalized = orion_gb_rmsnorm(g, hidden, rms_weight,
                                      ORION_VOXTREAM_DEPTH_RMS_EPS, "rms2", d, s);
    snprintf(path, sizeof(path), "@model_path/weights/layer%d/w1.bin", layer);
    int up = orion_gb_linear(g, normalized, "w1", d, h, s, path, NULL);
    orion_gb_output(g, up, "hidden_out");
    return g;
}

static OrionGraph* build_qrope(int layer) {
    if (layer < 0 || layer >= ORION_VOXTREAM_DEPTH_LAYERS) return NULL;
    const int d = ORION_VOXTREAM_DEPTH_DIM;
    const int s = ORION_VOXTREAM_DEPTH_ANE_SEQ;
    OrionGraph* g = orion_graph_create();
    if (!g) return NULL;
    int hidden_shape[4] = {1, d, 1, s};
    int scale_shape[4] = {1, d, 1, 1};
    int hidden = orion_gb_input(g, "hidden", ORION_DTYPE_FP16, hidden_shape);
    char path[256];
    snprintf(path, sizeof(path), "@model_path/weights/layer%d/rms1.bin", layer);
    int rms_weight = orion_gb_const_weight(g, "rms1_weight", ORION_DTYPE_FP16,
                                            scale_shape, path, 64);
    int normalized = orion_gb_rmsnorm(g, hidden, rms_weight,
                                      ORION_VOXTREAM_DEPTH_RMS_EPS, "rms1", d, s);
    snprintf(path, sizeof(path), "@model_path/weights/layer%d/wq.bin", layer);
    int q = orion_gb_linear(g, normalized, "q", d, d, s, path, NULL);
    int q_rope = orion_pattern_rope_interleaved_elementwise(
        g, q, "@model_path/constants/rope_q_cos.bin",
        "@model_path/constants/rope_q_sin.bin",
        "@model_path/constants/rope_q_perm.bin",
        ORION_VOXTREAM_DEPTH_HEADS, ORION_VOXTREAM_DEPTH_HEAD_DIM, s, "q_rope");
    orion_gb_output(g, q_rope, "hidden_out");
    return g;
}

static OrionGraph* build_mlp_gate(int layer) {
    if (layer < 0 || layer >= ORION_VOXTREAM_DEPTH_LAYERS) return NULL;
    const int d = ORION_VOXTREAM_DEPTH_DIM;
    const int h = ORION_VOXTREAM_DEPTH_HIDDEN_DIM;
    const int s = ORION_VOXTREAM_DEPTH_ANE_SEQ;
    OrionGraph* g = orion_graph_create();
    if (!g) return NULL;
    int hidden_shape[4] = {1, d, 1, s};
    int scale_shape[4] = {1, d, 1, 1};
    int hidden = orion_gb_input(g, "hidden", ORION_DTYPE_FP16, hidden_shape);
    char path[256];
    snprintf(path, sizeof(path), "@model_path/weights/layer%d/rms2.bin", layer);
    int rms_weight = orion_gb_const_weight(g, "rms2_weight", ORION_DTYPE_FP16,
                                            scale_shape, path, 64);
    int normalized = orion_gb_rmsnorm(g, hidden, rms_weight,
                                      ORION_VOXTREAM_DEPTH_RMS_EPS, "rms2", d, s);
    char w1[256], w3[256];
    snprintf(w1, sizeof(w1), "@model_path/weights/layer%d/w1.bin", layer);
    snprintf(w3, sizeof(w3), "@model_path/weights/layer%d/w3.bin", layer);
    int up = orion_gb_linear(g, normalized, "w1", d, h, s, w1, NULL);
    int gate = orion_gb_linear(g, normalized, "w3", d, h, s, w3, NULL);
    int activated = orion_gb_silu(g, up, "w1_silu", h, s);
    int output = orion_gb_mul(g, activated, gate, "hidden_out");
    orion_gb_output(g, output, "hidden_out");
    return g;
}

static OrionGraph* build_mlp_two_up(int layer) {
    if (layer < 0 || layer >= ORION_VOXTREAM_DEPTH_LAYERS) return NULL;
    const int d = ORION_VOXTREAM_DEPTH_DIM;
    const int h = ORION_VOXTREAM_DEPTH_HIDDEN_DIM;
    const int s = ORION_VOXTREAM_DEPTH_ANE_SEQ;
    OrionGraph* g = orion_graph_create();
    if (!g) return NULL;
    int hidden_shape[4] = {1, d, 1, s};
    int scale_shape[4] = {1, d, 1, 1};
    int hidden = orion_gb_input(g, "hidden", ORION_DTYPE_FP16, hidden_shape);
    char path[256], w1[256], w3[256];
    snprintf(path, sizeof(path), "@model_path/weights/layer%d/rms2.bin", layer);
    int rms_weight = orion_gb_const_weight(g, "rms2_weight", ORION_DTYPE_FP16,
                                            scale_shape, path, 64);
    int normalized = orion_gb_rmsnorm(g, hidden, rms_weight,
                                      ORION_VOXTREAM_DEPTH_RMS_EPS, "rms2", d, s);
    snprintf(w1, sizeof(w1), "@model_path/weights/layer%d/w1.bin", layer);
    snprintf(w3, sizeof(w3), "@model_path/weights/layer%d/w3.bin", layer);
    int up = orion_gb_linear(g, normalized, "w1", d, h, s, w1, NULL);
    int gate = orion_gb_linear(g, normalized, "w3", d, h, s, w3, NULL);
    int output = orion_gb_add(g, up, gate, "hidden_out");
    orion_gb_output(g, output, "hidden_out");
    return g;
}

static OrionGraph* build_mlp_silu(int layer) {
    if (layer < 0 || layer >= ORION_VOXTREAM_DEPTH_LAYERS) return NULL;
    const int d = ORION_VOXTREAM_DEPTH_DIM;
    const int h = ORION_VOXTREAM_DEPTH_HIDDEN_DIM;
    const int s = ORION_VOXTREAM_DEPTH_ANE_SEQ;
    OrionGraph* g = orion_graph_create();
    if (!g) return NULL;
    int hidden_shape[4] = {1, d, 1, s};
    int scale_shape[4] = {1, d, 1, 1};
    int hidden = orion_gb_input(g, "hidden", ORION_DTYPE_FP16, hidden_shape);
    char path[256];
    snprintf(path, sizeof(path), "@model_path/weights/layer%d/rms2.bin", layer);
    int rms_weight = orion_gb_const_weight(g, "rms2_weight", ORION_DTYPE_FP16,
                                            scale_shape, path, 64);
    int normalized = orion_gb_rmsnorm(g, hidden, rms_weight,
                                      ORION_VOXTREAM_DEPTH_RMS_EPS, "rms2", d, s);
    snprintf(path, sizeof(path), "@model_path/weights/layer%d/w1.bin", layer);
    int up = orion_gb_linear(g, normalized, "w1", d, h, s, path, NULL);
    int output = orion_gb_silu(g, up, "w1_silu", h, s);
    orion_gb_output(g, output, "hidden_out");
    return g;
}

OrionGraph* orion_frontend_voxtream_depth_layer(int layer) {
    if (layer < 0 || layer >= ORION_VOXTREAM_DEPTH_LAYERS) return NULL;
    const int d = ORION_VOXTREAM_DEPTH_DIM;
    const int h = ORION_VOXTREAM_DEPTH_HIDDEN_DIM;
    const int s = ORION_VOXTREAM_DEPTH_ANE_SEQ;
    OrionGraph* g = orion_graph_create();
    if (!g) return NULL;

    int hidden_shape[4] = {1, d, 1, s};
    int scale_shape[4] = {1, d, 1, 1};
    int hidden = orion_gb_input(g, "hidden", ORION_DTYPE_FP16, hidden_shape);
    char path[256];

    snprintf(path, sizeof(path), "@model_path/weights/layer%d/rms1.bin", layer);
    int rms1_weight = orion_gb_const_weight(g, "rms1_weight", ORION_DTYPE_FP16,
                                             scale_shape, path, 64);
    int normalized = orion_gb_rmsnorm(g, hidden, rms1_weight,
                                      ORION_VOXTREAM_DEPTH_RMS_EPS,
                                      "rms1", d, s);
    const int kv = ORION_VOXTREAM_DEPTH_KV_HEADS * ORION_VOXTREAM_DEPTH_HEAD_DIM;
    snprintf(path, sizeof(path), "@model_path/weights/layer%d/wq.bin", layer);
    int q = orion_gb_linear(g, normalized, "q", d, d, s, path, NULL);
    snprintf(path, sizeof(path), "@model_path/weights/layer%d/wk.bin", layer);
    int k = orion_gb_linear(g, normalized, "k", d, kv, s, path, NULL);
    snprintf(path, sizeof(path), "@model_path/weights/layer%d/wv.bin", layer);
    int v = orion_gb_linear(g, normalized, "v", d, kv, s, path, NULL);
    int q_rope = orion_pattern_rope_interleaved_elementwise(
        g, q, "@model_path/constants/rope_q_cos.bin",
        "@model_path/constants/rope_q_sin.bin",
        "@model_path/constants/rope_q_perm.bin",
        ORION_VOXTREAM_DEPTH_HEADS, ORION_VOXTREAM_DEPTH_HEAD_DIM, s, "q_rope");
    int k_rope = orion_pattern_rope_interleaved_elementwise(
        g, k, "@model_path/constants/rope_k_cos.bin",
        "@model_path/constants/rope_k_sin.bin",
        "@model_path/constants/rope_k_perm.bin",
        ORION_VOXTREAM_DEPTH_KV_HEADS, ORION_VOXTREAM_DEPTH_HEAD_DIM, s, "k_rope");
    int attention = orion_pattern_gqa_attention(
        g, q_rope, k_rope, v, "@model_path/constants/causal_mask.bin",
        ORION_VOXTREAM_DEPTH_HEADS, ORION_VOXTREAM_DEPTH_KV_HEADS,
        ORION_VOXTREAM_DEPTH_HEAD_DIM, s, "attention");
    snprintf(path, sizeof(path), "@model_path/weights/layer%d/wo.bin", layer);
    int projected = orion_gb_linear(g, attention, "wo", d, d, s, path, NULL);
    int attention_residual = orion_pattern_residual(g, hidden, projected,
                                                     "attention_residual");
    snprintf(path, sizeof(path), "@model_path/weights/layer%d/rms2.bin", layer);
    int rms2_weight = orion_gb_const_weight(g, "rms2_weight", ORION_DTYPE_FP16,
                                             scale_shape, path, 64);
    int mlp_input = orion_gb_rmsnorm(g, attention_residual, rms2_weight,
                                     ORION_VOXTREAM_DEPTH_RMS_EPS,
                                     "rms2", d, s);
    char w1[256], w2[256], w3[256];
    snprintf(w1, sizeof(w1), "@model_path/weights/layer%d/w1.bin", layer);
    snprintf(w2, sizeof(w2), "@model_path/weights/layer%d/w2.bin", layer);
    snprintf(w3, sizeof(w3), "@model_path/weights/layer%d/w3.bin", layer);
    int mlp = orion_pattern_swiglu_ffn(g, mlp_input, "mlp", d, h, s, w1, w3, w2);
    int output = orion_pattern_residual(g, attention_residual, mlp, "hidden_out");
    orion_gb_output(g, output, "hidden_out");
    return g;
}

OrionGraph* orion_frontend_voxtream_depth_layer_adapter(
    int layer, int bucket, const OrionModelConfig* config) {
    if (!fixed_config(bucket, config)) return NULL;
    return orion_frontend_voxtream_depth_layer(layer);
}

OrionGraph* orion_frontend_voxtream_depth_attention_adapter(
    int layer, int bucket, const OrionModelConfig* config) {
    if (!fixed_config(bucket, config)) return NULL;
    return build_attention(layer);
}

OrionGraph* orion_frontend_voxtream_depth_mlp_adapter(
    int layer, int bucket, const OrionModelConfig* config) {
    if (!fixed_config(bucket, config)) return NULL;
    return build_mlp(layer);
}

OrionGraph* orion_frontend_voxtream_depth_qproj_adapter(
    int layer, int bucket, const OrionModelConfig* config) {
    if (!fixed_config(bucket, config)) return NULL;
    return build_qproj(layer);
}

OrionGraph* orion_frontend_voxtream_depth_mlp_up_adapter(
    int layer, int bucket, const OrionModelConfig* config) {
    if (!fixed_config(bucket, config)) return NULL;
    return build_mlp_up(layer);
}

OrionGraph* orion_frontend_voxtream_depth_qrope_adapter(
    int layer, int bucket, const OrionModelConfig* config) {
    if (!fixed_config(bucket, config)) return NULL;
    return build_qrope(layer);
}

OrionGraph* orion_frontend_voxtream_depth_mlp_gate_adapter(
    int layer, int bucket, const OrionModelConfig* config) {
    if (!fixed_config(bucket, config)) return NULL;
    return build_mlp_gate(layer);
}

OrionGraph* orion_frontend_voxtream_depth_mlp_two_up_adapter(
    int layer, int bucket, const OrionModelConfig* config) {
    if (!fixed_config(bucket, config)) return NULL;
    return build_mlp_two_up(layer);
}

OrionGraph* orion_frontend_voxtream_depth_mlp_silu_adapter(
    int layer, int bucket, const OrionModelConfig* config) {
    if (!fixed_config(bucket, config)) return NULL;
    return build_mlp_silu(layer);
}

OrionGraph* orion_frontend_voxtream_depth_final_norm(void) {
    const int d = ORION_VOXTREAM_DEPTH_DIM;
    const int s = ORION_VOXTREAM_DEPTH_ANE_SEQ;
    OrionGraph* g = orion_graph_create();
    if (!g) return NULL;
    int hidden_shape[4] = {1, d, 1, s};
    int scale_shape[4] = {1, d, 1, 1};
    int hidden = orion_gb_input(g, "hidden", ORION_DTYPE_FP16, hidden_shape);
    int weight = orion_gb_const_weight(
        g, "final_norm_weight", ORION_DTYPE_FP16, scale_shape,
        "@model_path/weights/final_norm.bin", 64);
    int output = orion_gb_rmsnorm(
        g, hidden, weight, ORION_VOXTREAM_DEPTH_RMS_EPS,
        "hidden", d, s);
    orion_gb_output(g, output, "hidden_out");
    return g;
}

static OrionGraph* build_probe_identity(void) {
    const char* override = getenv("VOXTREAM_PROBE_CHANNELS");
    const int d = override ? atoi(override) : ORION_VOXTREAM_DEPTH_DIM;
    const char* seq_override = getenv("VOXTREAM_PROBE_SEQ");
    const int s = seq_override ? atoi(seq_override) : ORION_VOXTREAM_DEPTH_BUCKET;
    OrionGraph* g = orion_graph_create();
    if (!g) return NULL;
    int hidden_shape[4] = {1, d, 1, s};
    int hidden = orion_gb_input(g, "hidden", ORION_DTYPE_FP16, hidden_shape);
    int out = orion_gb_add(g, hidden, hidden, "hidden_out");
    orion_gb_output(g, out, "hidden_out");
    return g;
}

static OrionGraph* build_probe_conv(int layer) {
    if (layer < 0 || layer >= ORION_VOXTREAM_DEPTH_LAYERS) return NULL;
    // ORX-102: honor VOXTREAM_PROBE_CHANNELS so the conv (linear/conv1x1) MAC
    // routing can be probed at the channel-packed dimension (2048), not just
    // the default 1024. build_probe_identity already proved the [1,2048,1,32]
    // I/O surface compiles (commit 0c729dd); this probe tests whether the
    // compiler routes a d×d conv at that channel count — the load-bearing op
    // for the real channel-packed depth layer.
    const char* channels_override = getenv("VOXTREAM_PROBE_CHANNELS");
    const int d = channels_override ? atoi(channels_override) : ORION_VOXTREAM_DEPTH_DIM;
    if (d <= 0) return NULL;
    const char* seq_override = getenv("VOXTREAM_PROBE_SEQ");
    const int s = seq_override ? atoi(seq_override) : ORION_VOXTREAM_DEPTH_BUCKET;
    OrionGraph* g = orion_graph_create();
    if (!g) return NULL;
    int hidden_shape[4] = {1, d, 1, s};
    int hidden = orion_gb_input(g, "hidden", ORION_DTYPE_FP16, hidden_shape);
    char path[256];
    snprintf(path, sizeof(path), "@model_path/weights/layer%d/wq.bin", layer);
    int q = orion_gb_linear(g, hidden, "q", d, d, s, path, NULL);
    orion_gb_output(g, q, "hidden_out");
    return g;
}

static OrionGraph* build_probe_rms(int layer) {
    if (layer < 0 || layer >= ORION_VOXTREAM_DEPTH_LAYERS) return NULL;
    // ORX-102: honor VOXTREAM_PROBE_CHANNELS for RMSNorm-at-channel-count probing.
    const char* channels_override = getenv("VOXTREAM_PROBE_CHANNELS");
    const int d = channels_override ? atoi(channels_override) : ORION_VOXTREAM_DEPTH_DIM;
    if (d <= 0) return NULL;
    const char* seq_override = getenv("VOXTREAM_PROBE_SEQ");
    const int s = seq_override ? atoi(seq_override) : ORION_VOXTREAM_DEPTH_BUCKET;
    OrionGraph* g = orion_graph_create();
    if (!g) return NULL;
    int hidden_shape[4] = {1, d, 1, s};
    int scale_shape[4] = {1, d, 1, 1};
    int hidden = orion_gb_input(g, "hidden", ORION_DTYPE_FP16, hidden_shape);
    char path[256];
    snprintf(path, sizeof(path), "@model_path/weights/layer%d/rms1.bin", layer);
    int rms_weight = orion_gb_const_weight(g, "rms1_weight", ORION_DTYPE_FP16,
                                            scale_shape, path, 64);
    int normalized = orion_gb_rmsnorm(g, hidden, rms_weight,
                                      ORION_VOXTREAM_DEPTH_RMS_EPS, "rms1", d, s);
    orion_gb_output(g, normalized, "hidden_out");
    return g;
}

OrionGraph* orion_frontend_voxtream_depth_probe_identity_adapter(
    int layer __attribute__((unused)), int bucket __attribute__((unused)),
    const OrionModelConfig* config __attribute__((unused))) {
    return build_probe_identity();
}

OrionGraph* orion_frontend_voxtream_depth_probe_conv_adapter(
    int layer, int bucket __attribute__((unused)),
    const OrionModelConfig* config __attribute__((unused))) {
    // Probes read channel/seq from VOXTREAM_PROBE_CHANNELS/SEQ env, not from a
    // production OrionModelConfig — mirror the identity probe adapter and do
    // NOT gate on fixed_config (which would reject the NULL config the probe
    // driver passes). Only the layer range is validated, inside build_probe_conv.
    return build_probe_conv(layer);
}

OrionGraph* orion_frontend_voxtream_depth_probe_rms_adapter(
    int layer, int bucket __attribute__((unused)),
    const OrionModelConfig* config __attribute__((unused))) {
    return build_probe_rms(layer);
}

// ORX-003: full channel-packed depth-layer graph for compile-acceptance probing.
// Mirrors the production orion_frontend_voxtream_depth_layer but doubles the
// channel axis (d -> 2d = 2048) so both CFG branches run in ONE ANE eval.
// Weights are block-diagonal: [branch0_weights 0; 0 branch1_weights] so the
// two 1024-wide branches cannot mix. n_heads 8->16, n_kv_heads 2->4,
// head_dim 128 unchanged (per ORX-102 recall). This is a COMPILE-ACCEPTANCE
// probe: it uses synthetic block-diagonal weight blobs (diagonal = the layer's
// real production weight, off-diagonal = zero), not production parity. The
// probe validates that the full graph (RMSNorm -> Q/K/V -> RoPE -> GQA -> Wo ->
// MLP, all at 2048 channels) compiles and produces finite output.
static OrionGraph* build_channel_packed_depth_layer(int layer) {
    if (layer < 0 || layer >= ORION_VOXTREAM_DEPTH_LAYERS) return NULL;
    const char* channels_override = getenv("VOXTREAM_PROBE_CHANNELS");
    const int d_packed = channels_override
        ? atoi(channels_override)
        : (2 * ORION_VOXTREAM_DEPTH_DIM);  // default 2048
    if (d_packed <= 0) return NULL;
    const int d = ORION_VOXTREAM_DEPTH_DIM;       // per-branch width (1024)
    if (d_packed != 2 * d) return NULL;            // channel-pack = exactly 2 branches
    const int h = 2 * ORION_VOXTREAM_DEPTH_HIDDEN_DIM;  // MLP hidden, doubled
    const int s = ORION_VOXTREAM_DEPTH_ANE_SEQ;
    const int n_heads = 2 * ORION_VOXTREAM_DEPTH_HEADS;       // 16
    const int n_kv_heads = 2 * ORION_VOXTREAM_DEPTH_KV_HEADS; // 4
    const int head_dim = ORION_VOXTREAM_DEPTH_HEAD_DIM;       // 128
    const int kv = n_kv_heads * head_dim;                      // 512

    OrionGraph* g = orion_graph_create();
    if (!g) return NULL;

    int hidden_shape[4] = {1, d_packed, 1, s};
    int scale_shape[4] = {1, d_packed, 1, 1};
    int hidden = orion_gb_input(g, "hidden", ORION_DTYPE_FP16, hidden_shape);
    char path[256];

    // RMSNorm1 (per-channel scale, doubled).
    snprintf(path, sizeof(path), "@model_path/weights/layer%d/rms1.bin", layer);
    int rms1_weight = orion_gb_const_weight(g, "rms1_weight", ORION_DTYPE_FP16,
                                             scale_shape, path, 64);
    int normalized = orion_gb_rmsnorm(g, hidden, rms1_weight,
                                      ORION_VOXTREAM_DEPTH_RMS_EPS,
                                      "rms1", d_packed, s);

    // Q/K/V projections at the packed dimension (block-diagonal weights).
    snprintf(path, sizeof(path), "@model_path/weights/layer%d/wq.bin", layer);
    int q = orion_gb_linear(g, normalized, "q", d_packed, d_packed, s, path, NULL);
    snprintf(path, sizeof(path), "@model_path/weights/layer%d/wk.bin", layer);
    int k = orion_gb_linear(g, normalized, "k", d_packed, kv, s, path, NULL);
    snprintf(path, sizeof(path), "@model_path/weights/layer%d/wv.bin", layer);
    int v = orion_gb_linear(g, normalized, "v", d_packed, kv, s, path, NULL);

    // RoPE (duplicated tables per branch — position reset, per ORX-103 proof).
    int q_rope = orion_pattern_rope_interleaved_elementwise(
        g, q, "@model_path/constants/rope_q_cos.bin",
        "@model_path/constants/rope_q_sin.bin",
        "@model_path/constants/rope_q_perm.bin",
        n_heads, head_dim, s, "q_rope");
    int k_rope = orion_pattern_rope_interleaved_elementwise(
        g, k, "@model_path/constants/rope_k_cos.bin",
        "@model_path/constants/rope_k_sin.bin",
        "@model_path/constants/rope_k_perm.bin",
        n_kv_heads, head_dim, s, "k_rope");

    // GQA attention with the packed head counts + block-diagonal causal mask.
    int attention = orion_pattern_gqa_attention(
        g, q_rope, k_rope, v, "@model_path/constants/causal_mask.bin",
        n_heads, n_kv_heads, head_dim, s, "attention");

    snprintf(path, sizeof(path), "@model_path/weights/layer%d/wo.bin", layer);
    int projected = orion_gb_linear(g, attention, "wo", d_packed, d_packed, s, path, NULL);
    int attention_residual = orion_pattern_residual(g, hidden, projected,
                                                     "attention_residual");

    snprintf(path, sizeof(path), "@model_path/weights/layer%d/rms2.bin", layer);
    int rms2_weight = orion_gb_const_weight(g, "rms2_weight", ORION_DTYPE_FP16,
                                             scale_shape, path, 64);
    int mlp_input = orion_gb_rmsnorm(g, attention_residual, rms2_weight,
                                     ORION_VOXTREAM_DEPTH_RMS_EPS,
                                     "rms2", d_packed, s);

    char w1[256], w2[256], w3[256];
    snprintf(w1, sizeof(w1), "@model_path/weights/layer%d/w1.bin", layer);
    snprintf(w2, sizeof(w2), "@model_path/weights/layer%d/w2.bin", layer);
    snprintf(w3, sizeof(w3), "@model_path/weights/layer%d/w3.bin", layer);
    int mlp = orion_pattern_swiglu_ffn(g, mlp_input, "mlp", d_packed, h, s, w1, w3, w2);
    int output = orion_pattern_residual(g, attention_residual, mlp, "hidden_out");
    orion_gb_output(g, output, "hidden_out");
    return g;
}

OrionGraph* orion_frontend_voxtream_depth_probe_depth_layer_adapter(
    int layer, int bucket __attribute__((unused)),
    const OrionModelConfig* config __attribute__((unused))) {
    return build_channel_packed_depth_layer(layer);
}

// ORX-103: sequence-packed depth-layer graph for compile-acceptance probing.
// Packs both CFG branches along the SEQUENCE axis (branch 0 at slots [0,32),
// branch 1 at slots [32,64)) on one [1,1024,1,64] surface, so both run in ONE
// ANE eval. Unlike channel-pack (ORX-003, retired), this reuses ALL production
// weights UNCHANGED — projections are per-token and channel-complete, so they
// operate identically on slot t whether t belongs to branch 0 or branch 1.
// The ONLY things that differ from the production layer are constants that
// encode position/mask structure (per docs/orion-ane-orx-103-sequence-pack-proof.md):
//   1. A BLOCK-DIAGONAL causal mask (two 32x32 triangles on the diagonal,
//      off-diagonal 32x32 blocks solid -65504) so branches cannot attend
//      across each other.
//   2. DUPLICATED RoPE cos/sin tables ([table32, table32] along seq) so
//      branch 1's positions RESTART at 0 (not continue from 32).
// n_heads (8), n_kv_heads (2), head_dim (128), dim (1024) are all UNCHANGED.
// This is a COMPILE-ACCEPTANCE probe using synthetic weights/constants; the
// real packed constants are generated by VoXtream blobs.py when this graph
// ships on the production plan path.
static OrionGraph* build_sequence_packed_depth_layer(int layer) {
    if (layer < 0 || layer >= ORION_VOXTREAM_DEPTH_LAYERS) return NULL;
    // Geometry (a) from the proof: [1, 1024, 1, 64] (32 + 32 physical).
    // Each branch keeps its own 16-logical + 16-zero-pad. Branch 0: [0,32);
    // branch 1: [32,64). seq=64 may or may not be accepted by the runtime;
    // geometry (b) [1,1024,1,32] (16+16, no padding) is the denser fallback.
    const int d = ORION_VOXTREAM_DEPTH_DIM;
    const int h = ORION_VOXTREAM_DEPTH_HIDDEN_DIM;
    const char* seq_override = getenv("VOXTREAM_PROBE_SEQ");
    const int s = seq_override ? atoi(seq_override) : (2 * ORION_VOXTREAM_DEPTH_ANE_SEQ);
    if (s <= 0) return NULL;
    OrionGraph* g = orion_graph_create();
    if (!g) return NULL;

    int hidden_shape[4] = {1, d, 1, s};
    int scale_shape[4] = {1, d, 1, 1};
    int hidden = orion_gb_input(g, "hidden", ORION_DTYPE_FP16, hidden_shape);
    char path[256];

    // RMSNorm1 — per-channel scale, REUSED UNCHANGED (position-independent).
    snprintf(path, sizeof(path), "@model_path/weights/layer%d/rms1.bin", layer);
    int rms1_weight = orion_gb_const_weight(g, "rms1_weight", ORION_DTYPE_FP16,
                                             scale_shape, path, 64);
    int normalized = orion_gb_rmsnorm(g, hidden, rms1_weight,
                                      ORION_VOXTREAM_DEPTH_RMS_EPS,
                                      "rms1", d, s);

    // Q/K/V projections — REUSED UNCHANGED (per-token, channel-complete).
    const int kv = ORION_VOXTREAM_DEPTH_KV_HEADS * ORION_VOXTREAM_DEPTH_HEAD_DIM;
    snprintf(path, sizeof(path), "@model_path/weights/layer%d/wq.bin", layer);
    int q = orion_gb_linear(g, normalized, "q", d, d, s, path, NULL);
    snprintf(path, sizeof(path), "@model_path/weights/layer%d/wk.bin", layer);
    int k = orion_gb_linear(g, normalized, "k", d, kv, s, path, NULL);
    snprintf(path, sizeof(path), "@model_path/weights/layer%d/wv.bin", layer);
    int v = orion_gb_linear(g, normalized, "v", d, kv, s, path, NULL);

    // RoPE — DUPLICATED tables per branch (position reset). The packed cos/sin
    // constants are [table32, table32] along seq; the perm constant is
    // position-independent and REUSED UNCHANGED. n_heads/n_kv_heads UNCHANGED.
    int q_rope = orion_pattern_rope_interleaved_elementwise(
        g, q, "@model_path/constants/rope_q_cos_packed.bin",
        "@model_path/constants/rope_q_sin_packed.bin",
        "@model_path/constants/rope_q_perm.bin",
        ORION_VOXTREAM_DEPTH_HEADS, ORION_VOXTREAM_DEPTH_HEAD_DIM, s, "q_rope");
    int k_rope = orion_pattern_rope_interleaved_elementwise(
        g, k, "@model_path/constants/rope_k_cos_packed.bin",
        "@model_path/constants/rope_k_sin_packed.bin",
        "@model_path/constants/rope_k_perm.bin",
        ORION_VOXTREAM_DEPTH_KV_HEADS, ORION_VOXTREAM_DEPTH_HEAD_DIM, s, "k_rope");

    // GQA attention with the BLOCK-DIAGONAL causal mask. n_heads/n_kv_heads/
    // head_dim UNCHANGED. The mask constant is the new block-diagonal one.
    int attention = orion_pattern_gqa_attention(
        g, q_rope, k_rope, v, "@model_path/constants/causal_mask_packed.bin",
        ORION_VOXTREAM_DEPTH_HEADS, ORION_VOXTREAM_DEPTH_KV_HEADS,
        ORION_VOXTREAM_DEPTH_HEAD_DIM, s, "attention");

    // Wo projection — REUSED UNCHANGED.
    snprintf(path, sizeof(path), "@model_path/weights/layer%d/wo.bin", layer);
    int projected = orion_gb_linear(g, attention, "wo", d, d, s, path, NULL);
    int attention_residual = orion_pattern_residual(g, hidden, projected,
                                                     "attention_residual");

    // RMSNorm2 + SwiGLU MLP — ALL REUSED UNCHANGED (per-token).
    snprintf(path, sizeof(path), "@model_path/weights/layer%d/rms2.bin", layer);
    int rms2_weight = orion_gb_const_weight(g, "rms2_weight", ORION_DTYPE_FP16,
                                             scale_shape, path, 64);
    int mlp_input = orion_gb_rmsnorm(g, attention_residual, rms2_weight,
                                     ORION_VOXTREAM_DEPTH_RMS_EPS,
                                     "rms2", d, s);
    char w1[256], w2[256], w3[256];
    snprintf(w1, sizeof(w1), "@model_path/weights/layer%d/w1.bin", layer);
    snprintf(w2, sizeof(w2), "@model_path/weights/layer%d/w2.bin", layer);
    snprintf(w3, sizeof(w3), "@model_path/weights/layer%d/w3.bin", layer);
    int mlp = orion_pattern_swiglu_ffn(g, mlp_input, "mlp", d, h, s, w1, w3, w2);
    int output = orion_pattern_residual(g, attention_residual, mlp, "hidden_out");
    orion_gb_output(g, output, "hidden_out");
    return g;
}

OrionGraph* orion_frontend_voxtream_depth_probe_seq_pack_adapter(
    int layer, int bucket __attribute__((unused)),
    const OrionModelConfig* config __attribute__((unused))) {
    return build_sequence_packed_depth_layer(layer);
}

static OrionGraph* build_qperm(int layer) {
    if (layer < 0 || layer >= ORION_VOXTREAM_DEPTH_LAYERS) return NULL;
    const int d = ORION_VOXTREAM_DEPTH_DIM;
    const int s = ORION_VOXTREAM_DEPTH_ANE_SEQ;
    OrionGraph* g = orion_graph_create();
    if (!g) return NULL;
    int hidden_shape[4] = {1, d, 1, s};
    int scale_shape[4] = {1, d, 1, 1};
    int hidden = orion_gb_input(g, "hidden", ORION_DTYPE_FP16, hidden_shape);
    char path[256];
    snprintf(path, sizeof(path), "@model_path/weights/layer%d/rms1.bin", layer);
    int rms_weight = orion_gb_const_weight(g, "rms1_weight", ORION_DTYPE_FP16,
                                            scale_shape, path, 64);
    int normalized = orion_gb_rmsnorm(g, hidden, rms_weight,
                                      ORION_VOXTREAM_DEPTH_RMS_EPS, "rms1", d, s);
    snprintf(path, sizeof(path), "@model_path/weights/layer%d/wq.bin", layer);
    int q = orion_gb_linear(g, normalized, "q", d, d, s, path, NULL);
    int perm_shape[4] = {d, d, 1, 1};
    int perm = orion_gb_const_weight(g, "q_rope_perm", ORION_DTYPE_FP16,
                                     perm_shape,
                                     "@model_path/constants/rope_q_perm.bin", 64);
    int swapped = orion_gb_conv1x1(g, q, perm, -1, "q_rope_swapped", d, s);
    orion_gb_output(g, swapped, "hidden_out");
    return g;
}

static OrionGraph* build_qmul(int layer) {
    if (layer < 0 || layer >= ORION_VOXTREAM_DEPTH_LAYERS) return NULL;
    const int d = ORION_VOXTREAM_DEPTH_DIM;
    const int s = ORION_VOXTREAM_DEPTH_ANE_SEQ;
    OrionGraph* g = orion_graph_create();
    if (!g) return NULL;
    int hidden_shape[4] = {1, d, 1, s};
    int scale_shape[4] = {1, d, 1, 1};
    int table_shape[4] = {1, d, 1, s};
    int hidden = orion_gb_input(g, "hidden", ORION_DTYPE_FP16, hidden_shape);
    char path[256];
    snprintf(path, sizeof(path), "@model_path/weights/layer%d/rms1.bin", layer);
    int rms_weight = orion_gb_const_weight(g, "rms1_weight", ORION_DTYPE_FP16,
                                            scale_shape, path, 64);
    int normalized = orion_gb_rmsnorm(g, hidden, rms_weight,
                                      ORION_VOXTREAM_DEPTH_RMS_EPS, "rms1", d, s);
    snprintf(path, sizeof(path), "@model_path/weights/layer%d/wq.bin", layer);
    int q = orion_gb_linear(g, normalized, "q", d, d, s, path, NULL);
    int cos = orion_gb_const_weight(g, "q_rope_cos", ORION_DTYPE_FP16,
                                    table_shape,
                                    "@model_path/constants/rope_q_cos.bin", 64);
    int sin = orion_gb_const_weight(g, "q_rope_sin", ORION_DTYPE_FP16,
                                    table_shape,
                                    "@model_path/constants/rope_q_sin.bin", 64);
    int cos_scaled = orion_gb_mul(g, q, cos, "q_rope_cos_scaled");
    int sin_scaled = orion_gb_mul(g, q, sin, "q_rope_sin_scaled");
    int out = orion_gb_add(g, cos_scaled, sin_scaled, "hidden_out");
    orion_gb_output(g, out, "hidden_out");
    return g;
}

OrionGraph* orion_frontend_voxtream_depth_qperm_adapter(
    int layer, int bucket, const OrionModelConfig* config) {
    if (!fixed_config(bucket, config)) return NULL;
    return build_qperm(layer);
}

OrionGraph* orion_frontend_voxtream_depth_qmul_adapter(
    int layer, int bucket, const OrionModelConfig* config) {
    if (!fixed_config(bucket, config)) return NULL;
    return build_qmul(layer);
}

static OrionGraph* build_qchain(int layer, int mode) {
    if (layer < 0 || layer >= ORION_VOXTREAM_DEPTH_LAYERS) return NULL;
    const int d = ORION_VOXTREAM_DEPTH_DIM;
    const char* seq_override = getenv("VOXTREAM_PROBE_SEQ");
    const int s = seq_override ? atoi(seq_override) : ORION_VOXTREAM_DEPTH_ANE_SEQ;
    OrionGraph* g = orion_graph_create();
    if (!g) return NULL;
    int hidden_shape[4] = {1, d, 1, s};
    int scale_shape[4] = {1, d, 1, 1};
    int hidden = orion_gb_input(g, "hidden", ORION_DTYPE_FP16, hidden_shape);
    char path[256];
    snprintf(path, sizeof(path), "@model_path/weights/layer%d/rms1.bin", layer);
    int rms_weight = orion_gb_const_weight(g, "rms1_weight", ORION_DTYPE_FP16,
                                            scale_shape, path, 64);
    int normalized = orion_gb_rmsnorm(g, hidden, rms_weight,
                                      ORION_VOXTREAM_DEPTH_RMS_EPS, "rms1", d, s);
    snprintf(path, sizeof(path), "@model_path/weights/layer%d/wq.bin", layer);
    int q = orion_gb_linear(g, normalized, "q", d, d, s, path, NULL);
    int out;
    if (mode == 0) {
        snprintf(path, sizeof(path), "@model_path/weights/layer%d/wo.bin", layer);
        out = orion_gb_linear(g, q, "wo", d, d, s, path, NULL);
    } else if (mode == 3) {
        const int kv = ORION_VOXTREAM_DEPTH_KV_HEADS * ORION_VOXTREAM_DEPTH_HEAD_DIM;
        snprintf(path, sizeof(path), "@model_path/weights/layer%d/wk.bin", layer);
        out = orion_gb_linear(g, q, "k", d, kv, s, path, NULL);
    } else if (mode == 4) {
        out = orion_gb_mul(g, q, q, "q_mul");
    } else if (mode == 5) {
        int spacer = orion_gb_add(g, q, q, "q_spacer");
        snprintf(path, sizeof(path), "@model_path/weights/layer%d/wo.bin", layer);
        out = orion_gb_linear(g, spacer, "wo", d, d, s, path, NULL);
    } else if (mode == 1) {
        out = orion_gb_add(g, q, q, "q_add");
    } else {
        int table_shape[4] = {1, d, 1, s};
        int cos = orion_gb_const_weight(g, "q_rope_cos", ORION_DTYPE_FP16,
                                        table_shape,
                                        "@model_path/constants/rope_q_cos.bin", 64);
        out = orion_gb_mul(g, q, cos, "q_cos_mul");
    }
    orion_gb_output(g, out, "hidden_out");
    return g;
}

OrionGraph* orion_frontend_voxtream_depth_qchain_conv_adapter(
    int layer, int bucket, const OrionModelConfig* config) {
    if (!fixed_config(bucket, config)) return NULL;
    return build_qchain(layer, 0);
}

OrionGraph* orion_frontend_voxtream_depth_qchain_add_adapter(
    int layer, int bucket, const OrionModelConfig* config) {
    if (!fixed_config(bucket, config)) return NULL;
    return build_qchain(layer, 1);
}

OrionGraph* orion_frontend_voxtream_depth_qchain_mul_adapter(
    int layer, int bucket, const OrionModelConfig* config) {
    if (!fixed_config(bucket, config)) return NULL;
    return build_qchain(layer, 2);
}

OrionGraph* orion_frontend_voxtream_depth_qchain_kv_adapter(
    int layer, int bucket, const OrionModelConfig* config) {
    if (!fixed_config(bucket, config)) return NULL;
    return build_qchain(layer, 3);
}

OrionGraph* orion_frontend_voxtream_depth_qchain_qmul_adapter(
    int layer, int bucket, const OrionModelConfig* config) {
    if (!fixed_config(bucket, config)) return NULL;
    return build_qchain(layer, 4);
}

OrionGraph* orion_frontend_voxtream_depth_qchain_spacer_adapter(
    int layer, int bucket, const OrionModelConfig* config) {
    if (!fixed_config(bucket, config)) return NULL;
    return build_qchain(layer, 5);
}

static OrionGraph* build_three_scales(void) {
    const int d = ORION_VOXTREAM_DEPTH_DIM;
    const int s = ORION_VOXTREAM_DEPTH_ANE_SEQ;
    OrionGraph* g = orion_graph_create();
    if (!g) return NULL;
    int hidden_shape[4] = {1, d, 1, s};
    int scale_shape[4] = {1, d, 1, 1};
    int hidden = orion_gb_input(g, "hidden", ORION_DTYPE_FP16, hidden_shape);
    char name[64], path[256];
    int x = hidden;
    for (int i = 0; i < 3; i++) {
        snprintf(name, sizeof(name), "scale%d_w", i);
        snprintf(path, sizeof(path), "@model_path/weights/layer%d/rms1.bin", i);
        int w = orion_gb_const_weight(g, name, ORION_DTYPE_FP16, scale_shape,
                                      path, 64);
        snprintf(name, sizeof(name), "scale%d", i);
        x = orion_gb_mul(g, x, w, name);
    }
    orion_gb_output(g, x, "hidden_out");
    return g;
}

OrionGraph* orion_frontend_voxtream_depth_three_scales_adapter(
    int layer __attribute__((unused)), int bucket,
    const OrionModelConfig* config) {
    if (!fixed_config(bucket, config)) return NULL;
    return build_three_scales();
}
