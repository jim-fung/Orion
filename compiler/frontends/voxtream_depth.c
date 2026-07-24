// Fixed VoXtream depth-transformer layer frontend.

#include "voxtream_depth.h"
#include "../builder.h"
#include "../patterns.h"
#include <stdio.h>

OrionGraph* orion_frontend_voxtream_depth_layer(int layer) {
    if (layer < 0 || layer >= ORION_VOXTREAM_DEPTH_LAYERS) return NULL;

    const int d = ORION_VOXTREAM_DEPTH_DIM;
    const int kv = ORION_VOXTREAM_DEPTH_KV_HEADS * ORION_VOXTREAM_DEPTH_HEAD_DIM;
    const int h = ORION_VOXTREAM_DEPTH_HIDDEN_DIM;
    const int s = ORION_VOXTREAM_DEPTH_BUCKET;
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

    int q_rope = orion_pattern_rope_interleaved(
        g, q, "@model_path/constants/rope_q.bin",
        ORION_VOXTREAM_DEPTH_HEADS, ORION_VOXTREAM_DEPTH_HEAD_DIM, s, "q_rope");
    int k_rope = orion_pattern_rope_interleaved(
        g, k, "@model_path/constants/rope_k.bin",
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
    int mlp = orion_pattern_swiglu_ffn(g, mlp_input, "mlp", d, h, s,
                                       w1, w3, w2);
    int output = orion_pattern_residual(g, attention_residual, mlp, "hidden_out");
    orion_gb_output(g, output, "hidden_out");
    return g;
}

OrionGraph* orion_frontend_voxtream_depth_layer_adapter(
    int layer, int bucket, const OrionModelConfig* config) {
    if (!config || bucket != ORION_VOXTREAM_DEPTH_BUCKET ||
        config->n_layer != ORION_VOXTREAM_DEPTH_LAYERS ||
        config->n_head != ORION_VOXTREAM_DEPTH_HEADS ||
        config->d_model != ORION_VOXTREAM_DEPTH_DIM ||
        config->head_dim != ORION_VOXTREAM_DEPTH_HEAD_DIM ||
        config->hidden_dim != ORION_VOXTREAM_DEPTH_HIDDEN_DIM ||
        config->max_seq != ORION_VOXTREAM_DEPTH_BUCKET) {
        return NULL;
    }
    return orion_frontend_voxtream_depth_layer(layer);
}

OrionGraph* orion_frontend_voxtream_depth_final_norm(void) {
    const int d = ORION_VOXTREAM_DEPTH_DIM;
    const int s = ORION_VOXTREAM_DEPTH_BUCKET;
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
