// Fixed VoXtream depth-transformer layer frontend.
#ifndef ORION_FRONTEND_VOXTREAM_DEPTH_H
#define ORION_FRONTEND_VOXTREAM_DEPTH_H

#include "../graph.h"
#include "../model_config.h"

#define ORION_VOXTREAM_DEPTH_LAYERS 4
#define ORION_VOXTREAM_DEPTH_DIM 1024
#define ORION_VOXTREAM_DEPTH_HEADS 8
#define ORION_VOXTREAM_DEPTH_KV_HEADS 2
#define ORION_VOXTREAM_DEPTH_HEAD_DIM 128
#define ORION_VOXTREAM_DEPTH_HIDDEN_DIM 8192
#define ORION_VOXTREAM_DEPTH_BUCKET 16
// Padded execution sequence: the private ANE runtime rejects channel counts
// above 768 when the sequence axis is below 32, so graphs execute 32 token
// slots. Slots [16, 32) are zero padding whose outputs are discarded; all
// operations are per-token or causally masked, keeping tokens [0, 16) exact.
#define ORION_VOXTREAM_DEPTH_ANE_SEQ 32
#define ORION_VOXTREAM_DEPTH_RMS_EPS 1e-5f

// Complete fixed-position layer: RMSNorm, GQA projections, scaled interleaved
// RoPE, manual causal attention, attention residual, RMSNorm, SwiGLU, residual.
OrionGraph* orion_frontend_voxtream_depth_layer(int layer);

// Adapter-compatible entry point. It rejects any non-fixed config or bucket
// so production compiler callers cannot silently produce a different ABI.
OrionGraph* orion_frontend_voxtream_depth_layer_adapter(
    int layer, int bucket, const OrionModelConfig* config);

// Independently compilable portions of a layer for ANE feasibility isolation.
// The attention stage returns hidden + Wo residual; the MLP stage accepts that
// residual and returns the completed layer output.
OrionGraph* orion_frontend_voxtream_depth_attention_adapter(
    int layer, int bucket, const OrionModelConfig* config);
OrionGraph* orion_frontend_voxtream_depth_mlp_adapter(
    int layer, int bucket, const OrionModelConfig* config);

// Minimal weighted probes used to isolate private compiler acceptance.
OrionGraph* orion_frontend_voxtream_depth_qproj_adapter(
    int layer, int bucket, const OrionModelConfig* config);
OrionGraph* orion_frontend_voxtream_depth_mlp_up_adapter(
    int layer, int bucket, const OrionModelConfig* config);
OrionGraph* orion_frontend_voxtream_depth_qrope_adapter(
    int layer, int bucket, const OrionModelConfig* config);
OrionGraph* orion_frontend_voxtream_depth_mlp_gate_adapter(
    int layer, int bucket, const OrionModelConfig* config);
OrionGraph* orion_frontend_voxtream_depth_mlp_two_up_adapter(
    int layer, int bucket, const OrionModelConfig* config);
OrionGraph* orion_frontend_voxtream_depth_mlp_silu_adapter(
    int layer, int bucket, const OrionModelConfig* config);

// Final decoder RMSNorm applied after layer 3 and before the acoustic head.
OrionGraph* orion_frontend_voxtream_depth_final_norm(void);

OrionGraph* orion_frontend_voxtream_depth_probe_identity_adapter(
    int layer, int bucket, const OrionModelConfig* config);
OrionGraph* orion_frontend_voxtream_depth_probe_conv_adapter(
    int layer, int bucket, const OrionModelConfig* config);
OrionGraph* orion_frontend_voxtream_depth_probe_rms_adapter(
    int layer, int bucket, const OrionModelConfig* config);

OrionGraph* orion_frontend_voxtream_depth_qperm_adapter(
    int layer, int bucket, const OrionModelConfig* config);
OrionGraph* orion_frontend_voxtream_depth_qmul_adapter(
    int layer, int bucket, const OrionModelConfig* config);

OrionGraph* orion_frontend_voxtream_depth_qchain_conv_adapter(
    int layer, int bucket, const OrionModelConfig* config);
OrionGraph* orion_frontend_voxtream_depth_qchain_add_adapter(
    int layer, int bucket, const OrionModelConfig* config);
OrionGraph* orion_frontend_voxtream_depth_qchain_mul_adapter(
    int layer, int bucket, const OrionModelConfig* config);

OrionGraph* orion_frontend_voxtream_depth_qchain_kv_adapter(
    int layer, int bucket, const OrionModelConfig* config);

OrionGraph* orion_frontend_voxtream_depth_qchain_qmul_adapter(
    int layer, int bucket, const OrionModelConfig* config);
OrionGraph* orion_frontend_voxtream_depth_qchain_spacer_adapter(
    int layer, int bucket, const OrionModelConfig* config);

OrionGraph* orion_frontend_voxtream_depth_three_scales_adapter(
    int layer, int bucket, const OrionModelConfig* config);

#endif // ORION_FRONTEND_VOXTREAM_DEPTH_H
