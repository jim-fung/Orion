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
#define ORION_VOXTREAM_DEPTH_RMS_EPS 1e-5f

// Complete fixed-position layer: RMSNorm, GQA projections, scaled interleaved
// RoPE, manual causal attention, attention residual, RMSNorm, SwiGLU, residual.
OrionGraph* orion_frontend_voxtream_depth_layer(int layer);

// Adapter-compatible entry point. It rejects any non-fixed config or bucket
// so production compiler callers cannot silently produce a different ABI.
OrionGraph* orion_frontend_voxtream_depth_layer_adapter(
    int layer, int bucket, const OrionModelConfig* config);

// Final decoder RMSNorm applied after layer 3 and before the acoustic head.
OrionGraph* orion_frontend_voxtream_depth_final_norm(void);

#endif // ORION_FRONTEND_VOXTREAM_DEPTH_H
