// Decomposed VoXtream depth-transformer layer frontends.
//
// The M5 Pro private compiler accepts at most one unique large BLOBFILE array
// per program (tiny scale constants are exempt, byte-identical large arrays
// are deduplicated). A complete layer is therefore executed as a chain of
// small programs, each holding at most one large weight. This file defines
// the stage graphs for a numerically equivalent fp16 execution with explicit
// host-visible intermediate boundaries.
#ifndef ORION_FRONTEND_VOXTREAM_DEPTH_CHAIN_H
#define ORION_FRONTEND_VOXTREAM_DEPTH_CHAIN_H

#include "../graph.h"
#include "../model_config.h"

// Stage identifiers in execution order.
typedef enum {
    ORION_VDX_WQ = 0,       // rms1 + wq             -> q
    ORION_VDX_WK,           // rms1 + wk             -> k
    ORION_VDX_WV,           // rms1 + wv             -> v
    ORION_VDX_QSWAP,        // conv1x1(rope_q_perm)  -> q_swapped
    ORION_VDX_QCOS,         // q * rope_q_cos        -> q_cos
    ORION_VDX_QSIN_ADD,     // q_cos + q_swapped*sin -> q_rope
    ORION_VDX_KSWAP,        // conv1x1(rope_k_perm)  -> k_swapped
    ORION_VDX_KCOS,         // k * rope_k_cos        -> k_cos
    ORION_VDX_KSIN_ADD,     // k_cos + k_swapped*sin -> k_rope
    ORION_VDX_ATTN_CORE,    // GQA scores/mask/softmax/context -> attn_ctx
    ORION_VDX_WO_RES,       // wo(attn_ctx) + hidden -> attn_res
    ORION_VDX_W1,           // rms2 + w1             -> mlp_up
    ORION_VDX_W3,           // rms2 + w3             -> mlp_gate_in
    ORION_VDX_GATE,         // silu(mlp_up) * mlp_gate_in -> gated
    ORION_VDX_W2_RES,       // w2(gated) + attn_res  -> hidden_out
    ORION_VDX_STAGE_COUNT
} OrionVoxtreamDepthStage;

// Direct builder: graph for one stage of one transformer layer.
OrionGraph* orion_frontend_voxtream_depth_chain_stage_graph(int layer,
                                                            int stage);

// Number of programs in the decomposed chain.
int orion_voxtream_depth_chain_stage_count(void);

// Stable stage name for artifacts and logging (e.g. "wq", "attn_core").
const char* orion_voxtream_depth_chain_stage_name(int stage);

// Adapter-compatible frontend for a fixed stage. The returned function
// validates the fixed VoXtream depth config and builds `stage`.
OrionGraph* orion_frontend_voxtream_depth_chain_adapter(
    int layer, int bucket, const OrionModelConfig* config);

// Select the stage built by orion_frontend_voxtream_depth_chain_adapter.
// Selection is thread-local because OrionFrontendFn has no context argument.
// Returns false for an invalid stage.
bool orion_voxtream_depth_chain_set_stage(int stage);

#endif // ORION_FRONTEND_VOXTREAM_DEPTH_CHAIN_H
