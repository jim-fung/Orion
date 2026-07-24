// compiler/frontends/stories_train.h — T136: Stories110M training frontends
#ifndef ORION_FRONTEND_STORIES_TRAIN_H
#define ORION_FRONTEND_STORIES_TRAIN_H

#include "../graph.h"
#include "../model_config.h"

// Forward attention stage 1: x -> RMSNorm -> QKV.
// Outputs: q_out, k_out, v_out, rms1_out (all fp16)
OrionGraph* orion_frontend_fwd_attn(int layer, const OrionModelConfig* cfg);

// Forward attention stage 2: [q,k,v] -> SDPA -> Wo.
// Outputs: attn_out, wo_out (all fp16)
OrionGraph* orion_frontend_fwd_attn_core(int layer, const OrionModelConfig* cfg);

// Forward FFN with taps: x -> RMSNorm -> SwiGLU -> W2
// Outputs: w2_out, w1_out, w3_out, gate, rms2_out (all fp16)
OrionGraph* orion_frontend_fwd_ffn(int layer, const OrionModelConfig* cfg);

// FFN backward: [dffn, h1, h3] -> backprop through SwiGLU
// Outputs: dx, dh1, dh3 (all fp16)
OrionGraph* orion_frontend_ffn_bwd(int layer, const OrionModelConfig* cfg);

// SDPA backward part 1: [qf, kf, vf, dx2f] -> backprop through Wo + recompute attn
// Outputs: dvf, pf, dpf (all fp16)
OrionGraph* orion_frontend_sdpa_bwd1(int layer, const OrionModelConfig* cfg);

// SDPA backward part 2: [pf, dpf, qf, kf] -> softmax backward -> dQ, dK
// Outputs: dqf, dkf (all fp16)
OrionGraph* orion_frontend_sdpa_bwd2(int layer, const OrionModelConfig* cfg);

// QKV backward: [dq, dk, dv] -> backprop through Wq, Wk, Wv -> dx
// Output: dx (fp16)
OrionGraph* orion_frontend_qkv_bwd(int layer, const OrionModelConfig* cfg);

#endif // ORION_FRONTEND_STORIES_TRAIN_H
