#ifndef ORION_STORIES110M_H
#define ORION_STORIES110M_H

#import "../../core/ane_runtime.h"

// Stories110M model configuration
// Matches ANEgpt stories_config.h
// Architecture: 12-layer transformer with RMSNorm and SwiGLU.
//
// IMPORTANT — positional encoding contract:
//   The active training frontend (compiler/frontends/stories_train.c) computes
//   attention as raw Q@K^T followed by a causal mask. It applies NO positional
//   encoding: there is no RoPE, learned position embedding, or position input.
//   The trained model is therefore POSITIONLESS — token order is visible to the
//   model only through the causal mask, not through any positional operation.
//   This is an intentional documented contract, not a missing feature. Do not
//   claim Llama2-style RoPE equivalence for this model until a verified RoPE
//   forward/backward path is implemented and independently tested. See
//   backlog task ORION-0070.
// Total params: ~109.53M (84.95M transformer + 24.58M embedding)

static const OrionModelConfig kStories110M = {
    .n_layer    = 12,
    .n_head     = 12,
    .d_model    = 768,
    .head_dim   = 64,    // 768 / 12
    .hidden_dim = 2048,  // SwiGLU hidden size
    .vocab      = 32000, // Llama2 BPE vocabulary
    .max_seq    = 256,
};

// Training constants (from ANEgpt)
#define STORIES_ACCUM_STEPS    10   // Gradient accumulation steps
#define STORIES_MAX_COMPILES   100  // ANE compile budget per process
#define STORIES_KERNELS_PER_LAYER 5
#define STORIES_TOTAL_WEIGHT_KERNELS (STORIES_KERNELS_PER_LAYER * 12) // = 60

// Checkpoint magic
#define ORION_CKPT_MAGIC   0x424C5A54  // "BLZT"
#define ORION_CKPT_VERSION 2

#endif // ORION_STORIES110M_H
