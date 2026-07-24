// compiler/patterns.h — T119: Composite graph patterns
// Higher-level graph building patterns used by frontends.

#ifndef ORION_PATTERNS_H
#define ORION_PATTERNS_H

#include "builder.h"

// Attention pattern: decomposed SDPA (Q@K^T -> mask -> softmax -> @V)
// Returns output node index. mask_path is BLOBFILE path for causal mask.
int orion_pattern_attention(OrionGraph* g, int q, int k, int v,
                             const char* mask_path,
                             int n_heads, int head_dim, int seq,
                             const char* prefix);

// Fixed-position interleaved RoPE. The BLOBFILE contains one 2x2 rotation
// matrix per head pair and sequence position.
int orion_pattern_rope_interleaved(OrionGraph* g, int input,
                                   const char* rotations_path,
                                   int n_heads, int head_dim, int seq,
                                   const char* prefix);

// Grouped-query attention without K/V concatenation or physical repetition.
// Query heads are grouped as [kv_heads, queries_per_kv, S, head_dim], while
// K/V use [kv_heads, 1, S, head_dim] and broadcast across each query group.
int orion_pattern_gqa_attention(OrionGraph* g, int q, int k, int v,
                                const char* mask_path,
                                int n_heads, int n_kv_heads,
                                int head_dim, int seq,
                                const char* prefix);

// FFN pattern: linear -> activation -> linear
// activation: 0=GELU, 1=SiLU
// Returns output node index.
int orion_pattern_ffn(OrionGraph* g, int input, const char* prefix,
                      int in_dim, int hidden_dim, int seq,
                      const char* w1_path, const char* b1_path,
                      const char* w2_path, const char* b2_path,
                      int activation);

// SwiGLU FFN pattern: W1->SiLU * W3 -> W2
int orion_pattern_swiglu_ffn(OrionGraph* g, int input, const char* prefix,
                             int in_dim, int hidden_dim, int seq,
                             const char* w1_path, const char* w3_path,
                             const char* w2_path);

// Residual: out = x + sublayer_output
int orion_pattern_residual(OrionGraph* g, int x, int sublayer_output,
                           const char* name);

// Cast fp32->fp16 (for input) and fp16->fp32 (for output)
int orion_pattern_cast_to_fp16(OrionGraph* g, int input, const char* name, int dim, int seq);
int orion_pattern_cast_to_fp32(OrionGraph* g, int input, const char* name, int dim, int seq);

int orion_pattern_rope_interleaved_elementwise(
    OrionGraph* g, int input, const char* cos_path, const char* sin_path,
    const char* perm_path, int n_heads, int head_dim, int seq,
    const char* prefix);

#endif // ORION_PATTERNS_H
