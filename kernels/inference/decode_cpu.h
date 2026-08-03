#ifndef ORION_DECODE_CPU_H
#define ORION_DECODE_CPU_H

#import "../../core/ane_runtime.h"
#import "../../model/weight_loader.h"
#import "kv_cache.h"

/// T032-T036: CPU-based GPT-2 inference (prefill + decode).

#pragma mark - CPU Ops (T033)

/// GPT-2 LayerNorm: y = gamma * (x - mean) / sqrt(var + eps) + beta
/// @param x     Input vector [dim]
/// @param gamma Scale [dim]
/// @param beta  Bias [dim]
/// @param dim   Vector dimension
/// @param out   Output vector [dim] (may alias x)
void orion_cpu_layernorm(const float* x, const float* gamma, const float* beta,
                          int dim, float* out);

#pragma mark - Decode Workspace (T039)

/// Persistent workspace for the single-token decode hot path.
///
/// The decode loop previously allocated/free'd several buffers on every
/// generated token: per-token hidden/ln/attn/ffn vectors, per-layer Q/K/V and
/// output-projection vectors, and a per-head attention-score vector. Over a
/// long generation (n_layer × n_head × n_tokens) this is heavy allocator lock
/// traffic, zeroing, and cache churn for buffers that are always the same size.
///
/// The workspace is sized once to the model/cache and reused across every step
/// and every layer/head inside a step. Create it before the decode loop and
/// free it after.
typedef struct {
    float* x;            // [d_model]  hidden state for the single active token
    float* ln_out;       // [d_model]  layernorm output
    float* attn_out;     // [d_model]  attention output (all heads concatenated)
    float* ffn_out;      // [d_model]  FFN output
    float* q;            // [d_model]  Q projection (per-token, reused per layer)
    float* k_new;        // [d_model]  K projection (per-token, reused per layer)
    float* v_new;        // [d_model]  V projection (per-token, reused per layer)
    float* proj_out;     // [d_model]  attention output projection
    float* scores;       // [max_seq]  attention score buffer (reused per head)
    int d_model;
    int max_seq;
} OrionDecodeWorkspace;

/// Allocate a decode workspace sized to the model and KV cache. Returns NULL
/// on allocation failure.
OrionDecodeWorkspace* orion_decode_workspace_create(const OrionGPT2Weights* w,
                                                     const OrionKVCache* kv);

/// Free a decode workspace (NULL-safe).
void orion_decode_workspace_free(OrionDecodeWorkspace* ws);

#pragma mark - Prefill Forward Pass (T036)

/// Run full CPU GPT-2 forward pass (prefill mode).
/// Processes all tokens in parallel and returns logits for the last position.
/// @param w         Loaded GPT-2 weights
/// @param tokens    Input token ids [seq_len]
/// @param seq_len   Number of tokens
/// @param logits    Output logits buffer [vocab] (last position only)
void orion_gpt2_forward_cpu(const OrionGPT2Weights* w,
                             const int* tokens, int seq_len,
                             float* logits);

/// Run full CPU GPT-2 forward pass returning all logits.
/// @param w         Loaded GPT-2 weights
/// @param tokens    Input token ids [seq_len]
/// @param seq_len   Number of tokens
/// @param all_logits Output logits buffer [seq_len * vocab]
void orion_gpt2_forward_cpu_all(const OrionGPT2Weights* w,
                                 const int* tokens, int seq_len,
                                 float* all_logits);

#pragma mark - Prefill + KV Cache (T037)

/// Run prefill forward pass and populate KV cache.
/// Returns true on success. Returns false (without mutating the cache) if
/// seq_len exceeds max_seq — an oversized prompt is a hard failure rather than
/// a silent zeroed/over-length cache. On success the cache is populated and
/// current_len is advanced to seq_len.
/// @param w         Loaded GPT-2 weights
/// @param tokens    Input token ids [seq_len]
/// @param seq_len   Number of tokens
/// @param kv        KV cache (will be populated)
/// @param logits    Output logits buffer [vocab]
bool orion_gpt2_prefill_kv(const OrionGPT2Weights* w,
                             const int* tokens, int seq_len,
                             OrionKVCache* kv, float* logits);

#pragma mark - Decode Step (T039)

/// Run a single decode step on CPU using KV cache.
/// @param w         Loaded GPT-2 weights
/// @param kv        KV cache (read for past K,V; updated with new K,V)
/// @param token     Input token id for this step
/// @param logits    Output logits buffer [vocab]
/// @param ws        Persistent decode workspace (create once before the loop).
///                  If NULL, a transient workspace is allocated/freed per call
///                  (backwards-compatible but loses the reuse benefit).
void orion_gpt2_decode_step(const OrionGPT2Weights* w,
                              OrionKVCache* kv,
                              int token,
                              float* logits,
                              OrionDecodeWorkspace* ws);

#pragma mark - Sampling (T040)

/// Sample a token from logits with temperature and top-p.
int orion_sample_token(const float* logits, int vocab_size,
                       float temperature, float top_p, uint64_t* rng_state);

#endif // ORION_DECODE_CPU_H
