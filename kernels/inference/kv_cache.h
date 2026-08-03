#ifndef ORION_KV_CACHE_H
#define ORION_KV_CACHE_H

#import "../../core/ane_runtime.h"

/// KV cache for autoregressive decode.
/// Stores K and V projections from prefill and appends new K,V per decode step.

typedef struct {
    int n_layer;
    int n_head;
    int head_dim;
    int max_seq;
    int current_len;   // Number of tokens currently cached
    float* k_cache;    // [n_layer, n_head, max_seq, head_dim]
    float* v_cache;    // [n_layer, n_head, max_seq, head_dim]
} OrionKVCache;

/// Allocate KV cache for the given model config.
OrionKVCache* orion_kv_cache_create(const OrionModelConfig* cfg);

/// Store K,V from prefill (all positions at once).
/// Returns true on success. Returns false (without writing) if seq_len is out
/// of range [0, max_seq] or `layer` is out of range [0, n_layer) — callers must
/// propagate this as a hard failure rather than advancing current_len, so that
/// an oversized prompt fails loudly instead of leaving a zeroed/over-length cache.
bool orion_kv_cache_store_prefill(OrionKVCache* cache, int layer,
                                   const float* k, const float* v, int seq_len);

/// Append K,V for a single new decode position (writes at current_len).
/// Returns true on success. Returns false (without writing) if the cache is
/// already full (current_len >= max_seq) or `layer` is out of range
/// [0, n_layer). Callers must check the return and stop decoding on failure;
/// continuing would read uninitialized/over-length cache rows.
bool orion_kv_cache_append(OrionKVCache* cache, int layer,
                            const float* k, const float* v);

/// Free KV cache.
void orion_kv_cache_free(OrionKVCache* cache);

#endif // ORION_KV_CACHE_H
