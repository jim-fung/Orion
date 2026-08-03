#import "kv_cache.h"
#import <stdlib.h>
#import <string.h>
#import <stdio.h>

// T037-T038: KV Cache Management
//
// Layout: [n_layer, n_head, max_seq, head_dim]
// For layer L, head H, position S:
//   offset = L * (n_head * max_seq * head_dim) + H * (max_seq * head_dim) + S * head_dim

OrionKVCache* orion_kv_cache_create(const OrionModelConfig* cfg) {
    OrionKVCache* cache = calloc(1, sizeof(OrionKVCache));
    cache->n_layer = cfg->n_layer;
    cache->n_head = cfg->n_head;
    cache->head_dim = cfg->head_dim;
    cache->max_seq = cfg->max_seq;
    cache->current_len = 0;

    size_t size = (size_t)cfg->n_layer * cfg->n_head * cfg->max_seq * cfg->head_dim * sizeof(float);
    cache->k_cache = calloc(1, size);
    cache->v_cache = calloc(1, size);

    return cache;
}

bool orion_kv_cache_store_prefill(OrionKVCache* cache, int layer,
                                   const float* k, const float* v, int seq_len) {
    // k, v are [seq_len, n_head * head_dim] (heads interleaved)
    // Rearrange to [n_head, seq_len, head_dim] in cache
    int nh = cache->n_head;
    int hd = cache->head_dim;
    int ms = cache->max_seq;

    // Bounds check: a prompt longer than max_seq would write past the allocated
    // K/V cache, and a stray layer index would overflow in the layer dimension.
    // Fail fast (return false, no write) so callers can propagate a hard error
    // rather than advancing current_len on a zeroed/over-length cache.
    if (seq_len < 0 || seq_len > ms || layer < 0 || layer >= cache->n_layer) {
        fprintf(stderr, "orion_kv_cache_store_prefill: out of bounds "
                "(layer=%d n_layer=%d seq_len=%d max_seq=%d)\n",
                layer, cache->n_layer, seq_len, ms);
        return false;
    }

    size_t layer_stride = (size_t)nh * ms * hd;

    for (int h = 0; h < nh; h++) {
        for (int s = 0; s < seq_len; s++) {
            size_t cache_off = layer * layer_stride + h * ms * hd + s * hd;
            size_t src_off = s * (nh * hd) + h * hd;
            memcpy(cache->k_cache + cache_off, k + src_off, hd * sizeof(float));
            memcpy(cache->v_cache + cache_off, v + src_off, hd * sizeof(float));
        }
    }
    return true;
}

bool orion_kv_cache_append(OrionKVCache* cache, int layer,
                            const float* k, const float* v) {
    // k, v are [n_head * head_dim] for a single position
    // Write to position current_len in each head
    int nh = cache->n_head;
    int hd = cache->head_dim;
    int ms = cache->max_seq;
    int pos = cache->current_len;

    // Bounds check: once the cache is full, a further append would write past
    // the allocated buffer; a stray layer index would overflow in the layer
    // dimension. Fail fast (return false, no write) so callers stop decoding
    // rather than reading an over-length/uninitialized cache row.
    if (pos < 0 || pos >= ms || layer < 0 || layer >= cache->n_layer) {
        fprintf(stderr, "orion_kv_cache_append: out of bounds "
                "(layer=%d n_layer=%d pos=%d max_seq=%d)\n",
                layer, cache->n_layer, pos, ms);
        return false;
    }

    size_t layer_stride = (size_t)nh * ms * hd;

    for (int h = 0; h < nh; h++) {
        size_t cache_off = layer * layer_stride + h * ms * hd + pos * hd;
        size_t src_off = h * hd;
        memcpy(cache->k_cache + cache_off, k + src_off, hd * sizeof(float));
        memcpy(cache->v_cache + cache_off, v + src_off, hd * sizeof(float));
    }
    return true;
}

void orion_kv_cache_free(OrionKVCache* cache) {
    if (!cache) return;
    free(cache->k_cache);
    free(cache->v_cache);
    free(cache);
}
