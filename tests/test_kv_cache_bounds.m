// tests/test_kv_cache_bounds.m — KV cache bounds regression.
//
// Regression for C2/C3: the KV-cache mutators must fail closed (return false,
// no write, current_len unchanged) when a caller would overflow the allocated
// cache. Previously they early-returned void but their callers ignored the
// skip and advanced current_len anyway, leaving a zeroed/over-length cache
// that silently corrupted decode (C2) and overflowed the fixed score buffer
// (C3). This test exercises the mutators directly with no model blobs or ANE.

#include <stdio.h>
#include <string.h>

#import "kernels/inference/kv_cache.h"

static int g_passed = 0, g_failed = 0;

#define TEST(name) do { \
    printf("  %-50s ", #name); \
    if (test_##name()) { printf("PASS\n"); g_passed++; } \
    else { printf("FAIL\n"); g_failed++; } \
} while (0)

// Returns true if every float in layer (layer)'s K/V cache region is 0.
// Inspects exactly one layer's stride (n_head * max_seq * head_dim) so the
// helper can never read past the cache allocation — the prior version took a
// caller-supplied count and could over-read into the next layer / out of bounds.
static bool layer_kv_is_zero(const OrionKVCache* kv, int layer) {
    size_t layer_stride = (size_t)kv->n_head * kv->max_seq * kv->head_dim;
    size_t base = (size_t)layer * layer_stride;
    for (size_t i = 0; i < layer_stride; i++) {
        if (kv->k_cache[base + i] != 0.0f) return false;
        if (kv->v_cache[base + i] != 0.0f) return false;
    }
    return true;
}

// A tiny config: 2 layers, 1 head, head_dim 2, max_seq 4. Each layer's K/V
// region is 1*4*2 = 8 floats; the whole cache holds 2*8 = 16 floats per buffer.
static OrionKVCache* make_tiny_cache(void) {
    OrionModelConfig cfg = { .n_layer = 2, .n_head = 1, .head_dim = 2, .max_seq = 4 };
    return orion_kv_cache_create(&cfg);
}

// store_prefill must reject seq_len > max_seq without writing or advancing len.
static bool test_store_prefill_rejects_oversized_prompt(void) {
    OrionKVCache* kv = make_tiny_cache();
    // 5 floats of K and V (heads interleaved: seq_len * n_head * head_dim).
    float k[5] = {1, 2, 3, 4, 5};
    float v[5] = {6, 7, 8, 9, 10};

    bool ok = orion_kv_cache_store_prefill(kv, /*layer*/ 0, k, v, /*seq_len*/ 5);
    if (ok) { printf("(expected false) "); goto fail; }   // must reject
    if (kv->current_len != 0) goto fail;                   // must not advance
    if (!layer_kv_is_zero(kv, 0)) goto fail;               // must not write anything
    orion_kv_cache_free(kv);
    return true;
fail:
    orion_kv_cache_free(kv);
    return false;
}

// store_prefill must reject negative seq_len.
static bool test_store_prefill_rejects_negative_seq_len(void) {
    OrionKVCache* kv = make_tiny_cache();
    float k[1] = {0}, v[1] = {0};
    bool ok = orion_kv_cache_store_prefill(kv, 0, k, v, -1);
    if (ok) goto fail;
    if (kv->current_len != 0) goto fail;
    orion_kv_cache_free(kv);
    return true;
fail:
    orion_kv_cache_free(kv);
    return false;
}

// store_prefill must reject an out-of-range layer index.
static bool test_store_prefill_rejects_bad_layer(void) {
    OrionKVCache* kv = make_tiny_cache();
    float k[2] = {1, 2}, v[2] = {3, 4};
    // layer 2 is out of range (n_layer == 2).
    bool ok = orion_kv_cache_store_prefill(kv, 2, k, v, 1);
    if (ok) goto fail;
    // Every layer's cache must still be zero.
    for (int L = 0; L < kv->n_layer; L++) {
        if (!layer_kv_is_zero(kv, L)) goto fail;
    }
    orion_kv_cache_free(kv);
    return true;
fail:
    orion_kv_cache_free(kv);
    return false;
}

// store_prefill within bounds succeeds and writes the expected bytes.
static bool test_store_prefill_succeeds_within_bounds(void) {
    OrionKVCache* kv = make_tiny_cache();
    // seq_len 4 == max_seq is the boundary; it must succeed.
    float k[8] = {1, 2, 3, 4, 5, 6, 7, 8};   // [4, n_head*head_dim=2]
    float v[8] = {9, 10, 11, 12, 13, 14, 15, 16};
    bool ok = orion_kv_cache_store_prefill(kv, 0, k, v, 4);
    if (!ok) goto fail;
    // The cache is rearranged to [n_head, seq_len, head_dim]; for n_head=1 this
    // is a straight copy of the interleaved source rows, so cache = source.
    size_t layer_stride = (size_t)kv->n_head * kv->max_seq * kv->head_dim;
    for (int i = 0; i < 8; i++) {
        if (kv->k_cache[i] != (float)(i + 1)) goto fail;
        if (kv->v_cache[i] != (float)(i + 9)) goto fail;
    }
    (void)layer_stride;
    orion_kv_cache_free(kv);
    return true;
fail:
    orion_kv_cache_free(kv);
    return false;
}

// append must reject when the cache is already full (current_len >= max_seq).
static bool test_append_rejects_when_full(void) {
    OrionKVCache* kv = make_tiny_cache();
    kv->current_len = kv->max_seq;   // simulate a full cache (prefill filled it)
    float k[2] = {1, 1}, v[2] = {2, 2};

    bool ok = orion_kv_cache_append(kv, 0, k, v);
    if (ok) { printf("(expected false) "); goto fail; }    // must reject
    if (kv->current_len != kv->max_seq) goto fail;         // must not advance past full
    if (!layer_kv_is_zero(kv, 0)) goto fail;               // must not write at the OOB slot
    orion_kv_cache_free(kv);
    return true;
fail:
    orion_kv_cache_free(kv);
    return false;
}

// append must reject a negative current_len (defensive; should never happen).
static bool test_append_rejects_negative_len(void) {
    OrionKVCache* kv = make_tiny_cache();
    kv->current_len = -1;
    float k[2] = {1, 1}, v[2] = {2, 2};
    bool ok = orion_kv_cache_append(kv, 0, k, v);
    if (ok) goto fail;
    orion_kv_cache_free(kv);
    return true;
fail:
    orion_kv_cache_free(kv);
    return false;
}

// append must reject an out-of-range layer index.
static bool test_append_rejects_bad_layer(void) {
    OrionKVCache* kv = make_tiny_cache();
    kv->current_len = 0;
    float k[2] = {1, 1}, v[2] = {2, 2};
    bool ok = orion_kv_cache_append(kv, /*layer*/ 99, k, v);
    if (ok) goto fail;
    orion_kv_cache_free(kv);
    return true;
fail:
    orion_kv_cache_free(kv);
    return false;
}

// append within bounds succeeds and writes at the current_len slot.
static bool test_append_succeeds_within_bounds(void) {
    OrionKVCache* kv = make_tiny_cache();
    kv->current_len = 0;
    float k[2] = {7, 8}, v[2] = {9, 10};
    bool ok = orion_kv_cache_append(kv, 0, k, v);
    if (!ok) goto fail;
    // Wrote at position 0, head 0: offset 0 in the cache.
    if (kv->k_cache[0] != 7.0f || kv->k_cache[1] != 8.0f) goto fail;
    if (kv->v_cache[0] != 9.0f || kv->v_cache[1] != 10.0f) goto fail;
    orion_kv_cache_free(kv);
    return true;
fail:
    orion_kv_cache_free(kv);
    return false;
}

int main(void) {
    printf("test_kv_cache_bounds:\n");
    TEST(store_prefill_rejects_oversized_prompt);
    TEST(store_prefill_rejects_negative_seq_len);
    TEST(store_prefill_rejects_bad_layer);
    TEST(store_prefill_succeeds_within_bounds);
    TEST(append_rejects_when_full);
    TEST(append_rejects_negative_len);
    TEST(append_rejects_bad_layer);
    TEST(append_succeeds_within_bounds);
    printf("\n%d/%d passed\n", g_passed, g_passed + g_failed);
    return g_failed ? 1 : 0;
}
