#import "decode_cpu.h"
#import <Accelerate/Accelerate.h>
#import <math.h>
#import <stdlib.h>
#import <string.h>

// T032-T036: CPU GPT-2 Forward Pass
// Uses Accelerate (cblas_sgemm) for matrix multiplications.
//
// Weight layout: converter transposes Conv1D weights [in, out] → blob [out, in].
// CPU matmul: Y = X @ W_original = X @ W_blob^T → use CblasTrans on W_blob.

static const float LAYERNORM_EPS = 1e-5f;

#pragma mark - T033: LayerNorm

void orion_cpu_layernorm(const float* x, const float* gamma, const float* beta,
                          int dim, float* out) {
    // mean = sum(x) / dim
    float mean = 0.0f;
    vDSP_meanv(x, 1, &mean, dim);

    // var = sum((x - mean)^2) / dim
    float neg_mean = -mean;
    float* tmp = (float*)malloc(dim * sizeof(float));
    vDSP_vsadd(x, 1, &neg_mean, tmp, 1, dim);  // tmp = x - mean
    float var = 0.0f;
    vDSP_dotpr(tmp, 1, tmp, 1, &var, dim);      // var = sum(tmp^2)
    var /= dim;

    // rstd = 1 / sqrt(var + eps)
    float rstd = 1.0f / sqrtf(var + LAYERNORM_EPS);

    // out = gamma * (x - mean) * rstd + beta
    vDSP_vsmul(tmp, 1, &rstd, tmp, 1, dim);     // tmp = (x - mean) * rstd
    vDSP_vmul(tmp, 1, gamma, 1, out, 1, dim);   // out = tmp * gamma
    vDSP_vadd(out, 1, beta, 1, out, 1, dim);    // out = out + beta

    free(tmp);
}

#pragma mark - Internal: Linear (matmul + bias)

/// Y[M, N] = X[M, K] @ W_blob[N, K]^T + bias[N]
/// W_blob is stored [N, K] (transposed from Conv1D).
static void linear(const float* x, const float* w, const float* bias,
                    int M, int K, int N, float* out) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, N, K,
                1.0f, x, K, w, K,
                0.0f, out, N);
    if (bias) {
        for (int i = 0; i < M; i++) {
            vDSP_vadd(out + i * N, 1, bias, 1, out + i * N, 1, N);
        }
    }
}

#pragma mark - Internal: GELU

/// GELU activation (tanh approximation matching GPT-2).
/// gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
static void gelu(float* x, int n) {
    const float c = sqrtf(2.0f / M_PI);
    for (int i = 0; i < n; i++) {
        float v = x[i];
        float v3 = v * v * v;
        float t = c * (v + 0.044715f * v3);
        x[i] = 0.5f * v * (1.0f + tanhf(t));
    }
}

#pragma mark - T032: Token + Positional Embedding

/// Embed tokens and add positional embedding.
/// out[seq_len, d_model] = wte[tokens[i]] + wpe[i]
static void embed(const OrionGPT2Weights* w, const int* tokens,
                   int seq_len, float* out) {
    int d = w->d_model;
    for (int i = 0; i < seq_len; i++) {
        const float* tok_emb = w->wte + tokens[i] * d;
        const float* pos_emb = w->wpe + i * d;
        vDSP_vadd(tok_emb, 1, pos_emb, 1, out + i * d, 1, d);
    }
}

#pragma mark - T034: Multi-Head Attention (Prefill)

/// Full multi-head self-attention for prefill (all positions at once).
/// x[seq, d_model] → attn_out[seq, d_model]
static void attention_prefill(const OrionGPT2LayerWeights* l,
                                const float* x, int seq_len,
                                int n_head, int d_model, float* out) {
    int head_dim = d_model / n_head;

    // Q, K, V projections: [seq, d_model] @ [d_model, d_model]^T + bias
    float* q = (float*)malloc(seq_len * d_model * sizeof(float));
    float* k = (float*)malloc(seq_len * d_model * sizeof(float));
    float* v = (float*)malloc(seq_len * d_model * sizeof(float));

    linear(x, l->wq, l->bq, seq_len, d_model, d_model, q);
    linear(x, l->wk, l->bk, seq_len, d_model, d_model, k);
    linear(x, l->wv, l->bv, seq_len, d_model, d_model, v);

    // Attention per head
    float scale = 1.0f / sqrtf((float)head_dim);
    float* attn_out = (float*)calloc(seq_len * d_model, sizeof(float));
    float* scores = (float*)malloc(seq_len * seq_len * sizeof(float));
    float* softmax_buf = (float*)malloc(seq_len * sizeof(float));

    for (int h = 0; h < n_head; h++) {
        // Extract Q_h[seq, head_dim], K_h[seq, head_dim], V_h[seq, head_dim]
        // They are interleaved in [seq, d_model] as head-contiguous blocks
        // Q[pos][h * head_dim ... (h+1) * head_dim]
        float* qh = (float*)malloc(seq_len * head_dim * sizeof(float));
        float* kh = (float*)malloc(seq_len * head_dim * sizeof(float));
        float* vh = (float*)malloc(seq_len * head_dim * sizeof(float));

        for (int s = 0; s < seq_len; s++) {
            memcpy(qh + s * head_dim, q + s * d_model + h * head_dim, head_dim * sizeof(float));
            memcpy(kh + s * head_dim, k + s * d_model + h * head_dim, head_dim * sizeof(float));
            memcpy(vh + s * head_dim, v + s * d_model + h * head_dim, head_dim * sizeof(float));
        }

        // scores[seq, seq] = Q_h @ K_h^T * scale
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    seq_len, seq_len, head_dim,
                    scale, qh, head_dim, kh, head_dim,
                    0.0f, scores, seq_len);

        // Apply causal mask + softmax
        for (int i = 0; i < seq_len; i++) {
            // Mask future positions
            for (int j = i + 1; j < seq_len; j++) {
                scores[i * seq_len + j] = -INFINITY;
            }

            // Softmax over row i
            float max_val = scores[i * seq_len];
            for (int j = 1; j <= i; j++) {
                if (scores[i * seq_len + j] > max_val)
                    max_val = scores[i * seq_len + j];
            }

            float sum = 0.0f;
            for (int j = 0; j <= i; j++) {
                scores[i * seq_len + j] = expf(scores[i * seq_len + j] - max_val);
                sum += scores[i * seq_len + j];
            }
            for (int j = 0; j <= i; j++) {
                scores[i * seq_len + j] /= sum;
            }
            // Future positions already -inf → exp → 0, but set explicitly
            for (int j = i + 1; j < seq_len; j++) {
                scores[i * seq_len + j] = 0.0f;
            }
        }

        // attn_h[seq, head_dim] = scores[seq, seq] @ V_h[seq, head_dim]
        float* attn_h = (float*)malloc(seq_len * head_dim * sizeof(float));
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    seq_len, head_dim, seq_len,
                    1.0f, scores, seq_len, vh, head_dim,
                    0.0f, attn_h, head_dim);

        // Write back to attn_out[seq, d_model]
        for (int s = 0; s < seq_len; s++) {
            memcpy(attn_out + s * d_model + h * head_dim, attn_h + s * head_dim, head_dim * sizeof(float));
        }

        free(qh); free(kh); free(vh); free(attn_h);
    }

    // Output projection: [seq, d_model] @ [d_model, d_model]^T + bias
    linear(attn_out, l->wo, l->bo, seq_len, d_model, d_model, out);

    free(q); free(k); free(v);
    free(attn_out); free(scores); free(softmax_buf);
}

#pragma mark - T035: FFN

/// FFN: Linear → GELU → Linear
/// x[seq, d_model] → out[seq, d_model]
static void ffn(const OrionGPT2LayerWeights* l,
                 const float* x, int seq_len,
                 int d_model, int d_ff, float* out) {
    // fc = x @ wfc^T + bfc  → [seq, d_ff]
    float* fc = (float*)malloc(seq_len * d_ff * sizeof(float));
    linear(x, l->wfc, l->bfc, seq_len, d_model, d_ff, fc);

    // gelu activation in-place
    gelu(fc, seq_len * d_ff);

    // proj = fc @ wproj^T + bproj → [seq, d_model]
    linear(fc, l->wproj, l->bproj, seq_len, d_ff, d_model, out);

    free(fc);
}

#pragma mark - T036: Full Forward Pass

void orion_gpt2_forward_cpu(const OrionGPT2Weights* w,
                             const int* tokens, int seq_len,
                             float* logits) {
    int vocab = w->vocab;

    // Allocate full logits and call the _all version, then copy last row
    float* all_logits = (float*)malloc(seq_len * vocab * sizeof(float));
    orion_gpt2_forward_cpu_all(w, tokens, seq_len, all_logits);

    // Copy last position logits
    memcpy(logits, all_logits + (seq_len - 1) * vocab, vocab * sizeof(float));
    free(all_logits);
}

void orion_gpt2_forward_cpu_all(const OrionGPT2Weights* w,
                                 const int* tokens, int seq_len,
                                 float* all_logits) {
    int d = w->d_model;
    int n_layer = w->n_layer;
    int n_head = 12;  // GPT-2 124M
    int d_ff = w->d_ff;
    int vocab = w->vocab;

    // x[seq, d_model] — hidden state
    float* x = (float*)malloc(seq_len * d * sizeof(float));
    float* ln_out = (float*)malloc(seq_len * d * sizeof(float));
    float* attn_out = (float*)malloc(seq_len * d * sizeof(float));
    float* ffn_out = (float*)malloc(seq_len * d * sizeof(float));

    // T032: Token + positional embedding
    embed(w, tokens, seq_len, x);

    // Transformer layers
    for (int layer = 0; layer < n_layer; layer++) {
        const OrionGPT2LayerWeights* l = &w->layers[layer];

        // Pre-attention LayerNorm
        for (int s = 0; s < seq_len; s++) {
            orion_cpu_layernorm(x + s * d, l->ln1_g, l->ln1_b, d, ln_out + s * d);
        }

        // Multi-head self-attention
        attention_prefill(l, ln_out, seq_len, n_head, d, attn_out);

        // Residual connection: x = x + attn_out
        vDSP_vadd(x, 1, attn_out, 1, x, 1, seq_len * d);

        // Pre-FFN LayerNorm
        for (int s = 0; s < seq_len; s++) {
            orion_cpu_layernorm(x + s * d, l->ln2_g, l->ln2_b, d, ln_out + s * d);
        }

        // FFN
        ffn(l, ln_out, seq_len, d, d_ff, ffn_out);

        // Residual connection: x = x + ffn_out
        vDSP_vadd(x, 1, ffn_out, 1, x, 1, seq_len * d);
    }

    // Final LayerNorm
    for (int s = 0; s < seq_len; s++) {
        orion_cpu_layernorm(x + s * d, w->ln_f_g, w->ln_f_b, d, x + s * d);
    }

    // Logits: x[seq, d_model] @ wte[vocab, d_model]^T → [seq, vocab]
    // wte is NOT transposed (stored as [vocab, d_model]), so use CblasTrans
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                seq_len, vocab, d,
                1.0f, x, d, w->wte, d,
                0.0f, all_logits, vocab);

    free(x); free(ln_out); free(attn_out); free(ffn_out);
}

#pragma mark - T037: Prefill with KV Cache

bool orion_gpt2_prefill_kv(const OrionGPT2Weights* w,
                             const int* tokens, int seq_len,
                             OrionKVCache* kv, float* logits) {
    int d = w->d_model;
    int n_layer = w->n_layer;
    int n_head = 12;
    int d_ff = w->d_ff;
    int vocab = w->vocab;

    // Fail closed before any allocation if the prompt cannot fit. An oversized
    // prompt would otherwise leave current_len advanced past a zeroed cache and
    // poison every subsequent decode step (silent corruption).
    if (seq_len < 0 || seq_len > kv->max_seq) {
        fprintf(stderr, "orion_gpt2_prefill_kv: seq_len %d exceeds max_seq %d\n",
                seq_len, kv->max_seq);
        return false;
    }

    float* x = (float*)malloc(seq_len * d * sizeof(float));
    float* ln_out = (float*)malloc(seq_len * d * sizeof(float));
    float* attn_out = (float*)malloc(seq_len * d * sizeof(float));
    float* ffn_out = (float*)malloc(seq_len * d * sizeof(float));

    embed(w, tokens, seq_len, x);

    for (int layer = 0; layer < n_layer; layer++) {
        const OrionGPT2LayerWeights* l = &w->layers[layer];

        for (int s = 0; s < seq_len; s++) {
            orion_cpu_layernorm(x + s * d, l->ln1_g, l->ln1_b, d, ln_out + s * d);
        }

        // Compute Q, K, V and store K, V in cache
        float* q = (float*)malloc(seq_len * d * sizeof(float));
        float* k = (float*)malloc(seq_len * d * sizeof(float));
        float* v = (float*)malloc(seq_len * d * sizeof(float));
        linear(ln_out, l->wq, l->bq, seq_len, d, d, q);
        linear(ln_out, l->wk, l->bk, seq_len, d, d, k);
        linear(ln_out, l->wv, l->bv, seq_len, d, d, v);

        // Store K, V into cache for this layer. If the store fails (it should
        // not, since we checked seq_len above), bail without advancing
        // current_len so the caller sees a hard failure instead of a corrupt cache.
        if (!orion_kv_cache_store_prefill(kv, layer, k, v, seq_len)) {
            free(q); free(k); free(v);
            free(x); free(ln_out); free(attn_out); free(ffn_out);
            return false;
        }

        // Run attention using the freshly computed Q, K, V (same as prefill)
        attention_prefill(l, ln_out, seq_len, n_head, d, attn_out);

        vDSP_vadd(x, 1, attn_out, 1, x, 1, seq_len * d);

        for (int s = 0; s < seq_len; s++) {
            orion_cpu_layernorm(x + s * d, l->ln2_g, l->ln2_b, d, ln_out + s * d);
        }
        ffn(l, ln_out, seq_len, d, d_ff, ffn_out);
        vDSP_vadd(x, 1, ffn_out, 1, x, 1, seq_len * d);

        free(q); free(k); free(v);
    }

    for (int s = 0; s < seq_len; s++) {
        orion_cpu_layernorm(x + s * d, w->ln_f_g, w->ln_f_b, d, x + s * d);
    }

    // Only compute logits for last position
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                1, vocab, d,
                1.0f, x + (seq_len - 1) * d, d, w->wte, d,
                0.0f, logits, vocab);

    // Only advance the cache length once the full prefill succeeded.
    kv->current_len = seq_len;

    free(x); free(ln_out); free(attn_out); free(ffn_out);
    return true;
}

#pragma mark - T039: Single-Token Decode with KV Cache

/// Single-head attention for decode: Q[1, head_dim] against cached K,V.
/// scores_buf is a caller-owned [max_seq] scratch buffer, reused across heads
/// and tokens to avoid per-head malloc/free in the inner attention loop.
static void attention_decode_head(const float* q_h, const float* k_cache_h,
                                    const float* v_cache_h, int cache_len,
                                    int head_dim, int max_seq,
                                    float* scores_buf, float* out_h) {
    (void)max_seq;  // scores_buf is already sized to max_seq by the workspace
    float scale = 1.0f / sqrtf((float)head_dim);

    // scores[cache_len] = Q_h @ K_h^T * scale
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                cache_len, head_dim,
                scale, k_cache_h, head_dim, q_h, 1,
                0.0f, scores_buf, 1);

    // Softmax (no causal mask needed — we attend to all cached positions)
    float max_val = scores_buf[0];
    for (int j = 1; j < cache_len; j++) {
        if (scores_buf[j] > max_val) max_val = scores_buf[j];
    }
    float sum = 0.0f;
    for (int j = 0; j < cache_len; j++) {
        scores_buf[j] = expf(scores_buf[j] - max_val);
        sum += scores_buf[j];
    }
    for (int j = 0; j < cache_len; j++) {
        scores_buf[j] /= sum;
    }

    // out_h[head_dim] = scores[cache_len] @ V_h[cache_len, head_dim]
    cblas_sgemv(CblasRowMajor, CblasTrans,
                cache_len, head_dim,
                1.0f, v_cache_h, head_dim, scores_buf, 1,
                0.0f, out_h, 1);
}

OrionDecodeWorkspace* orion_decode_workspace_create(const OrionGPT2Weights* w,
                                                     const OrionKVCache* kv) {
    if (!w || !kv) return NULL;
    OrionDecodeWorkspace* ws = (OrionDecodeWorkspace*)calloc(1, sizeof(*ws));
    if (!ws) return NULL;
    ws->d_model = w->d_model;
    ws->max_seq = kv->max_seq;
    size_t d_bytes = (size_t)w->d_model * sizeof(float);
    ws->x        = (float*)malloc(d_bytes);
    ws->ln_out   = (float*)malloc(d_bytes);
    ws->attn_out = (float*)malloc(d_bytes);
    ws->ffn_out  = (float*)malloc(d_bytes);
    ws->q        = (float*)malloc(d_bytes);
    ws->k_new    = (float*)malloc(d_bytes);
    ws->v_new    = (float*)malloc(d_bytes);
    ws->proj_out = (float*)malloc(d_bytes);
    ws->scores   = (float*)malloc((size_t)kv->max_seq * sizeof(float));
    if (!ws->x || !ws->ln_out || !ws->attn_out || !ws->ffn_out ||
        !ws->q || !ws->k_new || !ws->v_new || !ws->proj_out || !ws->scores) {
        orion_decode_workspace_free(ws);
        return NULL;
    }
    return ws;
}

void orion_decode_workspace_free(OrionDecodeWorkspace* ws) {
    if (!ws) return;
    free(ws->x); free(ws->ln_out); free(ws->attn_out); free(ws->ffn_out);
    free(ws->q); free(ws->k_new); free(ws->v_new); free(ws->proj_out);
    free(ws->scores);
    free(ws);
}

void orion_gpt2_decode_step(const OrionGPT2Weights* w,
                              OrionKVCache* kv,
                              int token,
                              float* logits,
                              OrionDecodeWorkspace* ws) {
    int d = w->d_model;
    int n_layer = w->n_layer;
    int n_head = kv->n_head;
    int head_dim = kv->head_dim;
    int d_ff = w->d_ff;
    int vocab = w->vocab;
    int pos = kv->current_len;

    // Fail closed when the cache is full: continuing would compute
    // cache_len = pos+1 == max_seq+1, overreading the cache by one row and
    // overflowing the fixed ws->scores buffer. Leave logits untouched and do
    // NOT advance current_len; callers must stop generating when this fires.
    if (pos < 0 || pos >= kv->max_seq) {
        fprintf(stderr, "orion_gpt2_decode_step: KV cache full "
                "(current_len=%d max_seq=%d)\n", pos, kv->max_seq);
        return;
    }

    // Use the caller's persistent workspace when provided; otherwise allocate a
    // transient one for backwards compatibility (tests/diagnostics). The
    // production decode loop should create the workspace once before the loop.
    bool transient = false;
    if (!ws) {
        ws = orion_decode_workspace_create(w, kv);
        if (!ws) return;
        transient = true;
    }

    float* x = ws->x;
    float* ln_out = ws->ln_out;
    float* attn_out_full = ws->attn_out;
    float* ffn_out = ws->ffn_out;

    // Embed: wte[token] + wpe[pos]
    for (int i = 0; i < d; i++) {
        x[i] = w->wte[token * d + i] + w->wpe[pos * d + i];
    }

    size_t layer_stride = (size_t)n_head * kv->max_seq * head_dim;

    for (int layer = 0; layer < n_layer; layer++) {
        const OrionGPT2LayerWeights* l = &w->layers[layer];

        // Pre-attention LayerNorm
        orion_cpu_layernorm(x, l->ln1_g, l->ln1_b, d, ln_out);

        // Q, K, V projections for single token — reused per layer from the
        // workspace instead of malloc/free on every layer of every token.
        linear(ln_out, l->wq, l->bq, 1, d, d, ws->q);
        linear(ln_out, l->wk, l->bk, 1, d, d, ws->k_new);
        linear(ln_out, l->wv, l->bv, 1, d, d, ws->v_new);

        // Append K, V to cache
        orion_kv_cache_append(kv, layer, ws->k_new, ws->v_new);

        // Multi-head attention using cached K, V. The score buffer is reused
        // across all heads (and all tokens) from the workspace.
        int cache_len = pos + 1;  // includes current position
        for (int h = 0; h < n_head; h++) {
            float* q_h = ws->q + h * head_dim;
            float* k_cache_h = kv->k_cache + layer * layer_stride + h * kv->max_seq * head_dim;
            float* v_cache_h = kv->v_cache + layer * layer_stride + h * kv->max_seq * head_dim;

            attention_decode_head(q_h, k_cache_h, v_cache_h, cache_len,
                                   head_dim, kv->max_seq, ws->scores,
                                   attn_out_full + h * head_dim);
        }

        // Output projection — proj_out reused from the workspace.
        linear(attn_out_full, l->wo, l->bo, 1, d, d, ws->proj_out);

        // Residual
        vDSP_vadd(x, 1, ws->proj_out, 1, x, 1, d);

        // Pre-FFN LayerNorm
        orion_cpu_layernorm(x, l->ln2_g, l->ln2_b, d, ln_out);

        // FFN (single token)
        ffn(l, ln_out, 1, d, d_ff, ffn_out);
        vDSP_vadd(x, 1, ffn_out, 1, x, 1, d);
    }

    // Final LayerNorm
    orion_cpu_layernorm(x, w->ln_f_g, w->ln_f_b, d, x);

    // Logits
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                1, vocab, d,
                1.0f, x, d, w->wte, d,
                0.0f, logits, vocab);

    kv->current_len = pos + 1;

    if (transient) orion_decode_workspace_free(ws);
}

#pragma mark - T040: Sampling

/// Simple xorshift64 RNG
static uint64_t xorshift64(uint64_t* state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

// Comparator for top-p sampling (BSD qsort_r context-then-elements order):
// sort indices by probability descending, breaking ties by index ascending.
// This matches the stable ordering the prior insertion sort produced, so
// nucleus membership and sampling are deterministic for a given RNG state.
static int cmp_prob_desc(void* ctx, const void* a, const void* b) {
    const float* probs = (const float*)ctx;
    int ia = *(const int*)a;
    int ib = *(const int*)b;
    float pa = probs[ia];
    float pb = probs[ib];
    if (pa != pb) return (pa < pb) ? 1 : -1;  // higher prob first
    return ia - ib;                             // tie: lower index first
}

int orion_sample_token(const float* logits, int vocab_size,
                       float temperature, float top_p, uint64_t* rng_state) {
    // Argmax for temp=0
    if (temperature <= 0.0f) {
        int max_idx = 0;
        float max_val = logits[0];
        for (int i = 1; i < vocab_size; i++) {
            if (logits[i] > max_val) {
                max_val = logits[i];
                max_idx = i;
            }
        }
        return max_idx;
    }

    // Temperature scaling + softmax
    float* probs = (float*)malloc(vocab_size * sizeof(float));
    float max_val = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > max_val) max_val = logits[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        probs[i] = expf((logits[i] - max_val) / temperature);
        sum += probs[i];
    }
    for (int i = 0; i < vocab_size; i++) {
        probs[i] /= sum;
    }

    // Top-p (nucleus) sampling
    // Sort indices by probability descending. The prior code insertion-sorted
    // the entire 50,257-entry vocabulary every generated token — O(vocab²)
    // worst case. qsort_r is O(n log n) and preserves the same deterministic
    // tie ordering (index ascending for equal probabilities).
    int* indices = (int*)malloc(vocab_size * sizeof(int));
    for (int i = 0; i < vocab_size; i++) indices[i] = i;
    qsort_r(indices, vocab_size, sizeof(int), probs, cmp_prob_desc);

    // Truncate to top-p cumulative probability
    float cum = 0.0f;
    int n_top = 0;
    for (int i = 0; i < vocab_size; i++) {
        cum += probs[indices[i]];
        n_top = i + 1;
        if (cum >= top_p) break;
    }

    // Renormalize
    float norm = 0.0f;
    for (int i = 0; i < n_top; i++) {
        norm += probs[indices[i]];
    }

    // Sample
    float r = (float)(xorshift64(rng_state) & 0xFFFFFFFF) / 4294967296.0f;
    float acc = 0.0f;
    int sampled = indices[0];
    for (int i = 0; i < n_top; i++) {
        acc += probs[indices[i]] / norm;
        if (r < acc) {
            sampled = indices[i];
            break;
        }
    }

    free(probs);
    free(indices);
    return sampled;
}
