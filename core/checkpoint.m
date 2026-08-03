#import "checkpoint.h"
#import "../model/configs/stories110m.h"
#import <stdio.h>
#import <string.h>
#import <stdlib.h>
#import <unistd.h>
#import <libgen.h>

// T075: Checkpoint save
// T076: Checkpoint resume
//
// Binary format:
//   OrionCkptHdr (fixed size)
//   Per-layer weights (fp32): rms_att, wq, wk, wv, wo, rms_ffn, w1, w3, w2
//   Per-layer Adam m: same order as weights
//   Per-layer Adam v: same order as weights
//   Embedding (fp32): [vocab * dim]
//   rms_final (fp32): [dim]
//   Adam embed m,v: [vocab * dim] each
//   Adam rms_final m,v: [dim] each

#pragma mark - Helpers

static bool write_buf(FILE *f, const void *buf, size_t bytes) {
    return fwrite(buf, 1, bytes, f) == bytes;
}

static bool read_buf(FILE *f, void *buf, size_t bytes) {
    return fread(buf, 1, bytes, f) == bytes;
}

// Write all 9 weight arrays for a layer
static bool write_layer_weights(FILE *f, const OrionLayerWeights *w,
                                 int d, int h) {
    return write_buf(f, w->rms_att, d * sizeof(float)) &&
           write_buf(f, w->wq, d*d * sizeof(float)) &&
           write_buf(f, w->wk, d*d * sizeof(float)) &&
           write_buf(f, w->wv, d*d * sizeof(float)) &&
           write_buf(f, w->wo, d*d * sizeof(float)) &&
           write_buf(f, w->rms_ffn, d * sizeof(float)) &&
           write_buf(f, w->w1, h*d * sizeof(float)) &&
           write_buf(f, w->w3, h*d * sizeof(float)) &&
           write_buf(f, w->w2, d*h * sizeof(float));
}

static bool read_layer_weights(FILE *f, OrionLayerWeights *w,
                                int d, int h) {
    return read_buf(f, w->rms_att, d * sizeof(float)) &&
           read_buf(f, w->wq, d*d * sizeof(float)) &&
           read_buf(f, w->wk, d*d * sizeof(float)) &&
           read_buf(f, w->wv, d*d * sizeof(float)) &&
           read_buf(f, w->wo, d*d * sizeof(float)) &&
           read_buf(f, w->rms_ffn, d * sizeof(float)) &&
           read_buf(f, w->w1, h*d * sizeof(float)) &&
           read_buf(f, w->w3, h*d * sizeof(float)) &&
           read_buf(f, w->w2, d*h * sizeof(float));
}

// Write Adam m+v for a layer (same field order as weights)
static bool write_layer_adam(FILE *f, const OrionLayerAdam *a,
                              int d, int h) {
    // m arrays
    if (!write_buf(f, a->m_rms_att, d * sizeof(float))) return false;
    if (!write_buf(f, a->m_wq, d*d * sizeof(float))) return false;
    if (!write_buf(f, a->m_wk, d*d * sizeof(float))) return false;
    if (!write_buf(f, a->m_wv, d*d * sizeof(float))) return false;
    if (!write_buf(f, a->m_wo, d*d * sizeof(float))) return false;
    if (!write_buf(f, a->m_rms_ffn, d * sizeof(float))) return false;
    if (!write_buf(f, a->m_w1, h*d * sizeof(float))) return false;
    if (!write_buf(f, a->m_w3, h*d * sizeof(float))) return false;
    if (!write_buf(f, a->m_w2, d*h * sizeof(float))) return false;
    // v arrays
    if (!write_buf(f, a->v_rms_att, d * sizeof(float))) return false;
    if (!write_buf(f, a->v_wq, d*d * sizeof(float))) return false;
    if (!write_buf(f, a->v_wk, d*d * sizeof(float))) return false;
    if (!write_buf(f, a->v_wv, d*d * sizeof(float))) return false;
    if (!write_buf(f, a->v_wo, d*d * sizeof(float))) return false;
    if (!write_buf(f, a->v_rms_ffn, d * sizeof(float))) return false;
    if (!write_buf(f, a->v_w1, h*d * sizeof(float))) return false;
    if (!write_buf(f, a->v_w3, h*d * sizeof(float))) return false;
    if (!write_buf(f, a->v_w2, d*h * sizeof(float))) return false;
    return true;
}

static bool read_layer_adam(FILE *f, OrionLayerAdam *a,
                             int d, int h) {
    if (!read_buf(f, a->m_rms_att, d * sizeof(float))) return false;
    if (!read_buf(f, a->m_wq, d*d * sizeof(float))) return false;
    if (!read_buf(f, a->m_wk, d*d * sizeof(float))) return false;
    if (!read_buf(f, a->m_wv, d*d * sizeof(float))) return false;
    if (!read_buf(f, a->m_wo, d*d * sizeof(float))) return false;
    if (!read_buf(f, a->m_rms_ffn, d * sizeof(float))) return false;
    if (!read_buf(f, a->m_w1, h*d * sizeof(float))) return false;
    if (!read_buf(f, a->m_w3, h*d * sizeof(float))) return false;
    if (!read_buf(f, a->m_w2, d*h * sizeof(float))) return false;
    if (!read_buf(f, a->v_rms_att, d * sizeof(float))) return false;
    if (!read_buf(f, a->v_wq, d*d * sizeof(float))) return false;
    if (!read_buf(f, a->v_wk, d*d * sizeof(float))) return false;
    if (!read_buf(f, a->v_wv, d*d * sizeof(float))) return false;
    if (!read_buf(f, a->v_wo, d*d * sizeof(float))) return false;
    if (!read_buf(f, a->v_rms_ffn, d * sizeof(float))) return false;
    if (!read_buf(f, a->v_w1, h*d * sizeof(float))) return false;
    if (!read_buf(f, a->v_w3, h*d * sizeof(float))) return false;
    if (!read_buf(f, a->v_w2, d*h * sizeof(float))) return false;
    return true;
}

// Total bytes a checkpoint occupies for a given config (header + all sections).
// Used to validate a file is complete before mutating trainer state on load.
static size_t checkpoint_total_bytes(const OrionModelConfig *cfg, int nl) {
    int d = cfg->d_model, h = cfg->hidden_dim, v = cfg->vocab;
    size_t layer_weights = (size_t)(d + 4*d*d + d + 2*h*d + d*h) * sizeof(float);
    size_t layer_adam = 2 * layer_weights;
    size_t embed_rms = ((size_t)v * d + d) * sizeof(float);
    size_t embed_rms_adam = 2 * embed_rms;
    return sizeof(OrionCkptHdr) + (size_t)nl * (layer_weights + layer_adam) +
           embed_rms + embed_rms_adam;
}

#pragma mark - Save

bool orion_checkpoint_save(const OrionTrainer* t, const char* path,
                           int step, float loss) {
    const OrionModelConfig *cfg = t->cfg;
    int d = cfg->d_model, h = cfg->hidden_dim, v = cfg->vocab;
    int nl = t->n_layers;

    // Transactional save: write to a temp file in the same directory, fsync,
    // then atomically rename over the destination. A crash or disk-full leaves
    // the previous good checkpoint intact rather than a truncated partial file.
    char dir[1024];
    strncpy(dir, path, sizeof(dir) - 1);
    dir[sizeof(dir) - 1] = '\0';
    char *dirpart = dirname(dir);
    char tmp[1024];
    if (snprintf(tmp, sizeof(tmp), "%s/.orion-ckpt-XXXXXX",
                 dirpart && dirpart[0] ? dirpart : ".") >= (int)sizeof(tmp)) {
        fprintf(stderr, "Checkpoint: path too long for temp staging\n");
        return false;
    }
    int fd = mkstemp(tmp);
    if (fd < 0) {
        fprintf(stderr, "Checkpoint: cannot create temp file: %s\n", tmp);
        return false;
    }
    FILE *f = fdopen(fd, "wb");
    if (!f) { close(fd); unlink(tmp); return false; }

    // Write header
    OrionCkptHdr hdr = {0};
    hdr.magic = ORION_CKPT_MAGIC;
    hdr.version = ORION_CKPT_VERSION;
    hdr.step = step;
    hdr.n_layers = nl;
    hdr.vocab_size = v;
    hdr.dim = d;
    hdr.hidden_dim = h;
    hdr.n_heads = cfg->n_head;
    hdr.seq_len = cfg->max_seq;
    hdr.lr = t->lr;
    hdr.loss = loss;
    hdr.adam_t = t->adam_t;
    bool ok = write_buf(f, &hdr, sizeof(hdr));

    // Per-layer weights
    for (int L = 0; L < nl && ok; L++) {
        ok = write_layer_weights(f, &t->weights[L], d, h);
    }

    // Per-layer Adam state
    for (int L = 0; L < nl && ok; L++) {
        ok = write_layer_adam(f, &t->adam[L], d, h);
    }

    // Embedding + final RMSNorm
    if (ok) ok = write_buf(f, t->embed, (size_t)v * d * sizeof(float));
    if (ok) ok = write_buf(f, t->rms_final, (size_t)d * sizeof(float));

    // Adam state for embed + rms_final
    if (ok) ok = write_buf(f, t->m_embed, (size_t)v * d * sizeof(float));
    if (ok) ok = write_buf(f, t->v_embed, (size_t)v * d * sizeof(float));
    if (ok) ok = write_buf(f, t->m_rms_final, (size_t)d * sizeof(float));
    if (ok) ok = write_buf(f, t->v_rms_final, (size_t)d * sizeof(float));

    // Flush to OS, fsync for durability, then close.
    if (ok && fflush(f) != 0) ok = false;
    if (ok && fdatasync(fd) != 0) ok = false;
    fclose(f);

    if (!ok) {
        fprintf(stderr, "Checkpoint: write/sync failed; keeping previous file\n");
        unlink(tmp);
        return false;
    }

    // Atomically replace the destination. rename() on the same filesystem is
    // atomic: readers see either the old or the new file, never a partial one.
    if (rename(tmp, path) != 0) {
        fprintf(stderr, "Checkpoint: rename(%s -> %s) failed\n", tmp, path);
        unlink(tmp);
        return false;
    }
    return true;
}

#pragma mark - Load

bool orion_checkpoint_load(OrionTrainer* t, const char* path,
                           int* out_step, float* out_loss) {
    FILE *f = fopen(path, "rb");
    if (!f) return false;

    const OrionModelConfig *cfg = t->cfg;
    int d = cfg->d_model, h = cfg->hidden_dim, v = cfg->vocab;
    int nl = t->n_layers;

    // Validate the file is complete BEFORE mutating any trainer state. A
    // truncated/partial checkpoint (e.g. from an interrupted save) must be
    // rejected up front rather than partway through restoring weights, which
    // would leave the trainer in a mixed half-restored state.
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return false; }
    long file_size = ftell(f);
    if (file_size < 0) { fclose(f); return false; }
    size_t expected = checkpoint_total_bytes(cfg, nl);
    if ((size_t)file_size != expected) {
        NSLog(@"Checkpoint: file size %ld bytes != expected %zu bytes (truncated or corrupt)",
              file_size, expected);
        fclose(f);
        return false;
    }
    rewind(f);

    // Read and validate header
    OrionCkptHdr hdr;
    if (!read_buf(f, &hdr, sizeof(hdr))) { fclose(f); return false; }

    if (hdr.magic != ORION_CKPT_MAGIC) {
        NSLog(@"Checkpoint: bad magic 0x%08x (expected 0x%08x)", hdr.magic, ORION_CKPT_MAGIC);
        fclose(f); return false;
    }
    if (hdr.version != ORION_CKPT_VERSION) {
        NSLog(@"Checkpoint: version %d (expected %d)", hdr.version, ORION_CKPT_VERSION);
        fclose(f); return false;
    }
    if (hdr.n_layers != nl || hdr.dim != d || hdr.hidden_dim != h || hdr.vocab_size != v) {
        NSLog(@"Checkpoint: config mismatch (layers=%d/%d, dim=%d/%d, hidden=%d/%d, vocab=%d/%d)",
              hdr.n_layers, nl, hdr.dim, d, hdr.hidden_dim, h, hdr.vocab_size, v);
        fclose(f); return false;
    }

    // Restore training state from header
    t->lr = hdr.lr;
    t->adam_t = hdr.adam_t;

    if (out_step) *out_step = hdr.step;
    if (out_loss) *out_loss = hdr.loss;

    // Per-layer weights — the file is known-complete, so a failed read here is
    // a genuine I/O error, not truncation.
    for (int L = 0; L < nl; L++) {
        if (!read_layer_weights(f, &t->weights[L], d, h)) { fclose(f); return false; }
    }

    // Per-layer Adam state
    for (int L = 0; L < nl; L++) {
        if (!read_layer_adam(f, &t->adam[L], d, h)) { fclose(f); return false; }
    }

    // Embedding + final RMSNorm
    if (!read_buf(f, t->embed, (size_t)v * d * sizeof(float))) { fclose(f); return false; }
    if (!read_buf(f, t->rms_final, (size_t)d * sizeof(float))) { fclose(f); return false; }

    // Adam state for embed + rms_final
    if (!read_buf(f, t->m_embed, (size_t)v * d * sizeof(float))) { fclose(f); return false; }
    if (!read_buf(f, t->v_embed, (size_t)v * d * sizeof(float))) { fclose(f); return false; }
    if (!read_buf(f, t->m_rms_final, (size_t)d * sizeof(float))) { fclose(f); return false; }
    if (!read_buf(f, t->v_rms_final, (size_t)d * sizeof(float))) { fclose(f); return false; }

    fclose(f);
    return true;
}
