#import "checkpoint.h"
#import "../model/configs/stories110m.h"
#import <stdio.h>
#import <string.h>
#import <stdlib.h>
#import <unistd.h>
#import <libgen.h>
#import <errno.h>
#import <fcntl.h>

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
    // Reject an oversized path up front rather than silently truncating it into
    // a fixed buffer (which could stage the temp file in the wrong directory
    // and break the same-filesystem atomic-rename guarantee).
    char dir[1024];
    size_t path_len = path ? strlen(path) : 0;
    if (path_len == 0 || path_len >= sizeof(dir)) {
        fprintf(stderr, "Checkpoint: destination path too long (%zu bytes)\n", path_len);
        return false;
    }
    memcpy(dir, path, path_len + 1);  // path_len < sizeof(dir), includes NUL
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

    // Flush to OS, then force full barrier durability. fclose can surface a
    // deferred write error, so its return is part of the success predicate.
    if (ok && fflush(f) != 0) ok = false;
    // F_FULLFSYNC forces the drive to flush its volatile write cache to stable
    // storage, giving the strongest macOS power-loss guarantee (stronger than
    // fdatasync/fsync, which may only flush to the device). Fall back to
    // fdatasync if the filesystem rejects F_FULLFSYNC.
    if (ok) {
        if (fcntl(fd, F_FULLFSYNC) != 0 && fdatasync(fd) != 0) ok = false;
    }
    if (fclose(f) != 0) ok = false;

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

    // The rename is atomic but not crash-durable until the parent directory
    // entry update is flushed. fsync the directory so a reported-successful
    // checkpoint cannot revert to the previous file after power loss. Unlike
    // the file fsync, failure here is fatal to "success": the function must not
    // report success if the rename is not durable. Note that rename has already
    // committed the new pathname, so we cannot restore the old file; we report
    // the failure honestly so the caller knows durability was not guaranteed.
    const char *dir_for_sync = dirpart && dirpart[0] ? dirpart : ".";
    int dirfd = open(dir_for_sync, O_RDONLY);
    if (dirfd < 0) {
        fprintf(stderr, "Checkpoint: replaced %s but could not open parent dir '%s' "
                "for fsync (errno=%d); rename may not survive power loss\n",
                path, dir_for_sync, errno);
        return false;
    }
    if (fsync(dirfd) != 0) {
        fprintf(stderr, "Checkpoint: replaced %s but directory fsync failed (errno=%d); "
                "rename may not survive power loss\n", path, errno);
        close(dirfd);
        return false;
    }
    if (close(dirfd) != 0) {
        fprintf(stderr, "Checkpoint: replaced %s but directory close failed (errno=%d); "
                "rename may not survive power loss\n", path, errno);
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

    // Read the entire validated file into one in-memory buffer. Every section
    // is then copied out of this buffer with memcpy (which cannot fail with a
    // short read), so a mid-load I/O error cannot leave the trainer partially
    // restored: either the whole buffer loads and every copy succeeds, or no
    // live trainer field is touched.
    unsigned char* buf = (unsigned char*)malloc(expected);
    if (!buf) { fclose(f); return false; }
    size_t got = fread(buf, 1, expected, f);
    fclose(f);
    if (got != expected) {
        NSLog(@"Checkpoint: short read %zu/%zu bytes", got, expected);
        free(buf);
        return false;
    }
    size_t off = 0;

    #define LOAD_FROM_BUF(dst, bytes) do { \
        if (off + (bytes) > expected) { free(buf); return false; } \
        memcpy((dst), buf + off, (bytes)); \
        off += (bytes); \
    } while (0)

    // Read and validate header
    OrionCkptHdr hdr;
    if (off + sizeof(hdr) > expected) { free(buf); return false; }
    memcpy(&hdr, buf + off, sizeof(hdr));
    off += sizeof(hdr);

    if (hdr.magic != ORION_CKPT_MAGIC) {
        NSLog(@"Checkpoint: bad magic 0x%08x (expected 0x%08x)", hdr.magic, ORION_CKPT_MAGIC);
        free(buf); return false;
    }
    if (hdr.version != ORION_CKPT_VERSION) {
        NSLog(@"Checkpoint: version %d (expected %d)", hdr.version, ORION_CKPT_VERSION);
        free(buf); return false;
    }
    if (hdr.n_layers != nl || hdr.dim != d || hdr.hidden_dim != h || hdr.vocab_size != v ||
        hdr.n_heads != cfg->n_head || hdr.seq_len != cfg->max_seq) {
        NSLog(@"Checkpoint: config mismatch (layers=%d/%d, dim=%d/%d, hidden=%d/%d, "
              "vocab=%d/%d, heads=%d/%d, seq=%d/%d)",
              hdr.n_layers, nl, hdr.dim, d, hdr.hidden_dim, h, hdr.vocab_size, v,
              hdr.n_heads, cfg->n_head, hdr.seq_len, cfg->max_seq);
        free(buf); return false;
    }

    // Per-layer weights — copied from the fully validated buffer.
    for (int L = 0; L < nl; L++) {
        OrionLayerWeights* w = &t->weights[L];
        LOAD_FROM_BUF(w->rms_att, (size_t)d * sizeof(float));
        LOAD_FROM_BUF(w->wq,     (size_t)d * d * sizeof(float));
        LOAD_FROM_BUF(w->wk,     (size_t)d * d * sizeof(float));
        LOAD_FROM_BUF(w->wv,     (size_t)d * d * sizeof(float));
        LOAD_FROM_BUF(w->wo,     (size_t)d * d * sizeof(float));
        LOAD_FROM_BUF(w->rms_ffn,(size_t)d * sizeof(float));
        LOAD_FROM_BUF(w->w1,     (size_t)h * d * sizeof(float));
        LOAD_FROM_BUF(w->w3,     (size_t)h * d * sizeof(float));
        LOAD_FROM_BUF(w->w2,     (size_t)d * h * sizeof(float));
    }

    // Per-layer Adam state
    for (int L = 0; L < nl; L++) {
        OrionLayerAdam* a = &t->adam[L];
        LOAD_FROM_BUF(a->m_rms_att, (size_t)d * sizeof(float));
        LOAD_FROM_BUF(a->m_wq,      (size_t)d * d * sizeof(float));
        LOAD_FROM_BUF(a->m_wk,      (size_t)d * d * sizeof(float));
        LOAD_FROM_BUF(a->m_wv,      (size_t)d * d * sizeof(float));
        LOAD_FROM_BUF(a->m_wo,      (size_t)d * d * sizeof(float));
        LOAD_FROM_BUF(a->m_rms_ffn, (size_t)d * sizeof(float));
        LOAD_FROM_BUF(a->m_w1,      (size_t)h * d * sizeof(float));
        LOAD_FROM_BUF(a->m_w3,      (size_t)h * d * sizeof(float));
        LOAD_FROM_BUF(a->m_w2,      (size_t)d * h * sizeof(float));
        LOAD_FROM_BUF(a->v_rms_att, (size_t)d * sizeof(float));
        LOAD_FROM_BUF(a->v_wq,      (size_t)d * d * sizeof(float));
        LOAD_FROM_BUF(a->v_wk,      (size_t)d * d * sizeof(float));
        LOAD_FROM_BUF(a->v_wv,      (size_t)d * d * sizeof(float));
        LOAD_FROM_BUF(a->v_wo,      (size_t)d * d * sizeof(float));
        LOAD_FROM_BUF(a->v_rms_ffn, (size_t)d * sizeof(float));
        LOAD_FROM_BUF(a->v_w1,      (size_t)h * d * sizeof(float));
        LOAD_FROM_BUF(a->v_w3,      (size_t)h * d * sizeof(float));
        LOAD_FROM_BUF(a->v_w2,      (size_t)d * h * sizeof(float));
    }

    // Embedding + final RMSNorm
    LOAD_FROM_BUF(t->embed,     (size_t)v * d * sizeof(float));
    LOAD_FROM_BUF(t->rms_final, (size_t)d * sizeof(float));

    // Adam state for embed + rms_final
    LOAD_FROM_BUF(t->m_embed,     (size_t)v * d * sizeof(float));
    LOAD_FROM_BUF(t->v_embed,     (size_t)v * d * sizeof(float));
    LOAD_FROM_BUF(t->m_rms_final, (size_t)d * sizeof(float));
    LOAD_FROM_BUF(t->v_rms_final, (size_t)d * sizeof(float));

    #undef LOAD_FROM_BUF

    // All sections copied successfully: commit the scalar training state last,
    // so a failed/aborted load never leaves lr/adam_t from a different step
    // than the weight/Adam tensors that were just restored.
    t->lr = hdr.lr;
    t->adam_t = hdr.adam_t;
    if (out_step) *out_step = hdr.step;
    if (out_loss) *out_loss = hdr.loss;

    free(buf);
    return true;
}
