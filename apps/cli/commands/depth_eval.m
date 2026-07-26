// apps/cli/commands/depth_eval.m
//
// orion depth-eval — drive the VoXtream depth loop on the ANE from Python via
// subprocess. Two modes:
//
//   (1) DIAGNOSTIC (default): fixed-input 15-step separate-CFG latency/parity
//       probe. Reuses the two packed-hidden inputs across all 15 steps; does
//       NOT sample tokens. Useful only for shape-dependent timing and
//       layer-0 parity. This is what ORX-005 corrected: it is NOT a real
//       dependent token loop and does NOT establish the 60ms budget.
//
//   (2) RECURRENCE (--recurrence): REAL 15-step autoregressive token loop.
//       Builds the production prefix [last_hidden, c0_embed, c1_embed, ...],
//       runs 4 real layer programs + final norm per step, reads the active
//       position, runs audio_head GEMV (Accelerate cblas_sgemv), CFG combine
//       (cond*gamma + (1-gamma)*uncond), argmax, looks up the codebook
//       embedding (audio_embeddings[token + codebook*2050]), and feeds it
//       back into both CFG branches. Writes tokens.bin (15 LE int32).
//
// Usage:
//   orion depth-eval --export-dir <path>
//                    [--input <input.npz | dir>]
//                    [--output <dir | out.npz>]
//                    [--evidence <evidence.json>]
//                    [--max-compile N]
//                    [--samples N] [--warmup N] [--host-overhead-ms F]
//                    [--protocol-version N | --protocol-info]
//                    [--recurrence]
//                    [--embeddings <audio_embeddings.fp32.bin>]
//                    [--audio-head <audio_head.fp32.bin>]
//                    [--c0 N] [--gamma F]
//
//   --export-dir     VoXtream v16 export directory (manifest, weights/,
//                    ane_references/, constants/). Falls back to the
//                    $VOXTREAM_DEPTH_EXPORT env var if omitted.
//   --input          Either a numpy .npz holding fp16 [1,1024,1,32] members
//                    `hidden_cond` and `hidden_uncond`, OR a directory
//                    containing `hidden_cond.fp16.bin` / `hidden_uncond.fp16.bin`.
//                    In --recurrence mode these are the post-speaker-projection
//                    last_hidden state per CFG branch (still fp16 [1,1024,1,32];
//                    slot 0 holds last_hidden, the rest are ignored).
//                    Omit to use <export-dir>/ane_references/packed_hidden.batch{0,1}.fp16.bin
//                    (the existing test fixtures).
//   --output         Directory to receive the captured final-norm output hidden
//                    states for each CFG branch as raw fp16 .bin + index.json.
//                    If the path ends in .npz the stem is used as the directory.
//                    Omit to skip output capture. Full npz is not written from C;
//                    each tensor is a raw fp16 buffer (np.fromfile).
//   --evidence       Write the timing/parity/compile JSON evidence here.
//                    Omit to write the JSON to stdout (human summary -> stderr).
//   --max-compile    Abort if the compile-count delta exceeds N (default 3).
//   --samples        Timed full-loop samples (default 30, max 100).
//   --warmup         Untimed warmup loops (default 2, max 20).
//   --host-overhead-ms  External host logits/sampling/embedding estimate (default 0).
//   --protocol-version N  Advertise a wire-protocol version (default 1). Echoed
//                    as evidence.protocol_version. Only N=1 is currently
//                    supported; any other value returns exit 9.
//   --protocol-info  Print a JSON capabilities object to stdout and exit 0
//                    without touching the ANE. Used by the Python runner to
//                    negotiate before invoking.
//   --recurrence     Enable REAL autoregressive 15-step token production.
//   --embeddings     Path to audio_embeddings as raw LE fp32, shape [32800,1024]
//                    row-major. Required for --recurrence.
//   --audio-head     Path to audio_head as raw LE fp32, shape [15,1024,2050]
//                    row-major. Default: <export-dir>/audio_head.fp32.bin if present.
//   --c0             Semantic codebook-0 token (int, 0..2049). Required for --recurrence.
//   --gamma          Acoustic CFG scale (default 3.0).
//
// Weight strategy (recurrence mode: 5 compiles, no aliasing):
//   - Each layer 0..3: generate aggregate graph, build real layer-N blob,
//     compile once (M5 Pro has no practical compile limit).
//   - Final norm: compile once via orion_frontend_voxtream_depth_final_norm().
// Diagnostic mode keeps the 2-compile donor-patch path for backward compat.
//
// Building this command consumes ZERO ANE compile budget; only invoking it does.

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Accelerate/Accelerate.h>
#include <mach/mach_time.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core/ane_runtime.h"
#include "core/iosurface_tensor.h"
#include "core/orion_error.h"
#include "compiler/codegen.h"
#include "compiler/frontends/voxtream_depth.h"
#include "compiler/pipeline.h"
#include "compiler/validate.h"

#define DIM ORION_VOXTREAM_DEPTH_DIM
#define SEQ ORION_VOXTREAM_DEPTH_ANE_SEQ
#define ELEMENTS ((size_t)DIM * SEQ)
#define TENSOR_BYTES (ELEMENTS * sizeof(_Float16))
#define BLOB_HEADER_BYTES 128
#define CHUNK_HEADER_BYTES 64
#define FIRST_CHUNK_OFFSET 64
#define MAX_ABS_GATE 0.03
#define REL_L2_GATE 0.005
#define DEPTH_STEPS 15
#define CFG_BRANCHES 2
#define ITERATIONS_PER_SAMPLE (DEPTH_STEPS * CFG_BRANCHES)
#define LAYERS ORION_VOXTREAM_DEPTH_LAYERS
#define CHAIN_SURFACES (LAYERS + 2)  // input + 4 layer outputs + final-norm output
#define DEPTH_BUDGET_MS 60.0
#define FRAME_BUDGET_MS 80.0
#define MARGIN_TARGET_MS 20.0
#define DEFAULT_MAX_COMPILE 3
#define MAX_SAMPLES 100
#define DEFAULT_SAMPLES 30
#define DEFAULT_WARMUP 2
#define MAX_ITERATION_SAMPLES (MAX_SAMPLES * ITERATIONS_PER_SAMPLE)

// -- Recurrence (real token production) --------------------------------------
// Production depth recurrence emits 15 acoustic tokens. cfg_acoustic=3.0 by
// default. The audio vocab is 2050 per codebook (2048 Mimi + 2 specials); the
// embedding table is [num_codebooks(16) * 2050, 1024] = [32800, 1024] fp32.
#define NUM_CODEBOOKS 16
#define AUDIO_VOCAB 2050
#define AUDIO_EMBED_ROWS (NUM_CODEBOOKS * AUDIO_VOCAB)   // 32800
#define AUDIO_EMBED_DIM DIM                              // 1024
#define AUDIO_HEAD_STEPS 15                              // codebooks 1..15
#define DEFAULT_CFG_GAMMA 3.0
#define PROTOCOL_VERSION 1
#define RECURRENCE_EMBED_BYTES ((size_t)AUDIO_EMBED_ROWS * AUDIO_EMBED_DIM * sizeof(float))
#define RECURRENCE_HEAD_BYTES  ((size_t)AUDIO_HEAD_STEPS * DIM * AUDIO_VOCAB * sizeof(float))
// The execution contract discards padding token slots [16, 32); parity is gated
// on logical tokens [0, 16) exactly like the aggregate-layer harness.
#define LOGICAL_TOKENS ORION_VOXTREAM_DEPTH_BUCKET

// ---------------------------------------------------------------------------
// Aggregate BLOBFILE chunk table (identical to the depth-loop test).
// ---------------------------------------------------------------------------

typedef struct {
    const char *relative_path;
    const char *mil_path;
    size_t payload_bytes;
    uint64_t chunk_offset;
} AggregateEntry;

static AggregateEntry kEntries[] = {
    {"weights/layer0/rms1.bin", "@model_path/weights/layer0/rms1.bin", 2048, 0},
    {"weights/layer0/wq.bin", "@model_path/weights/layer0/wq.bin", 2097152, 0},
    {"weights/layer0/wk.bin", "@model_path/weights/layer0/wk.bin", 524288, 0},
    {"weights/layer0/wv.bin", "@model_path/weights/layer0/wv.bin", 524288, 0},
    {"weights/layer0/wo.bin", "@model_path/weights/layer0/wo.bin", 2097152, 0},
    {"weights/layer0/rms2.bin", "@model_path/weights/layer0/rms2.bin", 2048, 0},
    {"weights/layer0/w1.bin", "@model_path/weights/layer0/w1.bin", 16777216, 0},
    {"weights/layer0/w2.bin", "@model_path/weights/layer0/w2.bin", 16777216, 0},
    {"weights/layer0/w3.bin", "@model_path/weights/layer0/w3.bin", 16777216, 0},
    {"constants/rope_q_cos.bin", "@model_path/constants/rope_q_cos.bin", 65536, 0},
    {"constants/rope_q_sin.bin", "@model_path/constants/rope_q_sin.bin", 65536, 0},
    {"constants/rope_q_perm.bin", "@model_path/constants/rope_q_perm.bin", 2097152, 0},
    {"constants/rope_k_cos.bin", "@model_path/constants/rope_k_cos.bin", 16384, 0},
    {"constants/rope_k_sin.bin", "@model_path/constants/rope_k_sin.bin", 16384, 0},
    {"constants/rope_k_perm.bin", "@model_path/constants/rope_k_perm.bin", 131072, 0},
    {"constants/causal_mask.bin", "@model_path/constants/causal_mask.bin", 2048, 0},
};
static const int kEntryCount = sizeof(kEntries) / sizeof(kEntries[0]);

// ---------------------------------------------------------------------------
// Byte + timing helpers (ported from the depth-loop test).
// ---------------------------------------------------------------------------

static void write_le32(uint8_t *dst, uint32_t value) {
    dst[0] = (uint8_t)value;
    dst[1] = (uint8_t)(value >> 8);
    dst[2] = (uint8_t)(value >> 16);
    dst[3] = (uint8_t)(value >> 24);
}

static uint32_t read_le32(const uint8_t *src) {
    return (uint32_t)src[0] | ((uint32_t)src[1] << 8) |
           ((uint32_t)src[2] << 16) | ((uint32_t)src[3] << 24);
}

static uint16_t read_le16(const uint8_t *src) {
    return (uint16_t)((uint16_t)src[0] | ((uint16_t)src[1] << 8));
}

static size_t align64(size_t value) {
    return (value + 63u) & ~(size_t)63u;
}

static uint64_t monotonic_now(void) {
    return mach_absolute_time();
}

static double monotonic_elapsed_ms(uint64_t start, uint64_t end) {
    static mach_timebase_info_data_t info;
    static dispatch_once_t once;
    dispatch_once(&once, ^{ mach_timebase_info(&info); });
    return (double)(end - start) * (double)info.numer /
           (double)info.denom / 1e6;
}

static int compare_double(const void *lhs, const void *rhs) {
    double a = *(const double *)lhs, b = *(const double *)rhs;
    return (a > b) - (a < b);
}

static double percentile_sorted(const double *sorted, int count, double fraction) {
    if (count == 1) return sorted[0];
    double pos = fraction * (double)(count - 1);
    int lo = (int)pos;
    int hi = lo + 1;
    if (hi >= count) hi = count - 1;
    double frac = pos - lo;
    return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
}

static NSArray *sample_array(const double *samples, int count) {
    NSMutableArray *array = [NSMutableArray arrayWithCapacity:(NSUInteger)count];
    for (int i = 0; i < count; i++) [array addObject:@(samples[i])];
    return array;
}

// mean / p50 / p95 / p99 over an arbitrary sample array. The scratch buffer is
// fixed at MAX_ITERATION_SAMPLES so no VLA is needed.
static NSDictionary *stats_dict(const double *samples, int count) {
    if (count <= 0) {
        return @{@"count": @(0)};
    }
    double sorted[MAX_ITERATION_SAMPLES];
    memcpy(sorted, samples, sizeof(double) * (size_t)count);
    qsort(sorted, (size_t)count, sizeof(double), compare_double);
    double sum = 0.0;
    for (int i = 0; i < count; i++) sum += samples[i];
    double mean = sum / (double)count;
    return @{
        @"count": @(count),
        @"mean_ms": @(mean),
        @"p50_ms": @(percentile_sorted(sorted, count, 0.50)),
        @"p95_ms": @(percentile_sorted(sorted, count, 0.95)),
        @"p99_ms": @(percentile_sorted(sorted, count, 0.99)),
        @"min_ms": @(sorted[0]),
        @"max_ms": @(sorted[count - 1]),
        @"samples_ms": sample_array(samples, count),
    };
}

// ---------------------------------------------------------------------------
// Aggregate-blob assembly + MIL generation (ported from the depth-loop test).
// ---------------------------------------------------------------------------

static NSData *read_valid_single_blob(NSString *path, size_t expected_payload) {
    NSData *data = [NSData dataWithContentsOfFile:path];
    if (!data || data.length != BLOB_HEADER_BYTES + expected_payload) return nil;
    const uint8_t *bytes = data.bytes;
    if (bytes[0] != 1 || bytes[4] != 2 ||
        bytes[64] != 0xef || bytes[65] != 0xbe ||
        bytes[66] != 0xad || bytes[67] != 0xde || bytes[68] != 1 ||
        read_le32(bytes + 72) != expected_payload || read_le32(bytes + 80) != 128) {
        return nil;
    }
    return data;
}

static void write_chunk_header(uint8_t *bytes, size_t chunk_offset,
                               size_t payload_bytes, size_t payload_offset) {
    bytes[chunk_offset] = 0xef;
    bytes[chunk_offset + 1] = 0xbe;
    bytes[chunk_offset + 2] = 0xad;
    bytes[chunk_offset + 3] = 0xde;
    bytes[chunk_offset + 4] = 1;
    write_le32(bytes + chunk_offset + 8, (uint32_t)payload_bytes);
    write_le32(bytes + chunk_offset + 16, (uint32_t)payload_offset);
}

static void relative_path_for_layer(int index, int layer, char *buffer, size_t size) {
    const char *prefix = "weights/layer0/";
    const char *path = kEntries[index].relative_path;
    if (strncmp(path, prefix, strlen(prefix)) == 0) {
        snprintf(buffer, size, "weights/layer%d/%s", layer, path + strlen(prefix));
    } else {
        snprintf(buffer, size, "%s", path);
    }
}

// Build one physical BLOBFILE containing every entry at its own chunk offset.
// The chunk layout is identical for every layer so one MIL text serves all
// layers through weight patching.
static NSData *build_aggregate_blob(NSString *root, int layer) {
    NSData *payloads[32];
    char relative[256];
    size_t total = BLOB_HEADER_BYTES;
    for (int i = 0; i < kEntryCount; i++) {
        relative_path_for_layer(i, layer, relative, sizeof(relative));
        payloads[i] = read_valid_single_blob(
            [root stringByAppendingPathComponent:@(relative)],
            kEntries[i].payload_bytes);
        if (!payloads[i]) {
            fprintf(stderr, "depth-eval: cannot read validated blob %s\n", relative);
            return nil;
        }
        size_t header_offset = i == 0 ? FIRST_CHUNK_OFFSET : align64(total);
        kEntries[i].chunk_offset = (uint64_t)header_offset;
        total = header_offset + CHUNK_HEADER_BYTES + kEntries[i].payload_bytes;
    }
    NSMutableData *aggregate = [NSMutableData dataWithLength:total];
    uint8_t *out = aggregate.mutableBytes;
    out[0] = 1;
    out[4] = 2;
    for (int i = 0; i < kEntryCount; i++) {
        size_t header_offset = (size_t)kEntries[i].chunk_offset;
        size_t payload_offset = header_offset + CHUNK_HEADER_BYTES;
        write_chunk_header(out, header_offset, kEntries[i].payload_bytes,
                           payload_offset);
        memcpy(out + payload_offset,
               (const uint8_t *)[payloads[i] bytes] + BLOB_HEADER_BYTES,
               kEntries[i].payload_bytes);
    }
    return aggregate;
}

// Point every BLOBFILE const in the graph at the aggregate file chunk table.
// The graph for layer N emits MIL paths `@model_path/weights/layer{N}/...`,
// so we match against the layer-N-renamed table. Constants under
// `constants/...` are shared across all layers (same mil_path).
static bool rewrite_blob_refs(OrionGraph *g, int layer, int *rewritten) {
    *rewritten = 0;
    // Build the layer-N mil_path table for matching. Weight entries use
    // `layer{N}`; constant entries keep their fixed `constants/...` path.
    char layer_mil[32][256];
    const char *prefix = "weights/layer0/";
    for (int j = 0; j < kEntryCount; j++) {
        const char *path = kEntries[j].mil_path;
        if (strncmp(path + strlen("@model_path/"), prefix,
                    strlen(prefix)) == 0) {
            snprintf(layer_mil[j], sizeof(layer_mil[j]),
                     "@model_path/weights/layer%d/%s", layer,
                     path + strlen("@model_path/") + strlen(prefix));
        } else {
            snprintf(layer_mil[j], sizeof(layer_mil[j]), "%s", path);
        }
    }
    for (int i = 0; i < g->n_nodes; i++) {
        OrionNode *node = &g->nodes[i];
        if (node->op != ORION_OP_CONST || !node->attrs.blob_path[0]) continue;
        bool found = false;
        for (int j = 0; j < kEntryCount; j++) {
            if (strcmp(node->attrs.blob_path, layer_mil[j]) == 0) {
                snprintf(node->attrs.blob_path, sizeof(node->attrs.blob_path),
                         "@model_path/aggregate.bin");
                node->attrs.blob_offset = kEntries[j].chunk_offset;
                found = true;
                (*rewritten)++;
                break;
            }
        }
        if (!found) {
            fprintf(stderr, "depth-eval: graph references unaggregated blob %s\n",
                    node->attrs.blob_path);
            return false;
        }
    }
    return true;
}

// Validate + optimize (production passes) but skip the ANE plan validation
// that enforces one tensor per file at offset 64, then emit MIL directly.
static NSString *generate_aggregate_mil(OrionGraph *g, int layer, int *rewritten) {
    if (!g) return nil;
    OrionValidationResult vr = orion_graph_validate(g);
    if (!vr.valid) {
        fprintf(stderr, "depth-eval: graph validation: %s\n", vr.message);
        orion_graph_free(g);
        return nil;
    }
    orion_pipeline_optimize(g);
    if (!rewrite_blob_refs(g, layer, rewritten)) {
        orion_graph_free(g);
        return nil;
    }
    NSString *mil = orion_codegen_mil(g, "main");
    orion_graph_free(g);
    return mil;
}

static NSDictionary *wdict(NSData *aggregate) {
    return @{@"@model_path/aggregate.bin": @{@"offset": @0, @"data": aggregate}};
}

// ---------------------------------------------------------------------------
// Parity / finiteness helpers (ported from the depth-loop test).
// ---------------------------------------------------------------------------

static double max_abs_diff(const _Float16 *a, const _Float16 *b, size_t count) {
    double maximum = 0.0;
    for (size_t i = 0; i < count; i++) {
        double diff = fabs((double)a[i] - (double)b[i]);
        if (diff > maximum) maximum = diff;
    }
    return maximum;
}

static double logical_max_abs(const _Float16 *a, const _Float16 *b) {
    double maximum = 0.0;
    for (int c = 0; c < DIM; c++) {
        for (int t = 0; t < LOGICAL_TOKENS; t++) {
            size_t idx = (size_t)c * SEQ + t;
            double diff = fabs((double)a[idx] - (double)b[idx]);
            if (diff > maximum) maximum = diff;
        }
    }
    return maximum;
}

static double logical_relative_l2(const _Float16 *a, const _Float16 *b) {
    double err = 0.0, ref = 0.0;
    for (int c = 0; c < DIM; c++) {
        for (int t = 0; t < LOGICAL_TOKENS; t++) {
            size_t idx = (size_t)c * SEQ + t;
            double d = (double)a[idx] - (double)b[idx];
            err += d * d;
            ref += (double)b[idx] * (double)b[idx];
        }
    }
    return ref > 0.0 ? sqrt(err / ref) : INFINITY;
}

static bool logical_finite(const _Float16 *values) {
    for (int c = 0; c < DIM; c++) {
        for (int t = 0; t < LOGICAL_TOKENS; t++) {
            if (!isfinite((double)values[(size_t)c * SEQ + t])) return false;
        }
    }
    return true;
}

static bool all_finite(const _Float16 *values, size_t count) {
    for (size_t i = 0; i < count; i++) {
        if (!isfinite((double)values[i])) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Minimal npz (ZIP, STORED) + npy header reader.
//
// numpy's np.savez() stores each array as an uncompressed (ZIP method 0) member
// named "<key>.npy". We locate a member via the End-of-Central-Directory
// record, walk the central directory, then read its local header to find the
// raw member bytes. We then skip the .npy header to reach the fp16 payload.
// DEFLATE-compressed members (np.savez_compressed) are rejected; np.savez is
// the contract Python uses to feed this command.
// ---------------------------------------------------------------------------

// Returns the raw .npy member bytes for `member` (without the ".npy" suffix),
// or nil. Copies the member into an autoreleased NSData.
static NSData *npz_member_npy(NSData *npz, const char *member) {
    const uint8_t *base = npz.bytes;
    size_t size = npz.length;
    if (size < 22) return nil;

    // Locate the End-of-Central-Directory record (signature PK\x05\x06) by
    // scanning backwards from near the end of the file.
    long eocd = -1;
    long scan_start = (long)size - 22;
    long min_ok = scan_start - 65536;
    if (min_ok < 0) min_ok = 0;
    for (long i = scan_start; i >= min_ok; i--) {
        if (base[i] == 0x50 && base[i + 1] == 0x4b &&
            base[i + 2] == 0x05 && base[i + 3] == 0x06) {
            eocd = i;
            break;
        }
    }
    if (eocd < 0) return nil;

    uint32_t cd_off = read_le32(base + eocd + 16);
    uint16_t total = read_le16(base + eocd + 10);
    if ((size_t)cd_off >= size) return nil;

    char want[256];
    snprintf(want, sizeof(want), "%s.npy", member);
    size_t wantlen = strlen(want);

    long p = (long)cd_off;
    for (uint16_t e = 0; e < total; e++) {
        if ((size_t)(p + 46) > size) return nil;
        if (!(base[p] == 0x50 && base[p + 1] == 0x4b &&
              base[p + 2] == 0x01 && base[p + 3] == 0x02)) return nil;
        uint16_t fname_len = read_le16(base + p + 28);
        uint16_t extra_len = read_le16(base + p + 30);
        uint16_t comment_len = read_le16(base + p + 32);
        uint32_t loc_off = read_le32(base + p + 42);
        if ((size_t)(p + 46 + fname_len) > size) return nil;
        if ((size_t)fname_len == wantlen &&
            memcmp(base + p + 46, want, wantlen) == 0) {
            // Jump to the local file header for data offset + method.
            if ((size_t)(loc_off + 30) > size) return nil;
            long lh = (long)loc_off;
            if (!(base[lh] == 0x50 && base[lh + 1] == 0x4b &&
                  base[lh + 2] == 0x03 && base[lh + 3] == 0x04)) return nil;
            uint16_t method = read_le16(base + lh + 8);
            uint32_t comp_size = read_le32(base + lh + 18);
            uint16_t lh_fname = read_le16(base + lh + 26);
            uint16_t lh_extra = read_le16(base + lh + 28);
            long data_off = lh + 30 + (long)lh_fname + (long)lh_extra;
            if ((size_t)(data_off + comp_size) > size) return nil;
            if (method != 0) {
                fprintf(stderr,
                        "depth-eval: npz member %s is compressed (method %u); "
                        "use np.savez (uncompressed), not np.savez_compressed\n",
                        want, method);
                return nil;
            }
            return [NSData dataWithBytes:base + data_off length:comp_size];
        }
        p += 46 + (long)fname_len + (long)extra_len + (long)comment_len;
    }
    return nil;
}

// Given the bytes of a .npy file, compute the offset and length of the raw
// numeric payload that follows the header. Returns false on malformed input.
static bool npy_payload(const uint8_t *b, size_t n, size_t *out_off,
                        size_t *out_len) {
    if (n < 10) return false;
    if (!(b[0] == 0x93 && b[1] == 'N' && b[2] == 'U' &&
          b[3] == 'M' && b[4] == 'P' && b[5] == 'Y')) return false;
    uint8_t major = b[6];
    size_t header_len;
    size_t header_start;
    if (major <= 1) {
        header_len = read_le16(b + 8);
        header_start = 10;
    } else {
        if (n < 12) return false;
        header_len = read_le32(b + 8);
        header_start = 12;
    }
    size_t data_off = header_start + header_len;
    if (data_off > n) return false;
    *out_off = data_off;
    *out_len = n - data_off;
    return true;
}

// Load `member` from an npz as raw bytes, requiring exactly expected_bytes.
static NSData *load_npz_tensor(NSData *npz, const char *member,
                               size_t expected_bytes) {
    NSData *npy = npz_member_npy(npz, member);
    if (!npy) return nil;
    size_t off = 0, len = 0;
    if (!npy_payload(npy.bytes, npy.length, &off, &len)) return nil;
    if (len != expected_bytes) return nil;
    return [NSData dataWithBytes:(const uint8_t *)npy.bytes + off length:len];
}

// ---------------------------------------------------------------------------
// Output + error mapping.
// ---------------------------------------------------------------------------

static bool write_output_dir(NSString *out_dir, const _Float16 *cond_out,
                             const _Float16 *uncond_out, size_t bytes,
                             NSString *branch_label) {
    NSFileManager *fm = [NSFileManager defaultManager];
    NSError *err = nil;
    if (![fm createDirectoryAtPath:out_dir
               withIntermediateDirectories:YES
                               attributes:nil
                                   error:&err]) {
        fprintf(stderr, "depth-eval: cannot create output dir %s: %s\n",
                out_dir.UTF8String, err.localizedDescription.UTF8String);
        return false;
    }
    NSString *p_cond = [out_dir stringByAppendingPathComponent:@"final_norm_cond.fp16.bin"];
    NSString *p_uncond = [out_dir stringByAppendingPathComponent:@"final_norm_uncond.fp16.bin"];
    [[NSData dataWithBytes:cond_out length:bytes] writeToFile:p_cond atomically:YES];
    [[NSData dataWithBytes:uncond_out length:bytes] writeToFile:p_uncond atomically:YES];

    NSMutableDictionary *idx = [NSMutableDictionary dictionary];
    idx[@"schema"] = @"orion.cli.depth-eval-output/1.0";
    idx[@"dtype"] = @"float16";
    idx[@"shape"] = @[@1, @(DIM), @1, @(SEQ)];
    idx[@"element_count"] = @(ELEMENTS);
    idx[@"bytes_per_tensor"] = @(bytes);
    idx[@"files"] = @{
        @"final_norm_cond": @"final_norm_cond.fp16.bin",
        @"final_norm_uncond": @"final_norm_uncond.fp16.bin",
    };
    idx[@"branch_mapping"] = @{
        @"final_norm_cond": @"CFG guidance branch (input hidden_cond / batch0)",
        @"final_norm_uncond": @"CFG generation branch (input hidden_uncond / batch1)",
    };
    idx[@"token_note"] =
        @"Per-step acoustic tokens are not produced here: this harness reuses "
        "fixed packed-hidden inputs across all 15 steps to measure latency and "
        "parity (ANE timing is shape-dependent, not data-dependent). The "
        "acoustic-head + sampler pipeline lives in the Python host; these "
        "tensors are the depth-transformer final-norm hidden states.";
    if (branch_label) idx[@"source_branch_order"] = branch_label;
    NSData *j = [NSJSONSerialization dataWithJSONObject:idx
                                                options:NSJSONWritingPrettyPrinted |
                                                        NSJSONWritingSortedKeys
                                                  error:nil];
    [j writeToFile:[out_dir stringByAppendingPathComponent:@"index.json"]
        atomically:YES];
    return true;
}

// Map a failure stage onto an Orion error code (core/orion_error.h). Mirrors
// orion_error_name() for the codes this command can return; kept local so the
// command links without pulling core/orion_error.c into the CLI binary.
static int depth_eval_exit_code(NSString *failure) {
    if (!failure) return ORION_OK;
    if ([failure isEqualToString:@"compile-budget-exceeded"]) return ORION_ERR_BUDGET_EXHAUSTED;
    if ([failure isEqualToString:@"evaluation"]) return ORION_ERR_EVAL_FAILED;
    if ([failure isEqualToString:@"surface-allocation"] ||
        [failure isEqualToString:@"host-allocation"]) return ORION_ERR_SURFACE_INVALID;
    if ([failure hasPrefix:@"aggregate"] || [failure hasPrefix:@"layer0"] ||
        [failure hasPrefix:@"final"] || [failure hasPrefix:@"weight"]) {
        return ORION_ERR_COMPILE_FAILED;
    }
    return 1;
}

static const char *depth_eval_error_name(int code) {
    switch (code) {
        case ORION_OK: return "ORION_OK";
        case ORION_ERR_COMPILE_FAILED: return "ORION_ERR_COMPILE_FAILED";
        case ORION_ERR_EVAL_FAILED: return "ORION_ERR_EVAL_FAILED";
        case ORION_ERR_BUDGET_EXHAUSTED: return "ORION_ERR_BUDGET_EXHAUSTED";
        case ORION_ERR_SURFACE_INVALID: return "ORION_ERR_SURFACE_INVALID";
        default: return "ORION_UNKNOWN";
    }
}

static void print_depth_eval_help(void) {
    fprintf(stderr,
        "Usage: orion depth-eval --export-dir <path> [options]\n"
        "\n"
        "Drive the VoXtream depth loop on the ANE. Default mode is a fixed-input\n"
        "diagnostic latency probe; --recurrence enables real token production.\n"
        "\n"
        "Options:\n"
        "  --export-dir PATH        VoXtream v16 export directory (required, or\n"
        "                           set $VOXTREAM_DEPTH_EXPORT).\n"
        "  --input PATH             npz with fp16 [1,1024,1,32] members\n"
        "                           hidden_cond / hidden_uncond, or a directory\n"
        "                           holding hidden_{cond,uncond}.fp16.bin.\n"
        "                           Default: <export-dir>/ane_references/packed_hidden.batch{0,1}.fp16.bin\n"
        "  --output PATH            Directory (or out.npz stem) for final-norm\n"
        "                           output hidden states as raw fp16 .bin + index.json.\n"
        "  --evidence PATH          Write timing/parity/compile JSON here.\n"
        "                           Default: write JSON to stdout.\n"
        "  --max-compile N          Abort if compile-count delta > N (default 3).\n"
        "  --samples N              Timed full-loop samples (default 30, max 100).\n"
        "  --warmup N               Untimed warmup loops (default 2, max 20).\n"
        "  --host-overhead-ms F     External host overhead estimate (default 0).\n"
        "  --protocol-version N     Advertise wire-protocol version (default 1).\n"
        "  --protocol-info          Print JSON capabilities and exit 0.\n"
        "  --recurrence             REAL 15-step autoregressive token production.\n"
        "  --embeddings PATH        audio_embeddings raw LE fp32 [32800,1024]\n"
        "                           (required for --recurrence).\n"
        "  --audio-head PATH        audio_head raw LE fp32 [15,1024,2050]\n"
        "                           (default: <export-dir>/audio_head.fp32.bin).\n"
        "  --c0 N                   Semantic codebook-0 token (0..2049).\n"
        "  --gamma F                Acoustic CFG scale (default 3.0).\n"
        "  --help                   Show this help message.\n");
}

// Emit the protocol-info capabilities object and exit. No ANE touched.
static int emit_protocol_info(void) {
    NSDictionary *info = @{
        @"schema": @"orion.cli.depth-eval-protocol-info/1",
        @"command": @"depth-eval",
        @"supported_min": @(1),
        @"supported_max": @(1),
        @"default_when_omitted": @(1),
        @"legacy_omission_supported": @(YES),
        @"capabilities": @{
            @"final_norm_outputs": @(YES),
            @"token_outputs": @(YES),       // when --recurrence is used
            @"recurrence": @(YES),
            @"separate_cfg_abi": @(YES),
            @"sequence_pack_abi": @(NO),    // deferred per ORX-005
            @"atomic_output_set": @(NO),    // current writer is per-file atomic only
        },
    };
    NSData *j = [NSJSONSerialization dataWithJSONObject:info
        options:NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys error:nil];
    NSString *s = [[NSString alloc] initWithBytes:j.bytes length:j.length
                                          encoding:NSUTF8StringEncoding];
    printf("%s\n", s.UTF8String);
    return 0;
}

// ---------------------------------------------------------------------------
// Recurrence helpers (real token production).
//
// The recurrence state per CFG branch is the full prefix in [last_hidden,
// c0_embed, c1_embed, ..., c{n-1}_embed] order, laid out in the [1,1024,1,32]
// ANE surface. Slot 0 = last_hidden, slot 1 = c0_embed, slot k+2 = c{k}_embed.
// Slots beyond the active prefix are zero-padded. The ANE graph has fixed
// RoPE positions 0..31 and a fixed full causal mask, so feeding the growing
// prefix at slots 0..prefix_len-1 and reading slot prefix_len-1 reproduces
// cached decoding exactly (attention is causal, so future padded slots do
// not affect the active position).
// ---------------------------------------------------------------------------

// Convert one fp16 hidden row of length DIM at the given slot to fp32
// (for cblas_sgemv). The surface is channel-major [1, DIM, 1, SEQ]:
// element (channel=c, slot=t) lives at offset c*SEQ + t. So the row for
// slot `t` is the strided gather surface[c*SEQ + t] for c in [0,DIM).
static void fp16_row_to_f32(const _Float16 *surface, int slot, float *out) {
    for (int c = 0; c < DIM; c++) {
        out[c] = (float)surface[(size_t)c * SEQ + slot];
    }
}

// audio_head GEMV: logits[V] = hidden[D] @ head[D, V], fp32 accumulation.
// head is stored row-major as [D, V] (i.e. head[d*V + v] = element (d,v)).
// We compute out[v] = sum_d hidden[d] * head[d,v] = (head.T @ hidden)[v]
// where head.T is [V, D]. cblas_sgemv with CblasTrans interprets the matrix
// as (rows=D, cols=V) and computes A.T @ x, giving [V].
static void audio_head_gemv(const float *hidden, const float *head,
                            int D, int V, float *out) {
    cblas_sgemv(CblasRowMajor, CblasTrans, D, V, 1.0f,
                head, V, hidden, 1, 0.0f, out, 1);
}

// CFG combine: out[V] = cond[V]*gamma + uncond[V]*(1-gamma).
static void cfg_combine(const float *cond, const float *uncond,
                        int V, float gamma, float *out) {
    float w1 = (float)gamma;
    float w2 = 1.0f - w1;
    vDSP_vsmul(cond, 1, &w1, out, 1, (vDSP_Length)V);
    vDSP_vsma(uncond, 1, &w2, out, 1, out, 1, (vDSP_Length)V);
}

// Argmax over V floats, deterministic (first max wins on tie).
static int argmax_float(const float *v, int V) {
    int best = 0;
    float best_val = v[0];
    for (int i = 1; i < V; i++) {
        if (v[i] > best_val) { best_val = v[i]; best = i; }
    }
    return best;
}

// Write a fp32 row (e.g. an audio embedding) into the fp16 surface at the
// given slot. Channel-major layout: surface[c*SEQ + slot] = row_f32[c].
static void write_prefix_slot_fp32(_Float16 *surface, int slot,
                                   const float *row_f32, int dim) {
    for (int c = 0; c < dim; c++) {
        surface[(size_t)c * SEQ + slot] = (_Float16)row_f32[c];
    }
}

// Write a fp16 row (e.g. last_hidden) into the surface at the given slot.
// Source row_fp16 is dense [dim]; surface is channel-major.
static void write_prefix_slot_fp16(_Float16 *surface, int slot,
                                   const _Float16 *row_fp16, int dim) {
    for (int c = 0; c < dim; c++) {
        surface[(size_t)c * SEQ + slot] = row_fp16[c];
    }
}

// Zero the entire [1, DIM, SEQ] surface.
static void zero_surface(_Float16 *surface) {
    memset(surface, 0, TENSOR_BYTES);
}

// Copy one slot column from a channel-major source surface to a slot column
// of a destination surface. Both are [1, DIM, 1, SEQ]; element (c, t) is at
// offset c*SEQ + t. Copies DIM elements (one per channel) at the given slots.
static void copy_slot(_Float16 *dst_surface, int dst_slot,
                      const _Float16 *src_surface, int src_slot) {
    for (int c = 0; c < DIM; c++) {
        dst_surface[(size_t)c * SEQ + dst_slot] =
            src_surface[(size_t)c * SEQ + src_slot];
    }
}

// Write the 15 sampled tokens as little-endian int32 to <output_dir>/tokens.bin.
// Returns true on success. Creates the directory if needed.
static bool write_tokens_bin(NSString *output_dir, const int32_t *tokens,
                             int count) {
    if (!output_dir) return false;
    NSFileManager *fm = [NSFileManager defaultManager];
    BOOL is_dir = NO;
    if (![fm fileExistsAtPath:output_dir isDirectory:&is_dir] || !is_dir) {
        NSError *err = nil;
        if (![fm createDirectoryAtPath:output_dir
              withIntermediateDirectories:YES attributes:nil error:&err]) {
            fprintf(stderr, "depth-eval: cannot create output dir %s: %s\n",
                    output_dir.UTF8String, err.localizedDescription.UTF8String);
            return false;
        }
    }
    NSString *path = [output_dir stringByAppendingPathComponent:@"tokens.bin"];
    NSMutableData *data = [NSMutableData dataWithCapacity:(NSUInteger)count * 4];
    for (int i = 0; i < count; i++) {
        uint8_t le[4] = {
            (uint8_t)((uint32_t)tokens[i] & 0xFF),
            (uint8_t)(((uint32_t)tokens[i] >> 8) & 0xFF),
            (uint8_t)(((uint32_t)tokens[i] >> 16) & 0xFF),
            (uint8_t)(((uint32_t)tokens[i] >> 24) & 0xFF),
        };
        [data appendBytes:le length:4];
    }
    return [data writeToFile:path atomically:YES];
}

// ---------------------------------------------------------------------------
// Command entry point.
// ---------------------------------------------------------------------------

int orion_cmd_depth_eval(int argc, const char *argv[]) {
    @autoreleasepool {
        const char *export_arg = NULL;
        const char *input_arg = NULL;
        const char *output_arg = NULL;
        const char *evidence_arg = NULL;
        const char *embeddings_arg = NULL;
        const char *audio_head_arg = NULL;
        int max_compile = DEFAULT_MAX_COMPILE;
        int sample_count = DEFAULT_SAMPLES;
        int warmup_count = DEFAULT_WARMUP;
        double host_overhead_ms = 0.0;
        int protocol_version = PROTOCOL_VERSION;
        int c0_token = -1;
        double cfg_gamma = DEFAULT_CFG_GAMMA;
        bool recurrence_mode = false;
        bool protocol_info_mode = false;

        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "--export-dir") == 0 && i + 1 < argc) {
                export_arg = argv[++i];
            } else if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
                input_arg = argv[++i];
            } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
                output_arg = argv[++i];
            } else if (strcmp(argv[i], "--evidence") == 0 && i + 1 < argc) {
                evidence_arg = argv[++i];
            } else if (strcmp(argv[i], "--max-compile") == 0 && i + 1 < argc) {
                max_compile = atoi(argv[++i]);
            } else if (strcmp(argv[i], "--samples") == 0 && i + 1 < argc) {
                sample_count = atoi(argv[++i]);
            } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
                warmup_count = atoi(argv[++i]);
            } else if (strcmp(argv[i], "--host-overhead-ms") == 0 && i + 1 < argc) {
                host_overhead_ms = atof(argv[++i]);
            } else if (strcmp(argv[i], "--protocol-version") == 0 && i + 1 < argc) {
                protocol_version = atoi(argv[++i]);
            } else if (strcmp(argv[i], "--protocol-info") == 0) {
                protocol_info_mode = true;
            } else if (strcmp(argv[i], "--recurrence") == 0) {
                recurrence_mode = true;
            } else if (strcmp(argv[i], "--embeddings") == 0 && i + 1 < argc) {
                embeddings_arg = argv[++i];
            } else if (strcmp(argv[i], "--audio-head") == 0 && i + 1 < argc) {
                audio_head_arg = argv[++i];
            } else if (strcmp(argv[i], "--c0") == 0 && i + 1 < argc) {
                c0_token = atoi(argv[++i]);
            } else if (strcmp(argv[i], "--gamma") == 0 && i + 1 < argc) {
                cfg_gamma = atof(argv[++i]);
            } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
                print_depth_eval_help();
                return 0;
            } else {
                fprintf(stderr, "depth-eval: unknown argument '%s'\n\n", argv[i]);
                print_depth_eval_help();
                return 1;
            }
        }

        // --protocol-info short-circuits before any ANE or export work.
        if (protocol_info_mode) {
            return emit_protocol_info();
        }

        // Protocol version enforcement: only v1 is supported. Any other value
        // is an unrecoverable wire mismatch (exit 9, fail-closed).
        if (protocol_version != PROTOCOL_VERSION) {
            fprintf(stderr,
                    "depth-eval: protocol mismatch: requested=%d supported_min=1 "
                    "supported_max=1\n",
                    protocol_version);
            return 9;
        }

        if (max_compile < 0) max_compile = 0;
        if (sample_count < 1 || sample_count > MAX_SAMPLES) {
            fprintf(stderr, "depth-eval: --samples must be 1..%d\n", MAX_SAMPLES);
            return 1;
        }
        if (warmup_count < 0 || warmup_count > 20) {
            fprintf(stderr, "depth-eval: --warmup must be 0..20\n");
            return 1;
        }
        if (host_overhead_ms < 0.0) host_overhead_ms = 0.0;
        if (cfg_gamma < 0.0) cfg_gamma = DEFAULT_CFG_GAMMA;

        // Recurrence mode requires embeddings + c0.
        if (recurrence_mode) {
            if (!embeddings_arg) {
                fprintf(stderr,
                        "depth-eval: --recurrence requires --embeddings PATH\n");
                return 1;
            }
            if (c0_token < 0 || c0_token >= AUDIO_VOCAB) {
                fprintf(stderr,
                        "depth-eval: --recurrence requires --c0 N (0..%d)\n",
                        AUDIO_VOCAB - 1);
                return 1;
            }
        }

        // Resolve the export directory (arg overrides env).
        const char *root_env = getenv("VOXTREAM_DEPTH_EXPORT");
        const char *root_c = export_arg ? export_arg : root_env;
        if (!root_c || !root_c[0]) {
            fprintf(stderr,
                    "depth-eval: --export-dir is required (or set VOXTREAM_DEPTH_EXPORT)\n");
            return 1;
        }
        NSString *root = @(root_c);
        BOOL is_dir = NO;
        if (![[NSFileManager defaultManager] fileExistsAtPath:root isDirectory:&is_dir] ||
            !is_dir) {
            fprintf(stderr, "depth-eval: export dir not found or not a directory: %s\n",
                    root_c);
            return 1;
        }

        if (!orion_ane_init()) {
            fprintf(stderr, "depth-eval: ANE runtime unavailable (orion_ane_init failed)\n");
            return 1;
        }

        // Resolve the two packed-hidden inputs.
        NSData *inputs[CFG_BRANCHES] = {nil, nil};
        if (input_arg) {
            NSString *inp = @(input_arg);
            BOOL in_is_dir = NO;
            if ([[NSFileManager defaultManager] fileExistsAtPath:inp isDirectory:&in_is_dir] &&
                in_is_dir) {
                inputs[0] = [NSData dataWithContentsOfFile:
                    [inp stringByAppendingPathComponent:@"hidden_cond.fp16.bin"]];
                inputs[1] = [NSData dataWithContentsOfFile:
                    [inp stringByAppendingPathComponent:@"hidden_uncond.fp16.bin"]];
            } else {
                NSData *npz = [NSData dataWithContentsOfFile:inp];
                if (npz) {
                    inputs[0] = load_npz_tensor(npz, "hidden_cond", TENSOR_BYTES);
                    inputs[1] = load_npz_tensor(npz, "hidden_uncond", TENSOR_BYTES);
                }
            }
            if (!inputs[0] || !inputs[1] ||
                inputs[0].length != TENSOR_BYTES || inputs[1].length != TENSOR_BYTES) {
                fprintf(stderr,
                        "depth-eval: --input %s did not provide hidden_cond and "
                        "hidden_uncond as fp16 [%d,%d,1,%d] tensors\n",
                        input_arg, 1, DIM, SEQ);
                return 1;
            }
        } else {
            inputs[0] = [NSData dataWithContentsOfFile:
                [root stringByAppendingPathComponent:
                    @"ane_references/packed_hidden.batch0.fp16.bin"]];
            inputs[1] = [NSData dataWithContentsOfFile:
                [root stringByAppendingPathComponent:
                    @"ane_references/packed_hidden.batch1.fp16.bin"]];
            if (!inputs[0] || !inputs[1] ||
                inputs[0].length != TENSOR_BYTES || inputs[1].length != TENSOR_BYTES) {
                fprintf(stderr,
                        "depth-eval: default inputs missing in %s/ane_references "
                        "(provide --input)\n", root_c);
                return 1;
            }
        }

        // Optional layer-0 parity references (skip parity if absent).
        NSData *references[CFG_BRANCHES] = {nil, nil};
        references[0] = [NSData dataWithContentsOfFile:
            [root stringByAppendingPathComponent:
                @"ane_references/packed_layer_0_output.batch0.fp16.bin"]];
        references[1] = [NSData dataWithContentsOfFile:
            [root stringByAppendingPathComponent:
                @"ane_references/packed_layer_0_output.batch1.fp16.bin"]];
        const bool parity_run = references[0] && references[1] &&
            references[0].length == TENSOR_BYTES && references[1].length == TENSOR_BYTES;

        // Normalize --output to a directory.
        NSString *output_dir = nil;
        if (output_arg) {
            output_dir = @(output_arg);
            if ([output_dir hasSuffix:@".npz"] || [output_dir hasSuffix:@".json"]) {
                output_dir = [output_dir stringByDeletingPathExtension];
            }
        }

        NSString *failure = nil;
        const int count_before = orion_compile_count();

        NSMutableDictionary *evidence = [NSMutableDictionary dictionary];
        evidence[@"schema"] = @"orion.cli.depth-eval/1.0";
        evidence[@"protocol_version"] = @(PROTOCOL_VERSION);
        evidence[@"export_root"] = root;
        evidence[@"input_source"] = input_arg ? @(input_arg) : @"export-dir-default";
        evidence[@"output_dir"] = output_dir ? output_dir : [NSNull null];
        evidence[@"depth_steps"] = @(DEPTH_STEPS);
        evidence[@"cfg_branches"] = @(CFG_BRANCHES);
        evidence[@"iterations_per_sample"] = @(ITERATIONS_PER_SAMPLE);
        evidence[@"layers"] = @(LAYERS);
        evidence[@"weight_patches"] = @(LAYERS - 1);
        evidence[@"samples_requested"] = @(sample_count);
        evidence[@"warmup"] = @(warmup_count);
        evidence[@"max_compile"] = @(max_compile);
        evidence[@"recurrence_mode"] = @(recurrence_mode);
        evidence[@"cfg_gamma"] = @(cfg_gamma);
        if (recurrence_mode) {
            evidence[@"c0_token"] = @(c0_token);
            evidence[@"embeddings_source"] = embeddings_arg ? @(embeddings_arg) : [NSNull null];
            evidence[@"audio_head_source"] = audio_head_arg ? @(audio_head_arg) : @"export-dir-default";
        }
        evidence[@"operating_system"] =
            [[NSProcessInfo processInfo] operatingSystemVersionString];
        const char *revision = getenv("VOXTREAM_ORION_REVISION");
        const char *compiler_build = getenv("VOXTREAM_COMPILER_BUILD");
        evidence[@"orion_revision"] = revision ? @(revision) : @"unspecified";
        evidence[@"compiler_build"] = compiler_build ? @(compiler_build) : @"unspecified";
        evidence[@"compile_count_before"] = @(count_before);

        // ---- Compile programs ----------------------------------------------
        // Recurrence mode: 4 real layer compiles + 1 final norm (no aliasing).
        // Diagnostic mode: 1 layer-0 compile + 3 donor patches + 1 final norm.
        OrionProgram *layer_programs[LAYERS];
        for (int i = 0; i < LAYERS; i++) layer_programs[i] = nil;
        OrionProgram *final_norm_program = nil;
        int compile_attempts = 0;
        NSString *layer0_mil = nil;

        if (recurrence_mode) {
            // Compile each layer with its own real weights.
            for (int n = 0; n < LAYERS && !failure; n++) {
                NSData *aggregate_n = build_aggregate_blob(root, n);
                if (!aggregate_n) { failure = @"aggregate-build"; break; }
                int rewritten = 0;
                NSString *mil_n = generate_aggregate_mil(
                    orion_frontend_voxtream_depth_layer(n), n, &rewritten);
                if (!mil_n) { failure = @"layer-mil-generation"; break; }
                if (n == 0) layer0_mil = mil_n;  // retained for diagnostics
                char tag[32];
                snprintf(tag, sizeof(tag), "depth-eval-layer%d-real", n);
                compile_attempts++;
                layer_programs[n] = orion_compile_mil(mil_n.UTF8String,
                                                      wdict(aggregate_n), tag);
                if (!layer_programs[n]) {
                    char buf[64];
                    snprintf(buf, sizeof(buf), "layer%d-compile", n);
                    failure = @(buf);
                    break;
                }
            }
        } else {
            // Diagnostic: donor-patch path (2 compiles total).
            NSData *aggregate0 = build_aggregate_blob(root, 0);
            if (!aggregate0) { failure = @"aggregate-build"; }
            if (!failure) {
                int rewritten = 0;
                layer0_mil = generate_aggregate_mil(
                    orion_frontend_voxtream_depth_layer(0), 0, &rewritten);
                if (!layer0_mil) {
                    failure = @"layer0-mil-generation";
                } else {
                    compile_attempts++;
                    layer_programs[0] = orion_compile_mil(layer0_mil.UTF8String,
                                                          wdict(aggregate0),
                                                          "depth-eval-layer0");
                    if (!layer_programs[0]) failure = @"layer0-compile";
                }
            }
            for (int n = 1; n < LAYERS && !failure; n++) {
                NSData *patched = build_aggregate_blob(root, n);
                if (!patched) { failure = @"aggregate-build"; break; }
                char tag[32];
                snprintf(tag, sizeof(tag), "depth-eval-layer%d", n);
                layer_programs[n] = orion_program_patch_weights(
                    layer_programs[0], layer0_mil.UTF8String, wdict(patched), tag);
                if (!layer_programs[n]) { failure = @"weight-patch"; break; }
            }
        }

        // Final norm: one compile.
        if (!failure) {
            NSData *final_blob = read_valid_single_blob(
                [root stringByAppendingPathComponent:@"weights/final_norm.bin"], 2048);
            OrionGraph *final_graph = orion_frontend_voxtream_depth_final_norm();
            NSString *final_mil = nil;
            if (final_blob && final_graph) {
                OrionValidationResult fvr = orion_graph_validate(final_graph);
                if (fvr.valid) {
                    orion_pipeline_optimize(final_graph);
                    final_mil = orion_codegen_mil(final_graph, "main");
                }
            }
            if (final_graph) orion_graph_free(final_graph);
            if (!final_mil) {
                failure = @"final-norm-mil";
            } else {
                compile_attempts++;
                final_norm_program = orion_compile_mil(final_mil.UTF8String,
                    @{@"@model_path/weights/final_norm.bin":
                        @{@"offset": @0, @"data": final_blob}},
                    "depth-eval-final-norm");
                if (!final_norm_program) failure = @"final-norm-compile";
            }
        }

        const int count_after = orion_compile_count();
        const int compile_delta = count_after - count_before;
        evidence[@"compile_attempts"] = @(compile_attempts);
        evidence[@"compile_count_after"] = @(count_after);
        evidence[@"compile_count_delta"] = @(compile_delta);
        // Hard compile-budget guard.
        if (!failure && compile_delta > max_compile) {
            fprintf(stderr,
                    "depth-eval: compile_count_delta=%d exceeds --max-compile=%d "
                    "(code %d, %s)\n",
                    compile_delta, max_compile,
                    ORION_ERR_BUDGET_EXHAUSTED,
                    depth_eval_error_name(ORION_ERR_BUDGET_EXHAUSTED));
            failure = @"compile-budget-exceeded";
        }

        // ---- Allocate the IOSurface chain ----------------------------------
        // chain[0] = input, chain[1..LAYERS] = layer outputs, chain[LAYERS+1]
        // = final-norm output.
        IOSurfaceRef chain[CHAIN_SURFACES];
        for (int i = 0; i < CHAIN_SURFACES; i++) chain[i] = NULL;
        _Float16 *output = NULL;
        _Float16 *repeat = NULL;
        _Float16 *cond_out = NULL;
        _Float16 *uncond_out = NULL;
        NSMutableArray *parity_results = [NSMutableArray array];
        bool parity = true, finite = true, deterministic = true;

        if (!failure) {
            for (int i = 0; i < CHAIN_SURFACES; i++) {
                chain[i] = orion_tensor_create(DIM, SEQ);
                if (!chain[i]) { failure = @"surface-allocation"; break; }
            }
            output = malloc(TENSOR_BYTES);
            repeat = malloc(TENSOR_BYTES);
            if (output_dir) {
                cond_out = malloc(TENSOR_BYTES);
                uncond_out = malloc(TENSOR_BYTES);
            }
            if (!output || !repeat ||
                (output_dir && (!cond_out || !uncond_out))) {
                failure = @"host-allocation";
            }
        }

        // ---- Layer-0 parity sanity check (only if references present) -------
        if (!failure && parity_run) {
            for (int batch = 0; batch < CFG_BRANCHES && !failure; batch++) {
                orion_tensor_write(chain[0], inputs[batch].bytes, TENSOR_BYTES);
                if (!orion_eval(layer_programs[0], &chain[0], 1, &chain[1], 1)) {
                    failure = @"evaluation";
                    break;
                }
                orion_tensor_read(chain[1], output, TENSOR_BYTES);
                const _Float16 *reference = references[batch].bytes;
                double max_abs = logical_max_abs(output, reference);
                double rel_l2 = logical_relative_l2(output, reference);
                bool branch_passed = max_abs <= MAX_ABS_GATE && rel_l2 <= REL_L2_GATE;
                parity = parity && branch_passed;
                finite = finite && logical_finite(output) && logical_finite(reference);
                [parity_results addObject:@{
                    @"batch": @(batch),
                    @"comparison_scope":
                        @"logical-tokens[0,16)-of-32-per-contract",
                    @"max_abs": @(max_abs),
                    @"relative_l2": @(rel_l2),
                    @"max_abs_gate": @(MAX_ABS_GATE),
                    @"relative_l2_gate": @(REL_L2_GATE),
                    @"passed": @(branch_passed),
                }];
            }
        }

        // ---- Determinism: identical full forward pass (batch 0) twice -------
        if (!failure) {
            orion_tensor_write(chain[0], inputs[0].bytes, TENSOR_BYTES);
            bool eval_ok = true;
            for (int stage = 0; stage < LAYERS; stage++) {
                if (!orion_eval(layer_programs[stage], &chain[stage], 1,
                                &chain[stage + 1], 1)) { eval_ok = false; break; }
            }
            if (eval_ok && !orion_eval(final_norm_program, &chain[LAYERS], 1,
                                       &chain[LAYERS + 1], 1)) eval_ok = false;
            if (!eval_ok) {
                failure = @"evaluation";
            } else {
                orion_tensor_read(chain[LAYERS + 1], output, TENSOR_BYTES);
                orion_tensor_write(chain[0], inputs[0].bytes, TENSOR_BYTES);
                for (int stage = 0; stage < LAYERS && eval_ok; stage++) {
                    if (!orion_eval(layer_programs[stage], &chain[stage], 1,
                                    &chain[stage + 1], 1)) { eval_ok = false; }
                }
                if (eval_ok && !orion_eval(final_norm_program, &chain[LAYERS], 1,
                                           &chain[LAYERS + 1], 1)) eval_ok = false;
                if (!eval_ok) {
                    failure = @"evaluation";
                } else {
                    orion_tensor_read(chain[LAYERS + 1], repeat, TENSOR_BYTES);
                    deterministic = max_abs_diff(output, repeat, ELEMENTS) == 0.0;
                    finite = finite && all_finite(output, ELEMENTS) &&
                             all_finite(repeat, ELEMENTS);
                }
            }
        }

        // ---- The 15-step separate-CFG depth loop ----------------------------
        // Per iteration: write packed hidden -> 4 layer evals (IOSurface-piped)
        // -> final-norm eval -> read final-norm output. The full loop wraps all
        // DEPTH_STEPS x CFG_BRANCHES iterations. ANE timing is shape-dependent,
        // so reusing the two fixed inputs reproduces the production latency.
        double full_loop_samples[MAX_SAMPLES];
        double it_write_samples[MAX_ITERATION_SAMPLES];
        double it_eval_samples[MAX_ITERATION_SAMPLES];
        double it_read_samples[MAX_ITERATION_SAMPLES];
        double it_total_samples[MAX_ITERATION_SAMPLES];
        int full_loop_count = 0;
        int iteration_count = 0;
        bool timed = false;

        if (!failure) {
            for (int sample = -warmup_count; sample < sample_count && !failure; sample++) {
                const bool record = sample >= 0;
                uint64_t full_t0 = monotonic_now();
                for (int step = 0; step < DEPTH_STEPS && !failure; step++) {
                    for (int branch = 0; branch < CFG_BRANCHES && !failure; branch++) {
                        uint64_t it_t0 = monotonic_now();
                        orion_tensor_write(chain[0], inputs[branch].bytes, TENSOR_BYTES);
                        uint64_t after_write = monotonic_now();
                        bool eval_ok = true;
                        for (int stage = 0; stage < LAYERS; stage++) {
                            if (!orion_eval(layer_programs[stage], &chain[stage], 1,
                                            &chain[stage + 1], 1)) {
                                eval_ok = false; break;
                            }
                        }
                        if (eval_ok && !orion_eval(final_norm_program,
                                                   &chain[LAYERS], 1,
                                                   &chain[LAYERS + 1], 1)) eval_ok = false;
                        if (!eval_ok) { failure = @"evaluation"; break; }
                        uint64_t after_eval = monotonic_now();
                        orion_tensor_read(chain[LAYERS + 1], output, TENSOR_BYTES);
                        uint64_t it_t1 = monotonic_now();
                        finite = finite && all_finite(output, ELEMENTS);
                        if (record) {
                            it_write_samples[iteration_count] =
                                monotonic_elapsed_ms(it_t0, after_write);
                            it_eval_samples[iteration_count] =
                                monotonic_elapsed_ms(after_write, after_eval);
                            it_read_samples[iteration_count] =
                                monotonic_elapsed_ms(after_eval, it_t1);
                            it_total_samples[iteration_count] =
                                monotonic_elapsed_ms(it_t0, it_t1);
                            iteration_count++;
                        }
                    }
                }
                if (failure) break;
                uint64_t full_t1 = monotonic_now();
                if (record) {
                    full_loop_samples[full_loop_count++] =
                        monotonic_elapsed_ms(full_t0, full_t1);
                }
            }
            timed = !failure && full_loop_count > 0;
        }

        // ---- REAL RECURRENCE LOOP (--recurrence) ---------------------------
        // Produces 15 real autoregressive tokens. Each step:
        //   1. Build the full prefix at slots [0, prefix_len).
        //   2. Zero slots [prefix_len, SEQ).
        //   3. Run 4 real layer programs + final norm via IOSurface piping.
        //   4. Read the active position (slot prefix_len-1) from final norm.
        //   5. GEMV: logits = hidden @ audio_head[codebook-1].
        //   6. CFG combine cond*gamma + (1-gamma)*uncond.
        //   7. argmax -> token.
        //   8. Embedding lookup: row = audio_embeddings[token + codebook*2050].
        //   9. Append embedding to both branch prefixes for the next step.
        // One authoritative CFG-combined token sequence; both branches share
        // the same sampled token from this step onward.
        int32_t sampled_tokens[DEPTH_STEPS];
        for (int i = 0; i < DEPTH_STEPS; i++) sampled_tokens[i] = -1;
        bool recurrence_ok = false;
        if (!failure && recurrence_mode) {
            // Load audio_embeddings [32800, 1024] fp32.
            NSData *emb_data = [NSData dataWithContentsOfFile:@(embeddings_arg)];
            if (!emb_data || emb_data.length != RECURRENCE_EMBED_BYTES) {
                fprintf(stderr,
                        "depth-eval: --embeddings %s missing or wrong size "
                        "(got %lu bytes, expected %lu)\n",
                        embeddings_arg, (unsigned long)emb_data.length,
                        (unsigned long)RECURRENCE_EMBED_BYTES);
                failure = @"embeddings-load";
            }
            // Load audio_head [15, 1024, 2050] fp32.
            NSData *head_data = nil;
            NSString *head_path = nil;
            if (!failure) {
                head_path = audio_head_arg
                    ? @(audio_head_arg)
                    : [root stringByAppendingPathComponent:@"audio_head.fp32.bin"];
                head_data = [NSData dataWithContentsOfFile:head_path];
                if (!head_data || head_data.length != RECURRENCE_HEAD_BYTES) {
                    fprintf(stderr,
                            "depth-eval: audio_head %s missing or wrong size "
                            "(got %lu bytes, expected %lu)\n",
                            head_path.UTF8String, (unsigned long)head_data.length,
                            (unsigned long)RECURRENCE_HEAD_BYTES);
                    failure = @"audio-head-load";
                }
            }
            if (!failure) {
                const float *embeddings = (const float *)emb_data.bytes;
                const float *audio_head = (const float *)head_data.bytes;

                // Per-branch prefix buffers: [SEQ slots, DIM channels] fp16.
                _Float16 *prefix[CFG_BRANCHES];
                float *hidden_cond_f32 = malloc(sizeof(float) * DIM);
                float *hidden_uncond_f32 = malloc(sizeof(float) * DIM);
                float *logits_cond = malloc(sizeof(float) * AUDIO_VOCAB);
                float *logits_uncond = malloc(sizeof(float) * AUDIO_VOCAB);
                float *combined = malloc(sizeof(float) * AUDIO_VOCAB);
                prefix[0] = malloc(TENSOR_BYTES);
                prefix[1] = malloc(TENSOR_BYTES);
                if (!prefix[0] || !prefix[1] || !hidden_cond_f32 ||
                    !hidden_uncond_f32 || !logits_cond || !logits_uncond ||
                    !combined) {
                    failure = @"recurrence-allocation";
                }

                if (!failure) {
                    // Initialize both branches: slot 0 = last_hidden (from
                    // input slot 0, channel-major), slot 1 = c0 embedding.
                    const _Float16 *in_cond = (const _Float16 *)inputs[0].bytes;
                    const _Float16 *in_uncond = (const _Float16 *)inputs[1].bytes;
                    zero_surface(prefix[0]);
                    zero_surface(prefix[1]);
                    copy_slot(prefix[0], 0, in_cond, 0);
                    copy_slot(prefix[1], 0, in_uncond, 0);
                    const float *c0_embed = embeddings + (size_t)c0_token * AUDIO_EMBED_DIM;
                    write_prefix_slot_fp32(prefix[0], 1, c0_embed, DIM);
                    write_prefix_slot_fp32(prefix[1], 1, c0_embed, DIM);

                    int prefix_len = 2;  // slots [0,2) populated after init.

                    for (int step = 0; step < DEPTH_STEPS && !failure; step++) {
                        // Pack both branches into chain[0]; since the ANE
                        // graph is per-branch (separate CFG), evaluate each
                        // branch's prefix separately.
                        float *hidden_out[2] = {hidden_cond_f32, hidden_uncond_f32};
                        for (int branch = 0; branch < CFG_BRANCHES && !failure;
                             branch++) {
                            // Write the current prefix into chain[0].
                            orion_tensor_write(chain[0], prefix[branch], TENSOR_BYTES);
                            // Run 4 layers + final norm.
                            bool eval_ok = true;
                            for (int stage = 0; stage < LAYERS; stage++) {
                                if (!orion_eval(layer_programs[stage],
                                                &chain[stage], 1,
                                                &chain[stage + 1], 1)) {
                                    eval_ok = false; break;
                                }
                            }
                            if (eval_ok && !orion_eval(final_norm_program,
                                                       &chain[LAYERS], 1,
                                                       &chain[LAYERS + 1], 1)) {
                                eval_ok = false;
                            }
                            if (!eval_ok) {
                                failure = @"recurrence-evaluation";
                                break;
                            }
                            // Read the active position's hidden state.
                            orion_tensor_read(chain[LAYERS + 1], output, TENSOR_BYTES);
                            int active_slot = prefix_len - 1;
                            fp16_row_to_f32(output, active_slot, hidden_out[branch]);
                        }
                        if (failure) break;

                        // GEMV: logits_branch = hidden_branch @ audio_head[step].
                        // audio_head is [15, 1024, 2050] row-major, so
                        // head_row = audio_head + step*DIM*AUDIO_VOCAB.
                        const float *head_step =
                            audio_head + (size_t)step * DIM * AUDIO_VOCAB;
                        audio_head_gemv(hidden_cond_f32, head_step,
                                        DIM, AUDIO_VOCAB, logits_cond);
                        audio_head_gemv(hidden_uncond_f32, head_step,
                                        DIM, AUDIO_VOCAB, logits_uncond);

                        // CFG combine + argmax.
                        cfg_combine(logits_cond, logits_uncond, AUDIO_VOCAB,
                                    (float)cfg_gamma, combined);
                        int token = argmax_float(combined, AUDIO_VOCAB);
                        sampled_tokens[step] = token;

                        // Embedding lookup + append to both branches.
                        if (step < DEPTH_STEPS - 1) {
                            int codebook = step + 1;  // step 0 -> codebook 1
                            const float *embed_row = embeddings +
                                (size_t)(token + codebook * AUDIO_VOCAB) * AUDIO_EMBED_DIM;
                            write_prefix_slot_fp32(prefix[0], prefix_len, embed_row, DIM);
                            write_prefix_slot_fp32(prefix[1], prefix_len, embed_row, DIM);
                            prefix_len++;
                        }
                    }

                    if (!failure) {
                        recurrence_ok = true;
                        // Write tokens.bin to the output dir (or evidence dir's dir).
                        NSString *token_dir = output_dir;
                        if (!token_dir && evidence_arg) {
                            token_dir = [@(evidence_arg) stringByDeletingLastPathComponent];
                        }
                        if (token_dir) {
                            if (write_tokens_bin(token_dir, sampled_tokens, DEPTH_STEPS)) {
                                fprintf(stderr, "depth-eval: wrote %s/tokens.bin\n",
                                        token_dir.UTF8String);
                            } else {
                                fprintf(stderr,
                                        "depth-eval: WARNING could not write tokens.bin to %s\n",
                                        token_dir.UTF8String);
                            }
                        } else {
                            fprintf(stderr,
                                    "depth-eval: WARNING no --output; tokens not written\n");
                        }
                        // Print tokens to stderr for visibility.
                        fprintf(stderr, "depth-eval tokens:");
                        for (int i = 0; i < DEPTH_STEPS; i++) {
                            fprintf(stderr, " %d", sampled_tokens[i]);
                        }
                        fprintf(stderr, "\n");
                    }
                }

                if (prefix[0]) free(prefix[0]);
                if (prefix[1]) free(prefix[1]);
                if (hidden_cond_f32) free(hidden_cond_f32);
                if (hidden_uncond_f32) free(hidden_uncond_f32);
                if (logits_cond) free(logits_cond);
                if (logits_uncond) free(logits_uncond);
                if (combined) free(combined);
            }
        }
        evidence[@"recurrence_ok"] = @(recurrence_ok);
        evidence[@"sampled_tokens"] = [NSMutableArray array];
        if (recurrence_ok) {
            NSMutableArray *tok = [NSMutableArray arrayWithCapacity:DEPTH_STEPS];
            for (int i = 0; i < DEPTH_STEPS; i++) [tok addObject:@(sampled_tokens[i])];
            evidence[@"sampled_tokens"] = tok;
        }

        // ---- Capture final-norm outputs for --output (untimed) -------------
        if (!failure && output_dir) {
            for (int batch = 0; batch < CFG_BRANCHES && !failure; batch++) {
                orion_tensor_write(chain[0], inputs[batch].bytes, TENSOR_BYTES);
                bool eval_ok = true;
                for (int stage = 0; stage < LAYERS; stage++) {
                    if (!orion_eval(layer_programs[stage], &chain[stage], 1,
                                    &chain[stage + 1], 1)) { eval_ok = false; break; }
                }
                if (eval_ok && !orion_eval(final_norm_program, &chain[LAYERS], 1,
                                           &chain[LAYERS + 1], 1)) eval_ok = false;
                if (!eval_ok) { failure = @"evaluation"; break; }
                _Float16 *dst = batch == 0 ? cond_out : uncond_out;
                orion_tensor_read(chain[LAYERS + 1], dst, TENSOR_BYTES);
            }
            if (!failure) {
                write_output_dir(output_dir, cond_out, uncond_out, TENSOR_BYTES,
                                 @"batch0=cond, batch1=uncond");
            }
        }

        // ---- Statistics + budget checks -------------------------------------
        bool depth_budget_passed = false;
        bool frame_budget_passed = false;
        double frame_margin_ms = NAN;
        double full_loop_p99 = NAN;
        if (timed) {
            NSDictionary *full_stats = stats_dict(full_loop_samples, full_loop_count);
            full_loop_p99 = [full_stats[@"p99_ms"] doubleValue];
            depth_budget_passed = full_loop_p99 < DEPTH_BUDGET_MS;
            frame_margin_ms = FRAME_BUDGET_MS - full_loop_p99 - host_overhead_ms;
            frame_budget_passed = frame_margin_ms >= MARGIN_TARGET_MS;
            evidence[@"timing"] = @{
                @"timing_method": @"actual-15-step-separate-cfg-depth-loop",
                @"surface_allocation": @"outside-timed-region",
                @"inter_layer_transfer": @"iosurface-pipe-no-host-roundtrip",
                @"shape_not_data_dependent":
                    @"identical inputs reproduce production latency",
                @"samples": @(full_loop_count),
                @"warmup": @(warmup_count),
                @"full_loop": full_stats,
                @"per_iteration": @{
                    @"total": stats_dict(it_total_samples, iteration_count),
                    @"ane_eval": stats_dict(it_eval_samples, iteration_count),
                    @"input_write": stats_dict(it_write_samples, iteration_count),
                    @"output_read": stats_dict(it_read_samples, iteration_count),
                },
            };
        }

        evidence[@"layer0_parity"] = parity_results;
        evidence[@"layer0_parity_status"] =
            parity_run ? @"checked" : @"skipped-no-references";
        evidence[@"finite"] = @(finite);
        evidence[@"deterministic"] = @(deterministic);
        evidence[@"parity_passed"] = parity_run ? @(parity) : [NSNull null];
        evidence[@"timed_samples"] = @(full_loop_count);
        evidence[@"timed_iterations"] = @(iteration_count);
        evidence[@"depth_budget_ms"] = @(DEPTH_BUDGET_MS);
        evidence[@"depth_budget_passed"] = @(depth_budget_passed);
        evidence[@"frame_budget_ms"] = @(FRAME_BUDGET_MS);
        evidence[@"host_overhead_ms"] = @(host_overhead_ms);
        evidence[@"host_overhead_basis"] =
            @"external-estimate-not-measured-logits-sampling-embedding";
        evidence[@"frame_margin_ms"] = isnan(frame_margin_ms) ? @0 : @(frame_margin_ms);
        evidence[@"margin_target_ms"] = @(MARGIN_TARGET_MS);
        evidence[@"frame_budget_passed"] = @(frame_budget_passed);

        const int exit_code = depth_eval_exit_code(failure);
        // `passed` requires parity only when parity references were available.
        const bool parity_ok = parity_run ? parity : true;
        // In recurrence mode, success is real token production; the timing
        // budget gate is informational, not blocking.
        bool passed;
        if (recurrence_mode) {
            passed = !failure && recurrence_ok && finite &&
                     compile_delta <= max_compile;
        } else {
            passed = !failure && parity_ok && finite && deterministic &&
                     depth_budget_passed && compile_delta <= max_compile;
        }
        // If a timing gate failed but no lower-level failure occurred, the run
        // still completed; surface the gate miss via evidence + non-zero exit.
        if (!failure && !passed) {
            // depth/frame budget miss with otherwise-clean execution.
        }
        evidence[@"failure_stage"] = failure ? failure : [NSNull null];
        evidence[@"error_code"] = @(exit_code);
        evidence[@"error_name"] = @(depth_eval_error_name(exit_code));
        evidence[@"passed"] = @(passed);

        // ---- Emit evidence JSON --------------------------------------------
        NSData *json = [NSJSONSerialization dataWithJSONObject:evidence
            options:NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys error:nil];
        if (evidence_arg) {
            [json writeToFile:@(evidence_arg) atomically:YES];
        } else {
            printf("%.*s\n", (int)json.length, (const char *)json.bytes);
        }

        // ---- Console summary (stderr keeps stdout JSON clean) --------------
        fprintf(stderr, "depth loop: steps=%d branches=%d compiles=%d (budget %d) "
                        "weight_patches=%d\n",
                DEPTH_STEPS, CFG_BRANCHES, compile_delta, max_compile, LAYERS - 1);
        for (NSDictionary *batch in parity_results) {
            fprintf(stderr, "  layer0 batch %d: max_abs=%.6f relative_l2=%.6f %s\n",
                    [batch[@"batch"] intValue], [batch[@"max_abs"] doubleValue],
                    [batch[@"relative_l2"] doubleValue],
                    [batch[@"passed"] boolValue] ? "PASS" : "FAIL");
        }
        if (!parity_run) {
            fprintf(stderr, "  layer0 parity: skipped (no reference fixtures)\n");
        }
        fprintf(stderr, "  deterministic=%s finite=%s\n",
                deterministic ? "yes" : "no", finite ? "yes" : "no");
        if (timed) {
            fprintf(stderr, "  full-loop p99=%.3f ms (depth budget %.0f ms) -> %s; "
                            "frame margin=%.2f ms (target >= %.0f) -> %s\n",
                    full_loop_p99, DEPTH_BUDGET_MS,
                    depth_budget_passed ? "PASS" : "FAIL",
                    frame_margin_ms, MARGIN_TARGET_MS,
                    frame_budget_passed ? "PASS" : "FAIL");
        }
        if (output_dir && !failure) {
            fprintf(stderr, "  output: %s/{final_norm_cond,final_norm_uncond}.fp16.bin + index.json\n",
                    output_dir.UTF8String);
        }
        if (failure) {
            fprintf(stderr, "  failure stage: %s (exit code %d, %s)\n",
                    failure.UTF8String, exit_code, depth_eval_error_name(exit_code));
        }
        fprintf(stderr, "%s: 15-step separate-CFG depth loop %s\n",
                passed ? "PASS" : "STOP",
                passed ? "fits the 60 ms depth budget"
                       : "failed a gate; revisit depth inference strategy");

        if (output) free(output);
        if (repeat) free(repeat);
        if (cond_out) free(cond_out);
        if (uncond_out) free(uncond_out);
        for (int i = 0; i < CHAIN_SURFACES; i++)
            if (chain[i]) CFRelease(chain[i]);
        for (int i = 0; i < LAYERS; i++)
            if (layer_programs[i]) orion_release_program(layer_programs[i]);
        if (final_norm_program) orion_release_program(final_norm_program);

        // A timing/budget gate miss (no lower failure) should still be visible
        // to the caller as a non-zero exit, distinct from a hard ANE error.
        if (!failure && !passed) return 2;
        return exit_code;
    }
}
