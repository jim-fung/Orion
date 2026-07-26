// apps/cli/commands/depth_serve.m
//
// orion depth-serve — PERSISTENT depth worker for VoXtream.
//
// Compiles the four depth-layer programs + final norm ONCE at startup, then
// reads newline-delimited JSON frame requests from stdin and writes newline-
// delimited JSON responses to stdout. One process serves many frames,
// eliminating the ~3s/frame subprocess + recompile overhead of one-shot
// `depth-eval` invocations.
//
// Wire protocol (one JSON object per line):
//
//   REQUEST:  {"input_dir": "<path>", "output_dir": "<path>", "c0": 42,
//              "gamma": 3.0, "frame_id": 0}
//     input_dir contains hidden_cond.fp16.bin + hidden_uncond.fp16.bin
//     ([1,1024,1,32] fp16, channel-major). output_dir receives tokens.bin.
//
//   RESPONSE: {"ok": true, "frame_id": 0, "tokens": [t1,...,t15],
//              "eval_ms": 12.3, "compile_count": 5}
//     On error: {"ok": false, "frame_id": 0, "error": "..."}
//
//   SHUTDOWN: {"cmd": "shutdown"}  ->  {"ok": true, "shutdown": true}, then exit.
//
// Usage:
//   orion depth-serve --export-dir <path>
//                     --embeddings <audio_embeddings.fp32.bin>
//                     [--audio-head <audio_head.fp32.bin>]
//                     [--max-compile N]
//
// Building this command consumes ZERO ANE compile budget; the worker compiles
// on first connect.

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
#define DEPTH_STEPS 15
#define CFG_BRANCHES 2
#define LAYERS ORION_VOXTREAM_DEPTH_LAYERS
#define CHAIN_SURFACES (LAYERS + 2)
#define NUM_CODEBOOKS 16
#define AUDIO_VOCAB 2050
#define AUDIO_EMBED_ROWS (NUM_CODEBOOKS * AUDIO_VOCAB)
#define AUDIO_EMBED_DIM DIM
#define AUDIO_HEAD_STEPS 15
#define DEFAULT_CFG_GAMMA 3.0
#define DEFAULT_MAX_COMPILE 20
#define RECURRENCE_EMBED_BYTES ((size_t)AUDIO_EMBED_ROWS * AUDIO_EMBED_DIM * sizeof(float))
#define RECURRENCE_HEAD_BYTES  ((size_t)AUDIO_HEAD_STEPS * DIM * AUDIO_VOCAB * sizeof(float))

// ---------------------------------------------------------------------------
// Aggregate blob chunk table (same as depth_eval.m).
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
// Byte helpers (ported from depth_eval.m).
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

static size_t align64(size_t value) {
    return (value + 63u) & ~(size_t)63u;
}

static NSData *read_valid_single_blob(NSString *path, size_t expected_payload) {
    NSData *data = [NSData dataWithContentsOfFile:path];
    if (!data || data.length != BLOB_HEADER_BYTES + expected_payload) return nil;
    const uint8_t *bytes = data.bytes;
    if (bytes[0] != 1 || bytes[4] != 2 ||
        bytes[64] != 0xef || bytes[65] != 0xbe ||
        bytes[66] != 0xad || bytes[67] != 0xde || bytes[68] != 1 ||
        read_le32(bytes + 72) != (uint32_t)expected_payload || read_le32(bytes + 80) != 128) {
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
            fprintf(stderr, "depth-serve: cannot read validated blob %s\n", relative);
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

static bool rewrite_blob_refs(OrionGraph *g, int layer) {
    char layer_mil[32][256];
    const char *prefix = "weights/layer0/";
    for (int j = 0; j < kEntryCount; j++) {
        const char *path = kEntries[j].mil_path;
        if (strncmp(path + strlen("@model_path/"), prefix, strlen(prefix)) == 0) {
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
                break;
            }
        }
        if (!found) {
            fprintf(stderr, "depth-serve: graph references unaggregated blob %s\n",
                    node->attrs.blob_path);
            return false;
        }
    }
    return true;
}

static NSString *generate_aggregate_mil(OrionGraph *g, int layer) {
    if (!g) return nil;
    OrionValidationResult vr = orion_graph_validate(g);
    if (!vr.valid) {
        fprintf(stderr, "depth-serve: graph validation: %s\n", vr.message);
        orion_graph_free(g);
        return nil;
    }
    orion_pipeline_optimize(g);
    if (!rewrite_blob_refs(g, layer)) {
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
// Recurrence helpers (ported from depth_eval.m).
// ---------------------------------------------------------------------------

static void fp16_row_to_f32(const _Float16 *surface, int slot, float *out) {
    for (int c = 0; c < DIM; c++) {
        out[c] = (float)surface[(size_t)c * SEQ + slot];
    }
}

static void audio_head_gemv(const float *hidden, const float *head,
                            int D, int V, float *out) {
    cblas_sgemv(CblasRowMajor, CblasTrans, D, V, 1.0f,
                head, V, hidden, 1, 0.0f, out, 1);
}

static void cfg_combine(const float *cond, const float *uncond,
                        int V, float gamma, float *out) {
    float w1 = (float)gamma;
    float w2 = 1.0f - w1;
    vDSP_vsmul(cond, 1, &w1, out, 1, (vDSP_Length)V);
    vDSP_vsma(uncond, 1, &w2, out, 1, out, 1, (vDSP_Length)V);
}

static int argmax_float(const float *v, int V) {
    int best = 0;
    float best_val = v[0];
    for (int i = 1; i < V; i++) {
        if (v[i] > best_val) { best_val = v[i]; best = i; }
    }
    return best;
}

static void write_prefix_slot_fp32(_Float16 *surface, int slot,
                                   const float *row_f32, int dim) {
    for (int c = 0; c < dim; c++) {
        surface[(size_t)c * SEQ + slot] = (_Float16)row_f32[c];
    }
}

static void copy_slot(_Float16 *dst_surface, int dst_slot,
                      const _Float16 *src_surface, int src_slot) {
    for (int c = 0; c < DIM; c++) {
        dst_surface[(size_t)c * SEQ + dst_slot] =
            src_surface[(size_t)c * SEQ + src_slot];
    }
}

static void zero_surface(_Float16 *surface) {
    memset(surface, 0, TENSOR_BYTES);
}

static uint64_t monotonic_now(void) { return mach_absolute_time(); }

static double monotonic_elapsed_ms(uint64_t start, uint64_t end) {
    static mach_timebase_info_data_t info;
    static dispatch_once_t once;
    dispatch_once(&once, ^{ mach_timebase_info(&info); });
    return (double)(end - start) * (double)info.numer /
           (double)info.denom / 1e6;
}

// ---------------------------------------------------------------------------
// Worker state: compiled programs + persistent surfaces + embeddings/head.
// ---------------------------------------------------------------------------

typedef struct {
    OrionProgram *layer_programs[LAYERS];
    OrionProgram *final_norm_program;
    IOSurfaceRef chain[CHAIN_SURFACES];
    _Float16 *output;
    _Float16 *prefix[CFG_BRANCHES];
    float *hidden_cond_f32;
    float *hidden_uncond_f32;
    float *logits_cond;
    float *logits_uncond;
    float *combined;
    const float *embeddings;
    const float *audio_head;
    int compile_count;
    bool ready;
} DepthWorker;

static bool depth_worker_compile(DepthWorker *w, NSString *root, int max_compile) {
    for (int i = 0; i < LAYERS; i++) w->layer_programs[i] = nil;
    w->final_norm_program = nil;
    w->compile_count = 0;

    // Compile each of the 4 real layer programs.
    for (int n = 0; n < LAYERS; n++) {
        NSData *aggregate_n = build_aggregate_blob(root, n);
        if (!aggregate_n) return false;
        NSString *mil_n = generate_aggregate_mil(
            orion_frontend_voxtream_depth_layer(n), n);
        if (!mil_n) return false;
        char tag[32];
        snprintf(tag, sizeof(tag), "depth-serve-layer%d", n);
        w->compile_count++;
        w->layer_programs[n] = orion_compile_mil(mil_n.UTF8String,
                                                  wdict(aggregate_n), tag);
        if (!w->layer_programs[n]) return false;
    }

    // Final norm.
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
    if (!final_mil) return false;
    w->compile_count++;
    w->final_norm_program = orion_compile_mil(final_mil.UTF8String,
        @{@"@model_path/weights/final_norm.bin":
            @{@"offset": @0, @"data": final_blob}},
        "depth-serve-final-norm");
    if (!w->final_norm_program) return false;

    if (w->compile_count > max_compile) {
        fprintf(stderr, "depth-serve: compile count %d > max %d\n",
                w->compile_count, max_compile);
        return false;
    }

    // Allocate persistent IOSurface chain + scratch buffers.
    for (int i = 0; i < CHAIN_SURFACES; i++) {
        w->chain[i] = orion_tensor_create(DIM, SEQ);
        if (!w->chain[i]) return false;
    }
    w->output = malloc(TENSOR_BYTES);
    w->prefix[0] = malloc(TENSOR_BYTES);
    w->prefix[1] = malloc(TENSOR_BYTES);
    w->hidden_cond_f32 = malloc(sizeof(float) * DIM);
    w->hidden_uncond_f32 = malloc(sizeof(float) * DIM);
    w->logits_cond = malloc(sizeof(float) * AUDIO_VOCAB);
    w->logits_uncond = malloc(sizeof(float) * AUDIO_VOCAB);
    w->combined = malloc(sizeof(float) * AUDIO_VOCAB);
    if (!w->output || !w->prefix[0] || !w->prefix[1] ||
        !w->hidden_cond_f32 || !w->hidden_uncond_f32 ||
        !w->logits_cond || !w->logits_uncond || !w->combined) {
        return false;
    }
    w->ready = true;
    return true;
}

// Run one real recurrence frame. Inputs read from input_dir/hidden_*.fp16.bin
// (channel-major, slot 0 = last_hidden). Tokens written to output_dir/tokens.bin.
// Returns the 15 sampled tokens in `tokens_out` (caller-allocated, 15 ints).
// Returns true on success.
static bool depth_worker_run_frame(DepthWorker *w, NSString *input_dir,
                                   int c0_token, double cfg_gamma,
                                   int32_t *tokens_out, double *eval_ms_out) {
    if (!w->ready) return false;

    // Read input hidden states.
    NSData *in_cond = [NSData dataWithContentsOfFile:
        [input_dir stringByAppendingPathComponent:@"hidden_cond.fp16.bin"]];
    NSData *in_uncond = [NSData dataWithContentsOfFile:
        [input_dir stringByAppendingPathComponent:@"hidden_uncond.fp16.bin"]];
    if (!in_cond || !in_uncond ||
        in_cond.length != TENSOR_BYTES || in_uncond.length != TENSOR_BYTES) {
        return false;
    }
    const _Float16 *cond_src = (const _Float16 *)in_cond.bytes;
    const _Float16 *uncond_src = (const _Float16 *)in_uncond.bytes;

    // Initialize prefixes: slot 0 = last_hidden, slot 1 = c0 embedding.
    zero_surface(w->prefix[0]);
    zero_surface(w->prefix[1]);
    copy_slot(w->prefix[0], 0, cond_src, 0);
    copy_slot(w->prefix[1], 0, uncond_src, 0);
    const float *c0_embed = w->embeddings + (size_t)c0_token * AUDIO_EMBED_DIM;
    write_prefix_slot_fp32(w->prefix[0], 1, c0_embed, DIM);
    write_prefix_slot_fp32(w->prefix[1], 1, c0_embed, DIM);
    int prefix_len = 2;

    uint64_t t0 = monotonic_now();
    for (int step = 0; step < DEPTH_STEPS; step++) {
        float *hidden_out[2] = {w->hidden_cond_f32, w->hidden_uncond_f32};
        for (int branch = 0; branch < CFG_BRANCHES; branch++) {
            orion_tensor_write(w->chain[0], w->prefix[branch], TENSOR_BYTES);
            bool eval_ok = true;
            for (int stage = 0; stage < LAYERS; stage++) {
                if (!orion_eval(w->layer_programs[stage],
                                &w->chain[stage], 1,
                                &w->chain[stage + 1], 1)) {
                    eval_ok = false; break;
                }
            }
            if (eval_ok && !orion_eval(w->final_norm_program,
                                       &w->chain[LAYERS], 1,
                                       &w->chain[LAYERS + 1], 1)) {
                eval_ok = false;
            }
            if (!eval_ok) return false;
            orion_tensor_read(w->chain[LAYERS + 1], w->output, TENSOR_BYTES);
            fp16_row_to_f32(w->output, prefix_len - 1, hidden_out[branch]);
        }
        const float *head_step = w->audio_head + (size_t)step * DIM * AUDIO_VOCAB;
        audio_head_gemv(w->hidden_cond_f32, head_step, DIM, AUDIO_VOCAB, w->logits_cond);
        audio_head_gemv(w->hidden_uncond_f32, head_step, DIM, AUDIO_VOCAB, w->logits_uncond);
        cfg_combine(w->logits_cond, w->logits_uncond, AUDIO_VOCAB,
                    (float)cfg_gamma, w->combined);
        int token = argmax_float(w->combined, AUDIO_VOCAB);
        tokens_out[step] = token;
        if (step < DEPTH_STEPS - 1) {
            int codebook = step + 1;
            const float *embed_row = w->embeddings +
                (size_t)(token + codebook * AUDIO_VOCAB) * AUDIO_EMBED_DIM;
            write_prefix_slot_fp32(w->prefix[0], prefix_len, embed_row, DIM);
            write_prefix_slot_fp32(w->prefix[1], prefix_len, embed_row, DIM);
            prefix_len++;
        }
    }
    if (eval_ms_out) *eval_ms_out = monotonic_elapsed_ms(t0, monotonic_now());
    return true;
}

static void print_depth_serve_help(void) {
    fprintf(stderr,
        "Usage: orion depth-serve --export-dir <path> --embeddings <path> [options]\n"
        "\n"
        "Persistent depth worker: compiles once, serves many frames via stdin/stdout JSON.\n"
        "\n"
        "Options:\n"
        "  --export-dir PATH     VoXtream v16 export directory (required).\n"
        "  --embeddings PATH     audio_embeddings raw fp32 [32800,1024] (required).\n"
        "  --audio-head PATH     audio_head raw fp32 [15,1024,2050] (default: <export>/audio_head.fp32.bin).\n"
        "  --max-compile N       Abort if startup compile count > N (default 20).\n"
        "  --help                Show this help message.\n");
}

int orion_cmd_depth_serve(int argc, const char *argv[]) {
    @autoreleasepool {
        const char *export_arg = NULL;
        const char *embeddings_arg = NULL;
        const char *audio_head_arg = NULL;
        int max_compile = DEFAULT_MAX_COMPILE;

        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "--export-dir") == 0 && i + 1 < argc) {
                export_arg = argv[++i];
            } else if (strcmp(argv[i], "--embeddings") == 0 && i + 1 < argc) {
                embeddings_arg = argv[++i];
            } else if (strcmp(argv[i], "--audio-head") == 0 && i + 1 < argc) {
                audio_head_arg = argv[++i];
            } else if (strcmp(argv[i], "--max-compile") == 0 && i + 1 < argc) {
                max_compile = atoi(argv[++i]);
            } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
                print_depth_serve_help();
                return 0;
            } else {
                fprintf(stderr, "depth-serve: unknown argument '%s'\n\n", argv[i]);
                print_depth_serve_help();
                return 1;
            }
        }

        if (!export_arg || !embeddings_arg) {
            fprintf(stderr, "depth-serve: --export-dir and --embeddings are required\n");
            return 1;
        }

        NSString *root = @(export_arg);
        NSString *head_path = audio_head_arg
            ? @(audio_head_arg)
            : [root stringByAppendingPathComponent:@"audio_head.fp32.bin"];

        if (!orion_ane_init()) {
            fprintf(stderr, "depth-serve: ANE runtime unavailable\n");
            return 1;
        }

        // Load embeddings + audio head.
        NSData *emb_data = [NSData dataWithContentsOfFile:@(embeddings_arg)];
        NSData *head_data = [NSData dataWithContentsOfFile:head_path];
        if (!emb_data || emb_data.length != RECURRENCE_EMBED_BYTES) {
            fprintf(stderr, "depth-serve: embeddings missing or wrong size\n");
            return 1;
        }
        if (!head_data || head_data.length != RECURRENCE_HEAD_BYTES) {
            fprintf(stderr, "depth-serve: audio_head missing or wrong size\n");
            return 1;
        }

        // Compile programs + allocate surfaces (one-time cost).
        DepthWorker worker = {0};
        worker.embeddings = (const float *)emb_data.bytes;
        worker.audio_head = (const float *)head_data.bytes;
        fprintf(stderr, "depth-serve: compiling programs...\n");
        uint64_t t0 = monotonic_now();
        if (!depth_worker_compile(&worker, root, max_compile)) {
            fprintf(stderr, "depth-serve: compilation failed\n");
            return 1;
        }
        fprintf(stderr, "depth-serve: ready, %d compiles in %.0f ms\n",
                worker.compile_count,
                monotonic_elapsed_ms(t0, monotonic_now()));

        // Signal readiness on stdout (the Python side waits for this).
        printf("{\"ready\":true,\"compile_count\":%d}\n", worker.compile_count);
        fflush(stdout);

        // Frame request loop: read JSON lines from stdin, respond on stdout.
        char line[8192];
        while (fgets(line, sizeof(line), stdin)) {
            // Strip trailing newline.
            size_t len = strlen(line);
            while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) {
                line[--len] = '\0';
            }
            if (len == 0) continue;

            NSData *line_data = [NSData dataWithBytes:line length:len];
            NSDictionary *req = [NSJSONSerialization JSONObjectWithData:line_data
                options:0 error:nil];
            if (!req || ![req isKindOfClass:[NSDictionary class]]) {
                printf("{\"ok\":false,\"error\":\"invalid-json\"}\n");
                fflush(stdout);
                continue;
            }

            // Shutdown command.
            if ([req[@"cmd"] isEqualToString:@"shutdown"]) {
                printf("{\"ok\":true,\"shutdown\":true}\n");
                fflush(stdout);
                break;
            }

            NSString *input_dir = req[@"input_dir"];
            NSNumber *c0_num = req[@"c0"];
            NSNumber *frame_id = req[@"frame_id"];
            double gamma = req[@"gamma"] ? [req[@"gamma"] doubleValue] : DEFAULT_CFG_GAMMA;
            int frame_id_int = frame_id ? [frame_id intValue] : -1;

            if (!input_dir || !c0_num) {
                printf("{\"ok\":false,\"frame_id\":%d,\"error\":\"missing-input_dir-or-c0\"}\n",
                       frame_id_int);
                fflush(stdout);
                continue;
            }

            int32_t tokens[DEPTH_STEPS];
            double eval_ms = 0.0;
            bool ok = depth_worker_run_frame(&worker, input_dir,
                                             [c0_num intValue], gamma,
                                             tokens, &eval_ms);
            if (!ok) {
                printf("{\"ok\":false,\"frame_id\":%d,\"error\":\"eval-failed\"}\n",
                       frame_id_int);
                fflush(stdout);
                continue;
            }

            // Write tokens.bin to input_dir (so the Python side knows where to find it).
            NSString *tokens_path = [input_dir stringByAppendingPathComponent:@"tokens.bin"];
            NSMutableData *tdata = [NSMutableData dataWithCapacity:DEPTH_STEPS * 4];
            for (int i = 0; i < DEPTH_STEPS; i++) {
                uint8_t le[4] = {
                    (uint8_t)((uint32_t)tokens[i] & 0xFF),
                    (uint8_t)(((uint32_t)tokens[i] >> 8) & 0xFF),
                    (uint8_t)(((uint32_t)tokens[i] >> 16) & 0xFF),
                    (uint8_t)(((uint32_t)tokens[i] >> 24) & 0xFF),
                };
                [tdata appendBytes:le length:4];
            }
            [tdata writeToFile:tokens_path atomically:YES];

            // Build response with token array.
            NSMutableArray *tok_arr = [NSMutableArray arrayWithCapacity:DEPTH_STEPS];
            for (int i = 0; i < DEPTH_STEPS; i++) [tok_arr addObject:@(tokens[i])];
            NSDictionary *resp = @{
                @"ok": @(YES),
                @"frame_id": @(frame_id_int),
                @"tokens": tok_arr,
                @"eval_ms": @(eval_ms),
                @"compile_count": @(worker.compile_count),
            };
            NSData *j = [NSJSONSerialization dataWithJSONObject:resp
                options:0 error:nil];
            printf("%.*s\n", (int)j.length, (const char *)j.bytes);
            fflush(stdout);
        }

        // Cleanup.
        if (worker.output) free(worker.output);
        if (worker.prefix[0]) free(worker.prefix[0]);
        if (worker.prefix[1]) free(worker.prefix[1]);
        if (worker.hidden_cond_f32) free(worker.hidden_cond_f32);
        if (worker.hidden_uncond_f32) free(worker.hidden_uncond_f32);
        if (worker.logits_cond) free(worker.logits_cond);
        if (worker.logits_uncond) free(worker.logits_uncond);
        if (worker.combined) free(worker.combined);
        for (int i = 0; i < CHAIN_SURFACES; i++) {
            if (worker.chain[i]) CFRelease(worker.chain[i]);
        }
        for (int i = 0; i < LAYERS; i++) {
            if (worker.layer_programs[i]) orion_release_program(worker.layer_programs[i]);
        }
        if (worker.final_norm_program) orion_release_program(worker.final_norm_program);

        return 0;
    }
}
