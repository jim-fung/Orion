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
// Wire protocol v1 (one JSON object per line; protocol_version is required):
//
//   REQUEST:  {"protocol_version": 1, "input_dir": "<path>",
//              "output_dir": "<existing-path>", "c0": 42, "gamma": 3.0,
//              "frame_id": 0}
//     input_dir contains hidden_cond.fp16.bin + hidden_uncond.fp16.bin
//     ([1,1024,1,32] fp16, channel-major). output_dir receives tokens.bin.
//     Alternatively, provide both hidden_cond_b64 and hidden_uncond_b64 instead
//     of input_dir. The decoded payloads must each be exactly 65536 bytes.
//
//   RESPONSE: {"protocol_version": 1, "ok": true, "frame_id": 0,
//              "tokens": [t1,...,t15], "eval_ms": 12.3, "compile_count": 5}
//     Errors and the readiness response are likewise versioned.
//
//   SHUTDOWN: {"protocol_version": 1, "cmd": "shutdown"}
//       -> {"protocol_version": 1, "ok": true, "shutdown": true}, then exit.
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
#include <errno.h>
#include <limits.h>
#include <mach/mach_time.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "apps/cli/commands/depth_canonical.h"
#include "apps/cli/commands/depth_serve_support.h"
#include "core/ane_runtime.h"
#include "core/iosurface_tensor.h"
#include "core/orion_error.h"
#include "core/orion_compile_ledger.h"
#include "compiler/codegen.h"
#include "compiler/frontends/voxtream_depth.h"
#include "compiler/pipeline.h"
#include "compiler/validate.h"

#define DIM ORION_VOXTREAM_DEPTH_DIM
#define SEQ ORION_VOXTREAM_DEPTH_ANE_SEQ
#define ELEMENTS ((size_t)DIM * SEQ)
#define TENSOR_BYTES (ELEMENTS * sizeof(_Float16))
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
#define DEPTH_PROTOCOL_VERSION 1
#define MAX_REQUEST_BYTES (1024u * 1024u)
#define RECURRENCE_EMBED_BYTES ((size_t)AUDIO_EMBED_ROWS * AUDIO_EMBED_DIM * sizeof(float))
#define RECURRENCE_HEAD_BYTES  ((size_t)AUDIO_HEAD_STEPS * DIM * AUDIO_VOCAB * sizeof(float))

// ---------------------------------------------------------------------------
// Byte helpers. write_le32 emits little-endian int32 for tokens.bin. The
// canonical aggregate loader lives in the shared depth_canonical.{h,m} TU.
// ---------------------------------------------------------------------------

static void write_le32(uint8_t *dst, uint32_t value) {
    dst[0] = (uint8_t)value;
    dst[1] = (uint8_t)(value >> 8);
    dst[2] = (uint8_t)(value >> 16);
    dst[3] = (uint8_t)(value >> 24);
}

// ---------------------------------------------------------------------------
// Seqpack helpers.
// ---------------------------------------------------------------------------

// Read a per-layer seqpack manifest's chunk-name -> offset map.
static NSDictionary<NSString *, NSNumber *> *seqpack_offsets(NSString *root, int layer) {
    NSString *mp = [root stringByAppendingPathComponent:
        [NSString stringWithFormat:@"weights/layer%d_seqpack.manifest.json", layer]];
    NSData *md = [NSData dataWithContentsOfFile:mp];
    if (!md) return nil;
    NSDictionary *m = [NSJSONSerialization JSONObjectWithData:md options:0 error:nil];
    if (!m || !m[@"chunks"]) return nil;
    NSMutableDictionary *out = [NSMutableDictionary dictionary];
    for (NSDictionary *c in m[@"chunks"]) out[c[@"name"]] = c[@"offset"];
    return out;
}

// Rewrite all BLOBFILE const refs in a seqpack graph to point at layer-N's
// seqpack blob using the manifest's chunk offsets.
static bool rewrite_refs_seqpack(OrionGraph *g, NSString *blob_mil_path,
                                 NSDictionary<NSString *, NSNumber *> *offsets) {
    for (int i = 0; i < g->n_nodes; i++) {
        OrionNode *node = &g->nodes[i];
        if (node->op != ORION_OP_CONST || !node->attrs.blob_path[0]) continue;
        NSString *name = [NSString stringWithUTF8String:node->attrs.blob_path];
        NSString *chunk_name = [name.lastPathComponent stringByDeletingPathExtension];
        NSNumber *offset = offsets[chunk_name];
        if (!offset) {
            fprintf(stderr, "depth-serve: no chunk offset for %s\n",
                    chunk_name.UTF8String);
            return false;
        }
        snprintf(node->attrs.blob_path, sizeof(node->attrs.blob_path),
                 "%s", blob_mil_path.UTF8String);
        node->attrs.blob_offset = [offset unsignedLongLongValue];
    }
    return true;
}

// Generate MIL for a seqpack layer graph (with blob refs rewritten).
static NSString *generate_seqpack_mil(int layer, NSString *root) {
    NSDictionary *offsets = seqpack_offsets(root, layer);
    if (!offsets) {
        fprintf(stderr, "depth-serve: no seqpack manifest for layer %d\n", layer);
        return nil;
    }
    OrionGraph *g = orion_frontend_voxtream_depth_probe_seq_pack_adapter(layer, 0, NULL);
    if (!g) return nil;
    OrionValidationResult vr = orion_graph_validate(g);
    if (!vr.valid) {
        fprintf(stderr, "depth-serve: seqpack graph invalid: %s\n", vr.message);
        orion_graph_free(g);
        return nil;
    }
    orion_pipeline_optimize(g);
    NSString *blob_mil = [NSString stringWithFormat:@"@model_path/weights/layer%d_seqpack.blob", layer];
    if (!rewrite_refs_seqpack(g, blob_mil, offsets)) {
        orion_graph_free(g);
        return nil;
    }
    NSString *mil = orion_codegen_mil(g, "main");
    orion_graph_free(g);
    return mil;
}

// Compile a seqpack layer program from its blob + MIL.
static OrionProgram *compile_seqpack_layer(NSString *root, int layer,
                                           NSString *mil, int *compile_count,
                                           OrionCompileLedger *ledger) {
    NSString *blob_path = [root stringByAppendingPathComponent:
        [NSString stringWithFormat:@"weights/layer%d_seqpack.blob", layer]];
    NSData *blob = [NSData dataWithContentsOfFile:blob_path];
    if (!blob) {
        fprintf(stderr, "depth-serve: missing seqpack blob for layer %d\n", layer);
        return nil;
    }
    if (orion_ledger_exhausted(ledger)) {
        fprintf(stderr, "depth-serve: compile ledger exhausted before seqpack layer %d\n",
                layer);
        return nil;
    }
    NSString *blob_mil = [NSString stringWithFormat:@"@model_path/weights/layer%d_seqpack.blob", layer];
    NSDictionary *wd = @{blob_mil: @{@"offset": @0, @"data": blob}};
    char tag[40];
    snprintf(tag, sizeof(tag), "depth-serve-seqpack-layer%d", layer);
    OrionProgram *program = orion_compile_mil(mil.UTF8String, wd, tag);
    orion_ledger_record_compile(ledger, program != nil, /*cached=*/0);
    if (program) (*compile_count)++;
    return program;
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
    OrionBindingPlan layer_plans[LAYERS];   // separate-CFG canonical layer plans
    OrionProgram *final_norm_program;
    OrionBindingPlan final_norm_plan;       // separate-CFG canonical final-norm plan
    bool final_norm_plan_ready;
    IOSurfaceRef chain[CHAIN_SURFACES];
    _Float16 *output;
    _Float16 *prefix[CFG_BRANCHES];
    _Float16 *packed_prefix;     // seqpack: single [1,DIM,1,64] surface
    float *hidden_cond_f32;
    float *hidden_uncond_f32;
    float *logits_cond;
    float *logits_uncond;
    float *combined;
    const float *embeddings;
    const float *audio_head;
    const _Float16 *final_norm_scale;  // [DIM] fp16, for seqpack host RMSNorm
    NSData *final_norm_data;           // owned backing storage for final_norm_scale
    OrionCompileLedger compile_ledger;
    int compile_count;
    bool ready;
    bool seqpack;                // seqpack mode (packed CFG, halved evals)
    int seq_len;                 // 32 (separate) or 64 (seqpack)
} DepthWorker;

static void depth_worker_cleanup(DepthWorker *w) {
    if (!w) return;
    w->ready = false;
    for (int i = 0; i < LAYERS; i++) {
        if (w->layer_programs[i]) {
            orion_release_program(w->layer_programs[i]);
            w->layer_programs[i] = nil;
        }
    }
    if (w->final_norm_program) {
        orion_release_program(w->final_norm_program);
        w->final_norm_program = nil;
    }
    for (int i = 0; i < LAYERS; i++) w->layer_plans[i] = (OrionBindingPlan){0};
    w->final_norm_plan = (OrionBindingPlan){0};
    w->final_norm_plan_ready = false;
    for (int i = 0; i < CHAIN_SURFACES; i++) {
        if (w->chain[i]) {
            orion_tensor_release(w->chain[i]);
            w->chain[i] = NULL;
        }
    }
    free(w->output); w->output = NULL;
    free(w->prefix[0]); w->prefix[0] = NULL;
    free(w->prefix[1]); w->prefix[1] = NULL;
    free(w->packed_prefix); w->packed_prefix = NULL;
    free(w->hidden_cond_f32); w->hidden_cond_f32 = NULL;
    free(w->hidden_uncond_f32); w->hidden_uncond_f32 = NULL;
    free(w->logits_cond); w->logits_cond = NULL;
    free(w->logits_uncond); w->logits_uncond = NULL;
    free(w->combined); w->combined = NULL;
    w->final_norm_scale = NULL;
    w->final_norm_data = nil;
    w->compile_count = 0;
    memset(&w->compile_ledger, 0, sizeof(w->compile_ledger));
    w->seq_len = 0;
}

static bool depth_worker_compile(DepthWorker *w, NSString *root, int max_compile) {
    const int planned_compiles = w->seqpack ? LAYERS : LAYERS + 1;
    NSString *final_norm_path =
        [root stringByAppendingPathComponent:@"weights/final_norm.bin"];
    w->compile_count = 0;
    w->seq_len = w->seqpack ? (2 * SEQ) : SEQ;
    w->compile_ledger = orion_ledger_init(planned_compiles, max_compile);
    if (max_compile <= 0 || planned_compiles > max_compile) {
        fprintf(stderr, "depth-serve: compile preflight planned=%d exceeds max=%d\n",
                planned_compiles, max_compile);
        depth_worker_cleanup(w);
        return false;
    }

    // Final norm is mandatory. Validate its framing and every scale value before
    // a compile is attempted or readiness becomes observable.
    w->final_norm_data = [NSData dataWithContentsOfFile:final_norm_path];
    OrionDepthServeSupportStatus norm_status = orion_depth_serve_final_norm_scale(
        w->final_norm_data, DIM, &w->final_norm_scale);
    if (norm_status != ORION_DEPTH_SERVE_SUPPORT_OK) {
        fprintf(stderr, "depth-serve: invalid final norm %s: %s\n",
                final_norm_path.UTF8String,
                orion_depth_serve_support_status_string(norm_status));
        depth_worker_cleanup(w);
        return false;
    }
    for (int c = 0; c < DIM; c++) {
        if (!isfinite((float)w->final_norm_scale[c])) {
            fprintf(stderr, "depth-serve: nonfinite final norm scale %s at channel %d\n",
                    final_norm_path.UTF8String, c);
            depth_worker_cleanup(w);
            return false;
        }
    }

    if (w->seqpack) {
        // Seqpack uses checked CPU RMSNorm on seq64 output; compile exactly four layers.
        for (int n = 0; n < LAYERS; n++) {
            NSString *mil_n = generate_seqpack_mil(n, root);
            if (!mil_n) {
                depth_worker_cleanup(w);
                return false;
            }
            w->layer_programs[n] = compile_seqpack_layer(root, n, mil_n,
                                                         &w->compile_count,
                                                         &w->compile_ledger);
            if (!w->layer_programs[n]) {
                depth_worker_cleanup(w);
                return false;
            }
        }
    } else {
        // Separate-CFG: every layer is compiled from its exporter-published
        // canonical v1.5 aggregate blob via a finalized, plan-verified compile
        // (orion_compile_mil_with_plan). The legacy in-memory aggregate builder
        // (build_aggregate_blob) has been removed: an incomplete or non-canonical
        // export fails closed with canonical-aggregate-artifacts-unavailable
        // rather than reconstructing aggregates from raw per-weight blobs.
        const char *revision = getenv("VOXTREAM_ORION_REVISION");
        const char *compiler_build = getenv("VOXTREAM_COMPILER_BUILD");
        OrionBindingPlanIdentity identity = {
            .orion_revision = revision ?: "depth-serve-canonical-artifact",
            .compiler_build = compiler_build ?: "depth-serve-canonical-artifact",
            .weight_root = root.fileSystemRepresentation,
        };

        // Preflight the full canonical contract before any compile is attempted
        // or readiness becomes observable. Older / partly-produced exports never
        // reach graph construction or any legacy fallback path.
        for (int layer = 0; layer < LAYERS; layer++) {
            CanonicalLayerArtifact artifact;
            NSString *artifact_failure = nil;
            if (!depth_canonical_load_layer_artifact(root, layer, &artifact,
                                                     &artifact_failure)) {
                fprintf(stderr,
                        "depth-serve: canonical-aggregate-artifacts-unavailable "
                        "for layer%d: %s\n", layer,
                        artifact_failure.UTF8String ?: "unknown");
                depth_worker_cleanup(w);
                return false;
            }
        }

        for (int n = 0; n < LAYERS; n++) {
            NSString *artifact_failure = nil;
            NSString *layer_mil = nil;
            NSDictionary *layer_wdict = nil;
            char program_name[64];
            snprintf(program_name, sizeof(program_name), "depth-serve-layer%d", n);
            if (!depth_canonical_build_layer(root, n, program_name, &identity,
                    &w->layer_plans[n], &layer_mil, &layer_wdict,
                    &artifact_failure)) {
                fprintf(stderr,
                        "depth-serve: canonical-aggregate-artifacts-unavailable "
                        "layer%d: %s\n", n, artifact_failure.UTF8String ?: "unknown");
                depth_worker_cleanup(w);
                return false;
            }
            if (orion_ledger_exhausted(&w->compile_ledger)) {
                fprintf(stderr, "depth-serve: compile ledger exhausted before layer %d\n", n);
                depth_worker_cleanup(w);
                return false;
            }
            char tag[48], compile_error[512] = {0};
            snprintf(tag, sizeof(tag), "depth-serve-layer%d-canonical", n);
            w->layer_programs[n] = orion_compile_mil_with_plan(layer_mil.UTF8String,
                &w->layer_plans[n], layer_wdict, tag,
                compile_error, sizeof(compile_error));
            orion_ledger_record_compile(&w->compile_ledger,
                                        w->layer_programs[n] != nil, /*cached=*/0);
            if (!w->layer_programs[n]) {
                fprintf(stderr, "depth-serve: layer%d plan-verified compile: %s\n", n,
                        compile_error[0] ? compile_error : "compiler rejected MIL");
                depth_worker_cleanup(w);
                return false;
            }
            w->compile_count++;
        }

        // Final norm: separate canonical single-chunk artifact, plan-verified.
        NSString *final_failure = nil;
        NSString *final_mil = nil;
        if (!depth_canonical_build_final_norm(w->final_norm_data, "depth-serve-final-norm",
                &identity, &w->final_norm_plan, &final_mil, &final_failure)) {
            fprintf(stderr,
                    "depth-serve: canonical-aggregate-artifacts-unavailable "
                    "final-norm: %s\n", final_failure.UTF8String ?: "unknown");
            depth_worker_cleanup(w);
            return false;
        }
        w->final_norm_plan_ready = true;
        if (orion_ledger_exhausted(&w->compile_ledger)) {
            fprintf(stderr, "depth-serve: compile ledger exhausted before final norm\n");
            depth_worker_cleanup(w);
            return false;
        }
        char final_error[512] = {0};
        w->final_norm_program = orion_compile_mil_with_plan(final_mil.UTF8String,
            &w->final_norm_plan,
            @{@"@model_path/weights/final_norm.bin":
                @{@"offset": @0, @"data": w->final_norm_data}},
            "depth-serve-final-norm", final_error, sizeof(final_error));
        orion_ledger_record_compile(&w->compile_ledger,
                                    w->final_norm_program != nil, /*cached=*/0);
        if (!w->final_norm_program) {
            fprintf(stderr, "depth-serve: final-norm plan-verified compile: %s\n",
                    final_error[0] ? final_error : "compiler rejected MIL");
            depth_worker_cleanup(w);
            return false;
        }
        w->compile_count++;
    }

    if (w->compile_count != planned_compiles ||
        w->compile_ledger.planned != planned_compiles ||
        w->compile_ledger.successful != planned_compiles) {
        fprintf(stderr, "depth-serve: compile ledger mismatch planned=%d count=%d ledger-planned=%d successful=%d\n",
                planned_compiles, w->compile_count, w->compile_ledger.planned,
                w->compile_ledger.successful);
        depth_worker_cleanup(w);
        return false;
    }

    int surf_bytes = DIM * w->seq_len * sizeof(_Float16);
    for (int i = 0; i < CHAIN_SURFACES; i++) {
        OrionSurfaceStatus surface_status = ORION_SURFACE_OK;
        if (w->seqpack) {
            // Seqpack has no canonical plans; allocate the fixed [DIM, seq64]
            // surface directly.
            w->chain[i] = orion_tensor_create(DIM, w->seq_len);
        } else if (i == 0) {
            w->chain[i] = orion_tensor_create_for_binding(
                &w->layer_plans[0].inputs[0], &surface_status);
        } else if (i < LAYERS) {
            w->chain[i] = orion_tensor_create_for_binding(
                &w->layer_plans[i - 1].outputs[0], &surface_status);
        } else if (i == LAYERS) {
            w->chain[i] = orion_tensor_create_for_binding(
                &w->layer_plans[LAYERS - 1].outputs[0], &surface_status);
        } else {
            w->chain[i] = orion_tensor_create_for_binding(
                &w->final_norm_plan.outputs[0], &surface_status);
        }
        if (!w->chain[i] || (!w->seqpack && surface_status != ORION_SURFACE_OK)) {
            fprintf(stderr, "depth-serve: surface allocation failed at chain[%d]\n", i);
            depth_worker_cleanup(w);
            return false;
        }
    }
    w->output = malloc(surf_bytes);
    w->prefix[0] = malloc(surf_bytes);
    w->prefix[1] = malloc(surf_bytes);
    if (w->seqpack) w->packed_prefix = malloc(surf_bytes);
    w->hidden_cond_f32 = malloc(sizeof(float) * DIM);
    w->hidden_uncond_f32 = malloc(sizeof(float) * DIM);
    w->logits_cond = malloc(sizeof(float) * AUDIO_VOCAB);
    w->logits_uncond = malloc(sizeof(float) * AUDIO_VOCAB);
    w->combined = malloc(sizeof(float) * AUDIO_VOCAB);
    if (!w->output || !w->prefix[0] || !w->prefix[1] ||
        (w->seqpack && !w->packed_prefix) ||
        !w->hidden_cond_f32 || !w->hidden_uncond_f32 ||
        !w->logits_cond || !w->logits_uncond || !w->combined) {
        fprintf(stderr, "depth-serve: allocation failed during worker startup\n");
        depth_worker_cleanup(w);
        return false;
    }
    w->ready = true;
    return true;
}

// Run one real recurrence frame. Inputs are passed either as base64 in the
// request ("hidden_cond_b64"/"hidden_uncond_b64") OR via input_dir files
// (legacy). Base64 avoids the temp-dir file round-trip (~ms/frame).
// Returns the 15 sampled tokens in `tokens_out` (caller-allocated, 15 ints).
// Returns true on success.
static bool depth_worker_run_frame(DepthWorker *w, NSString *input_dir,
                                   NSData *in_cond, NSData *in_uncond,
                                   int c0_token, double cfg_gamma,
                                   int32_t *tokens_out, double *eval_ms_out) {
    if (!w->ready) return false;
    if (!in_cond || !in_uncond ||
        in_cond.length != TENSOR_BYTES || in_uncond.length != TENSOR_BYTES) {
        return false;
    }
    const _Float16 *cond_src = (const _Float16 *)in_cond.bytes;
    const _Float16 *uncond_src = (const _Float16 *)in_uncond.bytes;

    int surf_bytes = DIM * w->seq_len * sizeof(_Float16);

    if (w->seqpack) {
        // ---- Seqpack: pack both branches into one [1,DIM,1,64] surface ----
        // Branch 0 occupies slots [0,32), branch 1 slots [32,64).
        memset(w->packed_prefix, 0, surf_bytes);
        // Slot 0 = cond last_hidden, slot 32 = uncond last_hidden.
        for (int c = 0; c < DIM; c++) {
            w->packed_prefix[(size_t)c * w->seq_len + 0] = cond_src[(size_t)c * SEQ + 0];
            w->packed_prefix[(size_t)c * w->seq_len + 32] = uncond_src[(size_t)c * SEQ + 0];
        }
        // Slot 1 and slot 33 = c0 embedding (both branches share it).
        const float *c0_embed = w->embeddings + (size_t)c0_token * AUDIO_EMBED_DIM;
        for (int c = 0; c < DIM; c++) {
            _Float16 e = (_Float16)c0_embed[c];
            w->packed_prefix[(size_t)c * w->seq_len + 1] = e;
            w->packed_prefix[(size_t)c * w->seq_len + 33] = e;
        }
        int prefix_len = 2;  // active prefix length per branch

        uint64_t t0 = monotonic_now();
        for (int step = 0; step < DEPTH_STEPS; step++) {
            orion_tensor_write(w->chain[0], w->packed_prefix, surf_bytes);
            bool eval_ok = true;
            for (int stage = 0; stage < LAYERS; stage++) {
                if (!orion_eval(w->layer_programs[stage],
                                &w->chain[stage], 1,
                                &w->chain[stage + 1], 1)) {
                    eval_ok = false; break;
                }
            }
            if (!eval_ok) return false;
            // Read the final layer output (pre-norm) for the active positions.
            orion_tensor_read(w->chain[LAYERS], w->output, surf_bytes);
            int active = prefix_len - 1;
            // Branch 0 active slot: `active`. Branch 1 active slot: `32 + active`.
            // Apply final RMSNorm host-side (the seq32 final-norm graph is
            // incompatible with seq64 surfaces, so we read layer-3 raw output
            // and normalize on the CPU: out = x/rms(x) * scale).
            // Apply the checked CPU RMSNorm to each branch (seq32 ANE norm graph
            // is incompatible with seq64 surfaces). Any helper failure fails the
            // frame: no identity/pre-norm fallback is permitted.
            OrionDepthServeSupportStatus cond_norm_status =
                orion_depth_serve_host_rmsnorm_checked(
                    w->output, DIM, w->seq_len, (size_t)active,
                    w->final_norm_scale, ORION_VOXTREAM_DEPTH_RMS_EPS,
                    w->hidden_cond_f32);
            OrionDepthServeSupportStatus uncond_norm_status =
                orion_depth_serve_host_rmsnorm_checked(
                    w->output, DIM, w->seq_len, (size_t)(32 + active),
                    w->final_norm_scale, ORION_VOXTREAM_DEPTH_RMS_EPS,
                    w->hidden_uncond_f32);
            if (cond_norm_status != ORION_DEPTH_SERVE_SUPPORT_OK ||
                uncond_norm_status != ORION_DEPTH_SERVE_SUPPORT_OK) {
                fprintf(stderr, "depth-serve: seqpack final RMSNorm failed at step %d (%s/%s)\n",
                        step,
                        orion_depth_serve_support_status_string(cond_norm_status),
                        orion_depth_serve_support_status_string(uncond_norm_status));
                return false;
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
                int slot0 = prefix_len;
                int slot1 = 32 + prefix_len;
                for (int c = 0; c < DIM; c++) {
                    _Float16 e = (_Float16)embed_row[c];
                    w->packed_prefix[(size_t)c * w->seq_len + slot0] = e;
                    w->packed_prefix[(size_t)c * w->seq_len + slot1] = e;
                }
                prefix_len++;
            }
        }
        if (eval_ms_out) *eval_ms_out = monotonic_elapsed_ms(t0, monotonic_now());
        return true;
    }

    // ---- Separate CFG (legacy path) -------------------------------------
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
                if (!orion_eval_with_plan(w->layer_programs[stage],
                                          &w->layer_plans[stage],
                                          &w->chain[stage], 1,
                                          &w->chain[stage + 1], 1)) {
                    eval_ok = false; break;
                }
            }
            if (eval_ok && !orion_eval_with_plan(w->final_norm_program,
                                                 &w->final_norm_plan,
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

static void emit_depth_response(NSDictionary *response) {
    NSMutableDictionary *versioned = [response mutableCopy];
    versioned[@"protocol_version"] = @(DEPTH_PROTOCOL_VERSION);
    NSError *error = nil;
    NSData *json = [NSJSONSerialization dataWithJSONObject:versioned options:0 error:&error];
    if (json) {
        fwrite(json.bytes, 1, json.length, stdout);
        fputc('\n', stdout);
    } else {
        // This is a fixed, valid protocol error fallback; never emit diagnostics
        // to stdout, which is reserved exclusively for JSONL responses.
        fprintf(stdout, "{\"protocol_version\":1,\"ok\":false,\"error\":\"response-encode-failed\"}\n");
    }
    fflush(stdout);
}

static void emit_depth_error(NSNumber *frame_id, NSString *error) {
    NSMutableDictionary *response = [@{ @"ok": @NO, @"error": error } mutableCopy];
    if (frame_id) response[@"frame_id"] = frame_id;
    emit_depth_response(response);
}

static bool is_json_number(id value) {
    return [value isKindOfClass:[NSNumber class]] &&
           CFGetTypeID((__bridge CFTypeRef)value) != CFBooleanGetTypeID();
}

static bool json_exact_integer(id value, int64_t min, int64_t max, int64_t *out) {
    if (!is_json_number(value)) return false;
    double number = [value doubleValue];
    if (!isfinite(number) || trunc(number) != number ||
        number < (double)min || number > (double)max) {
        return false;
    }
    if (out) *out = (int64_t)number;
    return true;
}

static bool json_finite_number_in_range(id value, double min, double max,
                                        double *out) {
    if (!is_json_number(value)) return false;
    double number = [value doubleValue];
    if (!isfinite(number) || number < min || number > max) return false;
    if (out) *out = number;
    return true;
}

static bool strict_base64_tensor(NSString *encoded, NSData **decoded_out) {
    if (![encoded isKindOfClass:[NSString class]]) return false;
    NSData *decoded = [[NSData alloc] initWithBase64EncodedString:encoded options:0];
    if (!decoded || decoded.length != TENSOR_BYTES) return false;
    // Require canonical base64 so whitespace, ignored characters, and malformed
    // padding cannot be silently accepted by Foundation's decoder.
    if (![[decoded base64EncodedStringWithOptions:0] isEqualToString:encoded]) return false;
    if (decoded_out) *decoded_out = decoded;
    return true;
}

static bool existing_directory(NSString *path) {
    if (![path isKindOfClass:[NSString class]] || path.length == 0) return false;
    BOOL is_dir = NO;
    return [[NSFileManager defaultManager] fileExistsAtPath:path isDirectory:&is_dir] && is_dir;
}

// Check the entry's type and size before Foundation reads it, preventing a
// request from making the worker ingest a device, FIFO, directory, or huge file.
static bool regular_file_with_exact_size(NSString *path, size_t expected_size) {
    if (![path isKindOfClass:[NSString class]] || path.length == 0) return false;
    struct stat st;
    const char *fs_path = path.fileSystemRepresentation;
    if (!fs_path || stat(fs_path, &st) != 0 || !S_ISREG(st.st_mode) ||
        st.st_size < 0 || (uintmax_t)st.st_size != (uintmax_t)expected_size) {
        return false;
    }
    return true;
}

static bool request_has_exact_keys(NSDictionary *request, NSArray<NSString *> *keys) {
    if (request.count != keys.count) return false;
    for (NSString *key in keys) {
        if (request[key] == nil) return false;
    }
    return true;
}

static bool parse_max_compile(const char *text, int *out) {
    if (!text || !*text || !out) return false;
    errno = 0;
    char *end = NULL;
    long value = strtol(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0' ||
        value < 1 || value > INT_MAX) {
        return false;
    }
    *out = (int)value;
    return true;
}

// Write exactly one complete tokens.bin into an already existing output directory.
static bool write_tokens_bin_checked(NSString *output_dir, const int32_t *tokens) {
    if (!existing_directory(output_dir) || !tokens) return false;
    uint8_t bytes[DEPTH_STEPS * sizeof(int32_t)];
    for (int i = 0; i < DEPTH_STEPS; i++) {
        write_le32(bytes + (size_t)i * sizeof(int32_t), (uint32_t)tokens[i]);
    }
    NSError *error = nil;
    NSString *path = [output_dir stringByAppendingPathComponent:@"tokens.bin"];
    return [[NSData dataWithBytes:bytes length:sizeof(bytes)] writeToFile:path
        options:NSDataWritingAtomic error:&error];
}

static void discard_stdin_line_remainder(void) {
    int ch;
    while ((ch = fgetc(stdin)) != '\n' && ch != EOF) {}
}

// Returns 0 at clean EOF, 1 for a complete request, 2 after discarding an
// overlong request, or 3 for a non-empty unterminated EOF request. `length`
// excludes the newline delimiter.
static int read_bounded_request_line(uint8_t *buffer, size_t *length) {
    size_t len = 0;
    int ch;
    while ((ch = fgetc(stdin)) != EOF) {
        if (ch == '\n') {
            *length = len;
            return 1;
        }
        // Allow one extra byte only for CR in a CRLF-delimited max-size line.
        if (len == MAX_REQUEST_BYTES + 1u) {
            discard_stdin_line_remainder();
            return 2;
        }
        buffer[len++] = (uint8_t)ch;
    }
    if (len == 0) return 0;
    *length = len;
    return 3;
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
        "  --seqpack             Use sequence-packed CFG (halves ANE evals).\n"
        "                         Requires a seqpack export (4 layer_seqpack.blob files).\n"
        "  --help                Show this help message.\n");
}

int orion_cmd_depth_serve(int argc, const char *argv[]) {
    @autoreleasepool {
        const char *export_arg = NULL;
        const char *embeddings_arg = NULL;
        const char *audio_head_arg = NULL;
        int max_compile = DEFAULT_MAX_COMPILE;
        bool seqpack_mode = false;

        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "--export-dir") == 0 && i + 1 < argc) {
                export_arg = argv[++i];
            } else if (strcmp(argv[i], "--embeddings") == 0 && i + 1 < argc) {
                embeddings_arg = argv[++i];
            } else if (strcmp(argv[i], "--audio-head") == 0 && i + 1 < argc) {
                audio_head_arg = argv[++i];
            } else if (strcmp(argv[i], "--max-compile") == 0 && i + 1 < argc) {
                if (!parse_max_compile(argv[++i], &max_compile)) {
                    fprintf(stderr, "depth-serve: --max-compile must be an integer from 1 to %d\n",
                            INT_MAX);
                    return 1;
                }
            } else if (strcmp(argv[i], "--seqpack") == 0) {
                seqpack_mode = true;
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

        // Allocate the bounded request buffer before readiness, so a ready
        // worker always has the storage required to accept a request.
        uint8_t *line = malloc(MAX_REQUEST_BYTES + 1u);  // plus an optional CR
        if (!line) {
            fprintf(stderr, "depth-serve: cannot allocate request buffer\n");
            return 1;
        }

        // Compile programs + allocate surfaces (one-time cost).
        DepthWorker worker = {0};
        worker.seqpack = seqpack_mode;
        worker.embeddings = (const float *)emb_data.bytes;
        worker.audio_head = (const float *)head_data.bytes;
        fprintf(stderr, "depth-serve: compiling programs...\n");
        uint64_t t0 = monotonic_now();
        if (!depth_worker_compile(&worker, root, max_compile)) {
            fprintf(stderr, "depth-serve: compilation failed\n");
            depth_worker_cleanup(&worker);
            free(line);
            return 1;
        }
        fprintf(stderr, "depth-serve: ready, %d compiles in %.0f ms\n",
                worker.compile_count,
                monotonic_elapsed_ms(t0, monotonic_now()));

        // Signal readiness on stdout (the Python side waits for this).
        emit_depth_response(@{ @"ready": @YES,
                               @"compile_count": @(worker.compile_count) });

        // Read bounded JSONL requests. Discard an oversized line completely so
        // malformed input never desynchronizes recovery of later requests.
        size_t len = 0;
        int line_status = 0;
        while ((line_status = read_bounded_request_line(line, &len)) != 0) {
            @autoreleasepool {
                @try {
            if (line_status == 3) {
                emit_depth_error(nil, @"unterminated-request");
                break;
            }
            if (line_status == 2) {
                emit_depth_error(nil, @"request-too-large");
                continue;
            }
            if (len > 0 && line[len - 1] == '\r') len--;
            if (len > MAX_REQUEST_BYTES) {
                emit_depth_error(nil, @"request-too-large");
                continue;
            }
            if (len == 0) {
                emit_depth_error(nil, @"empty-request");
                continue;
            }

            id parsed = [NSJSONSerialization JSONObjectWithData:
                [NSData dataWithBytes:line length:len] options:0 error:nil];
            if (![parsed isKindOfClass:[NSDictionary class]]) {
                emit_depth_error(nil, @"invalid-json-object");
                continue;
            }
            NSDictionary *req = parsed;

            // Every request, including shutdown, must explicitly negotiate v1.
            int64_t protocol_version = 0;
            if (!json_exact_integer(req[@"protocol_version"], DEPTH_PROTOCOL_VERSION,
                                    DEPTH_PROTOCOL_VERSION, &protocol_version)) {
                emit_depth_error(nil, @"unsupported-protocol-version");
                continue;
            }

            id cmd = req[@"cmd"];
            if (cmd) {
                if (![cmd isKindOfClass:[NSString class]] ||
                    ![(NSString *)cmd isEqualToString:@"shutdown"] ||
                    !request_has_exact_keys(req, @[@"protocol_version", @"cmd"])) {
                    emit_depth_error(nil, @"invalid-command");
                    continue;
                }
                emit_depth_response(@{ @"ok": @YES, @"shutdown": @YES });
                break;
            }

            bool has_input_dir = req[@"input_dir"] != nil;
            bool has_cond_b64 = req[@"hidden_cond_b64"] != nil;
            bool has_uncond_b64 = req[@"hidden_uncond_b64"] != nil;
            bool has_b64_pair = has_cond_b64 && has_uncond_b64;
            if ((has_input_dir && (has_cond_b64 || has_uncond_b64)) ||
                (!has_input_dir && !has_b64_pair) ||
                (has_cond_b64 != has_uncond_b64)) {
                emit_depth_error(nil, @"invalid-hidden-input-method");
                continue;
            }
            NSArray<NSString *> *frame_keys = has_input_dir
                ? @[@"protocol_version", @"input_dir", @"output_dir", @"c0", @"gamma", @"frame_id"]
                : @[@"protocol_version", @"hidden_cond_b64", @"hidden_uncond_b64",
                    @"output_dir", @"c0", @"gamma", @"frame_id"];
            if (!request_has_exact_keys(req, frame_keys)) {
                emit_depth_error(nil, @"invalid-request-schema");
                continue;
            }

            int64_t frame_id_value = 0;
            if (!json_exact_integer(req[@"frame_id"], INT32_MIN, INT32_MAX,
                                    &frame_id_value)) {
                emit_depth_error(nil, @"invalid-frame-id");
                continue;
            }
            NSNumber *frame_id = @((int32_t)frame_id_value);

            int64_t c0 = 0;
            if (!json_exact_integer(req[@"c0"], 0, AUDIO_VOCAB - 1, &c0)) {
                emit_depth_error(frame_id, @"invalid-c0");
                continue;
            }

            double gamma = 0.0;
            if (!json_finite_number_in_range(req[@"gamma"], 0.0, 100.0, &gamma)) {
                emit_depth_error(frame_id, @"invalid-gamma");
                continue;
            }

            NSString *output_dir = req[@"output_dir"];
            if (!existing_directory(output_dir)) {
                emit_depth_error(frame_id, @"invalid-output-dir");
                continue;
            }

            id input_dir_value = req[@"input_dir"];
            id cond_b64_value = req[@"hidden_cond_b64"];
            id uncond_b64_value = req[@"hidden_uncond_b64"];

            NSString *input_dir = nil;
            NSData *in_cond = nil;
            NSData *in_uncond = nil;
            if (has_input_dir) {
                if (!existing_directory(input_dir_value)) {
                    emit_depth_error(frame_id, @"invalid-input-dir");
                    continue;
                }
                input_dir = input_dir_value;
                NSString *cond_path = [input_dir stringByAppendingPathComponent:
                    @"hidden_cond.fp16.bin"];
                NSString *uncond_path = [input_dir stringByAppendingPathComponent:
                    @"hidden_uncond.fp16.bin"];
                if (!regular_file_with_exact_size(cond_path, TENSOR_BYTES) ||
                    !regular_file_with_exact_size(uncond_path, TENSOR_BYTES)) {
                    emit_depth_error(frame_id, @"invalid-hidden-tensor-files");
                    continue;
                }
                in_cond = [NSData dataWithContentsOfFile:cond_path];
                in_uncond = [NSData dataWithContentsOfFile:uncond_path];
                if (in_cond.length != TENSOR_BYTES || in_uncond.length != TENSOR_BYTES) {
                    emit_depth_error(frame_id, @"invalid-hidden-tensor-files");
                    continue;
                }
            } else if (!strict_base64_tensor(cond_b64_value, &in_cond) ||
                       !strict_base64_tensor(uncond_b64_value, &in_uncond)) {
                emit_depth_error(frame_id, @"invalid-hidden-tensor-base64");
                continue;
            }

            int32_t tokens[DEPTH_STEPS];
            double eval_ms = 0.0;
            if (!depth_worker_run_frame(&worker, input_dir, in_cond, in_uncond,
                                        (int)c0, gamma, tokens, &eval_ms)) {
                emit_depth_error(frame_id, @"eval-failed");
                continue;
            }
            if (!write_tokens_bin_checked(output_dir, tokens)) {
                emit_depth_error(frame_id, @"output-write-failed");
                continue;
            }

            NSMutableArray *tok_arr = [NSMutableArray arrayWithCapacity:DEPTH_STEPS];
            for (int i = 0; i < DEPTH_STEPS; i++) [tok_arr addObject:@(tokens[i])];
            emit_depth_response(@{
                @"ok": @YES,
                @"frame_id": frame_id,
                @"tokens": tok_arr,
                @"eval_ms": @(eval_ms),
                @"compile_count": @(worker.compile_count),
            });
                } @catch (NSException *exception) {
                    (void)exception;
                    emit_depth_error(nil, @"request-handler-failed");
                }
            }
        }

        // Cleanup: release programs/surfaces/buffers through the shared helper
        // so the normal path uses the same release sequence as startup failures.
        depth_worker_cleanup(&worker);

        if (line) free(line);
        return 0;
    }
}
