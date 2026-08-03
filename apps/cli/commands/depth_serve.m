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
// Wire protocol v2 is defined normatively in docs/depth_serve_protocol.md.
// Launch requires --launch-challenge. Every frame carries strict correlation,
// tensor, semantic-request, and program-set identities and returns a chained
// operational receipt with exact private-evaluate counters.
//
// Usage:
//   orion depth-serve --export-dir <path>
//                     --embeddings <audio_embeddings.fp32.bin>
//                     --launch-challenge <lowercase-64-hex>
//                     [--audio-head <audio_head.fp32.bin>]
//                     [--max-compile N]
//                     [--seqpack]
//                     [--phase-timings]
//
// Building this command consumes ZERO ANE compile budget; the worker compiles
// on first connect.

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Accelerate/Accelerate.h>
#import <CommonCrypto/CommonDigest.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <mach/mach_time.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

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
#define DEPTH_PROTOCOL_VERSION 2
#define DEPTH_RECEIPT_SCHEMA @"voxtream.orion.depth-operational-receipt/2"
#define DEPTH_PHASE_TIMINGS_SCHEMA @"voxtream.orion.depth-phase-timings/1"
#define ZERO_SHA256 @"0000000000000000000000000000000000000000000000000000000000000000"
#define MAX_SAFE_JSON_INTEGER INT64_C(9007199254740991)
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
static OrionProgram *compile_seqpack_layer(int layer, NSString *mil, NSData *blob,
                                           int *compile_count,
                                           OrionCompileLedger *ledger) {
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

static uint64_t monotonic_ns(void) {
    static mach_timebase_info_data_t info;
    static dispatch_once_t once;
    dispatch_once(&once, ^{ mach_timebase_info(&info); });
    uint64_t ticks = monotonic_now();
    __uint128_t ns = (__uint128_t)ticks * info.numer / info.denom;
    return ns > UINT64_MAX ? UINT64_MAX : (uint64_t)ns;
}

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
    NSString *program_set_digest;
    NSMutableArray<NSDictionary *> *program_identities;
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
    w->program_set_digest = nil;
    w->program_identities = nil;
    w->compile_count = 0;
    memset(&w->compile_ledger, 0, sizeof(w->compile_ledger));
    w->seq_len = 0;
}

static bool depth_worker_compile(DepthWorker *w, NSString *root, int max_compile,
                                 NSData *emb_data, NSData *head_data) {
    const int planned_compiles = w->seqpack ? LAYERS : LAYERS + 1;
    w->program_identities = [NSMutableArray arrayWithCapacity:LAYERS];
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
            NSString *blob_path = [root stringByAppendingPathComponent:
                [NSString stringWithFormat:@"weights/layer%d_seqpack.blob", n]];
            NSData *blob_n = [NSData dataWithContentsOfFile:blob_path];
            NSString *mil_sha = mil_n ? orion_depth_serve_sha256_data(
                [mil_n dataUsingEncoding:NSUTF8StringEncoding]) : nil;
            NSString *blob_sha = orion_depth_serve_sha256_data(blob_n);
            if (!mil_n || !blob_n || !mil_sha || !blob_sha) {
                depth_worker_cleanup(w);
                return false;
            }
            w->layer_programs[n] = compile_seqpack_layer(n, mil_n, blob_n,
                                                         &w->compile_count,
                                                         &w->compile_ledger);
            if (!w->layer_programs[n]) {
                depth_worker_cleanup(w);
                return false;
            }
            [w->program_identities addObject:@{
                @"name": [NSString stringWithFormat:@"layer%d", n],
                @"mil_sha256": mil_sha,
                @"weights_sha256": blob_sha,
            }];
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
            NSString *weights_sha = orion_depth_serve_plan_weights_sha256(
                &w->layer_plans[n]);
            NSString *mil_sha = @(w->layer_plans[n].mil_identity);
            if (!weights_sha || !orion_depth_serve_is_lower_hex_sha256(mil_sha)) {
                fprintf(stderr, "depth-serve: layer%d program identity unavailable\n", n);
                depth_worker_cleanup(w);
                return false;
            }
            [w->program_identities addObject:@{
                @"name": [NSString stringWithFormat:@"layer%d", n],
                @"mil_sha256": mil_sha,
                @"weights_sha256": weights_sha,
            }];
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

    NSString *final_norm_program_sha = w->seqpack
        ? orion_depth_serve_sha256_data([@"cpu-checked-rmsnorm-v1"
            dataUsingEncoding:NSUTF8StringEncoding])
        : @(w->final_norm_plan.mil_identity);
    NSString *final_norm_sha = orion_depth_serve_sha256_data(w->final_norm_data);
    NSString *embeddings_sha = orion_depth_serve_sha256_data(emb_data);
    NSString *audio_head_sha = orion_depth_serve_sha256_data(head_data);
    w->program_set_digest = orion_depth_serve_program_set_sha256(
        w->seqpack ? @"seqpack" : @"separate", w->program_identities,
        w->seqpack ? @"cpu-checked-rmsnorm" : @"ane-private",
        final_norm_program_sha, final_norm_sha, embeddings_sha, audio_head_sha,
        @"cpu-accelerate", @"cpu-vdsp", @"cpu-argmax");
    if (!w->program_set_digest) {
        fprintf(stderr, "depth-serve: stable program-set identity unavailable\n");
        depth_worker_cleanup(w);
        return false;
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
            // FP16 surface with the metadata required by checked I/O.
            w->chain[i] = orion_tensor_create_checked(
                ORION_SURFACE_DTYPE_FP16, DIM, w->seq_len,
                (size_t)surf_bytes, &surface_status);
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
        if (!w->chain[i] || surface_status != ORION_SURFACE_OK) {
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

typedef enum {
    DEPTH_FRAME_FAILURE_NONE = 0,
    DEPTH_FRAME_FAILURE_PRIVATE_EVALUATION,
    DEPTH_FRAME_FAILURE_EVALUATION_PREPARATION,
    DEPTH_FRAME_FAILURE_HOST_RMSNORM,
} DepthFrameFailure;

typedef struct {
    uint64_t surface_write_ns;
    uint64_t private_evaluation_ns;
    uint64_t private_evaluation_by_program_ns[LAYERS + 1];
    uint64_t surface_read_ns;
    uint64_t host_rmsnorm_ns;
    uint64_t audio_head_ns;
    uint64_t cfg_combine_ns;
    uint64_t argmax_ns;
    uint64_t embedding_update_ns;
    uint64_t token_encoding_ns;
    uint64_t output_write_ns;
} DepthPhaseTimings;

typedef struct {
    uint32_t expected[LAYERS + 1];
    uint32_t actual[LAYERS + 1];
    uint32_t successful[LAYERS + 1];
    uint32_t failed[LAYERS + 1];
} DepthEvalCounters;

static uint64_t phase_start(const DepthPhaseTimings *timings) {
    return timings ? monotonic_ns() : 0;
}

static void phase_accumulate(DepthPhaseTimings *timings, uint64_t start,
                             uint64_t *field) {
    if (!timings || !field) return;
    *field += monotonic_ns() - start;
}

static void depth_eval_counters_init(DepthEvalCounters *counters, bool seqpack) {
    memset(counters, 0, sizeof(*counters));
    for (int i = 0; i < LAYERS; i++) counters->expected[i] = seqpack ? DEPTH_STEPS : DEPTH_STEPS * CFG_BRANCHES;
    counters->expected[LAYERS] = seqpack ? 0 : DEPTH_STEPS * CFG_BRANCHES;
}

static bool depth_eval_counted(DepthWorker *w, int program_index,
                               IOSurfaceRef *input, IOSurfaceRef *output,
                               DepthEvalCounters *counters,
                               DepthPhaseTimings *timings,
                               DepthFrameFailure *failure) {
    uint64_t eval_start = phase_start(timings);
    OrionEvalResult result = program_index < LAYERS
        ? (w->seqpack
            ? orion_eval_report(w->layer_programs[program_index], input, 1, output, 1)
            : orion_eval_with_plan_report(w->layer_programs[program_index],
                                          &w->layer_plans[program_index], input, 1,
                                          output, 1))
        : orion_eval_with_plan_report(w->final_norm_program, &w->final_norm_plan,
                                      input, 1, output, 1);
    if (timings) {
        uint64_t elapsed = monotonic_ns() - eval_start;
        timings->private_evaluation_ns += elapsed;
        timings->private_evaluation_by_program_ns[program_index] += elapsed;
    }
    if (result == ORION_EVAL_NOT_INVOKED) {
        if (failure) *failure = DEPTH_FRAME_FAILURE_EVALUATION_PREPARATION;
        return false;
    }
    counters->actual[program_index]++;
    if (result == ORION_EVAL_PRIVATE_SUCCEEDED) {
        counters->successful[program_index]++;
        return true;
    }
    counters->failed[program_index]++;
    if (failure) *failure = DEPTH_FRAME_FAILURE_PRIVATE_EVALUATION;
    return false;
}

// Run one real recurrence frame. Inputs are passed either as base64 in the
// request ("hidden_cond_b64"/"hidden_uncond_b64") OR via input_dir files.
// Returns the 15 sampled tokens and exact private evaluate counters.
static bool depth_worker_run_frame(DepthWorker *w,
                                   NSData *in_cond, NSData *in_uncond,
                                   int c0_token, double cfg_gamma,
                                   int32_t *tokens_out,
                                   DepthEvalCounters *counters,
                                   DepthPhaseTimings *timings,
                                   DepthFrameFailure *failure) {
    if (failure) *failure = DEPTH_FRAME_FAILURE_NONE;
    if (!w->ready || !counters || !failure) return false;
    depth_eval_counters_init(counters, w->seqpack);
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

        for (int step = 0; step < DEPTH_STEPS; step++) {
            uint64_t write_start = phase_start(timings);
            OrionSurfaceStatus write_status = orion_tensor_write_packed_checked(
                w->chain[0], ORION_SURFACE_DTYPE_FP16, w->packed_prefix,
                (size_t)surf_bytes);
            if (timings)
                phase_accumulate(timings, write_start, &timings->surface_write_ns);
            if (write_status != ORION_SURFACE_OK) {
                *failure = DEPTH_FRAME_FAILURE_EVALUATION_PREPARATION;
                return false;
            }
            bool eval_ok = true;
            for (int stage = 0; stage < LAYERS; stage++) {
                if (!depth_eval_counted(w, stage, &w->chain[stage],
                                        &w->chain[stage + 1], counters, timings,
                                        failure)) {
                    eval_ok = false; break;
                }
            }
            if (!eval_ok) return false;
            // Read the final layer output (pre-norm) for the active positions.
            uint64_t read_start = phase_start(timings);
            OrionSurfaceStatus read_status = orion_tensor_read_packed_checked(
                w->chain[LAYERS], ORION_SURFACE_DTYPE_FP16, w->output,
                (size_t)surf_bytes);
            if (timings)
                phase_accumulate(timings, read_start, &timings->surface_read_ns);
            if (read_status != ORION_SURFACE_OK) {
                *failure = DEPTH_FRAME_FAILURE_EVALUATION_PREPARATION;
                return false;
            }
            int active = prefix_len - 1;
            // Branch 0 active slot: `active`. Branch 1 active slot: `32 + active`.
            // Apply final RMSNorm host-side (the seq32 final-norm graph is
            // incompatible with seq64 surfaces, so we read layer-3 raw output
            // and normalize on the CPU: out = x/rms(x) * scale).
            // Apply the checked CPU RMSNorm to each branch (seq32 ANE norm graph
            // is incompatible with seq64 surfaces). Any helper failure fails the
            // frame: no identity/pre-norm fallback is permitted.
            uint64_t rmsnorm_start = phase_start(timings);
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
            if (timings)
                phase_accumulate(timings, rmsnorm_start, &timings->host_rmsnorm_ns);
            if (cond_norm_status != ORION_DEPTH_SERVE_SUPPORT_OK ||
                uncond_norm_status != ORION_DEPTH_SERVE_SUPPORT_OK) {
                *failure = DEPTH_FRAME_FAILURE_HOST_RMSNORM;
                fprintf(stderr, "depth-serve: seqpack final RMSNorm failed at step %d (%s/%s)\n",
                        step,
                        orion_depth_serve_support_status_string(cond_norm_status),
                        orion_depth_serve_support_status_string(uncond_norm_status));
                return false;
            }

            const float *head_step = w->audio_head + (size_t)step * DIM * AUDIO_VOCAB;
            uint64_t audio_head_start = phase_start(timings);
            audio_head_gemv(w->hidden_cond_f32, head_step, DIM, AUDIO_VOCAB, w->logits_cond);
            audio_head_gemv(w->hidden_uncond_f32, head_step, DIM, AUDIO_VOCAB, w->logits_uncond);
            if (timings)
                phase_accumulate(timings, audio_head_start, &timings->audio_head_ns);
            uint64_t cfg_start = phase_start(timings);
            cfg_combine(w->logits_cond, w->logits_uncond, AUDIO_VOCAB,
                        (float)cfg_gamma, w->combined);
            if (timings)
                phase_accumulate(timings, cfg_start, &timings->cfg_combine_ns);
            uint64_t argmax_start = phase_start(timings);
            int token = argmax_float(w->combined, AUDIO_VOCAB);
            if (timings)
                phase_accumulate(timings, argmax_start, &timings->argmax_ns);
            tokens_out[step] = token;
            if (step < DEPTH_STEPS - 1) {
                uint64_t embedding_start = phase_start(timings);
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
                if (timings)
                    phase_accumulate(timings, embedding_start,
                                     &timings->embedding_update_ns);
            }
        }
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

    for (int step = 0; step < DEPTH_STEPS; step++) {
        float *hidden_out[2] = {w->hidden_cond_f32, w->hidden_uncond_f32};
        for (int branch = 0; branch < CFG_BRANCHES; branch++) {
            uint64_t write_start = phase_start(timings);
            OrionSurfaceStatus write_status = orion_tensor_write_packed_checked(
                w->chain[0], ORION_SURFACE_DTYPE_FP16, w->prefix[branch],
                TENSOR_BYTES);
            if (timings)
                phase_accumulate(timings, write_start, &timings->surface_write_ns);
            if (write_status != ORION_SURFACE_OK) {
                *failure = DEPTH_FRAME_FAILURE_EVALUATION_PREPARATION;
                return false;
            }
            bool eval_ok = true;
            for (int stage = 0; stage < LAYERS; stage++) {
                if (!depth_eval_counted(w, stage, &w->chain[stage],
                                        &w->chain[stage + 1], counters, timings,
                                        failure)) {
                    eval_ok = false; break;
                }
            }
            if (eval_ok && !depth_eval_counted(w, LAYERS, &w->chain[LAYERS],
                                               &w->chain[LAYERS + 1], counters,
                                               timings, failure)) {
                eval_ok = false;
            }
            if (!eval_ok) return false;
            uint64_t read_start = phase_start(timings);
            OrionSurfaceStatus read_status = orion_tensor_read_packed_checked(
                w->chain[LAYERS + 1], ORION_SURFACE_DTYPE_FP16, w->output,
                TENSOR_BYTES);
            if (timings)
                phase_accumulate(timings, read_start, &timings->surface_read_ns);
            if (read_status != ORION_SURFACE_OK) {
                *failure = DEPTH_FRAME_FAILURE_EVALUATION_PREPARATION;
                return false;
            }
            fp16_row_to_f32(w->output, prefix_len - 1, hidden_out[branch]);
        }
        const float *head_step = w->audio_head + (size_t)step * DIM * AUDIO_VOCAB;
        uint64_t audio_head_start = phase_start(timings);
        audio_head_gemv(w->hidden_cond_f32, head_step, DIM, AUDIO_VOCAB, w->logits_cond);
        audio_head_gemv(w->hidden_uncond_f32, head_step, DIM, AUDIO_VOCAB, w->logits_uncond);
        if (timings)
            phase_accumulate(timings, audio_head_start, &timings->audio_head_ns);
        uint64_t cfg_start = phase_start(timings);
        cfg_combine(w->logits_cond, w->logits_uncond, AUDIO_VOCAB,
                    (float)cfg_gamma, w->combined);
        if (timings)
            phase_accumulate(timings, cfg_start, &timings->cfg_combine_ns);
        uint64_t argmax_start = phase_start(timings);
        int token = argmax_float(w->combined, AUDIO_VOCAB);
        if (timings)
            phase_accumulate(timings, argmax_start, &timings->argmax_ns);
        tokens_out[step] = token;
        if (step < DEPTH_STEPS - 1) {
            uint64_t embedding_start = phase_start(timings);
            int codebook = step + 1;
            const float *embed_row = w->embeddings +
                (size_t)(token + codebook * AUDIO_VOCAB) * AUDIO_EMBED_DIM;
            write_prefix_slot_fp32(w->prefix[0], prefix_len, embed_row, DIM);
            write_prefix_slot_fp32(w->prefix[1], prefix_len, embed_row, DIM);
            prefix_len++;
            if (timings)
                phase_accumulate(timings, embedding_start,
                                 &timings->embedding_update_ns);
        }
    }
    return true;
}

static NSData *serialize_depth_response(NSDictionary *response) {
    if (![response isKindOfClass:[NSDictionary class]]) return nil;
    NSMutableDictionary *versioned = [response mutableCopy];
    versioned[@"protocol_version"] = @(DEPTH_PROTOCOL_VERSION);
    NSError *error = nil;
    NSData *json = [NSJSONSerialization dataWithJSONObject:versioned options:0 error:&error];
    if (!json || error) return nil;
    NSMutableData *line = [json mutableCopy];
    const uint8_t newline = '\n';
    [line appendBytes:&newline length:1];
    return line;
}

static bool emit_serialized_depth_response(NSData *line) {
    if (!line || line.length == 0) return false;
    const uint8_t *bytes = line.bytes;
    size_t offset = 0;
    while (offset < line.length) {
        size_t written = fwrite(bytes + offset, 1, line.length - offset, stdout);
        if (written == 0) return false;
        offset += written;
    }
    return fflush(stdout) == 0 && !ferror(stdout);
}

static bool emit_depth_response(NSDictionary *response) {
    NSData *line = serialize_depth_response(response);
    return line && emit_serialized_depth_response(line);
}

static bool emit_depth_error(NSDictionary *correlation, NSString *error) {
    NSMutableDictionary *response = [@{ @"ok": @NO, @"error": error } mutableCopy];
    if ([correlation isKindOfClass:[NSDictionary class]])
        [response addEntriesFromDictionary:correlation];
    return emit_depth_response(response);
}

static void require_depth_emission(bool emitted) {
    if (emitted) return;
    fprintf(stderr, "depth-serve: protocol response write failed; terminating worker\n");
    _Exit(EXIT_FAILURE);
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
    if (out) *out = number == 0.0 ? 0.0 : number;
    return true;
}

static bool strict_protocol_string(id value) {
    if (![value isKindOfClass:[NSString class]]) return false;
    NSString *text = value;
    NSData *utf8 = [text dataUsingEncoding:NSUTF8StringEncoding allowLossyConversion:NO];
    return utf8.length >= 1 && utf8.length <= 128 &&
           [text rangeOfString:@"\0"].location == NSNotFound;
}

static NSMutableDictionary *validated_correlation(NSDictionary *request) {
    NSMutableDictionary *correlation = [NSMutableDictionary dictionary];
    if (strict_protocol_string(request[@"stream_session_id"]))
        correlation[@"stream_session_id"] = request[@"stream_session_id"];
    int64_t number = 0;
    if (json_exact_integer(request[@"logical_frame_id"], 0, MAX_SAFE_JSON_INTEGER, &number))
        correlation[@"logical_frame_id"] = @(number);
    if (strict_protocol_string(request[@"request_id"]))
        correlation[@"request_id"] = request[@"request_id"];
    if (json_exact_integer(request[@"request_sequence"], 0, MAX_SAFE_JSON_INTEGER, &number))
        correlation[@"request_sequence"] = @(number);
    return correlation;
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

static bool read_exact_regular_file(NSString *path, size_t expected_size,
                                    NSData **data_out) {
    if (data_out) *data_out = nil;
    if (![path isKindOfClass:[NSString class]] || path.length == 0 || !data_out)
        return false;
    const char *fs_path = path.fileSystemRepresentation;
    if (!fs_path) return false;
    int flags = O_RDONLY | O_CLOEXEC;
#ifdef O_NOFOLLOW
    flags |= O_NOFOLLOW;
#endif
    int fd = open(fs_path, flags);
    if (fd < 0) return false;
    bool ok = false;
    NSMutableData *data = nil;
    struct stat st;
    if (fstat(fd, &st) != 0 || !S_ISREG(st.st_mode) || st.st_size < 0 ||
        (uintmax_t)st.st_size != (uintmax_t)expected_size) goto done;
    data = [NSMutableData dataWithLength:expected_size];
    if (!data) goto done;
    uint8_t *bytes = data.mutableBytes;
    size_t offset = 0;
    while (offset < expected_size) {
        ssize_t count = read(fd, bytes + offset, expected_size - offset);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) goto done;
        offset += (size_t)count;
    }
    // Even if the initial fstat matched, reject a concurrently extended file.
    uint8_t trailing = 0;
    for (;;) {
        ssize_t count = read(fd, &trailing, 1);
        if (count < 0 && errno == EINTR) continue;
        if (count != 0) goto done;
        break;
    }
    *data_out = data;
    ok = true;
done:
    close(fd);
    return ok;
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

static NSData *tokens_data(const int32_t *tokens) {
    if (!tokens) return nil;
    uint8_t bytes[DEPTH_STEPS * sizeof(int32_t)];
    for (int i = 0; i < DEPTH_STEPS; i++)
        write_le32(bytes + (size_t)i * sizeof(int32_t), (uint32_t)tokens[i]);
    return [NSData dataWithBytes:bytes length:sizeof(bytes)];
}

// Write exactly one complete tokens.bin into an already existing output directory.
static bool write_tokens_bin_checked(NSString *output_dir, NSData *data) {
    if (!existing_directory(output_dir) || data.length != DEPTH_STEPS * sizeof(int32_t)) return false;
    NSError *error = nil;
    NSString *path = [output_dir stringByAppendingPathComponent:@"tokens.bin"];
    return [data writeToFile:path options:NSDataWritingAtomic error:&error];
}

static NSDictionary *eval_counts_dictionary(const DepthEvalCounters *counters) {
    NSMutableDictionary *result = [NSMutableDictionary dictionaryWithCapacity:LAYERS + 1];
    for (int i = 0; i <= LAYERS; i++) {
        NSString *name = i < LAYERS ? [NSString stringWithFormat:@"layer%d", i] : @"final_norm";
        result[name] = @{
            @"expected": @(counters->expected[i]),
            @"actual": @(counters->actual[i]),
            @"successful": @(counters->successful[i]),
            @"failed": @(counters->failed[i]),
        };
    }
    return result;
}

static NSDictionary *phase_timings_dictionary(const DepthPhaseTimings *timings) {
    if (!timings) return nil;
    NSMutableDictionary *by_program = [NSMutableDictionary dictionaryWithCapacity:LAYERS + 1];
    for (int i = 0; i <= LAYERS; i++) {
        NSString *name = i < LAYERS
            ? [NSString stringWithFormat:@"layer%d", i]
            : @"final_norm";
        by_program[name] = @(timings->private_evaluation_by_program_ns[i]);
    }
    return @{
        @"surface_write_ns": @(timings->surface_write_ns),
        @"private_evaluation_ns": @(timings->private_evaluation_ns),
        @"private_evaluation_by_program_ns": by_program,
        @"surface_read_ns": @(timings->surface_read_ns),
        @"host_rmsnorm_ns": @(timings->host_rmsnorm_ns),
        @"audio_head_ns": @(timings->audio_head_ns),
        @"cfg_combine_ns": @(timings->cfg_combine_ns),
        @"argmax_ns": @(timings->argmax_ns),
        @"embedding_update_ns": @(timings->embedding_update_ns),
        @"token_encoding_ns": @(timings->token_encoding_ns),
        @"output_write_ns": @(timings->output_write_ns),
    };
}

static bool configure_phase_timing_stderr(void) {
    int flags = fcntl(STDERR_FILENO, F_GETFL);
    if (flags < 0) return false;
    if (fcntl(STDERR_FILENO, F_SETFL, flags | O_NONBLOCK) != 0) return false;
    if (fcntl(STDERR_FILENO, F_SETNOSIGPIPE, 1) == 0) return true;
    (void)fcntl(STDERR_FILENO, F_SETFL, flags);
    return false;
}

static void emit_phase_timing_diagnostic(DepthPhaseTimings *timings,
                                         NSString *worker_session_id,
                                         NSDictionary *request,
                                         bool seqpack,
                                         bool ok,
                                         NSString *receipt_sha256,
                                         uint64_t frame_duration_ns) {
    if (!timings) return;
    @try {
        NSDictionary *diagnostic = @{
            @"diagnostic_schema": DEPTH_PHASE_TIMINGS_SCHEMA,
            @"clock": @"mach-absolute-time",
            @"worker_session_id": worker_session_id,
            @"stream_session_id": request[@"stream_session_id"],
            @"logical_frame_id": request[@"logical_frame_id"],
            @"request_id": request[@"request_id"],
            @"request_sequence": request[@"request_sequence"],
            @"mode": seqpack ? @"seqpack" : @"separate",
            @"ok": @(ok),
            @"receipt_sha256": receipt_sha256,
            @"frame_duration_ns": @(frame_duration_ns),
            @"phase_duration_ns": phase_timings_dictionary(timings),
        };
        NSError *error = nil;
        NSData *json = [NSJSONSerialization dataWithJSONObject:diagnostic options:0 error:&error];
        if (!json || error) return;
        NSMutableData *line = [json mutableCopy];
        const uint8_t newline = '\n';
        [line appendBytes:&newline length:1];
        const uint8_t *bytes = line.bytes;
        ssize_t written = write(STDERR_FILENO, bytes, line.length);
        (void)written;
    } @catch (__unused NSException *exception) {
        // Phase timing is best-effort and must never affect protocol lifetime.
    }
}

static uint32_t eval_count_total(const uint32_t values[LAYERS + 1]) {
    uint32_t total = 0;
    for (int i = 0; i <= LAYERS; i++) total += values[i];
    return total;
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
        "  --launch-challenge H  Required fresh lowercase 64-hex host challenge.\n"
        "  --max-compile N       Abort if startup compile count > N (default 20).\n"
        "  --seqpack             Use sequence-packed CFG (halves ANE evals).\n"
        "                         Requires a seqpack export (4 layer_seqpack.blob files).\n"
        "  --phase-timings       Emit correlated phase timing JSON to stderr.\n"
        "  --help                Show this help message.\n");
}

int orion_cmd_depth_serve(int argc, const char *argv[]) {
    @autoreleasepool {
        const char *export_arg = NULL;
        const char *embeddings_arg = NULL;
        const char *audio_head_arg = NULL;
        const char *launch_challenge_arg = NULL;
        int max_compile = DEFAULT_MAX_COMPILE;
        bool seqpack_mode = false;
        bool phase_timings_enabled = false;

        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "--export-dir") == 0 && i + 1 < argc) {
                export_arg = argv[++i];
            } else if (strcmp(argv[i], "--embeddings") == 0 && i + 1 < argc) {
                embeddings_arg = argv[++i];
            } else if (strcmp(argv[i], "--audio-head") == 0 && i + 1 < argc) {
                audio_head_arg = argv[++i];
            } else if (strcmp(argv[i], "--launch-challenge") == 0 && i + 1 < argc) {
                launch_challenge_arg = argv[++i];
            } else if (strcmp(argv[i], "--max-compile") == 0 && i + 1 < argc) {
                if (!parse_max_compile(argv[++i], &max_compile)) {
                    fprintf(stderr, "depth-serve: --max-compile must be an integer from 1 to %d\n",
                            INT_MAX);
                    return 1;
                }
            } else if (strcmp(argv[i], "--seqpack") == 0) {
                seqpack_mode = true;
            } else if (strcmp(argv[i], "--phase-timings") == 0) {
                phase_timings_enabled = true;
            } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
                print_depth_serve_help();
                return 0;
            } else {
                fprintf(stderr, "depth-serve: unknown argument '%s'\n\n", argv[i]);
                print_depth_serve_help();
                return 1;
            }
        }

        if (phase_timings_enabled && !configure_phase_timing_stderr()) {
            fprintf(stderr, "depth-serve: cannot configure safe phase timing stderr; disabling diagnostics\n");
            phase_timings_enabled = false;
        }

        if (!export_arg || !embeddings_arg || !launch_challenge_arg) {
            fprintf(stderr, "depth-serve: --export-dir, --embeddings, and --launch-challenge are required\n");
            return 1;
        }
        NSString *launch_challenge = @(launch_challenge_arg);
        if (!orion_depth_serve_is_lower_hex_sha256(launch_challenge)) {
            fprintf(stderr, "depth-serve: --launch-challenge must be lowercase 64-hex\n");
            return 1;
        }
        NSString *worker_session_id = [NSUUID UUID].UUIDString.lowercaseString;
        uint64_t start_monotonic_ns = monotonic_ns();

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
        if (!depth_worker_compile(&worker, root, max_compile, emb_data, head_data)) {
            fprintf(stderr, "depth-serve: compilation failed\n");
            depth_worker_cleanup(&worker);
            free(line);
            return 1;
        }
        fprintf(stderr, "depth-serve: ready, %d compiles in %.0f ms\n",
                worker.compile_count,
                monotonic_elapsed_ms(t0, monotonic_now()));

        // Signal strict v2 readiness only after all startup identities exist.
        require_depth_emission(emit_depth_response(@{
            @"ready": @YES,
            @"receipt_schema": DEPTH_RECEIPT_SCHEMA,
            @"worker_session_id": worker_session_id,
            @"launch_challenge": launch_challenge,
            @"pid": @((uint64_t)getpid()),
            @"start_monotonic_ns": @(start_monotonic_ns),
            @"mode": seqpack_mode ? @"seqpack" : @"separate",
            @"startup_compile_count": @(worker.compile_count),
            @"expected_private_evals_per_frame": @(seqpack_mode ? 60 : 150),
            @"final_norm_placement": seqpack_mode ? @"cpu-checked-rmsnorm" : @"ane-private",
            @"audio_head_placement": @"cpu-accelerate",
            @"cfg_placement": @"cpu-vdsp",
            @"sampler_placement": @"cpu-argmax",
            @"program_set_digest": worker.program_set_digest,
        }));
        NSString *previous_receipt_sha256 = ZERO_SHA256;

        // Read bounded JSONL requests. Discard an oversized line completely so
        // malformed input never desynchronizes recovery of later requests.
        size_t len = 0;
        int line_status = 0;
        while ((line_status = read_bounded_request_line(line, &len)) != 0) {
            @autoreleasepool {
            bool accepted_frame = false;
                @try {
            if (line_status == 3) {
                require_depth_emission(emit_depth_error(nil, @"unterminated-request"));
                break;
            }
            if (line_status == 2) {
                require_depth_emission(emit_depth_error(nil, @"request-too-large"));
                continue;
            }
            if (len > 0 && line[len - 1] == '\r') len--;
            if (len > MAX_REQUEST_BYTES) {
                require_depth_emission(emit_depth_error(nil, @"request-too-large"));
                continue;
            }
            if (len == 0) {
                require_depth_emission(emit_depth_error(nil, @"empty-request"));
                continue;
            }

            id parsed = [NSJSONSerialization JSONObjectWithData:
                [NSData dataWithBytes:line length:len] options:0 error:nil];
            if (![parsed isKindOfClass:[NSDictionary class]]) {
                require_depth_emission(emit_depth_error(nil, @"invalid-json-object"));
                continue;
            }
            NSDictionary *req = parsed;

            // Every request, including shutdown, must explicitly negotiate v2.
            int64_t protocol_version = 0;
            if (!json_exact_integer(req[@"protocol_version"], DEPTH_PROTOCOL_VERSION,
                                    DEPTH_PROTOCOL_VERSION, &protocol_version)) {
                require_depth_emission(emit_depth_error(nil, @"unsupported-protocol-version"));
                continue;
            }

            id cmd = req[@"cmd"];
            if (cmd) {
                if (![cmd isKindOfClass:[NSString class]] ||
                    ![(NSString *)cmd isEqualToString:@"shutdown"] ||
                    !request_has_exact_keys(req, @[@"protocol_version", @"cmd"])) {
                    require_depth_emission(emit_depth_error(nil, @"invalid-command"));
                    continue;
                }
                require_depth_emission(emit_depth_response(@{ @"ok": @YES, @"shutdown": @YES,
                                       @"worker_session_id": worker_session_id }));
                break;
            }

            bool has_input_dir = req[@"input_dir"] != nil;
            bool has_cond_b64 = req[@"hidden_cond_b64"] != nil;
            bool has_uncond_b64 = req[@"hidden_uncond_b64"] != nil;
            bool has_b64_pair = has_cond_b64 && has_uncond_b64;
            if ((has_input_dir && (has_cond_b64 || has_uncond_b64)) ||
                (!has_input_dir && !has_b64_pair) ||
                (has_cond_b64 != has_uncond_b64)) {
                require_depth_emission(emit_depth_error(validated_correlation(req), @"invalid-hidden-input-method"));
                continue;
            }
            NSArray<NSString *> *frame_keys = has_input_dir
                ? @[@"protocol_version", @"input_dir", @"output_dir", @"c0", @"gamma",
                    @"stream_session_id", @"logical_frame_id", @"request_id",
                    @"request_sequence", @"challenge_nonce", @"hidden_cond_sha256",
                    @"hidden_uncond_sha256", @"semantic_request_sha256",
                    @"program_set_digest"]
                : @[@"protocol_version", @"hidden_cond_b64", @"hidden_uncond_b64",
                    @"output_dir", @"c0", @"gamma", @"stream_session_id",
                    @"logical_frame_id", @"request_id", @"request_sequence",
                    @"challenge_nonce", @"hidden_cond_sha256", @"hidden_uncond_sha256",
                    @"semantic_request_sha256", @"program_set_digest"];
            if (!request_has_exact_keys(req, frame_keys)) {
                require_depth_emission(emit_depth_error(validated_correlation(req), @"invalid-request-schema"));
                continue;
            }

            NSMutableDictionary *correlation = validated_correlation(req);
            if (!strict_protocol_string(req[@"stream_session_id"])) {
                require_depth_emission(emit_depth_error(correlation, @"invalid-stream-session-id"));
                continue;
            }
            int64_t logical_frame_id = 0;
            if (!json_exact_integer(req[@"logical_frame_id"], 0, MAX_SAFE_JSON_INTEGER,
                                    &logical_frame_id)) {
                require_depth_emission(emit_depth_error(correlation, @"invalid-logical-frame-id"));
                continue;
            }
            if (!strict_protocol_string(req[@"request_id"])) {
                require_depth_emission(emit_depth_error(correlation, @"invalid-request-id"));
                continue;
            }
            int64_t request_sequence = 0;
            if (!json_exact_integer(req[@"request_sequence"], 0, MAX_SAFE_JSON_INTEGER,
                                    &request_sequence)) {
                require_depth_emission(emit_depth_error(correlation, @"invalid-request-sequence"));
                continue;
            }
            NSString *challenge_nonce = req[@"challenge_nonce"];
            if (!orion_depth_serve_is_lower_hex_sha256(challenge_nonce) ||
                [challenge_nonce isEqualToString:launch_challenge]) {
                require_depth_emission(emit_depth_error(correlation, @"invalid-challenge-nonce"));
                continue;
            }
            int64_t c0 = 0;
            if (!json_exact_integer(req[@"c0"], 0, AUDIO_VOCAB - 1, &c0)) {
                require_depth_emission(emit_depth_error(correlation, @"invalid-c0"));
                continue;
            }
            double gamma = 0.0;
            if (!json_finite_number_in_range(req[@"gamma"], 0.0, 100.0, &gamma)) {
                require_depth_emission(emit_depth_error(correlation, @"invalid-gamma"));
                continue;
            }
            if (![req[@"program_set_digest"] isKindOfClass:[NSString class]] ||
                ![req[@"program_set_digest"] isEqualToString:worker.program_set_digest]) {
                require_depth_emission(emit_depth_error(correlation, @"program-set-digest-mismatch"));
                continue;
            }

            NSString *output_dir = req[@"output_dir"];
            if (!existing_directory(output_dir)) {
                require_depth_emission(emit_depth_error(correlation, @"invalid-output-dir"));
                continue;
            }

            NSData *in_cond = nil;
            NSData *in_uncond = nil;
            if (has_input_dir) {
                NSString *input_dir = req[@"input_dir"];
                if (!existing_directory(input_dir)) {
                    require_depth_emission(emit_depth_error(correlation, @"invalid-input-dir"));
                    continue;
                }
                NSString *cond_path = [input_dir stringByAppendingPathComponent:
                    @"hidden_cond.fp16.bin"];
                NSString *uncond_path = [input_dir stringByAppendingPathComponent:
                    @"hidden_uncond.fp16.bin"];
                if (!read_exact_regular_file(cond_path, TENSOR_BYTES, &in_cond) ||
                    !read_exact_regular_file(uncond_path, TENSOR_BYTES, &in_uncond)) {
                    require_depth_emission(emit_depth_error(correlation, @"invalid-hidden-tensor-files"));
                    continue;
                }
            } else if (!strict_base64_tensor(req[@"hidden_cond_b64"], &in_cond) ||
                       !strict_base64_tensor(req[@"hidden_uncond_b64"], &in_uncond)) {
                require_depth_emission(emit_depth_error(correlation, @"invalid-hidden-tensor-base64"));
                continue;
            }

            NSString *cond_sha = orion_depth_serve_sha256_data(in_cond);
            NSString *uncond_sha = orion_depth_serve_sha256_data(in_uncond);
            if (!orion_depth_serve_is_lower_hex_sha256(req[@"hidden_cond_sha256"]) ||
                ![cond_sha isEqualToString:req[@"hidden_cond_sha256"]]) {
                require_depth_emission(emit_depth_error(correlation, @"hidden-cond-sha256-mismatch"));
                continue;
            }
            if (!orion_depth_serve_is_lower_hex_sha256(req[@"hidden_uncond_sha256"]) ||
                ![uncond_sha isEqualToString:req[@"hidden_uncond_sha256"]]) {
                require_depth_emission(emit_depth_error(correlation, @"hidden-uncond-sha256-mismatch"));
                continue;
            }
            NSString *semantic_sha = orion_depth_serve_semantic_request_sha256(
                worker_session_id, req[@"stream_session_id"], (uint64_t)logical_frame_id,
                req[@"request_id"], (uint64_t)request_sequence, launch_challenge,
                challenge_nonce, (uint64_t)c0, gamma, cond_sha, uncond_sha,
                worker.program_set_digest);
            if (!orion_depth_serve_is_lower_hex_sha256(req[@"semantic_request_sha256"]) ||
                ![semantic_sha isEqualToString:req[@"semantic_request_sha256"]]) {
                require_depth_emission(emit_depth_error(correlation, @"semantic-request-sha256-mismatch"));
                continue;
            }
            accepted_frame = true;

            int32_t tokens[DEPTH_STEPS] = {0};
            DepthEvalCounters counters;
            DepthPhaseTimings phase_timings = {0};
            DepthPhaseTimings *phase_timings_ptr =
                phase_timings_enabled ? &phase_timings : NULL;
            DepthFrameFailure frame_failure = DEPTH_FRAME_FAILURE_NONE;
            uint64_t frame_start_ns = monotonic_ns();
            int compile_before = orion_compile_count();
            bool eval_ok = depth_worker_run_frame(&worker, in_cond, in_uncond,
                                                  (int)c0, gamma, tokens, &counters,
                                                  phase_timings_ptr, &frame_failure);
            NSData *token_data = nil;
            if (eval_ok) {
                uint64_t token_encoding_start = phase_start(phase_timings_ptr);
                token_data = tokens_data(tokens);
                if (phase_timings_ptr)
                    phase_accumulate(phase_timings_ptr, token_encoding_start,
                                     &phase_timings.token_encoding_ns);
            }
            int compile_after = orion_compile_count();
            int compile_delta = compile_after - compile_before;
            bool counts_complete =
                eval_count_total(counters.actual) == eval_count_total(counters.expected) &&
                eval_count_total(counters.successful) == eval_count_total(counters.expected) &&
                eval_count_total(counters.failed) == 0;
            bool output_ok = false;
            NSString *frame_error = nil;
            if (frame_failure == DEPTH_FRAME_FAILURE_PRIVATE_EVALUATION)
                frame_error = @"private-evaluation-failed";
            else if (frame_failure == DEPTH_FRAME_FAILURE_EVALUATION_PREPARATION)
                frame_error = @"evaluation-preparation-failed";
            else if (frame_failure == DEPTH_FRAME_FAILURE_HOST_RMSNORM)
                frame_error = @"host-rmsnorm-failed";
            else if (!eval_ok || !counts_complete)
                frame_error = @"private-invocation-proof-incomplete";
            else if (compile_delta != 0) frame_error = @"compile-count-changed";
            else {
                uint64_t output_write_start = phase_start(phase_timings_ptr);
                output_ok = write_tokens_bin_checked(output_dir, token_data);
                if (phase_timings_ptr)
                    phase_accumulate(phase_timings_ptr, output_write_start,
                                     &phase_timings.output_write_ns);
                if (!output_ok) frame_error = @"output-write-failed";
            }
            uint64_t frame_end_ns = monotonic_ns();

            NSMutableDictionary *receipt = [@{
                @"receipt_schema": DEPTH_RECEIPT_SCHEMA,
                @"ok": (frame_error == nil) ? @YES : @NO,
                @"worker_session_id": worker_session_id,
                @"stream_session_id": req[@"stream_session_id"],
                @"logical_frame_id": @(logical_frame_id),
                @"request_id": req[@"request_id"],
                @"request_sequence": @(request_sequence),
                @"challenge_nonce": challenge_nonce,
                @"hidden_cond_sha256": cond_sha,
                @"hidden_uncond_sha256": uncond_sha,
                @"semantic_request_sha256": semantic_sha,
                @"program_set_digest": worker.program_set_digest,
                @"previous_receipt_sha256": previous_receipt_sha256,
                @"private_invocation_known": @YES,
                @"private_invocations_expected": @(eval_count_total(counters.expected)),
                @"private_invocations_actual": @(eval_count_total(counters.actual)),
                @"private_invocations_successful": @(eval_count_total(counters.successful)),
                @"private_invocations_failed": @(eval_count_total(counters.failed)),
                @"private_invocations_by_program": eval_counts_dictionary(&counters),
                @"compile_count_before": @(compile_before),
                @"compile_count_after": @(compile_after),
                @"compile_count_delta": @(compile_delta),
                @"native_fallback_used": @NO,
                @"frame_start_monotonic_ns": @(frame_start_ns),
                @"frame_end_monotonic_ns": @(frame_end_ns),
                @"frame_duration_ns": @(frame_end_ns - frame_start_ns),
                @"output_placement": @"host-memory+atomic-tokens-bin",
            } mutableCopy];
            if (frame_error) {
                receipt[@"error"] = frame_error;
            } else {
                NSMutableArray *tok_arr = [NSMutableArray arrayWithCapacity:DEPTH_STEPS];
                for (int i = 0; i < DEPTH_STEPS; i++) [tok_arr addObject:@(tokens[i])];
                NSString *token_sha = orion_depth_serve_sha256_data(token_data);
                receipt[@"tokens"] = tok_arr;
                receipt[@"token_sha256"] = token_sha;
                receipt[@"tokens_bin_sha256"] = token_sha;
            }
            NSString *receipt_sha = orion_depth_serve_receipt_sha256(receipt);
            if (!receipt_sha) {
                char *field = orion_depth_serve_receipt_sha256_failure_field(receipt);
                fprintf(stderr, "depth-serve: accepted receipt hashing failed at field '%s'; terminating worker\n",
                        field ?: "unknown");
                if (field) free(field);
                _Exit(EXIT_FAILURE);
            }
            receipt[@"receipt_sha256"] = receipt_sha;
            NSData *serialized_receipt = serialize_depth_response(receipt);
            if (!serialized_receipt) {
                fprintf(stderr, "depth-serve: accepted receipt encoding failed; terminating worker\n");
                _Exit(EXIT_FAILURE);
            }
            require_depth_emission(emit_serialized_depth_response(serialized_receipt));
            previous_receipt_sha256 = receipt_sha;
            emit_phase_timing_diagnostic(
                phase_timings_ptr, worker_session_id, req, seqpack_mode,
                frame_error == nil, receipt_sha, frame_end_ns - frame_start_ns);
                } @catch (NSException *exception) {
                    if (!accepted_frame) {
                        require_depth_emission(emit_depth_error(nil, @"request-handler-failed"));
                        continue;
                    }
                    fprintf(stderr, "depth-serve: request handler exception after input acceptance boundary: %s; terminating worker\n",
                            exception.reason.UTF8String ?: "unknown");
                    _Exit(EXIT_FAILURE);
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
