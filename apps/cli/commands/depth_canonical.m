// apps/cli/commands/depth_canonical.m
//
// Shared canonical VoXtream v1.5 aggregate-artifact + plan-aware compile
// helpers for depth-eval and depth-serve. See depth_canonical.h.
//
// The artifact-validation logic is ported verbatim from the depth_eval.m
// implementation that already passes recurrence against the canonical export
// (orion-depth-export-v16-canonical); both commands now share it.

#import "apps/cli/commands/depth_canonical.h"

#import <CommonCrypto/CommonDigest.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "compiler/codegen.h"
#include "compiler/frontends/voxtream_depth.h"
#include "compiler/pipeline.h"
#include "compiler/validate.h"
#include "core/ane_runtime.h"
#include "core/blobfile.h"

#define LAYERS ORION_VOXTREAM_DEPTH_LAYERS

// ---------------------------------------------------------------------------
// Local SHA/path helpers (file-local so this TU is self-contained).
// ---------------------------------------------------------------------------

static bool can_is_hex_sha256(NSString *value) {
    if (![value isKindOfClass:[NSString class]] || value.length != 64) return false;
    NSCharacterSet *non_hex = [[NSCharacterSet characterSetWithCharactersInString:
        @"0123456789abcdefABCDEF"] invertedSet];
    return [value rangeOfCharacterFromSet:non_hex].location == NSNotFound;
}

static NSString *can_sha256_data(NSData *data) {
    if (!data) return nil;
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(data.bytes, (CC_LONG)data.length, digest);
    NSMutableString *value = [NSMutableString stringWithCapacity:64];
    for (int i = 0; i < CC_SHA256_DIGEST_LENGTH; i++) [value appendFormat:@"%02x", digest[i]];
    return value;
}

// Resolve a manifest-relative path only when it remains inside the canonical
// export root after symlink resolution.  Rejecting absolute and dot paths is
// intentional: artifact metadata is an input trust boundary.
static NSString *can_safe_export_path(NSString *resolved_root, NSString *relative) {
    if (![relative isKindOfClass:[NSString class]] || !relative.length ||
        [relative isAbsolutePath]) return nil;
    for (NSString *part in relative.pathComponents)
        if ([part isEqualToString:@".."] || [part isEqualToString:@"."]) return nil;
    NSString *candidate = [[[resolved_root stringByAppendingPathComponent:relative]
        stringByStandardizingPath] stringByResolvingSymlinksInPath];
    NSString *prefix = [resolved_root stringByAppendingString:@"/"];
    return [candidate hasPrefix:prefix] ? candidate : nil;
}

static NSString *can_chunk_name_from_mil_path(const char *mil_path) {
    NSString *path = @(mil_path);
    return [path.lastPathComponent stringByDeletingPathExtension];
}

static bool can_rewrite_canonical_refs(OrionGraph *g, CanonicalLayerArtifact *artifact) {
    int rewritten = 0;
    for (int i = 0; i < g->n_nodes; i++) {
        OrionNode *node = &g->nodes[i];
        if (node->op != ORION_OP_CONST || !node->attrs.blob_path[0]) continue;
        NSString *name = can_chunk_name_from_mil_path(node->attrs.blob_path);
        NSNumber *offset = artifact->offset_by_name[name];
        if (!offset || [offset unsignedLongLongValue] < 64) {
            fprintf(stderr, "depth-canonical: graph blob %s has no canonical aggregate chunk '%s'\n",
                    node->attrs.blob_path, name.UTF8String);
            return false;
        }
        snprintf(node->attrs.blob_path, sizeof(node->attrs.blob_path), "%s",
                 artifact->mil_path.UTF8String);
        node->attrs.blob_offset = [offset unsignedLongLongValue];
        rewritten++;
    }
    if (!rewritten) {
        fprintf(stderr, "depth-canonical: canonical graph has no BLOBFILE references\n");
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Canonical layer artifact loading (full contract validation).
// ---------------------------------------------------------------------------

bool depth_canonical_load_layer_artifact(NSString *root, int wanted_layer,
                                         CanonicalLayerArtifact *out,
                                         NSString **failure_detail) {
    if (failure_detail) *failure_detail = nil;
    if (!out || wanted_layer < 0 || wanted_layer >= LAYERS) return false;
    out->mil_path = nil;
    out->data = nil;
    out->offset_by_name = nil;
    NSString *resolved_root = [[root stringByStandardizingPath] stringByResolvingSymlinksInPath];
    NSString *root_manifest_path = can_safe_export_path(resolved_root, @"depth_export.manifest.json");
    NSData *root_data = root_manifest_path ? [NSData dataWithContentsOfFile:root_manifest_path] : nil;
    NSDictionary *root_manifest = root_data ? [NSJSONSerialization JSONObjectWithData:root_data
        options:0 error:nil] : nil;
    if (![root_manifest isKindOfClass:[NSDictionary class]] ||
        ![root_manifest[@"schema_version"] isEqual:@"voxtream.orion.depth-export/1.5"]) {
        if (failure_detail) *failure_detail = @"root-manifest-contract";
        return false;
    }
    NSDictionary *aggregate = root_manifest[@"aggregate_blob_export"];
    NSArray *layers = [aggregate isKindOfClass:[NSDictionary class]] ? aggregate[@"layers"] : nil;
    if (![layers isKindOfClass:[NSArray class]] || layers.count != LAYERS) {
        if (failure_detail) *failure_detail = @"aggregate-layer-inventory";
        return false;
    }

    NSMutableDictionary<NSNumber *, NSDictionary *> *records = [NSMutableDictionary dictionary];
    for (id item in layers) {
        if (![item isKindOfClass:[NSDictionary class]]) { if (failure_detail) *failure_detail = @"aggregate-layer-record"; return false; }
        NSNumber *layer_number = item[@"layer"];
        if (![layer_number isKindOfClass:[NSNumber class]] ||
            CFGetTypeID((__bridge CFTypeRef)layer_number) == CFBooleanGetTypeID() ||
            layer_number.intValue < 0 || layer_number.intValue >= LAYERS || records[layer_number]) {
            if (failure_detail) *failure_detail = @"aggregate-layer-coverage";
            return false;
        }
        records[layer_number] = item;
    }
    for (int layer = 0; layer < LAYERS; layer++) if (!records[@(layer)]) {
        if (failure_detail) *failure_detail = @"aggregate-layer-coverage";
        return false;
    }

    NSDictionary *record = records[@(wanted_layer)];
    NSString *blob_rel = record[@"blob"], *sidecar_rel = record[@"manifest"];
    NSString *expected_schema = record[@"schema_version"], *expected_sha = record[@"sha256"];
    if (![blob_rel isKindOfClass:[NSString class]] || ![sidecar_rel isKindOfClass:[NSString class]] ||
        ![expected_schema isEqual:@"voxtream.orion.aggregate-blobs/1.1"] || !can_is_hex_sha256(expected_sha)) {
        if (failure_detail) *failure_detail = @"aggregate-record-contract";
        return false;
    }
    NSString *blob_path = can_safe_export_path(resolved_root, blob_rel);
    NSString *sidecar_path = can_safe_export_path(resolved_root, sidecar_rel);
    NSData *blob = blob_path ? [NSData dataWithContentsOfFile:blob_path] : nil;
    NSData *sidecar_data = sidecar_path ? [NSData dataWithContentsOfFile:sidecar_path] : nil;
    NSDictionary *sidecar = sidecar_data ? [NSJSONSerialization JSONObjectWithData:sidecar_data
        options:0 error:nil] : nil;
    if (!blob || blob.length < ORION_DEPTH_CANONICAL_BLOB_HEADER_BYTES || ![sidecar isKindOfClass:[NSDictionary class]] ||
        ![sidecar[@"schema_version"] isEqual:@"voxtream.orion.aggregate-blobs/1.1"] ||
        [sidecar[@"layer"] intValue] != wanted_layer ||
        [can_sha256_data(blob) caseInsensitiveCompare:expected_sha] != NSOrderedSame ||
        ![sidecar[@"sha256"] isEqual:expected_sha]) {
        if (failure_detail) *failure_detail = @"aggregate-file-or-sidecar";
        return false;
    }
    NSArray *chunks = sidecar[@"chunks"];
    NSNumber *declared_count = record[@"chunk_count"];
    if (![chunks isKindOfClass:[NSArray class]] || chunks.count != 16 ||
        (declared_count && declared_count.unsignedIntegerValue != chunks.count) ||
        ![sidecar[@"total_payload_bytes"] isEqual:record[@"total_payload_bytes"]]) {
        if (failure_detail) *failure_detail = @"aggregate-chunk-inventory";
        return false;
    }
    NSMutableDictionary<NSString *, NSNumber *> *offsets = [NSMutableDictionary dictionary];
    uint64_t previous_end = ORION_DEPTH_CANONICAL_BLOB_HEADER_BYTES;
    for (id item in chunks) {
        if (![item isKindOfClass:[NSDictionary class]]) { if (failure_detail) *failure_detail = @"aggregate-chunk-record"; return false; }
        NSString *name = item[@"mil_name"] ?: item[@"name"];
        // chunk_sha256 is the on-disk payload digest; tensor_sha256 is the
        // source-array digest and is not equal to the stored fp16 payload.
        NSString *payload_sha = item[@"chunk_sha256"] ?: item[@"tensor_sha256"];
        NSNumber *offset = item[@"offset"], *bytes = item[@"payload_bytes"];
        if (![name isKindOfClass:[NSString class]] || !name.length || offsets[name] ||
            !can_is_hex_sha256(payload_sha) ||
            ![offset isKindOfClass:[NSNumber class]] || ![bytes isKindOfClass:[NSNumber class]]) {
            if (failure_detail) *failure_detail = @"aggregate-chunk-record";
            return false;
        }
        uint64_t chunk_offset = offset.unsignedLongLongValue;
        uint64_t payload_bytes = bytes.unsignedLongLongValue;
        if (chunk_offset < ORION_DEPTH_CANONICAL_BLOB_HEADER_BYTES || payload_bytes == 0 ||
            chunk_offset < previous_end) {
            if (failure_detail) *failure_detail = @"aggregate-chunk-range";
            return false;
        }
        OrionBlobfileValidationOptions options = {
            .chunk_offset = chunk_offset,
            .expected_payload_size = payload_bytes,
            .expected_sha256 = payload_sha.UTF8String,
            .compute_sha256 = false,
        };
        OrionBlobfileChunk chunk;
        OrionBlobfileStatus status = orion_blobfile_validate_memory(
            blob.bytes, blob.length, &options, &chunk);
        if (status == ORION_BLOBFILE_SHA256_MISMATCH ||
            status == ORION_BLOBFILE_SHA256_READ_ERROR) {
            if (failure_detail) *failure_detail = @"aggregate-chunk-hash";
            return false;
        }
        if (status != ORION_BLOBFILE_OK) {
            if (failure_detail) *failure_detail = @"aggregate-chunk-header";
            return false;
        }
        offsets[name] = offset;
        previous_end = chunk.payload_end;
    }
    out->mil_path = [@"@model_path/" stringByAppendingString:blob_rel];
    out->data = blob;
    out->offset_by_name = offsets;
    return true;
}

// ---------------------------------------------------------------------------
// Plan-aware layer compile-unit construction.
// ---------------------------------------------------------------------------

bool depth_canonical_build_layer(NSString *root, int layer,
                                 const char *program_name_prefix,
                                 const OrionBindingPlanIdentity *identity,
                                 OrionBindingPlan *plan,
                                 NSString **mil_out,
                                 NSDictionary **wdict_out,
                                 NSString **failure_detail) {
    CanonicalLayerArtifact artifact;
    if (!depth_canonical_load_layer_artifact(root, layer, &artifact, failure_detail)) return false;
    OrionGraph *graph = orion_frontend_voxtream_depth_layer(layer);
    if (!graph) { if (failure_detail) *failure_detail = @"layer-graph"; return false; }
    OrionValidationResult vr = orion_graph_validate(graph);
    if (!vr.valid) {
        fprintf(stderr, "depth-canonical: layer%d graph validation: %s\n", layer, vr.message);
        orion_graph_free(graph);
        if (failure_detail) *failure_detail = @"layer-graph-validation";
        return false;
    }
    orion_pipeline_optimize(graph);
    if (!can_rewrite_canonical_refs(graph, &artifact)) {
        orion_graph_free(graph);
        if (failure_detail) *failure_detail = @"layer-sidecar-rewrite";
        return false;
    }
    NSString *mil = orion_codegen_mil(graph, "main");
    orion_graph_free(graph);
    if (!mil) { if (failure_detail) *failure_detail = @"layer-mil"; return false; }

    OrionGraph *plan_graph = orion_frontend_voxtream_depth_layer(layer);
    if (!plan_graph) { if (failure_detail) *failure_detail = @"layer-plan-graph"; return false; }
    orion_pipeline_optimize(plan_graph);
    if (!can_rewrite_canonical_refs(plan_graph, &artifact)) {
        orion_graph_free(plan_graph);
        if (failure_detail) *failure_detail = @"layer-plan-rewrite";
        return false;
    }
    char error[512] = {0};
    bool generated = orion_binding_plan_generate_with_identity(plan_graph, program_name_prefix,
        identity, plan, error, sizeof(error));
    orion_graph_free(plan_graph);
    if (!generated || !orion_binding_plan_set_mil_identity(plan, mil.UTF8String) ||
        !orion_binding_plan_validate_finalized(plan, error, sizeof(error)) ||
        plan->n_inputs != 1 || plan->n_outputs != 1) {
        fprintf(stderr, "depth-canonical: layer%d finalized plan rejected: %s\n", layer,
                error[0] ? error : "invalid ABI");
        if (failure_detail) *failure_detail = @"layer-finalized-plan";
        return false;
    }
    NSDictionary *wdict = @{artifact.mil_path: @{@"offset": @0, @"data": artifact.data}};
    if (!orion_validate_weight_dictionary_for_plan(plan, wdict)) {
        if (failure_detail) *failure_detail = @"layer-plan-weight-dictionary";
        return false;
    }
    *mil_out = mil;
    *wdict_out = wdict;
    return true;
}

// ---------------------------------------------------------------------------
// Plan-aware final-norm compile-unit construction.
//
// Ported from depth_eval.m's separate-CFG final-norm path: the final norm is a
// single-chunk BLOBFILE (weights/final_norm.bin), not part of any aggregate.
// ---------------------------------------------------------------------------

bool depth_canonical_build_final_norm(NSData *final_blob,
                                      const char *program_name,
                                      const OrionBindingPlanIdentity *identity,
                                      OrionBindingPlan *plan,
                                      NSString **mil_out,
                                      NSString **failure_detail) {
    if (failure_detail) *failure_detail = nil;
    OrionGraph *final_graph = orion_frontend_voxtream_depth_final_norm();
    if (!final_graph) {
        if (failure_detail) *failure_detail = @"final-norm-graph";
        return false;
    }
    NSString *final_mil = nil;
    OrionValidationResult fvr = orion_graph_validate(final_graph);
    if (!fvr.valid) {
        fprintf(stderr, "depth-canonical: final-norm graph validation: %s\n", fvr.message);
        orion_graph_free(final_graph);
        if (failure_detail) *failure_detail = @"final-norm-graph-validation";
        return false;
    }
    orion_pipeline_optimize(final_graph);
    char plan_error[512] = {0};
    bool ok = orion_binding_plan_generate_with_identity(
        final_graph, program_name, identity, plan, plan_error, sizeof(plan_error));
    if (!ok) {
        fprintf(stderr, "depth-canonical: final-norm plan generation: %s\n", plan_error);
        orion_graph_free(final_graph);
        if (failure_detail) *failure_detail = @"final-norm-plan";
        return false;
    }
    final_mil = orion_codegen_mil(final_graph, "main");
    orion_graph_free(final_graph);
    if (!final_mil ||
        !orion_binding_plan_set_mil_identity(plan, final_mil.UTF8String) ||
        !orion_binding_plan_validate_finalized(plan, plan_error, sizeof(plan_error))) {
        fprintf(stderr, "depth-canonical: final-norm finalized-plan validation failed: %s\n",
                plan_error[0] ? plan_error : "unable to bind MIL identity");
        if (failure_detail) *failure_detail = @"final-norm-finalized-plan";
        return false;
    }
    NSDictionary *wdict = @{@"@model_path/weights/final_norm.bin":
        @{@"offset": @0, @"data": final_blob}};
    if (!orion_validate_weight_dictionary_for_plan(plan, wdict)) {
        if (failure_detail) *failure_detail = @"final-norm-plan-weight-dictionary";
        return false;
    }
    *mil_out = final_mil;
    return true;
}
