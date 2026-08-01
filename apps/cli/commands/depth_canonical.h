// apps/cli/commands/depth_canonical.h
//
// Shared canonical VoXtream v1.5 aggregate-artifact + plan-aware compile
// helpers for the depth CLI commands (depth-eval, depth-serve).
//
// Both depth-eval and depth-serve consume the exporter-published canonical
// aggregate BLOBFILEs and their sidecar manifests. They never reconstruct
// aggregates from individual legacy blobs. Each layer graph is rewritten to
// its sidecar-declared offsets, then a binding plan is generated against the
// export root and finalized before compilation. Any malformed, incomplete,
// escaped, or stale artifact aborts before graph construction.
//
// chunk_sha256 (NOT tensor_sha256) is the on-disk payload digest and is what
// this loader verifies. tensor_sha256 is the source-array digest and is not
// equal to the stored fp16 payload.

#ifndef ORION_DEPTH_CANONICAL_H
#define ORION_DEPTH_CANONICAL_H

#import <Foundation/Foundation.h>

#include "compiler/binding_plan.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// The aggregate blob header is 128 bytes; each per-chunk header is 64 bytes.
// These mirror the on-disk framing published by the canonical exporter.
#define ORION_DEPTH_CANONICAL_BLOB_HEADER_BYTES 128u

// A loaded canonical aggregate artifact for one depth layer:
//   - mil_path:  the @model_path/... key under which the blob is published to
//                the compiled program's weight dictionary;
//   - data:      the full aggregate blob bytes (retained for the program's life);
//   - offset_by_name: chunk-mil-name -> absolute chunk offset in `data`.
typedef struct {
    NSString *mil_path;
    NSData *data;
    NSDictionary<NSString *, NSNumber *> *offset_by_name;
} CanonicalLayerArtifact;

// Load one layer's canonical aggregate artifact from the export root.
//
// Performs full contract validation: root manifest schema, layer inventory
// coverage, sidecar schema/sha, per-chunk range/header/hash (chunk_sha256 is
// the on-disk payload digest). `wanted_layer` must be in [0, LAYERS).
//
// On success fills `out` (mil_path/data/offset_by_name). On failure returns
// false and sets `*failure_detail` to a short, stable reason token suitable
// for evidence/error reporting. `root` is the canonical export root path.
bool depth_canonical_load_layer_artifact(NSString *root, int wanted_layer,
                                         CanonicalLayerArtifact *out,
                                         NSString **failure_detail);

// Build a finalized, plan-bound layer compile unit from the canonical export.
//
// Loads the layer artifact, constructs + validates + optimizes the layer graph,
// rewrites its BLOBFILE refs to the aggregate blob offsets, codegens MIL,
// generates and finalizes a binding plan against `identity`, and verifies the
// resulting weight dictionary against that plan.
//
// `program_name_prefix` becomes the plan's program_name (e.g. "depth-serve-layer0"),
// keeping depth-eval and depth-serve plan identities distinct. `layer` is the
// depth-transformer layer index in [0, LAYERS).
//
// On success fills `plan`, `mil_out`, and `wdict_out` (the weight dictionary
// {artifact.mil_path: {offset:0, data:artifact.data}}). Returns true. On
// failure returns false and sets `*failure_detail` to a stable reason token.
bool depth_canonical_build_layer(NSString *root, int layer,
                                 const char *program_name_prefix,
                                 const OrionBindingPlanIdentity *identity,
                                 OrionBindingPlan *plan,
                                 NSString **mil_out,
                                 NSDictionary **wdict_out,
                                 NSString **failure_detail);

// Build a finalized, plan-bound final-norm compile unit.
//
// The final norm is a separate single-chunk BLOBFILE artifact
// (weights/final_norm.bin) that is NOT part of any layer aggregate. This
// constructs the final-norm graph, generates + finalizes its plan against
// `identity`, and codegens its MIL. `program_name` becomes the plan's
// program_name (e.g. "depth-serve-final-norm").
//
// On success fills `plan` and `mil_out`, and returns true. `final_blob` is the
// already-loaded weights/final_norm.bin payload (the caller owns its lifetime).
// On failure returns false and sets `*failure_detail`.
bool depth_canonical_build_final_norm(NSData *final_blob,
                                      const char *program_name,
                                      const OrionBindingPlanIdentity *identity,
                                      OrionBindingPlan *plan,
                                      NSString **mil_out,
                                      NSString **failure_detail);

#ifdef __cplusplus
}
#endif

#endif  // ORION_DEPTH_CANONICAL_H
