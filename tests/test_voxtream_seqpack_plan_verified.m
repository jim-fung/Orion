// test_voxtream_seqpack_plan_verified.m
//
// ORX-103: plan-verified compile of the sequence-packed depth-layer graph.
//
// This test loads the sequence-packed export (weights/layer0_seqpack.blob +
// manifest), builds the seq-pack graph, generates a binding plan with identity,
// and compiles via orion_compile_mil_with_plan — the SAME production path the
// G009 test uses, NOT the raw diagnostic path that produced Code=10 rejections
// in ORX-003/004.
//
// This is the definitive test of whether sequence-packing is ANE-viable:
// if it compiles through the production path with real export weights, the
// Code=10 rejections were synthetic-weight artifacts; if it still fails, the
// topology itself is rejected.
//
// Compile budget: 1 compile on success, 0 on rejection (pre-counter). The
// test's hard guard is compile_delta <= 1.
//
// Usage:
//   VOXTREAM_DEPTH_EXPORT=/path/to/orion-depth-export-seqpack \
//     ./build/tests/test_voxtream_seqpack_plan_verified
//
// Skip behavior: returns 0 when VOXTREAM_DEPTH_EXPORT is unset or ANE
// unavailable (no budget consumed).

#import <Foundation/Foundation.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "compiler/graph.h"
#include "compiler/validate.h"
#include "compiler/pipeline.h"
#include "compiler/codegen.h"
#include "compiler/frontends/voxtream_depth.h"
#include "core/ane_runtime.h"

static NSString *kSeqpackBlobPath = @"@model_path/weights/layer0_seqpack.blob";

static bool rewrite_refs(OrionGraph *g, NSDictionary<NSString *, NSNumber *> *offset_by_name) {
    for (int i = 0; i < g->n_nodes; i++) {
        OrionNode *node = &g->nodes[i];
        if (node->op != ORION_OP_CONST || node->attrs.blob_path[0] == '\0') continue;
        NSString *name = [NSString stringWithUTF8String:node->attrs.blob_path];
        // Extract the chunk name from the path (last component without .bin).
        NSString *basename = name.lastPathComponent;
        NSString *chunk_name = [basename stringByDeletingPathExtension];
        NSNumber *offset = offset_by_name[chunk_name];
        if (!offset) {
            fprintf(stderr, "FAIL: graph blob ref %s has no matching chunk in manifest\n",
                    node->attrs.blob_path);
            return false;
        }
        // Rewrite to point at the aggregate blob with the chunk offset.
        strcpy(node->attrs.blob_path, kSeqpackBlobPath.UTF8String);
        node->attrs.blob_offset = [offset unsignedLongLongValue];
    }
    return true;
}

int main(void) {
    @autoreleasepool {
        const char *export_root = getenv("VOXTREAM_DEPTH_EXPORT");
        if (!export_root || !*export_root) {
            printf("SKIP: set VOXTREAM_DEPTH_EXPORT to a seq-pack export directory\n");
            return 0;
        }

        if (!orion_ane_init()) {
            printf("SKIP: ANE unavailable\n");
            return 0;
        }

        NSString *export = [NSString stringWithUTF8String:export_root];
        printf("=== ORX-103 sequence-packed plan-verified compile ===\n");
        printf("  export: %s\n", export_root);

        // 1. Read the seq-pack aggregate manifest.
        NSString *manifest_path = [export stringByAppendingPathComponent:
            @"weights/layer0_seqpack.manifest.json"];
        NSData *manifest_data = [NSData dataWithContentsOfFile:manifest_path];
        if (!manifest_data) {
            fprintf(stderr, "FAIL: cannot read seqpack manifest %s\n",
                    manifest_path.UTF8String);
            return 1;
        }
        NSDictionary *manifest = [NSJSONSerialization JSONObjectWithData:manifest_data
            options:0 error:nil];
        NSArray *chunks = manifest[@"chunks"];
        if (!chunks || chunks.count == 0) {
            fprintf(stderr, "FAIL: seqpack manifest has no chunks\n");
            return 1;
        }
        NSMutableDictionary<NSString *, NSNumber *> *offset_by_name = [NSMutableDictionary dictionary];
        for (NSDictionary *chunk in chunks) {
            offset_by_name[chunk[@"name"]] = chunk[@"offset"];
        }
        printf("  chunks: %lu\n", (unsigned long)chunks.count);

        // 2. Read the aggregate blob.
        NSString *blob_path = [export stringByAppendingPathComponent:
            @"weights/layer0_seqpack.blob"];
        NSData *aggregate = [NSData dataWithContentsOfFile:blob_path];
        if (!aggregate) {
            fprintf(stderr, "FAIL: cannot read seqpack blob %s\n", blob_path.UTF8String);
            return 1;
        }
        printf("  blob size: %lu bytes\n", (unsigned long)aggregate.length);

        // 3. Build the seq-pack graph, rewrite refs, generate MIL.
        OrionGraph *g = orion_frontend_voxtream_depth_probe_seq_pack_adapter(0, 0, NULL);
        if (!g) { fprintf(stderr, "FAIL: seq-pack graph build\n"); return 1; }
        OrionValidationResult vr = orion_graph_validate(g);
        if (!vr.valid) {
            fprintf(stderr, "FAIL: graph validation: %s\n", vr.message);
            orion_graph_free(g);
            return 1;
        }
        orion_pipeline_optimize(g);
        if (!rewrite_refs(g, offset_by_name)) { orion_graph_free(g); return 1; }
        NSString *mil = orion_codegen_mil(g, "main");
        orion_graph_free(g);
        if (!mil) { fprintf(stderr, "FAIL: MIL generation\n"); return 1; }
        printf("  MIL size: %lu bytes\n", (unsigned long)mil.length);

        // 4. Generate the binding plan.
        OrionGraph *plan_graph = orion_frontend_voxtream_depth_probe_seq_pack_adapter(0, 0, NULL);
        if (!plan_graph) { fprintf(stderr, "FAIL: plan graph\n"); return 1; }
        orion_pipeline_optimize(plan_graph);
        if (!rewrite_refs(plan_graph, offset_by_name)) { orion_graph_free(plan_graph); return 1; }

        OrionBindingPlanIdentity identity = {
            .orion_revision = getenv("VOXTREAM_ORION_REVISION") ?: "orx-103-seqpack-test",
            .compiler_build = getenv("VOXTREAM_COMPILER_BUILD") ?: "orx-103-seqpack-test",
            .weight_root = export.fileSystemRepresentation,
        };
        OrionBindingPlan plan;
        char error[512] = {0};
        bool plan_ok = orion_binding_plan_generate_with_identity(
            plan_graph, "orx-103-seqpack", &identity, &plan, error, sizeof(error));
        orion_graph_free(plan_graph);
        if (!plan_ok) {
            fprintf(stderr, "FAIL: plan generation: %s\n", error);
            return 1;
        }
        printf("  plan: %d weights, version=%u\n", plan.n_weights, plan.version);

        orion_binding_plan_set_mil_identity(&plan, mil.UTF8String);
        char val_error[512] = {0};
        if (!orion_binding_plan_validate_finalized(&plan, val_error, sizeof(val_error))) {
            fprintf(stderr, "FAIL: plan validation: %s\n", val_error);
            return 1;
        }
        printf("  plan validated\n");

        // 5. Build + validate the weight dictionary.
        NSDictionary *wdict = @{kSeqpackBlobPath: @{@"offset": @0, @"data": aggregate}};
        if (!orion_validate_weight_dictionary_for_plan(&plan, wdict)) {
            fprintf(stderr, "FAIL: weight dictionary validation\n");
            return 1;
        }
        printf("  weight dict validated\n");

        // 6. Compile via the plan-verified path.
        int count_before = orion_compile_count();
        printf("  compile_count_before: %d\n", count_before);
        char compile_error[512] = {0};
        OrionProgram *prog = orion_compile_mil_with_plan(
            mil.UTF8String, &plan, wdict, "orx-103-seqpack-plan-verified",
            compile_error, sizeof(compile_error));
        int count_after = orion_compile_count();
        int compile_delta = count_after - count_before;
        printf("  compile_count_after: %d\n", count_after);

        if (compile_delta > 1) {
            fprintf(stderr, "FAIL: budget guard: delta=%d (expected <=1)\n", compile_delta);
            if (prog) orion_release_program(prog);
            return 1;
        }

        if (prog) {
            printf("\nPASS: seq-pack plan-verified compile succeeded (delta=%d)\n",
                   compile_delta);
            printf("  -> Sequence-packed CFG is COMPILER-VIABLE through the production path.\n");
            orion_release_program(prog);
            return 0;
        } else {
            printf("\nPROBE FAIL: seq-pack plan-verified compile rejected (delta=%d)\n",
                   compile_delta);
            if (compile_error[0]) printf("  error: %s\n", compile_error);
            printf("  -> Sequence-packed CFG is NOT viable via the production path.\n");
            return 1;
        }
    }
}
