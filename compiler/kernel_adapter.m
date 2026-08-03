// compiler/kernel_adapter.m — T137: OrionKernel adapter for compiler frontends

#import "kernel_adapter.h"
#import "codegen.h"
#import "pipeline.h"
#import "validate.h"
#import "pass_ane_validate.h"
#include <string.h>
#include <stdio.h>
#include <limits.h>
#include <unistd.h>

// Publish JSON/Markdown artifact pair crash- and concurrency-safely.
// Uses mkdtemp for a unique generation directory on the same filesystem as
// the targets, writes both files there, then renames them atomically into
// place. Each individual file is published atomically; a crash between the
// two renames may leave one old and one new, but no partial/empty file is
// ever visible. Concurrent writers cannot interfere because each uses its
// own unique directory.
static bool write_artifact_pair(const OrionBindingPlan* plan,
                                const char* json_path, const char* markdown_path,
                                char* error, size_t error_size) {
    if (error && error_size) error[0] = '\0';

    const char* slash = strrchr(json_path, '/');

    char gen_dir[PATH_MAX];
    if (snprintf(gen_dir, sizeof(gen_dir), "%.*s/orion-plan-XXXXXX",
                 slash ? (int)(slash - json_path) : 0, json_path) >= (int)sizeof(gen_dir)) {
        if (error && error_size) snprintf(error, error_size, "artifact path is too long");
        return false;
    }
    if (!mkdtemp(gen_dir)) {
        if (error && error_size) snprintf(error, error_size, "cannot create generation directory");
        return false;
    }

    char temp_json[PATH_MAX], temp_markdown[PATH_MAX];
    if (snprintf(temp_json, sizeof(temp_json), "%s/plan.json", gen_dir) >= (int)sizeof(temp_json) ||
        snprintf(temp_markdown, sizeof(temp_markdown), "%s/plan.md", gen_dir) >= (int)sizeof(temp_markdown)) {
        if (error && error_size) snprintf(error, error_size, "artifact path is too long");
        rmdir(gen_dir);
        return false;
    }

    if (!orion_binding_plan_write_json(plan, temp_json, error, error_size) ||
        !orion_binding_plan_write_markdown(plan, temp_markdown, error, error_size)) {
        unlink(temp_json); unlink(temp_markdown); rmdir(gen_dir);
        return false;
    }

    // rename() atomically replaces any existing destination on the same filesystem.
    if (rename(temp_json, json_path) != 0 || rename(temp_markdown, markdown_path) != 0) {
        unlink(temp_json); unlink(temp_markdown); rmdir(gen_dir);
        if (error && error_size) snprintf(error, error_size, "failed publishing binding-plan artifacts");
        return false;
    }

    rmdir(gen_dir);
    return true;
}

// Direct API: generate MIL from a frontend function.
// Builds graph -> validates -> optimizes -> codegen.
static NSString* generate_mil(OrionGraph* g, const OrionBindingPlanIdentity* identity,
                              const char* json_path, const char* markdown_path,
                              OrionBindingPlan* out_plan) {
    if (!g) return nil;
    char error[256] = {0};

    // Validate
    OrionValidationResult vr = orion_graph_validate(g);
    if (!vr.valid) {
        NSLog(@"[kernel_adapter] validation failed: %s", vr.message);
        orion_graph_free(g);
        return nil;
    }

    // Optimize
    orion_pipeline_optimize(g);

    // Revalidate structurally after optimization. The optimization passes
    // mutate graph liveness, output references, and node topology; a pass bug
    // could create dead references, invalid shapes, or name collisions after
    // the only prior structural gate. Catch that here, before ANE validation
    // and plan generation consume the post-optimize graph.
    OrionValidationResult vr_post = orion_graph_validate(g);
    if (!vr_post.valid) {
        NSLog(@"[kernel_adapter] post-optimize validation failed: %s", vr_post.message);
        orion_graph_free(g);
        return nil;
    }

    // ANE validation is mandatory before codegen. Unlike the structural graph
    // validator, this checks external ABI constraints and banned operations.
    orion_pipeline_ane_passes(g);
    OrionANEValidationResult ane = orion_pass_ane_validate(g);
    if (!ane.valid) {
        NSLog(@"[kernel_adapter] ANE validation failed: %s", ane.errors[0].message);
        orion_graph_free(g);
        return nil;
    }

    if (out_plan) {
        if (!orion_binding_plan_generate_with_identity(
                g, "main", identity, out_plan, error, sizeof(error))) {
            NSLog(@"[kernel_adapter] binding plan failed: %s", error);
            orion_graph_free(g);
            return nil;
        }
    }

    // Codegen
    NSString* mil = orion_codegen_mil(g, "main");
    if (!mil) {
        orion_graph_free(g);
        return nil;
    }
    if (out_plan && !orion_binding_plan_set_mil_identity(out_plan, mil.UTF8String)) {
        orion_graph_free(g);
        return nil;
    }
    if (out_plan && json_path && markdown_path &&
        !write_artifact_pair(out_plan, json_path, markdown_path, error, sizeof(error))) {
        NSLog(@"[kernel_adapter] binding plan artifact failed: %s", error);
        orion_graph_free(g);
        return nil;
    }
    orion_graph_free(g);
    return mil;
}

NSString* orion_kernel_adapter_generate_mil(OrionFrontendFn frontend,
                                              int layer_idx, int bucket,
                                              const OrionModelConfig* cfg) {
    return generate_mil(frontend(layer_idx, bucket, cfg), NULL, NULL, NULL, NULL);
}

NSString* orion_kernel_adapter_generate_mil_with_plan(OrionFrontendFn frontend,
                                                        int layer_idx, int bucket,
                                                        const OrionModelConfig* cfg,
                                                        OrionBindingPlan* out_plan) {
    return orion_kernel_adapter_generate_mil_with_plan_identity(
        frontend, layer_idx, bucket, cfg, NULL, out_plan);
}

NSString* orion_kernel_adapter_generate_mil_with_plan_identity(
    OrionFrontendFn frontend, int layer_idx, int bucket,
    const OrionModelConfig* cfg, const OrionBindingPlanIdentity* identity,
    OrionBindingPlan* out_plan) {
    if (!frontend || !out_plan) return nil;
    return generate_mil(frontend(layer_idx, bucket, cfg), identity, NULL, NULL, out_plan);
}

NSString* orion_kernel_adapter_generate_mil_with_artifacts(
    OrionFrontendFn frontend, int layer_idx, int bucket,
    const OrionModelConfig* cfg, const OrionBindingPlanIdentity* identity,
    const char* json_path, const char* markdown_path, OrionBindingPlan* out_plan) {
    if (!out_plan || !identity || !json_path || !markdown_path) return nil;
    if (strcmp(json_path, markdown_path) == 0) return nil;
    return generate_mil(frontend(layer_idx, bucket, cfg), identity,
                        json_path, markdown_path, out_plan);
}

// Direct API: generate MIL from a 2-arg frontend (no bucket param).
NSString* orion_kernel_adapter_generate_mil_2arg(OrionFrontend2Fn frontend,
                                                  int layer_idx,
                                                  const OrionModelConfig* cfg) {
    OrionGraph* g = frontend(layer_idx, cfg);
    if (!g) return nil;

    return generate_mil(g, NULL, NULL, NULL, NULL);
}

// Direct API: generate MIL and a finalized binding plan from a 2-arg frontend.
NSString* orion_kernel_adapter_generate_mil_2arg_with_plan(OrionFrontend2Fn frontend,
                                                            int layer_idx,
                                                            const OrionModelConfig* cfg,
                                                            OrionBindingPlan* out_plan) {
    if (!frontend || !out_plan) return nil;
    OrionGraph* g = frontend(layer_idx, cfg);
    if (!g) return nil;
    return generate_mil(g, NULL, NULL, NULL, out_plan);
}

OrionKernel orion_kernel_from_frontend(const char* name,
                                        OrionFrontendFn frontend,
                                        OrionWDictFn wdict,
                                        int n_inputs, int n_outputs) {
    OrionKernel k = {0};
    k.name = name;
    k.generate_mil = NULL;
    k.build_wdict = wdict;
    k.adapter_frontend = (const void*)frontend;
    k.n_inputs = n_inputs;
    k.n_outputs = n_outputs;
    return k;
}
