#import "kernel.h"
#import "compiler/kernel_adapter.h"
#import <stdio.h>

static NSString* adapter_mil_and_plan(const OrionKernel* kernel, int layer_idx,
                                      int bucket, const OrionModelConfig* cfg,
                                      const char* blob_dir, OrionBindingPlan* plan) {
    OrionBindingPlanIdentity identity = {
        .orion_revision = "runtime-unqualified",
        .compiler_build = "kernel-adapter",
        .weight_root = blob_dir,
    };
    return orion_kernel_adapter_generate_mil_with_plan_identity(
        (OrionFrontendFn)kernel->adapter_frontend, layer_idx, bucket, cfg,
        &identity, plan);
}

OrionProgram* orion_kernel_compile(const OrionKernel* kernel,
                                   int layer_idx, int bucket,
                                   const OrionModelConfig* cfg,
                                   const char* blob_dir,
                                   const OrionWeightsBinding* wb) {
    if (!kernel || !wb || !kernel->name || !kernel->build_wdict) return NULL;

    // Adapted kernels derive the current graph identity before accepting a
    // cache hit; frontend/config/blob-root drift must not reuse old programs.
    OrionBindingPlan plan;
    NSString* mil = nil;
    if (kernel->adapter_frontend) {
        mil = adapter_mil_and_plan(kernel, layer_idx, bucket, cfg, blob_dir, &plan);
        if (!mil) return NULL;
        OrionProgram* cached = orion_cache_lookup(kernel->name, layer_idx, wb);
        const OrionBindingPlan* cached_plan = orion_program_plan(cached);
        if (cached_plan && strcmp(cached_plan->graph_identity, plan.graph_identity) == 0 &&
            strcmp(cached_plan->mil_identity, plan.mil_identity) == 0) return cached;
    } else {
        OrionProgram* cached = orion_cache_lookup(kernel->name, layer_idx, wb);
        if (cached) return cached;
        if (!kernel->generate_mil) return NULL;
        mil = kernel->generate_mil(layer_idx, bucket, cfg);
        if (!mil) return NULL;
    }

    // Cache miss — the plan/MIL above and dictionary must describe one ABI.
    NSDictionary* wdict = kernel->build_wdict(layer_idx, bucket, blob_dir);

    char tag[64];
    snprintf(tag, sizeof(tag), "%s_L%d", kernel->name, layer_idx);
    char plan_error[256] = {0};
    OrionProgram* prog = kernel->adapter_frontend
        ? orion_compile_mil_with_plan(mil.UTF8String, &plan, wdict, tag,
                                      plan_error, sizeof(plan_error))
        : orion_compile_mil(mil.UTF8String, wdict, tag);
    if (!prog) {
        fprintf(stderr, "orion_kernel_compile: compile failed for %s L%d\n",
                kernel->name, layer_idx);
        return NULL;
    }

    // 3. Store in cache (cache takes ownership)
    orion_cache_store(kernel->name, layer_idx, wb, prog);
    return prog;
}

bool orion_kernel_eval(const OrionKernel* kernel,
                       int layer_idx, int bucket,
                       const OrionModelConfig* cfg,
                       const char* blob_dir,
                       const OrionWeightsBinding* wb,
                       IOSurfaceRef* inputs, int n_inputs,
                       IOSurfaceRef* outputs, int n_outputs) {
    OrionProgram* prog = orion_kernel_compile(kernel, layer_idx, bucket,
                                              cfg, blob_dir, wb);
    if (!prog) return false;

    if (!kernel->adapter_frontend) return orion_eval(prog, inputs, n_inputs, outputs, n_outputs);
    const OrionBindingPlan* cached = orion_program_plan(prog);
    if (!cached) return false;
    return orion_eval_with_plan(prog, cached, inputs, n_inputs, outputs, n_outputs);
}
