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
    // Compile (or fetch from cache) first. On a cache hit we must hold a lease
    // for the duration of evaluation: a concurrent eviction could otherwise
    // remove the entry and free the program between lookup and eval.
    OrionBindingPlan plan;
    NSString* mil = nil;
    void* lease = NULL;
    OrionProgram* prog = NULL;

    if (kernel->adapter_frontend) {
        mil = adapter_mil_and_plan(kernel, layer_idx, bucket, cfg, blob_dir, &plan);
        if (!mil) return false;
        // Use the retained lookup so eval is protected from concurrent eviction.
        prog = orion_cache_lookup_retained(kernel->name, layer_idx, wb, &lease);
        const OrionBindingPlan* cached_plan = prog ? orion_program_plan(prog) : NULL;
        if (!prog || !cached_plan ||
            strcmp(cached_plan->graph_identity, plan.graph_identity) != 0 ||
            strcmp(cached_plan->mil_identity, plan.mil_identity) != 0) {
            // Miss or stale: drop the lease before compiling.
            orion_cache_release(lease);
            lease = NULL;
            prog = NULL;
        }
    }

    if (!prog) {
        // Cache miss path — compile, store, then take a fresh lease for eval.
        NSDictionary* wdict = kernel->build_wdict(layer_idx, bucket, blob_dir);
        char tag[64];
        snprintf(tag, sizeof(tag), "%s_L%d", kernel->name, layer_idx);
        char plan_error[256] = {0};
        prog = kernel->adapter_frontend
            ? orion_compile_mil_with_plan(mil.UTF8String, &plan, wdict, tag,
                                          plan_error, sizeof(plan_error))
            : (orion_compile_mil((kernel->generate_mil
                                  ? kernel->generate_mil(layer_idx, bucket, cfg)
                                  : nil).UTF8String, wdict, tag));
        if (!prog) return false;
        orion_cache_store(kernel->name, layer_idx, wb, prog);
        // Re-acquire a lease on the freshly stored program so eval is protected.
        prog = orion_cache_lookup_retained(kernel->name, layer_idx, wb, &lease);
        if (!prog) return false;
    }

    bool ok;
    if (!kernel->adapter_frontend) {
        ok = orion_eval(prog, inputs, n_inputs, outputs, n_outputs);
    } else {
        const OrionBindingPlan* cached = orion_program_plan(prog);
        ok = cached && orion_eval_with_plan(prog, cached, inputs, n_inputs, outputs, n_outputs);
    }

    // Drop the lease now that evaluation is complete.
    orion_cache_release(lease);
    return ok;
}
