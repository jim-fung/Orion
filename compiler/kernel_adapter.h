// compiler/kernel_adapter.h — T137: OrionKernel adapter for compiler frontends
// ObjC (bridges graph frontends to OrionKernel interface).

#ifndef ORION_KERNEL_ADAPTER_H
#define ORION_KERNEL_ADAPTER_H

#import "core/kernel.h"
#include "graph.h"
#include "binding_plan.h"

// Frontend function type: builds a graph from model config (3-arg: layer, bucket, cfg).
typedef OrionGraph* (*OrionFrontendFn)(int layer_idx, int bucket, const OrionModelConfig* cfg);

// Frontend function type for kernels that don't use bucket (2-arg: layer, cfg).
typedef OrionGraph* (*OrionFrontend2Fn)(int layer_idx, const OrionModelConfig* cfg);

// Direct API: generate MIL from a 3-arg frontend function.
// Builds graph -> validates -> optimizes -> codegen.
NSString* orion_kernel_adapter_generate_mil(OrionFrontendFn frontend,
                                             int layer_idx, int bucket,
                                              const OrionModelConfig* cfg);

// As above, but also returns a generated, validated ABI binding plan.
NSString* orion_kernel_adapter_generate_mil_with_plan(OrionFrontendFn frontend,
                                                        int layer_idx, int bucket,
                                                        const OrionModelConfig* cfg,
                                                         OrionBindingPlan* out_plan);

// As above, with the source identity required for weighted plans.
NSString* orion_kernel_adapter_generate_mil_with_plan_identity(
    OrionFrontendFn frontend, int layer_idx, int bucket,
    const OrionModelConfig* cfg, const OrionBindingPlanIdentity* identity,
    OrionBindingPlan* out_plan);

// Qualification entry point: MIL and plan come from the same optimized graph,
// and deterministic JSON/Markdown plan artifacts are emitted before return.
NSString* orion_kernel_adapter_generate_mil_with_artifacts(
    OrionFrontendFn frontend, int layer_idx, int bucket,
    const OrionModelConfig* cfg, const OrionBindingPlanIdentity* identity,
    const char* json_path, const char* markdown_path, OrionBindingPlan* out_plan);

// Direct API: generate MIL from a 2-arg frontend function (no bucket).
NSString* orion_kernel_adapter_generate_mil_2arg(OrionFrontend2Fn frontend,
                                                  int layer_idx,
                                                  const OrionModelConfig* cfg);

// Create an OrionKernel from a compiler frontend.
// The generate_mil implementation: call frontend -> optimize -> codegen -> return MIL string.
// This is a drop-in replacement for hand-written milgen functions.
OrionKernel orion_kernel_from_frontend(const char* name,
                                        OrionFrontendFn frontend,
                                        OrionWDictFn wdict,
                                        int n_inputs, int n_outputs);

#endif // ORION_KERNEL_ADAPTER_H
