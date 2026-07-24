// compiler/binding_plan.h — Generated ABI contract for ANE programs.
#ifndef ORION_BINDING_PLAN_H
#define ORION_BINDING_PLAN_H

#include "graph.h"
#include <stddef.h>

#define ORION_BINDING_PLAN_VERSION 3
#define ORION_ANE_MIN_ALLOCATION_BYTES 49152
#define ORION_MAX_PLAN_WEIGHTS 256
#define ORION_SHA256_HEX_LENGTH 64

typedef struct {
    char name[ORION_MAX_NAME];
    OrionDtype dtype;
    int declared_shape[4];
    size_t logical_bytes;
    size_t allocation_bytes;
    size_t packed_prefix_bytes;
    char layout[32];
    char allocation_layout[32];
} OrionBindingPlanEntry;

typedef struct {
    char name[ORION_MAX_NAME];
    char path[256];
    char sha256[ORION_SHA256_HEX_LENGTH + 1];
    uint64_t offset;
    OrionDtype dtype;
    int shape[4];
    size_t logical_bytes;
} OrionBindingPlanWeight;

typedef struct {
    const char* orion_revision;
    const char* compiler_build;
    // Filesystem root substituted for the @model_path prefix when hashing.
    const char* weight_root;
} OrionBindingPlanIdentity;

typedef struct {
    unsigned int version;
    char program_name[ORION_MAX_NAME];
    char graph_identity[ORION_SHA256_HEX_LENGTH + 1];
    // SHA-256 of the exact MIL text compiled with this plan. Empty only before
    // code generation; compilation rejects plans without this binding.
    char mil_identity[ORION_SHA256_HEX_LENGTH + 1];
    char orion_revision[65];
    char compiler_build[128];
    char position_semantics[64];
    char mask_semantics[64];
    char failure_semantics[128];
    int n_inputs;
    OrionBindingPlanEntry inputs[ORION_MAX_GRAPH_IO];
    int n_outputs;
    OrionBindingPlanEntry outputs[ORION_MAX_GRAPH_IO];
    int n_weights;
    OrionBindingPlanWeight weights[ORION_MAX_PLAN_WEIGHTS];
} OrionBindingPlan;

// Builds sorted, dtype-aware input/output ABI entries from graph metadata.
// Multi-I/O groups are padded to a shared allocation size and all external
// allocations meet Orion's empirically required minimum.
bool orion_binding_plan_generate(const OrionGraph* graph, const char* program_name,
                                 OrionBindingPlan* out_plan, char* error,
                                 size_t error_size);

bool orion_binding_plan_generate_with_identity(
    const OrionGraph* graph, const char* program_name,
    const OrionBindingPlanIdentity* identity, OrionBindingPlan* out_plan,
    char* error, size_t error_size);

// Verifies plan ordering, shape, byte accounting, and uniform allocation.
bool orion_binding_plan_validate(const OrionBindingPlan* plan, char* error,
                                 size_t error_size);

// As above, but also requires a valid SHA-256 mil_identity. Use at every trust
// boundary that consumes a finalized plan (artifact publication, runtime eval,
// weight-dictionary verification). The structural validator is used during
// generation before codegen sets mil_identity.
bool orion_binding_plan_validate_finalized(const OrionBindingPlan* plan, char* error,
                                            size_t error_size);

// Binds a generated plan to exact MIL text. Returns false on invalid input.
bool orion_binding_plan_set_mil_identity(OrionBindingPlan* plan, const char* mil_text);

// Writes stable, reviewable artifacts.
bool orion_binding_plan_write_json(const OrionBindingPlan* plan, const char* path,
                                   char* error, size_t error_size);
bool orion_binding_plan_write_markdown(const OrionBindingPlan* plan, const char* path,
                                       char* error, size_t error_size);

#endif
