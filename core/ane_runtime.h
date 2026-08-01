#ifndef ORION_ANE_RUNTIME_H
#define ORION_ANE_RUNTIME_H

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#include "compiler/binding_plan.h"
#include "model_config.h"

/// Opaque handle to a compiled ANE program.
typedef struct OrionProgram OrionProgram;

/// Initialize the ANE runtime (dlopen framework, resolve classes).
/// Must be called once before any other orion_* call.
/// @return true on success, false if ANE is unavailable.
bool orion_ane_init(void);

/// Compile MIL text with weight blobs into an ANE program.
/// @param mil_text       MIL program source text (null-terminated)
/// @param weight_dict    NSDictionary mapping BLOBFILE paths to @{@"offset":@N, @"data":NSData*},
///                       or nil/@{} for weight-free programs.
/// @param program_tag    Human-readable tag for debugging (may be NULL)
/// @return Compiled program handle, or NULL on failure.
OrionProgram* orion_compile_mil(
    const char* mil_text,
    NSDictionary* weight_dict,
    const char* program_tag
);

/// Verify that a BLOBFILE dictionary exactly matches a generated binding plan.
/// Each weight must be present through its BLOBFILE path, use dictionary offset
/// zero, contain one exact BLOBFILE, and match the plan's chunk payload range
/// and SHA-256. Aggregate BLOBFILE paths may satisfy multiple plan weights.
bool orion_validate_weight_dictionary_for_plan(
    const OrionBindingPlan* plan, NSDictionary* weight_dict
);

/// Compile only after validating the BLOBFILE dictionary against the plan.
/// On validation failure, sets @c error and returns NULL (distinguishable from
/// a private-compiler failure, which leaves @c error empty).
OrionProgram* orion_compile_mil_with_plan(
    const char* mil_text, const OrionBindingPlan* plan,
    NSDictionary* weight_dict, const char* program_tag,
    char* error, size_t error_size
);

/// Returns the binding plan cached inside a compiled program, or NULL if the
/// program was compiled without a plan (e.g. via orion_compile_mil).
/// The pointer is valid for the lifetime of @c prog.
const OrionBindingPlan* orion_program_plan(const OrionProgram* prog);

/// Returns the program's on-disk model directory, or NULL. Intended for
/// diagnostics and tooling that inspects compiled model artifacts.
/// The pointer is valid until the next call on the same thread.
const char* orion_program_model_dir(const OrionProgram* prog);

/// Evaluate a compiled ANE program.
/// @param prog        Compiled program handle
/// @param inputs      Array of input IOSurface tensors
/// @param num_inputs  Number of inputs
/// @param outputs     Array of output IOSurface tensors (pre-allocated)
/// @param num_outputs Number of outputs
/// @return true on success.
bool orion_eval(
    OrionProgram* prog,
    IOSurfaceRef* inputs, int num_inputs,
    IOSurfaceRef* outputs, int num_outputs
);

/// Validate surfaces against the generated ABI plan, then evaluate them in
/// the plan's alphabetical input/output binding order.
bool orion_eval_with_plan(
    OrionProgram* prog, const OrionBindingPlan* plan,
    IOSurfaceRef* inputs, int num_inputs,
    IOSurfaceRef* outputs, int num_outputs
);

/// Release a compiled program and its resources (unload from ANE, clean temp dir).
void orion_release_program(OrionProgram* prog);

/// Get the number of successful compilations this process has done.
/// Useful for tracking approach to the ~119 compile limit.
int orion_compile_count(void);

/// Create a new ANE program with patched weights, reusing compiled artifacts
/// from a donor program. Skips compilation entirely — only loads. The patch
/// is rejected when the private runtime would alias the donor's model path.
///
/// The ANE compiler's "data" file IS the BLOBFILE. By creating a new model
/// identity and copying the compiled net.plist + new BLOBFILE, we can load
/// new weights without compiling. For a plan-bound donor, the returned program
/// retains an independently rebound plan whose provenance identity is the
/// patched MIL text and whose weight hashes describe the replacement chunks.
/// Evaluation compares that exact cached provenance plan; callers must obtain
/// it with orion_program_plan() rather than reusing the donor plan.
///
/// @param donor        A previously compiled OrionProgram (provides net.plist)
/// @param mil_text     Same MIL text used to compile the donor
/// @param weight_dict  New weight dict (same keys, new data)
/// @param program_tag  Tag for debugging (may be NULL)
/// @return New program handle with patched weights, or NULL on failure.
///         Caller must release with orion_release_program().
///         Does NOT increment orion_compile_count().
OrionProgram* orion_program_patch_weights(
    OrionProgram* donor,
    const char* mil_text,
    NSDictionary* weight_dict,
    const char* program_tag
);

/// Legacy in-place reload is disabled because private model-file replacement
/// cannot be made transactional. Use orion_program_patch_weights instead.
///
/// @param prog         The program to reload (must be loaded)
/// @param weight_dict  New weight dict (same keys, new data)
/// @return Always false.
bool orion_program_reload_weights(
    OrionProgram* prog,
    NSDictionary* weight_dict
);

#endif // ORION_ANE_RUNTIME_H
