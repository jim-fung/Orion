// tests/test_voxtream_depth_ane.m — Opt-in real-artifact VoXtream ANE spike.
//
// Set VOXTREAM_DEPTH_EXPORT to a validated depth export directory. The test
// deliberately skips when the private runtime or checkpoint-bound artifacts
// are unavailable, so compiler-only CI does not claim hardware qualification.

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "compiler/frontends/voxtream_depth.h"
#include "compiler/kernel_adapter.h"
#include "core/ane_runtime.h"
#include "core/iosurface_tensor.h"
#include "core/kernel.h"

static const char* kFullBlobPaths[] = {
    "constants/causal_mask.bin",
    "constants/rope_k_cos.bin",
    "constants/rope_k_perm.bin",
    "constants/rope_k_sin.bin",
    "constants/rope_q_cos.bin",
    "constants/rope_q_perm.bin",
    "constants/rope_q_sin.bin",
    "weights/layer0/rms1.bin",
    "weights/layer0/rms2.bin",
    "weights/layer0/w1.bin",
    "weights/layer0/w2.bin",
    "weights/layer0/w3.bin",
    "weights/layer0/wk.bin",
    "weights/layer0/wo.bin",
    "weights/layer0/wq.bin",
    "weights/layer0/wv.bin",
};

static const char* kAttentionBlobPaths[] = {
    "constants/causal_mask.bin",
    "constants/rope_k_cos.bin",
    "constants/rope_k_perm.bin",
    "constants/rope_k_sin.bin",
    "constants/rope_q_cos.bin",
    "constants/rope_q_perm.bin",
    "constants/rope_q_sin.bin",
    "weights/layer0/rms1.bin",
    "weights/layer0/wk.bin",
    "weights/layer0/wo.bin",
    "weights/layer0/wq.bin",
    "weights/layer0/wv.bin",
};

static const char* kMlpBlobPaths[] = {
    "weights/layer0/rms2.bin",
    "weights/layer0/w1.bin",
    "weights/layer0/w2.bin",
    "weights/layer0/w3.bin",
};

static const char* kQprojBlobPaths[] = {
    "weights/layer0/rms1.bin",
    "weights/layer0/wq.bin",
};

static const char* kMlpUpBlobPaths[] = {
    "weights/layer0/rms2.bin",
    "weights/layer0/w1.bin",
};

static const char* kQropeBlobPaths[] = {
    "constants/rope_q_cos.bin",
    "constants/rope_q_perm.bin",
    "constants/rope_q_sin.bin",
    "weights/layer0/rms1.bin",
    "weights/layer0/wq.bin",
};

static const char* kMlpGateBlobPaths[] = {
    "weights/layer0/rms2.bin",
    "weights/layer0/w1.bin",
    "weights/layer0/w3.bin",
};

static const char* kMlpTwoUpBlobPaths[] = {
    "weights/layer0/rms2.bin",
    "weights/layer0/w1.bin",
    "weights/layer0/w3.bin",
};

static const char* kMlpSiluBlobPaths[] = {
    "weights/layer0/rms2.bin",
    "weights/layer0/w1.bin",
};

static const char* kProbeConvBlobPaths[] = {
    "weights/layer0/wq.bin",
};

static const char* kProbeRmsBlobPaths[] = {
    "weights/layer0/rms1.bin",
};

static const char* kQpermBlobPaths[] = {
    "constants/rope_q_perm.bin",
    "weights/layer0/rms1.bin",
    "weights/layer0/wq.bin",
};

static const char* kQmulBlobPaths[] = {
    "constants/rope_q_cos.bin",
    "constants/rope_q_sin.bin",
    "weights/layer0/rms1.bin",
    "weights/layer0/wq.bin",
};

static const char* kQchainConvBlobPaths[] = {
    "weights/layer0/rms1.bin",
    "weights/layer0/wo.bin",
    "weights/layer0/wq.bin",
};

static const char* kQchainAddBlobPaths[] = {
    "weights/layer0/rms1.bin",
    "weights/layer0/wq.bin",
};

static const char* kQchainMulBlobPaths[] = {
    "constants/rope_q_cos.bin",
    "weights/layer0/rms1.bin",
    "weights/layer0/wq.bin",
};

static const char* kQchainKvBlobPaths[] = {
    "weights/layer0/rms1.bin",
    "weights/layer0/wk.bin",
    "weights/layer0/wq.bin",
};

static const char* kQchainQmulBlobPaths[] = {
    "weights/layer0/rms1.bin",
    "weights/layer0/wq.bin",
};

static const char* kThreeScalesBlobPaths[] = {
    "weights/layer0/rms1.bin",
    "weights/layer1/rms1.bin",
    "weights/layer2/rms1.bin",
};

typedef struct {
    const char* name;
    OrionFrontendFn frontend;
    const char* const* paths;
    size_t path_count;
} SpikeStage;

static const SpikeStage* s_stage;

static NSDictionary* voxtream_depth_wdict(int layer __attribute__((unused)),
                                          int bucket __attribute__((unused)),
                                          const char* blob_dir) {
    if (!blob_dir) return nil;
    NSMutableDictionary* dictionary = [NSMutableDictionary dictionary];
    for (size_t i = 0; i < s_stage->path_count; i++) {
        NSString* relative = @(s_stage->paths[i]);
        NSString* file = [@(blob_dir) stringByAppendingPathComponent:relative];
        NSData* data = [NSData dataWithContentsOfFile:file];
        if (!data) return nil;
        dictionary[[NSString stringWithFormat:@"@model_path/%@", relative]] = @{
            @"offset": @0,
            @"data": data,
        };
    }
    return dictionary;
}

static bool write_output_if_requested(const _Float16* output, size_t bytes) {
    const char* output_path = getenv("VOXTREAM_DEPTH_ANE_OUTPUT");
    if (!output_path || !output_path[0]) return true;
    NSString* path = @(output_path);
    NSData* data = [NSData dataWithBytes:output length:bytes];
    return [data writeToFile:path atomically:YES];
}

int main(void) {
    @autoreleasepool {
        const char* blob_dir = getenv("VOXTREAM_DEPTH_EXPORT");
        if (!blob_dir || !blob_dir[0]) {
            printf("SKIP: set VOXTREAM_DEPTH_EXPORT to a validated export directory\n");
            return 0;
        }
        if (!orion_ane_init()) {
            printf("SKIP: ANE runtime unavailable\n");
            return 0;
        }

        static const SpikeStage stages[] = {
            {"full", orion_frontend_voxtream_depth_layer_adapter,
             kFullBlobPaths, sizeof(kFullBlobPaths) / sizeof(kFullBlobPaths[0])},
            {"attention", orion_frontend_voxtream_depth_attention_adapter,
             kAttentionBlobPaths, sizeof(kAttentionBlobPaths) / sizeof(kAttentionBlobPaths[0])},
            {"mlp", orion_frontend_voxtream_depth_mlp_adapter,
             kMlpBlobPaths, sizeof(kMlpBlobPaths) / sizeof(kMlpBlobPaths[0])},
            {"qproj", orion_frontend_voxtream_depth_qproj_adapter,
             kQprojBlobPaths, sizeof(kQprojBlobPaths) / sizeof(kQprojBlobPaths[0])},
            {"mlp-up", orion_frontend_voxtream_depth_mlp_up_adapter,
             kMlpUpBlobPaths, sizeof(kMlpUpBlobPaths) / sizeof(kMlpUpBlobPaths[0])},
            {"qrope", orion_frontend_voxtream_depth_qrope_adapter,
             kQropeBlobPaths, sizeof(kQropeBlobPaths) / sizeof(kQropeBlobPaths[0])},
            {"mlp-gate", orion_frontend_voxtream_depth_mlp_gate_adapter,
             kMlpGateBlobPaths, sizeof(kMlpGateBlobPaths) / sizeof(kMlpGateBlobPaths[0])},
            {"mlp-two-up", orion_frontend_voxtream_depth_mlp_two_up_adapter,
             kMlpTwoUpBlobPaths, sizeof(kMlpTwoUpBlobPaths) / sizeof(kMlpTwoUpBlobPaths[0])},
            {"mlp-silu", orion_frontend_voxtream_depth_mlp_silu_adapter,
             kMlpSiluBlobPaths, sizeof(kMlpSiluBlobPaths) / sizeof(kMlpSiluBlobPaths[0])},
            {"probe-identity", orion_frontend_voxtream_depth_probe_identity_adapter,
             NULL, 0},
            {"probe-conv", orion_frontend_voxtream_depth_probe_conv_adapter,
             kProbeConvBlobPaths, sizeof(kProbeConvBlobPaths) / sizeof(kProbeConvBlobPaths[0])},
            {"probe-rms", orion_frontend_voxtream_depth_probe_rms_adapter,
             kProbeRmsBlobPaths, sizeof(kProbeRmsBlobPaths) / sizeof(kProbeRmsBlobPaths[0])},
            {"qperm", orion_frontend_voxtream_depth_qperm_adapter,
             kQpermBlobPaths, sizeof(kQpermBlobPaths) / sizeof(kQpermBlobPaths[0])},
            {"qmul", orion_frontend_voxtream_depth_qmul_adapter,
             kQmulBlobPaths, sizeof(kQmulBlobPaths) / sizeof(kQmulBlobPaths[0])},
            {"qchain-conv", orion_frontend_voxtream_depth_qchain_conv_adapter,
             kQchainConvBlobPaths, sizeof(kQchainConvBlobPaths) / sizeof(kQchainConvBlobPaths[0])},
            {"qchain-add", orion_frontend_voxtream_depth_qchain_add_adapter,
             kQchainAddBlobPaths, sizeof(kQchainAddBlobPaths) / sizeof(kQchainAddBlobPaths[0])},
            {"qchain-mul", orion_frontend_voxtream_depth_qchain_mul_adapter,
             kQchainMulBlobPaths, sizeof(kQchainMulBlobPaths) / sizeof(kQchainMulBlobPaths[0])},
            {"qchain-kv", orion_frontend_voxtream_depth_qchain_kv_adapter,
             kQchainKvBlobPaths, sizeof(kQchainKvBlobPaths) / sizeof(kQchainKvBlobPaths[0])},
            {"qchain-qmul", orion_frontend_voxtream_depth_qchain_qmul_adapter,
             kQchainQmulBlobPaths, sizeof(kQchainQmulBlobPaths) / sizeof(kQchainQmulBlobPaths[0])},
            {"qchain-spacer", orion_frontend_voxtream_depth_qchain_spacer_adapter,
             kQchainConvBlobPaths, sizeof(kQchainConvBlobPaths) / sizeof(kQchainConvBlobPaths[0])},
            {"three-scales", orion_frontend_voxtream_depth_three_scales_adapter,
             kThreeScalesBlobPaths, sizeof(kThreeScalesBlobPaths) / sizeof(kThreeScalesBlobPaths[0])},
        };
        const char* requested_stage = getenv("VOXTREAM_DEPTH_STAGE");
        if (!requested_stage || !requested_stage[0]) requested_stage = "full";
        for (size_t i = 0; i < sizeof(stages) / sizeof(stages[0]); i++) {
            if (strcmp(requested_stage, stages[i].name) == 0) {
                s_stage = &stages[i];
                break;
            }
        }
        if (!s_stage) {
            fprintf(stderr, "FAIL: unknown VOXTREAM_DEPTH_STAGE '%s' (full, attention, mlp, qproj, mlp-up, qrope, mlp-gate, mlp-two-up, mlp-silu)\n",
                    requested_stage);
            return 1;
        }

        OrionModelConfig config = {
            .n_layer = ORION_VOXTREAM_DEPTH_LAYERS,
            .n_head = ORION_VOXTREAM_DEPTH_HEADS,
            .d_model = ORION_VOXTREAM_DEPTH_DIM,
            .head_dim = ORION_VOXTREAM_DEPTH_HEAD_DIM,
            .hidden_dim = ORION_VOXTREAM_DEPTH_HIDDEN_DIM,
            .max_seq = ORION_VOXTREAM_DEPTH_BUCKET,
        };
        NSString* artifact_root = @(blob_dir);
        NSString* stem = [NSString stringWithFormat:@"layer0_%s_ane", s_stage->name];
        NSString* json_path = [artifact_root stringByAppendingPathComponent:[stem stringByAppendingString:@".plan.json"]];
        NSString* markdown_path = [artifact_root stringByAppendingPathComponent:[stem stringByAppendingString:@".plan.md"]];
        OrionBindingPlanIdentity identity = {
            .orion_revision = "runtime-spike",
            .compiler_build = "voxtream-depth-spike",
            .weight_root = blob_dir,
        };
        OrionBindingPlan artifact_plan;
        NSString* mil = orion_kernel_adapter_generate_mil_with_artifacts(
            s_stage->frontend, 0,
            ORION_VOXTREAM_DEPTH_BUCKET, &config, &identity,
            json_path.UTF8String, markdown_path.UTF8String, &artifact_plan);
        if (!mil || ![mil writeToFile:[artifact_root stringByAppendingPathComponent:[stem stringByAppendingString:@".mil"]]
                            atomically:YES encoding:NSUTF8StringEncoding error:nil]) {
            fprintf(stderr, "FAIL: unable to generate VoXtream spike artifacts\n");
            return 1;
        }

        NSDictionary* compile_wdict = voxtream_depth_wdict(
            0, ORION_VOXTREAM_DEPTH_BUCKET, blob_dir);
        CFAbsoluteTime compile_start = CFAbsoluteTimeGetCurrent();
        char compile_error[256] = {0};
        OrionProgram* program = getenv("VOXTREAM_DEPTH_LEGACY_COMPILE")
            ? orion_compile_mil(mil.UTF8String, compile_wdict,
                                "voxtream_depth_spike_legacy")
            : orion_compile_mil_with_plan(
                  mil.UTF8String, &artifact_plan, compile_wdict,
                  "voxtream_depth_spike", compile_error,
                  sizeof(compile_error));
        double compile_ms = (CFAbsoluteTimeGetCurrent() - compile_start) * 1000.0;
        if (!program) {
            fprintf(stderr, "FAIL: VoXtream %s stage did not compile (%.2f ms) %s\n",
                    s_stage->name, compile_ms, compile_error);
            return 1;
        }

        const OrionBindingPlan* plan = orion_program_plan(program);
        if (!plan) plan = &artifact_plan;
        if (getenv("VOXTREAM_DEPTH_COMPILE_ONLY")) {
            printf("PASS: VoXtream %s stage compiled in %.2fms graph=%s mil=%s\n",
                   s_stage->name, compile_ms, plan->graph_identity, plan->mil_identity);
            orion_release_program(program);
            return 0;
        }
        OrionSurfaceStatus status = ORION_SURFACE_OK;
        const bool raw_surfaces = getenv("VOXTREAM_DEPTH_RAW_SURFACES") != NULL;
        const int in_channels = plan->inputs[0].declared_shape[1];
        const int in_seq = plan->inputs[0].declared_shape[3];
        IOSurfaceRef input = NULL;
        IOSurfaceRef output = NULL;
        if (raw_surfaces) {
            size_t in_bytes = plan->inputs[0].packed_prefix_bytes;
            input = orion_tensor_create_checked(ORION_SURFACE_DTYPE_FP16,
                                                in_channels, in_seq, in_bytes, &status);
            int out_channels = (int)(plan->outputs[0].packed_prefix_bytes /
                                     (plan->outputs[0].declared_shape[3] * sizeof(_Float16)));
            output = orion_tensor_create_checked(
                ORION_SURFACE_DTYPE_FP16, out_channels, plan->outputs[0].declared_shape[3],
                plan->outputs[0].packed_prefix_bytes, &status);
        } else {
            input = plan ? orion_tensor_create_for_binding(&plan->inputs[0], &status) : NULL;
            output = plan ? orion_tensor_create_for_binding(&plan->outputs[0], &status) : NULL;
        }
        const size_t input_elements = (size_t)in_channels * in_seq;
        const size_t output_elements = plan ?
            plan->outputs[0].packed_prefix_bytes / sizeof(_Float16) : 0;
        const size_t output_bytes = raw_surfaces
            ? plan->outputs[0].packed_prefix_bytes
            : output_elements * sizeof(_Float16);
        _Float16* input_data = calloc(input_elements, sizeof(*input_data));
        _Float16* output_data = calloc(output_bytes / sizeof(_Float16), sizeof(*output_data));
        bool ok = input && output && input_data && output_data &&
                  (status == ORION_SURFACE_OK);
        if (ok) {
            // Batch 0 of prefix-8-guided-step0: logical [S,D] is arange + 7,
            // then packed channel-major as [1,D,1,S].
            for (int channel = 0; channel < in_channels; channel++) {
                for (int token = 0; token < in_seq; token++) {
                    // Tokens [16, 32) are the padded ANE execution slots.
                    input_data[(size_t)channel * in_seq + token] =
                        token < ORION_VOXTREAM_DEPTH_BUCKET
                            ? (_Float16)(((float)(token * in_channels + channel + 7)) /
                                         in_channels)
                            : (_Float16)0;
                }
            }
            if (raw_surfaces) {
                ok = orion_tensor_write_packed_checked(
                         input, ORION_SURFACE_DTYPE_FP16, input_data,
                         plan->inputs[0].packed_prefix_bytes) == ORION_SURFACE_OK;
            } else {
                ok = orion_tensor_write_for_binding(
                         input, &plan->inputs[0], input_data, plan->inputs[0].packed_prefix_bytes) ==
                     ORION_SURFACE_OK;
            }
        }
        CFAbsoluteTime eval_start = CFAbsoluteTimeGetCurrent();
        if (ok) {
            ok = raw_surfaces
                ? orion_eval(program, &input, 1, &output, 1)
                : orion_eval_with_plan(program, plan, &input, 1, &output, 1);
        }
        double eval_ms = (CFAbsoluteTimeGetCurrent() - eval_start) * 1000.0;
        if (ok) {
            if (raw_surfaces) {
                ok = orion_tensor_read_packed_checked(
                         output, ORION_SURFACE_DTYPE_FP16, output_data, output_bytes) ==
                     ORION_SURFACE_OK;
            } else {
                ok = orion_tensor_read_for_binding(
                         output, &plan->outputs[0], output_data,
                         plan->outputs[0].packed_prefix_bytes) == ORION_SURFACE_OK;
            }
        }
        if (ok) {
            for (size_t i = 0; i < output_bytes / sizeof(_Float16); i++) {
                if (!isfinite((float)output_data[i])) {
                    ok = false;
                    break;
                }
            }
        }
        if (ok) ok = write_output_if_requested(output_data, output_bytes);

        free(input_data);
        free(output_data);
        if (input) CFRelease(input);
        if (output) CFRelease(output);
        orion_release_program(program);
        if (!ok) {
            fprintf(stderr, "FAIL: VoXtream %s stage eval/readback failed\n", s_stage->name);
            return 1;
        }
        printf("PASS: VoXtream %s stage compile=%.2fms eval=%.2fms graph=%s mil=%s\n",
               s_stage->name, compile_ms, eval_ms, plan->graph_identity, plan->mil_identity);
        return 0;
    }
}
