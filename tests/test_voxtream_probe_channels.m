// test_voxtream_probe_channels.m
//
// Bounded channel-count probe for the Orion ANE runtime (ORX-102).
//
// Determines whether the M5 Pro ANE compiler accepts a graph at a given channel
// count and sequence length. This is the decisive experiment for channel-packed
// CFG: if 2048 channels @ seq=32 compiles, the latency redesign (halving ANE
// evaluations from 2 to 1 per layer per codebook step) is viable.
//
// Two probe kinds, selected by VOXTREAM_PROBE_KIND:
//   identity (default) — weight-free elementwise add. Probes I/O surface /
//     channel-layout support only. Already PASSED at 2048ch @ seq=32 (0c729dd).
//   conv                — a d×d conv1x1 (linear) projection. Probes conv MAC
//     routing at the channel-packed dimension — the load-bearing op for the
//     real channel-packed depth layer. Requires a synthetic weight blob.
//
// Compile budget: exactly 1 compile on success, 0 on rejection (per Orion's
// success counter; the private compiler's internal limit may count failed
// attempts separately). Hard guard: compile_delta <= 1.
//
// Usage:
//   # identity (already passed — do not re-run on the last diagnostic attempt):
//   VOXTREAM_PROBE_CHANNELS=2048 VOXTREAM_PROBE_SEQ=32 ./build/tests/test_voxtream_probe_channels
//   # conv (the open ORX-102 question):
//   VOXTREAM_PROBE_KIND=conv VOXTREAM_PROBE_CHANNELS=2048 VOXTREAM_PROBE_SEQ=32 \
//       ./build/tests/test_voxtream_probe_channels
//
// Skip behavior: returns 0 (inert) when VOXTREAM_PROBE_CHANNELS is unset or
// the ANE is unavailable, so it does not consume budget under `make test`
// without explicit opt-in.

#import <Foundation/Foundation.h>
#include <stdlib.h>
#include <stdio.h>

#include "compiler/graph.h"
#include "compiler/validate.h"
#include "compiler/pipeline.h"
#include "compiler/codegen.h"
#include "compiler/frontends/voxtream_depth.h"
#include "core/ane_runtime.h"

int main(void) {
    @autoreleasepool {
        // --- gates: opt-in + ANE availability ---
        const char* channels_env = getenv("VOXTREAM_PROBE_CHANNELS");
        if (!channels_env || !*channels_env) {
            printf("SKIP: set VOXTREAM_PROBE_CHANNELS=<N> [VOXTREAM_PROBE_SEQ=<S>] "
                   "[VOXTREAM_PROBE_KIND=identity|conv] to run the channel-count probe\n");
            return 0;
        }
        const int d = atoi(channels_env);
        if (d <= 0) {
            fprintf(stderr, "FAIL: VOXTREAM_PROBE_CHANNELS=%s is not a positive int\n",
                    channels_env);
            return 1;
        }
        const char* seq_env = getenv("VOXTREAM_PROBE_SEQ");
        const int s = (seq_env && *seq_env) ? atoi(seq_env) : 32;
        const char* kind_env = getenv("VOXTREAM_PROBE_KIND");
        NSString* kind = (kind_env && *kind_env)
            ? [NSString stringWithUTF8String:kind_env]
            : @"identity";
        BOOL conv = [kind isEqualToString:@"conv"];

        if (!orion_ane_init()) {
            printf("SKIP: ANE unavailable (private runtime not accessible)\n");
            return 0;
        }

        printf("=== Orion channel-count probe (ORX-102/003) ===\n");
        printf("  channels (d): %d\n", d);
        printf("  seq (s):      %d\n", s);
        NSString* kind_label;
        if ([kind isEqualToString:@"conv"]) {
            kind_label = @"conv (d×d linear projection)";
        } else if ([kind isEqualToString:@"depth_layer"]) {
            kind_label = @"depth_layer (full channel-packed layer graph)";
        } else if ([kind isEqualToString:@"seq_pack"]) {
            kind_label = @"seq_pack (sequence-packed layer graph, 2 branches)";
        } else {
            kind_label = @"identity (weight-free elementwise add)";
        }
        printf("  kind:         %s\n", kind_label.UTF8String);

        // --- build the probe graph ---
        OrionGraph* graph;
        if ([kind isEqualToString:@"conv"]) {
            graph = orion_frontend_voxtream_depth_probe_conv_adapter(0, 0, NULL);
        } else if ([kind isEqualToString:@"depth_layer"]) {
            graph = orion_frontend_voxtream_depth_probe_depth_layer_adapter(0, 0, NULL);
        } else if ([kind isEqualToString:@"seq_pack"]) {
            graph = orion_frontend_voxtream_depth_probe_seq_pack_adapter(0, 0, NULL);
        } else {
            graph = orion_frontend_voxtream_depth_probe_identity_adapter(0, 0, NULL);
        }
        if (!graph) {
            fprintf(stderr, "FAIL: probe graph build returned NULL\n");
            return 1;
        }

        OrionValidationResult vr = orion_graph_validate(graph);
        if (!vr.valid) {
            fprintf(stderr, "FAIL: graph validation: %s\n",
                    vr.message[0] ? vr.message : "(no message)");
            orion_graph_free(graph);
            return 1;
        }

        orion_pipeline_optimize(graph);

        NSString* mil = orion_codegen_mil(graph, "probe_channels");
        orion_graph_free(graph);
        if (!mil || mil.length == 0) {
            fprintf(stderr, "FAIL: MIL codegen produced empty output\n");
            return 1;
        }

        printf("  MIL size:     %lu bytes\n", (unsigned long)mil.length);

        // --- weight dict builder: a valid single-tensor BLOBFILE for an
        // arbitrary rows×cols fp16 weight. Header format per ane_runtime.m:
        // bytes[0]=0x01, [4]=0x02, [64:68]=deadbeef, [68]=0x01,
        // le32([72])=payload_size, le32([80])=128. Payload at byte 128. ---
        NSData* (^make_blob)(int, int) = ^NSData*(int rows, int cols) {
            const size_t n = (size_t)rows * (size_t)cols;
            const size_t payload_bytes = n * sizeof(_Float16);
            NSMutableData* blob = [NSMutableData dataWithLength:128 + payload_bytes];
            uint8_t* bytes = blob.mutableBytes;
            bytes[0] = 0x01; bytes[4] = 0x02;
            bytes[64] = 0xef; bytes[65] = 0xbe; bytes[66] = 0xad; bytes[67] = 0xde;
            bytes[68] = 0x01;
            uint32_t p32 = (uint32_t)payload_bytes;
            bytes[72] = p32 & 0xff; bytes[73] = (p32 >> 8) & 0xff;
            bytes[74] = (p32 >> 16) & 0xff; bytes[75] = (p32 >> 24) & 0xff;
            bytes[80] = 128; bytes[81] = 0; bytes[82] = 0; bytes[83] = 0;
            _Float16* p = (_Float16*)(bytes + 128);
            for (size_t i = 0; i < n; i++) p[i] = (_Float16)((i % 7) * 0.001 - 0.003);
            return blob;
        };
        NSDictionary* (^entry)(NSData*) = ^NSDictionary*(NSData* blob) {
            return @{@"offset": @0, @"data": blob};
        };

        // --- assemble the weight dict per probe kind ---
        NSMutableDictionary* wdict = [NSMutableDictionary dictionary];
        if ([kind isEqualToString:@"conv"]) {
            wdict[@"@model_path/weights/layer0/wq.bin"] = entry(make_blob(d, d));
        } else if ([kind isEqualToString:@"depth_layer"]) {
            // Channel-packed dims: d_packed = 2*d = 2048 (set by the graph
            // builder from VOXTREAM_PROBE_CHANNELS). Per-branch head geometry
            // doubles: n_heads 8->16, n_kv_heads 2->4, head_dim 128.
            const int D = d;  // d from VOXTREAM_PROBE_CHANNELS = 2048
            const int kv = 4 * 128;  // n_kv_heads(4) * head_dim(128) = 512
            const int H = 2 * 8192;  // doubled hidden_dim = 16384
            // Weights: rms1(D), wq(D,D), wk(D,kv), wv(D,kv), wo(D,D),
            //          rms2(D), w1(H,D), w2(D,H), w3(H,D).
            wdict[@"@model_path/weights/layer0/rms1.bin"] = entry(make_blob(1, D));
            wdict[@"@model_path/weights/layer0/wq.bin"]   = entry(make_blob(D, D));
            wdict[@"@model_path/weights/layer0/wk.bin"]   = entry(make_blob(kv, D));
            wdict[@"@model_path/weights/layer0/wv.bin"]   = entry(make_blob(kv, D));
            wdict[@"@model_path/weights/layer0/wo.bin"]   = entry(make_blob(D, D));
            wdict[@"@model_path/weights/layer0/rms2.bin"] = entry(make_blob(1, D));
            wdict[@"@model_path/weights/layer0/w1.bin"]   = entry(make_blob(H, D));
            wdict[@"@model_path/weights/layer0/w2.bin"]   = entry(make_blob(D, H));
            wdict[@"@model_path/weights/layer0/w3.bin"]   = entry(make_blob(H, D));
            // Constants: rope tables + causal mask. Shapes match blobs.py.
            // rope_q: [1, D, 1, s], rope_k: [1, kv, 1, s], perm: [W, W, 1, 1],
            // causal_mask: [1, 1, s, s].
            wdict[@"@model_path/constants/rope_q_cos.bin"]  = entry(make_blob(D, s));
            wdict[@"@model_path/constants/rope_q_sin.bin"]  = entry(make_blob(D, s));
            wdict[@"@model_path/constants/rope_q_perm.bin"] = entry(make_blob(D, D));
            wdict[@"@model_path/constants/rope_k_cos.bin"]  = entry(make_blob(kv, s));
            wdict[@"@model_path/constants/rope_k_sin.bin"]  = entry(make_blob(kv, s));
            wdict[@"@model_path/constants/rope_k_perm.bin"] = entry(make_blob(kv, kv));
            wdict[@"@model_path/constants/causal_mask.bin"] = entry(make_blob(s, s));
        } else if ([kind isEqualToString:@"seq_pack"]) {
            // Sequence-pack uses PRODUCTION dims (d=1024, unchanged). The seq
            // axis is doubled (default 64 = 2x32) to hold both branches. All
            // projection weights are production-shaped; only the constants differ
            // (packed block-diagonal mask + duplicated RoPE tables, _packed.bin).
            const int D = 1024;  // production dim, NOT doubled
            const int kv = 256;  // n_kv_heads(2) * head_dim(128)
            const int H = 8192;  // production hidden_dim
            const int sp = s;    // packed seq (default 64)
            // Production weights (reused unchanged per the ORX-103 proof).
            wdict[@"@model_path/weights/layer0/rms1.bin"] = entry(make_blob(1, D));
            wdict[@"@model_path/weights/layer0/wq.bin"]   = entry(make_blob(D, D));
            wdict[@"@model_path/weights/layer0/wk.bin"]   = entry(make_blob(kv, D));
            wdict[@"@model_path/weights/layer0/wv.bin"]   = entry(make_blob(kv, D));
            wdict[@"@model_path/weights/layer0/wo.bin"]   = entry(make_blob(D, D));
            wdict[@"@model_path/weights/layer0/rms2.bin"] = entry(make_blob(1, D));
            wdict[@"@model_path/weights/layer0/w1.bin"]   = entry(make_blob(H, D));
            wdict[@"@model_path/weights/layer0/w2.bin"]   = entry(make_blob(D, H));
            wdict[@"@model_path/weights/layer0/w3.bin"]   = entry(make_blob(H, D));
            // Packed constants: duplicated RoPE tables + block-diagonal mask.
            // rope_q_packed: [1, D, 1, sp], rope_k_packed: [1, kv, 1, sp].
            wdict[@"@model_path/constants/rope_q_cos_packed.bin"]  = entry(make_blob(D, sp));
            wdict[@"@model_path/constants/rope_q_sin_packed.bin"]  = entry(make_blob(D, sp));
            wdict[@"@model_path/constants/rope_q_perm.bin"]        = entry(make_blob(D, D));
            wdict[@"@model_path/constants/rope_k_cos_packed.bin"]  = entry(make_blob(kv, sp));
            wdict[@"@model_path/constants/rope_k_sin_packed.bin"]  = entry(make_blob(kv, sp));
            wdict[@"@model_path/constants/rope_k_perm.bin"]        = entry(make_blob(kv, kv));
            wdict[@"@model_path/constants/causal_mask_packed.bin"] = entry(make_blob(sp, sp));
        }
        NSDictionary* weight_dict = wdict;

        // --- bounded compile (budget guard: delta <= 1) ---
        // Print ABSOLUTE counts before and after, per PRD §14.4 (not just delta).
        const int count_before = orion_compile_count();
        printf("  compile_count_before: %d\n", count_before);
        OrionProgram* prog = orion_compile_mil(
            mil.UTF8String, weight_dict, "probe-channels");
        const int count_after = orion_compile_count();
        const int compile_delta = count_after - count_before;
        printf("  compile_count_after:  %d\n", count_after);

        // Hard budget guard: this probe must never consume more than 1 compile.
        if (compile_delta > 1) {
            fprintf(stderr,
                    "FAIL: compile budget guard violated: delta=%d (expected <=1)\n",
                    compile_delta);
            if (prog) orion_release_program(prog);
            return 1;
        }

        // --- report ---
        if (prog) {
            printf("\n");
            printf("PROBE PASS: %s @ %d channels @ seq=%d COMPILED (delta=%d)\n",
                   conv ? "conv" : "identity", d, s, compile_delta);
            printf("  -> Channel-packed CFG at this dimension is COMPILER-VIABLE"
                   " for this op class.\n");
            orion_release_program(prog);
            return 0;
        } else {
            printf("\n");
            printf("PROBE FAIL: %s @ %d channels @ seq=%d REJECTED by ANE compiler "
                   "(delta=%d)\n",
                   conv ? "conv" : "identity", d, s, compile_delta);
            printf("  -> Channel-packed CFG at this dimension is NOT viable for this"
                   " op class. Fall back to sequence-pack or graph-fusion redesign.\n");
            return 1;
        }
    }
}
