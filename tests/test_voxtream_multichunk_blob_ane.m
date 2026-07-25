// tests/test_voxtream_multichunk_blob_ane.m
//
// Bounded compiler-format probe: determine whether two distinct large arrays
// in one physical BLOBFILE can be referenced at independent chunk offsets.
// This intentionally bypasses the production one-tensor-per-file plan contract.

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core/ane_runtime.h"
#include "core/iosurface_tensor.h"

#define PROBE_DIM 1024
#define PROBE_SEQ 32
#define BLOB_HEADER_BYTES 128
#define CHUNK_HEADER_BYTES 64
#define FIRST_CHUNK_OFFSET 64

static void write_le32(uint8_t *dst, uint32_t value) {
    dst[0] = (uint8_t)value;
    dst[1] = (uint8_t)(value >> 8);
    dst[2] = (uint8_t)(value >> 16);
    dst[3] = (uint8_t)(value >> 24);
}

static uint32_t read_le32(const uint8_t *src) {
    return (uint32_t)src[0] | ((uint32_t)src[1] << 8) |
           ((uint32_t)src[2] << 16) | ((uint32_t)src[3] << 24);
}

static size_t align64(size_t value) {
    return (value + 63u) & ~(size_t)63u;
}

static NSData *read_valid_single_blob(NSString *path, size_t expected_payload) {
    NSData *data = [NSData dataWithContentsOfFile:path];
    if (!data || data.length != BLOB_HEADER_BYTES + expected_payload) return nil;
    const uint8_t *bytes = data.bytes;
    if (bytes[0] != 1 || bytes[4] != 2 ||
        bytes[64] != 0xef || bytes[65] != 0xbe ||
        bytes[66] != 0xad || bytes[67] != 0xde || bytes[68] != 1 ||
        read_le32(bytes + 72) != expected_payload || read_le32(bytes + 80) != 128) {
        return nil;
    }
    return data;
}

static void write_chunk_header(uint8_t *bytes, size_t chunk_offset,
                               size_t payload_bytes, size_t payload_offset) {
    bytes[chunk_offset] = 0xef;
    bytes[chunk_offset + 1] = 0xbe;
    bytes[chunk_offset + 2] = 0xad;
    bytes[chunk_offset + 3] = 0xde;
    bytes[chunk_offset + 4] = 1;
    write_le32(bytes + chunk_offset + 8, (uint32_t)payload_bytes);
    write_le32(bytes + chunk_offset + 16, (uint32_t)payload_offset);
}

static NSData *aggregate_blob(NSData *first, NSData *second, int varied_chunk,
                              uint64_t *second_chunk_offset) {
    const size_t payload_bytes = (size_t)PROBE_DIM * PROBE_DIM * sizeof(_Float16);
    const size_t first_payload_offset = BLOB_HEADER_BYTES;
    const size_t second_header = align64(first_payload_offset + payload_bytes);
    const size_t second_payload_offset = second_header + CHUNK_HEADER_BYTES;
    const size_t total_bytes = second_payload_offset + payload_bytes;
    NSMutableData *aggregate = [NSMutableData dataWithLength:total_bytes];
    uint8_t *out = aggregate.mutableBytes;
    out[0] = 1;
    out[4] = 2;
    write_chunk_header(out, FIRST_CHUNK_OFFSET, payload_bytes, first_payload_offset);
    write_chunk_header(out, second_header, payload_bytes, second_payload_offset);
    memcpy(out + first_payload_offset,
           (const uint8_t *)first.bytes + BLOB_HEADER_BYTES, payload_bytes);
    memcpy(out + second_payload_offset,
           (const uint8_t *)second.bytes + BLOB_HEADER_BYTES, payload_bytes);

    // Scale exactly one chunk by 0.5. This remains finite fp16 data while
    // ensuring each aggregate variant has distinct bytes and semantics.
    if (varied_chunk == 1 || varied_chunk == 2) {
        size_t offset = varied_chunk == 1 ? first_payload_offset : second_payload_offset;
        _Float16 *values = (_Float16 *)(out + offset);
        size_t count = payload_bytes / sizeof(*values);
        for (size_t i = 0; i < count; i++) values[i] = (_Float16)((float)values[i] * 0.5f);
    }
    *second_chunk_offset = second_header;
    return aggregate;
}

static NSString *linear_mil(NSString *path, uint64_t offset, NSString *weight_name,
                            NSString *output_name) {
    return [NSString stringWithFormat:
        @"program(1.3)\n"
         "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n"
         "{\n"
         "    func main<ios18>(tensor<fp16, [1,1024,1,32]> x) {\n"
         "        string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n"
         "        tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n"
         "        tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
         "        tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n"
         "        int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"
         "        tensor<fp16, [1024,1024,1,1]> %@ = const()[name=string(\"%@\"), val=tensor<fp16, [1024,1024,1,1]>(BLOBFILE(path=string(\"%@\"), offset=uint64(%llu)))];\n"
         "        tensor<fp16, [1,1024,1,32]> %@ = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=%@, x=x)[name=string(\"%@\")];\n"
         "    } -> (%@);\n"
         "}\n",
         weight_name, weight_name, path, offset,
         output_name, weight_name, output_name, output_name];
}

static NSString *aggregate_mil(uint64_t second_offset) {
    NSString *first = linear_mil(@"@model_path/aggregate.bin", FIRST_CHUNK_OFFSET,
                                 @"wq", @"q");
    // Build the dependent two-convolution graph directly to avoid brittle text replacement.
    (void)first;
    return [NSString stringWithFormat:
        @"program(1.3)\n"
         "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n"
         "{\n"
         "    func main<ios18>(tensor<fp16, [1,1024,1,32]> x) {\n"
         "        string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n"
         "        tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n"
         "        tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
         "        tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n"
         "        int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"
         "        tensor<fp16, [1024,1024,1,1]> wq = const()[name=string(\"wq\"), val=tensor<fp16, [1024,1024,1,1]>(BLOBFILE(path=string(\"@model_path/aggregate.bin\"), offset=uint64(64)))];\n"
         "        tensor<fp16, [1024,1024,1,1]> wo = const()[name=string(\"wo\"), val=tensor<fp16, [1024,1024,1,1]>(BLOBFILE(path=string(\"@model_path/aggregate.bin\"), offset=uint64(%llu)))];\n"
         "        tensor<fp16, [1,1024,1,32]> q = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=wq, x=x)[name=string(\"q\")];\n"
         "        tensor<fp16, [1,1024,1,32]> y = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=wo, x=q)[name=string(\"y\")];\n"
         "    } -> (y);\n"
         "}\n", second_offset];
}

static NSDictionary *wdict(NSString *path, NSData *data) {
    return @{path: @{@"offset": @0, @"data": data}};
}

static bool eval_program(OrionProgram *program, const _Float16 *input, _Float16 *output) {
    const size_t elements = (size_t)PROBE_DIM * PROBE_SEQ;
    IOSurfaceRef in = orion_tensor_create(PROBE_DIM, PROBE_SEQ);
    IOSurfaceRef out = orion_tensor_create(PROBE_DIM, PROBE_SEQ);
    bool ok = in && out;
    if (ok) {
        orion_tensor_write(in, input, elements * sizeof(*input));
        ok = orion_eval(program, &in, 1, &out, 1);
    }
    if (ok) orion_tensor_read(out, output, elements * sizeof(*output));
    if (in) CFRelease(in);
    if (out) CFRelease(out);
    return ok;
}

static double max_abs(const _Float16 *a, const _Float16 *b, size_t count) {
    double maximum = 0.0;
    for (size_t i = 0; i < count; i++) {
        double diff = fabs((double)a[i] - (double)b[i]);
        if (diff > maximum) maximum = diff;
    }
    return maximum;
}

int main(void) {
    @autoreleasepool {
        const char *root_env = getenv("VOXTREAM_DEPTH_EXPORT");
        if (!root_env || !root_env[0]) {
            printf("SKIP: set VOXTREAM_DEPTH_EXPORT to a validated export directory\n");
            return 0;
        }
        if (!orion_ane_init()) {
            printf("SKIP: ANE runtime unavailable\n");
            return 0;
        }
        NSString *root = @(root_env);
        const size_t payload_bytes = (size_t)PROBE_DIM * PROBE_DIM * sizeof(_Float16);
        NSData *wq = read_valid_single_blob(
            [root stringByAppendingPathComponent:@"weights/layer0/wq.bin"], payload_bytes);
        NSData *wo = read_valid_single_blob(
            [root stringByAppendingPathComponent:@"weights/layer0/wo.bin"], payload_bytes);
        if (!wq || !wo) {
            fprintf(stderr, "FAIL: expected validated layer-0 wq/wo blobs\n");
            return 1;
        }

        uint64_t second_offset = 0;
        NSData *aggregate = aggregate_blob(wq, wo, 0, &second_offset);
        NSData *varied_wq = aggregate_blob(wq, wo, 1, &second_offset);
        NSData *varied_wo = aggregate_blob(wq, wo, 2, &second_offset);
        NSString *aggregate_text = aggregate_mil(second_offset);

        const int count_before = orion_compile_count();
        OrionProgram *combined = orion_compile_mil(
            aggregate_text.UTF8String,
            wdict(@"@model_path/aggregate.bin", aggregate), "multichunk-baseline");
        if (!combined) {
            printf("STOP: multi-chunk baseline rejected by private compiler; archive Orion depth path\n");
            return 1;
        }
        OrionProgram *combined_wq = orion_compile_mil(
            aggregate_text.UTF8String,
            wdict(@"@model_path/aggregate.bin", varied_wq), "multichunk-varied-wq");
        OrionProgram *combined_wo = orion_compile_mil(
            aggregate_text.UTF8String,
            wdict(@"@model_path/aggregate.bin", varied_wo), "multichunk-varied-wo");
        OrionProgram *control_wq = orion_compile_mil(
            linear_mil(@"@model_path/wq.bin", 64, @"wq", @"q").UTF8String,
            wdict(@"@model_path/wq.bin", wq), "multichunk-control-wq");
        OrionProgram *control_wo = orion_compile_mil(
            linear_mil(@"@model_path/wo.bin", 64, @"wo", @"y").UTF8String,
            wdict(@"@model_path/wo.bin", wo), "multichunk-control-wo");

        const size_t elements = (size_t)PROBE_DIM * PROBE_SEQ;
        _Float16 *input = calloc(elements, sizeof(*input));
        _Float16 *baseline = calloc(elements, sizeof(*baseline));
        _Float16 *changed_wq = calloc(elements, sizeof(*changed_wq));
        _Float16 *changed_wo = calloc(elements, sizeof(*changed_wo));
        _Float16 *intermediate = calloc(elements, sizeof(*intermediate));
        _Float16 *control = calloc(elements, sizeof(*control));
        for (size_t i = 0; i < elements; i++) input[i] = (_Float16)(((int)(i % 97) - 48) / 256.0f);

        bool ok = combined_wq && combined_wo && control_wq && control_wo &&
                  input && baseline && changed_wq && changed_wo && intermediate && control &&
                  eval_program(combined, input, baseline) &&
                  eval_program(combined_wq, input, changed_wq) &&
                  eval_program(combined_wo, input, changed_wo) &&
                  eval_program(control_wq, input, intermediate) &&
                  eval_program(control_wo, intermediate, control);
        double baseline_control = ok ? max_abs(baseline, control, elements) : INFINITY;
        double wq_influence = ok ? max_abs(baseline, changed_wq, elements) : 0.0;
        double wo_influence = ok ? max_abs(baseline, changed_wo, elements) : 0.0;
        int compiles = orion_compile_count() - count_before;
        ok = ok && compiles == 5 && baseline_control <= 0.03 &&
             wq_influence > 1e-4 && wo_influence > 1e-4;

        printf("multi-chunk probe: second_offset=%llu compiles=%d parity_max_abs=%.8f "
               "wq_influence=%.8f wo_influence=%.8f\n",
               second_offset, compiles, baseline_control, wq_influence, wo_influence);
        printf("%s: two distinct large tensors in one BLOBFILE %s\n",
               ok ? "PASS" : "STOP", ok ? "compile independently and preserve parity"
                                         : "failed compile/load/influence/parity gate");

        orion_release_program(combined);
        orion_release_program(combined_wq);
        orion_release_program(combined_wo);
        orion_release_program(control_wq);
        orion_release_program(control_wo);
        free(input); free(baseline); free(changed_wq); free(changed_wo);
        free(intermediate); free(control);
        return ok ? 0 : 1;
    }
}
