// tests/test_iosurface_tensor_checked.m — Checked binding-plan IOSurface ABI.

#import <Foundation/Foundation.h>
#include "core/iosurface_tensor.h"

static int g_passed = 0, g_failed = 0;

#define TEST(name) do { \
    printf("  %-50s ", #name); \
    if (test_##name()) { printf("PASS\n"); g_passed++; } \
    else { printf("FAIL\n"); g_failed++; } \
} while (0)

static bool test_prefix_round_trip(void) {
    OrionSurfaceStatus status = ORION_SURFACE_OK;
    IOSurfaceRef surface = orion_tensor_create_checked(
        ORION_SURFACE_DTYPE_FP16, 4, 4, 128, &status);
    uint8_t input[32], output[32];
    for (size_t i = 0; i < sizeof(input); i++) input[i] = (uint8_t)(i + 1);
    memset(output, 0, sizeof(output));
    bool ok = surface && status == ORION_SURFACE_OK &&
        orion_tensor_write_packed_checked(surface, ORION_SURFACE_DTYPE_FP16,
                                          input, sizeof(input)) == ORION_SURFACE_OK &&
        orion_tensor_read_packed_checked(surface, ORION_SURFACE_DTYPE_FP16,
                                         output, sizeof(output)) == ORION_SURFACE_OK &&
        memcmp(input, output, sizeof(input)) == 0;
    if (surface) CFRelease(surface);
    return ok;
}

static bool test_seqpack_surface_requires_dtype_metadata(void) {
    const int channels = 1024;
    const int seq_len = 64;
    const size_t bytes = (size_t)channels * seq_len * sizeof(_Float16);
    IOSurfaceRef legacy = orion_tensor_create(channels, seq_len);
    OrionSurfaceStatus status = ORION_SURFACE_OK;
    IOSurfaceRef checked = orion_tensor_create_checked(
        ORION_SURFACE_DTYPE_FP16, channels, seq_len, bytes, &status);
    uint8_t *input = calloc(bytes, 1);
    uint8_t *output = calloc(bytes, 1);
    if (input) input[bytes - 1] = 1;
    bool ok = legacy && checked && input && output &&
        status == ORION_SURFACE_OK &&
        orion_tensor_write_packed_checked(
            legacy, ORION_SURFACE_DTYPE_FP16, input,
            bytes) == ORION_SURFACE_DTYPE_MISMATCH &&
        orion_tensor_read_packed_checked(
            legacy, ORION_SURFACE_DTYPE_FP16, output,
            bytes) == ORION_SURFACE_DTYPE_MISMATCH &&
        orion_tensor_write_packed_checked(
            checked, ORION_SURFACE_DTYPE_FP16, input,
            bytes) == ORION_SURFACE_OK &&
        orion_tensor_read_packed_checked(
            checked, ORION_SURFACE_DTYPE_FP16, output,
            bytes) == ORION_SURFACE_OK &&
        memcmp(input, output, bytes) == 0;
    if (legacy) CFRelease(legacy);
    if (checked) CFRelease(checked);
    free(input);
    free(output);
    return ok;
}

static bool test_rejects_overrun_and_small_allocation(void) {
    OrionSurfaceStatus small_status = ORION_SURFACE_OK;
    IOSurfaceRef too_small = orion_tensor_create_checked(
        ORION_SURFACE_DTYPE_FP16, 4, 4, 31, &small_status);
    OrionSurfaceStatus status = ORION_SURFACE_OK;
    IOSurfaceRef surface = orion_tensor_create_checked(
        ORION_SURFACE_DTYPE_FP16, 4, 4, 32, &status);
    uint8_t input[33] = {0};
    bool ok = !too_small && small_status == ORION_SURFACE_ALLOCATION_MISMATCH &&
        status == ORION_SURFACE_OK && surface &&
        orion_tensor_write_packed_checked(surface, ORION_SURFACE_DTYPE_FP16,
                                          input, sizeof(input)) == ORION_SURFACE_DTYPE_MISMATCH;
    if (surface) CFRelease(surface);
    return ok;
}

static OrionBindingPlanEntry binding_entry(OrionDtype dtype) {
    OrionBindingPlanEntry entry = {0};
    snprintf(entry.name, sizeof(entry.name), "x");
    entry.dtype = dtype;
    entry.declared_shape[0] = 1;
    entry.declared_shape[1] = 4;
    entry.declared_shape[2] = 1;
    entry.declared_shape[3] = 4;
    entry.logical_bytes = 32;
    entry.packed_prefix_bytes = 32;
    entry.allocation_bytes = ORION_ANE_MIN_ALLOCATION_BYTES;
    snprintf(entry.layout, sizeof(entry.layout), "packed-NCHW-prefix");
    snprintf(entry.allocation_layout, sizeof(entry.allocation_layout), "flat-byte-buffer");
    return entry;
}

static bool test_plan_bound_prefix_and_dtype(void) {
    OrionBindingPlanEntry fp16 = binding_entry(ORION_DTYPE_FP16);
    OrionBindingPlanEntry fp32 = binding_entry(ORION_DTYPE_FP32);
    fp32.logical_bytes = fp32.packed_prefix_bytes = 64;
    OrionSurfaceStatus status = ORION_SURFACE_OK;
    IOSurfaceRef surface = orion_tensor_create_for_binding(&fp16, &status);
    uint8_t bytes[33] = {0};
    OrionBindingPlanEntry inconsistent = fp16;
    inconsistent.logical_bytes = inconsistent.packed_prefix_bytes = 64;
    bool ok = surface && status == ORION_SURFACE_OK &&
        IOSurfaceGetAllocSize(surface) == ORION_ANE_MIN_ALLOCATION_BYTES &&
        orion_tensor_write_for_binding(surface, &fp16, bytes, 32) == ORION_SURFACE_OK &&
        orion_tensor_write_for_binding(surface, &fp16, bytes, 33) == ORION_SURFACE_COPY_OUT_OF_BOUNDS &&
        orion_tensor_write_for_binding(surface, &fp32, bytes, 32) == ORION_SURFACE_DTYPE_MISMATCH &&
        orion_tensor_validate_for_binding(surface, &inconsistent) == ORION_SURFACE_ALLOCATION_MISMATCH;
    if (surface) CFRelease(surface);
    return ok;
}

static bool test_plan_bound_copy_regions(void) {
    OrionBindingPlanEntry entry = binding_entry(ORION_DTYPE_FP16);
    OrionBindingPlanEntry other = binding_entry(ORION_DTYPE_FP32);
    other.logical_bytes = other.packed_prefix_bytes = 64;
    OrionSurfaceStatus status = ORION_SURFACE_OK;
    IOSurfaceRef src = orion_tensor_create_for_binding(&entry, &status);
    IOSurfaceRef dst = orion_tensor_create_for_binding(&entry, &status);
    bool ok = src && dst &&
        orion_tensor_copy_for_bindings(dst, &entry, 16, src, &entry, 0, 16) == ORION_SURFACE_OK &&
        orion_tensor_copy_for_bindings(dst, &entry, 17, src, &entry, 0, 16) == ORION_SURFACE_COPY_OUT_OF_BOUNDS &&
        orion_tensor_copy_for_bindings(dst, &other, 0, src, &entry, 0, 16) == ORION_SURFACE_DTYPE_MISMATCH;
    if (src) CFRelease(src);
    if (dst) CFRelease(dst);
    return ok;
}

static bool test_same_surface_overlapping_copy(void) {
    OrionBindingPlanEntry entry = binding_entry(ORION_DTYPE_FP16);
    OrionSurfaceStatus status = ORION_SURFACE_OK;
    IOSurfaceRef surface = orion_tensor_create_for_binding(&entry, &status);
    uint8_t bytes[32], result[32] = {0};
    for (size_t i = 0; i < sizeof(bytes); i++) bytes[i] = (uint8_t)i;
    bool ok = surface &&
        orion_tensor_write_for_binding(surface, &entry, bytes, sizeof(bytes)) == ORION_SURFACE_OK &&
        orion_tensor_copy_for_bindings(surface, &entry, 8, surface, &entry, 0, 16) == ORION_SURFACE_OK &&
        orion_tensor_read_for_binding(surface, &entry, result, sizeof(result)) == ORION_SURFACE_OK &&
        memcmp(result + 8, bytes, 16) == 0;
    if (surface) CFRelease(surface);
    return ok;
}

static bool test_copy_bounds_and_f32_conversion(void) {
    OrionSurfaceStatus status = ORION_SURFACE_OK;
    IOSurfaceRef src = orion_tensor_create_checked(ORION_SURFACE_DTYPE_FP16, 4, 4, 64, &status);
    IOSurfaceRef dst = orion_tensor_create_checked(ORION_SURFACE_DTYPE_FP16, 4, 4, 64, &status);
    IOSurfaceRef fp32 = orion_tensor_create_checked(ORION_SURFACE_DTYPE_FP32, 4, 4, 64, &status);
    uint8_t bytes[32];
    float floats[4] = {1.0f, -2.0f, 3.5f, 0.25f};
    for (size_t i = 0; i < sizeof(bytes); i++) bytes[i] = (uint8_t)i;
    bool ok = src && dst &&
        orion_tensor_write_packed_checked(src, ORION_SURFACE_DTYPE_FP16, bytes, sizeof(bytes)) == ORION_SURFACE_OK &&
        orion_tensor_copy_packed_checked(dst, 16, src, 0, 32) == ORION_SURFACE_OK &&
        orion_tensor_copy_packed_checked(dst, 33, src, 0, 32) == ORION_SURFACE_COPY_OUT_OF_BOUNDS &&
        orion_tensor_write_f32_as_fp16_at_checked(dst, 0, floats, 4) == ORION_SURFACE_OK &&
        orion_tensor_write_f32_as_fp16_at_checked(fp32, 0, floats, 4) == ORION_SURFACE_DTYPE_MISMATCH &&
        orion_tensor_write_f32_as_fp16_at_checked(dst, 31, floats, 2) == ORION_SURFACE_COPY_OUT_OF_BOUNDS;
    if (src) CFRelease(src);
    if (dst) CFRelease(dst);
    if (fp32) CFRelease(fp32);
    return ok;
}

int main(void) {
    @autoreleasepool {
        printf("test_iosurface_tensor_checked:\n");
        TEST(prefix_round_trip);
        TEST(seqpack_surface_requires_dtype_metadata);
        TEST(rejects_overrun_and_small_allocation);
        TEST(copy_bounds_and_f32_conversion);
        TEST(plan_bound_prefix_and_dtype);
        TEST(plan_bound_copy_regions);
        TEST(same_surface_overlapping_copy);
        printf("\n%d/%d passed\n", g_passed, g_passed + g_failed);
        return g_failed ? 1 : 0;
    }
}
