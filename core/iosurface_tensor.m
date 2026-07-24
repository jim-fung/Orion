#import "iosurface_tensor.h"
#import <arm_neon.h>
#include <limits.h>

static CFStringRef kOrionSurfaceDtypeKey = CFSTR("com.orion.tensor.dtype");

// ANE tensor layout: fp16 [1, C, 1, S]
// IOSurface provides zero-copy shared memory between CPU and ANE.
// IOSurface uses a flat byte buffer; the 4D shape is conceptual (MIL assigns meaning).

static bool checked_mul(size_t left, size_t right, size_t *out) {
    if (left != 0 && right > SIZE_MAX / left) return false;
    *out = left * right;
    return true;
}

static bool checked_add(size_t left, size_t right, size_t *out) {
    if (right > SIZE_MAX - left) return false;
    *out = left + right;
    return true;
}

static size_t dtype_bytes(OrionSurfaceDtype dtype) {
    switch (dtype) {
        case ORION_SURFACE_DTYPE_FP16: return sizeof(_Float16);
        case ORION_SURFACE_DTYPE_FP32: return sizeof(float);
    }
    return 0;
}

static bool surface_has_dtype(IOSurfaceRef surface, OrionSurfaceDtype dtype) {
    if (!surface) return false;
    CFTypeRef stored = IOSurfaceCopyValue(surface, kOrionSurfaceDtypeKey);
    int value = -1;
    if (stored && CFGetTypeID(stored) == CFNumberGetTypeID())
        CFNumberGetValue(stored, kCFNumberIntType, &value);
    if (stored) CFRelease(stored);
    return value == (int)dtype;
}

static bool binding_dtype(OrionDtype dtype, OrionSurfaceDtype *out) {
    if (dtype == ORION_DTYPE_FP16) *out = ORION_SURFACE_DTYPE_FP16;
    else if (dtype == ORION_DTYPE_FP32) *out = ORION_SURFACE_DTYPE_FP32;
    else return false;
    return true;
}

static OrionSurfaceStatus validate_binding_surface(
    IOSurfaceRef surface, const OrionBindingPlanEntry *entry) {
    if (!surface || !entry) return ORION_SURFACE_NULL;
    OrionSurfaceDtype dtype;
    if (!binding_dtype(entry->dtype, &dtype)) return ORION_SURFACE_DTYPE_MISMATCH;
    if (!surface_has_dtype(surface, dtype)) return ORION_SURFACE_DTYPE_MISMATCH;
    size_t elements = 0, expected_bytes = 0;
    if (entry->declared_shape[0] != 1 || entry->declared_shape[1] <= 0 ||
        entry->declared_shape[2] != 1 || entry->declared_shape[3] <= 0 ||
        !checked_mul((size_t)entry->declared_shape[1],
                     (size_t)entry->declared_shape[3], &elements) ||
        !checked_mul(elements, dtype_bytes(dtype), &expected_bytes) ||
        entry->logical_bytes != expected_bytes ||
        entry->packed_prefix_bytes != entry->logical_bytes ||
        entry->allocation_bytes < ORION_ANE_MIN_ALLOCATION_BYTES ||
        IOSurfaceGetAllocSize(surface) != entry->allocation_bytes) {
        return ORION_SURFACE_ALLOCATION_MISMATCH;
    }
    return ORION_SURFACE_OK;
}

OrionSurfaceStatus orion_tensor_validate_for_binding(
    IOSurfaceRef surface, const OrionBindingPlanEntry *entry) {
    return validate_binding_surface(surface, entry);
}

static OrionSurfaceStatus validate_binding_range(
    IOSurfaceRef surface, const OrionBindingPlanEntry *entry,
    size_t offset, size_t byte_count) {
    OrionSurfaceStatus status = validate_binding_surface(surface, entry);
    if (status != ORION_SURFACE_OK) return status;
    size_t end = 0;
    if (!checked_add(offset, byte_count, &end)) return ORION_SURFACE_OVERFLOW;
    return end <= entry->packed_prefix_bytes
        ? ORION_SURFACE_OK : ORION_SURFACE_COPY_OUT_OF_BOUNDS;
}

static OrionSurfaceStatus validate_range(IOSurfaceRef surface, size_t offset,
                                         size_t byte_count) {
    if (!surface) return ORION_SURFACE_NULL;
    size_t end = 0;
    if (!checked_add(offset, byte_count, &end)) return ORION_SURFACE_OVERFLOW;
    if (end > IOSurfaceGetAllocSize(surface)) return ORION_SURFACE_COPY_OUT_OF_BOUNDS;
    return ORION_SURFACE_OK;
}

static OrionSurfaceStatus lock_surface(IOSurfaceRef surface, IOSurfaceLockOptions options) {
    if (!surface) return ORION_SURFACE_NULL;
    return IOSurfaceLock(surface, options, NULL) == kIOReturnSuccess
        ? ORION_SURFACE_OK : ORION_SURFACE_LOCK_FAILED;
}

IOSurfaceRef orion_tensor_create_checked(OrionSurfaceDtype dtype, int channels,
                                         int seq_len, size_t allocation_bytes,
                                         OrionSurfaceStatus *status) {
    if (status) *status = ORION_SURFACE_INVALID_ARGUMENT;
    size_t bytes_per_element = dtype_bytes(dtype);
    if (channels <= 0 || seq_len <= 0 || bytes_per_element == 0) return NULL;

    size_t elements = 0, logical_bytes = 0;
    if (!checked_mul((size_t)channels, (size_t)seq_len, &elements) ||
        !checked_mul(elements, bytes_per_element, &logical_bytes)) {
        if (status) *status = ORION_SURFACE_OVERFLOW;
        return NULL;
    }
    if (allocation_bytes < logical_bytes || allocation_bytes == 0) {
        if (status) *status = ORION_SURFACE_ALLOCATION_MISMATCH;
        return NULL;
    }

    NSDictionary *props = @{
        (id)kIOSurfaceWidth: @(allocation_bytes),
        (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1,
        (id)kIOSurfaceBytesPerRow: @(allocation_bytes),
        (id)kIOSurfaceAllocSize: @(allocation_bytes),
        (id)kIOSurfacePixelFormat: @0,
    };
    IOSurfaceRef surface = IOSurfaceCreate((__bridge CFDictionaryRef)props);
    if (!surface) {
        if (status) *status = ORION_SURFACE_NULL;
        return NULL;
    }
    int dtype_value = (int)dtype;
    CFNumberRef dtype_number = CFNumberCreate(
        kCFAllocatorDefault, kCFNumberIntType, &dtype_value);
    IOSurfaceSetValue(surface, kOrionSurfaceDtypeKey, dtype_number);
    CFRelease(dtype_number);
    if (status) *status = ORION_SURFACE_OK;
    return surface;
}

OrionSurfaceStatus orion_tensor_write_packed_checked(IOSurfaceRef surface,
                                                       OrionSurfaceDtype dtype,
                                                       const void *data,
                                                       size_t packed_prefix_bytes) {
    if (dtype_bytes(dtype) == 0) return ORION_SURFACE_DTYPE_MISMATCH;
    if (!surface_has_dtype(surface, dtype) ||
        packed_prefix_bytes % dtype_bytes(dtype) != 0)
        return ORION_SURFACE_DTYPE_MISMATCH;
    if (!data) return ORION_SURFACE_NULL;
    OrionSurfaceStatus status = validate_range(surface, 0, packed_prefix_bytes);
    if (status != ORION_SURFACE_OK) return status;
    status = lock_surface(surface, 0);
    if (status != ORION_SURFACE_OK) return status;
    void *base = IOSurfaceGetBaseAddress(surface);
    if (!base) {
        IOSurfaceUnlock(surface, 0, NULL);
        return ORION_SURFACE_NULL;
    }
    memcpy(base, data, packed_prefix_bytes);
    return IOSurfaceUnlock(surface, 0, NULL) == kIOReturnSuccess
        ? ORION_SURFACE_OK : ORION_SURFACE_LOCK_FAILED;
}

OrionSurfaceStatus orion_tensor_read_packed_checked(IOSurfaceRef surface,
                                                      OrionSurfaceDtype dtype,
                                                      void *data,
                                                      size_t packed_prefix_bytes) {
    if (dtype_bytes(dtype) == 0) return ORION_SURFACE_DTYPE_MISMATCH;
    if (!surface_has_dtype(surface, dtype) ||
        packed_prefix_bytes % dtype_bytes(dtype) != 0)
        return ORION_SURFACE_DTYPE_MISMATCH;
    if (!data) return ORION_SURFACE_NULL;
    OrionSurfaceStatus status = validate_range(surface, 0, packed_prefix_bytes);
    if (status != ORION_SURFACE_OK) return status;
    status = lock_surface(surface, kIOSurfaceLockReadOnly);
    if (status != ORION_SURFACE_OK) return status;
    const void *base = IOSurfaceGetBaseAddress(surface);
    if (!base) {
        IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, NULL);
        return ORION_SURFACE_NULL;
    }
    memcpy(data, base, packed_prefix_bytes);
    return IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, NULL) == kIOReturnSuccess
        ? ORION_SURFACE_OK : ORION_SURFACE_LOCK_FAILED;
}

OrionSurfaceStatus orion_tensor_copy_packed_checked(IOSurfaceRef dst,
                                                      size_t dst_offset_bytes,
                                                      IOSurfaceRef src,
                                                      size_t src_offset_bytes,
                                                      size_t byte_count) {
    OrionSurfaceStatus status = validate_range(src, src_offset_bytes, byte_count);
    if (status != ORION_SURFACE_OK) return status;
    status = validate_range(dst, dst_offset_bytes, byte_count);
    if (status != ORION_SURFACE_OK) return status;
    if (src == dst) {
        status = lock_surface(src, 0);
        if (status != ORION_SURFACE_OK) return status;
        uint8_t *base = IOSurfaceGetBaseAddress(src);
        if (!base) {
            IOSurfaceUnlock(src, 0, NULL);
            return ORION_SURFACE_NULL;
        }
        memmove(base + dst_offset_bytes, base + src_offset_bytes, byte_count);
        return IOSurfaceUnlock(src, 0, NULL) == kIOReturnSuccess
            ? ORION_SURFACE_OK : ORION_SURFACE_LOCK_FAILED;
    }
    status = lock_surface(src, kIOSurfaceLockReadOnly);
    if (status != ORION_SURFACE_OK) return status;
    status = lock_surface(dst, 0);
    if (status != ORION_SURFACE_OK) {
        IOSurfaceUnlock(src, kIOSurfaceLockReadOnly, NULL);
        return status;
    }
    const uint8_t *src_base = IOSurfaceGetBaseAddress(src);
    uint8_t *dst_base = IOSurfaceGetBaseAddress(dst);
    if (!src_base || !dst_base) {
        IOSurfaceUnlock(dst, 0, NULL);
        IOSurfaceUnlock(src, kIOSurfaceLockReadOnly, NULL);
        return ORION_SURFACE_NULL;
    }
    memcpy(dst_base + dst_offset_bytes, src_base + src_offset_bytes, byte_count);
    IOReturn dst_unlock = IOSurfaceUnlock(dst, 0, NULL);
    IOReturn src_unlock = IOSurfaceUnlock(src, kIOSurfaceLockReadOnly, NULL);
    return dst_unlock == kIOReturnSuccess && src_unlock == kIOReturnSuccess
        ? ORION_SURFACE_OK : ORION_SURFACE_LOCK_FAILED;
}

OrionSurfaceStatus orion_tensor_write_f32_as_fp16_at_checked(
    IOSurfaceRef dst, size_t dst_offset_elements, const float *data,
    size_t element_count) {
    if (!data) return ORION_SURFACE_NULL;
    if (!surface_has_dtype(dst, ORION_SURFACE_DTYPE_FP16))
        return ORION_SURFACE_DTYPE_MISMATCH;
    size_t offset_bytes = 0, byte_count = 0;
    if (!checked_mul(dst_offset_elements, sizeof(_Float16), &offset_bytes) ||
        !checked_mul(element_count, sizeof(_Float16), &byte_count)) {
        return ORION_SURFACE_OVERFLOW;
    }
    OrionSurfaceStatus status = validate_range(dst, offset_bytes, byte_count);
    if (status != ORION_SURFACE_OK) return status;
    status = lock_surface(dst, 0);
    if (status != ORION_SURFACE_OK) return status;
    uint8_t *base = IOSurfaceGetBaseAddress(dst);
    if (!base) {
        IOSurfaceUnlock(dst, 0, NULL);
        return ORION_SURFACE_NULL;
    }
    _Float16 *out = (_Float16 *)(base + offset_bytes);
    for (size_t index = 0; index < element_count; index++) out[index] = (_Float16)data[index];
    return IOSurfaceUnlock(dst, 0, NULL) == kIOReturnSuccess
        ? ORION_SURFACE_OK : ORION_SURFACE_LOCK_FAILED;
}

IOSurfaceRef orion_tensor_create_for_binding(const OrionBindingPlanEntry *entry,
                                              OrionSurfaceStatus *status) {
    if (status) *status = ORION_SURFACE_INVALID_ARGUMENT;
    if (!entry) return NULL;
    OrionSurfaceDtype dtype;
    if (!binding_dtype(entry->dtype, &dtype)) {
        if (status) *status = ORION_SURFACE_DTYPE_MISMATCH;
        return NULL;
    }
    size_t elements = 0, expected_bytes = 0;
    if (entry->declared_shape[0] != 1 || entry->declared_shape[1] <= 0 ||
        entry->declared_shape[2] != 1 || entry->declared_shape[3] <= 0 ||
        !checked_mul((size_t)entry->declared_shape[1],
                     (size_t)entry->declared_shape[3], &elements) ||
        !checked_mul(elements, dtype_bytes(dtype), &expected_bytes) ||
        entry->logical_bytes != expected_bytes ||
        entry->packed_prefix_bytes != expected_bytes ||
        entry->allocation_bytes < ORION_ANE_MIN_ALLOCATION_BYTES) return NULL;
    return orion_tensor_create_checked(dtype, entry->declared_shape[1],
                                       entry->declared_shape[3],
                                       entry->allocation_bytes, status);
}

OrionSurfaceStatus orion_tensor_write_for_binding(
    IOSurfaceRef surface, const OrionBindingPlanEntry *entry,
    const void *data, size_t byte_count) {
    OrionSurfaceStatus status = validate_binding_range(surface, entry, 0, byte_count);
    if (status != ORION_SURFACE_OK) return status;
    OrionSurfaceDtype dtype;
    if (!binding_dtype(entry->dtype, &dtype)) return ORION_SURFACE_DTYPE_MISMATCH;
    return orion_tensor_write_packed_checked(surface, dtype, data, byte_count);
}

OrionSurfaceStatus orion_tensor_read_for_binding(
    IOSurfaceRef surface, const OrionBindingPlanEntry *entry,
    void *data, size_t byte_count) {
    OrionSurfaceStatus status = validate_binding_range(surface, entry, 0, byte_count);
    if (status != ORION_SURFACE_OK) return status;
    OrionSurfaceDtype dtype;
    if (!binding_dtype(entry->dtype, &dtype)) return ORION_SURFACE_DTYPE_MISMATCH;
    return orion_tensor_read_packed_checked(surface, dtype, data, byte_count);
}

OrionSurfaceStatus orion_tensor_copy_for_bindings(
    IOSurfaceRef dst, const OrionBindingPlanEntry *dst_entry, size_t dst_offset_bytes,
    IOSurfaceRef src, const OrionBindingPlanEntry *src_entry, size_t src_offset_bytes,
    size_t byte_count) {
    if (!dst_entry || !src_entry) return ORION_SURFACE_NULL;
    if (dst_entry->dtype != src_entry->dtype) return ORION_SURFACE_DTYPE_MISMATCH;
    OrionSurfaceStatus status = validate_binding_range(
        src, src_entry, src_offset_bytes, byte_count);
    if (status != ORION_SURFACE_OK) return status;
    status = validate_binding_range(dst, dst_entry, dst_offset_bytes, byte_count);
    if (status != ORION_SURFACE_OK) return status;
    return orion_tensor_copy_packed_checked(
        dst, dst_offset_bytes, src, src_offset_bytes, byte_count);
}

#pragma mark - T012: Create

IOSurfaceRef orion_tensor_create(int channels, int seq_len) {
    if (channels <= 0 || seq_len <= 0) return NULL;

    // ANE uses fp16 (2 bytes per element). Total elements = C * S.
    size_t bytes = (size_t)channels * seq_len * sizeof(_Float16);

    // IOSurface properties: flat 1D byte buffer (width=bytes, height=1).
    // The 4D [1, C, 1, S] layout is defined by the MIL program, not the IOSurface.
    NSDictionary *props = @{
        (id)kIOSurfaceWidth:           @(bytes),
        (id)kIOSurfaceHeight:          @1,
        (id)kIOSurfaceBytesPerElement:  @1,
        (id)kIOSurfaceBytesPerRow:      @(bytes),
        (id)kIOSurfaceAllocSize:        @(bytes),
        (id)kIOSurfacePixelFormat:      @0
    };

    return IOSurfaceCreate((__bridge CFDictionaryRef)props);
}

IOSurfaceRef orion_tensor_create_f32(int channels, int seq_len) {
    if (channels <= 0 || seq_len <= 0) return NULL;

    // fp32: 4 bytes per element
    size_t bytes = (size_t)channels * seq_len * sizeof(float);

    NSDictionary *props = @{
        (id)kIOSurfaceWidth:           @(bytes),
        (id)kIOSurfaceHeight:          @1,
        (id)kIOSurfaceBytesPerElement:  @1,
        (id)kIOSurfaceBytesPerRow:      @(bytes),
        (id)kIOSurfaceAllocSize:        @(bytes),
        (id)kIOSurfacePixelFormat:      @0
    };

    return IOSurfaceCreate((__bridge CFDictionaryRef)props);
}

#pragma mark - T013: Raw fp16 Read/Write

void orion_tensor_write(IOSurfaceRef surface, const void* data, size_t size) {
    if (!surface || !data || size == 0) return;
    IOSurfaceLock(surface, 0, NULL);
    memcpy(IOSurfaceGetBaseAddress(surface), data, size);
    IOSurfaceUnlock(surface, 0, NULL);
}

void orion_tensor_read(IOSurfaceRef surface, void* data, size_t size) {
    if (!surface || !data || size == 0) return;
    IOSurfaceLock(surface, kIOSurfaceLockReadOnly, NULL);
    memcpy(data, IOSurfaceGetBaseAddress(surface), size);
    IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, NULL);
}

#pragma mark - T014: NEON fp16↔fp32 Conversion

void orion_tensor_write_f32(IOSurfaceRef surface, const float* data, int count) {
    if (!surface || !data || count <= 0) return;
    IOSurfaceLock(surface, 0, NULL);
    _Float16 *dst = (_Float16 *)IOSurfaceGetBaseAddress(surface);

    // NEON-accelerated f32 → f16 conversion (8 elements at a time)
    int i = 0;
    for (; i + 7 < count; i += 8) {
        float16x8_t h = vcombine_f16(
            vcvt_f16_f32(vld1q_f32(data + i)),
            vcvt_f16_f32(vld1q_f32(data + i + 4)));
        vst1q_f16((__fp16 *)(dst + i), h);
    }
    // Scalar tail
    for (; i < count; i++) {
        dst[i] = (_Float16)data[i];
    }

    IOSurfaceUnlock(surface, 0, NULL);
}

void orion_tensor_read_f32(IOSurfaceRef surface, float* data, int count) {
    if (!surface || !data || count <= 0) return;
    IOSurfaceLock(surface, kIOSurfaceLockReadOnly, NULL);
    const _Float16 *src = (const _Float16 *)IOSurfaceGetBaseAddress(surface);

    // NEON-accelerated f16 → f32 conversion (8 elements at a time)
    int i = 0;
    for (; i + 7 < count; i += 8) {
        float16x8_t h = vld1q_f16((const __fp16 *)(src + i));
        vst1q_f32(data + i,     vcvt_f32_f16(vget_low_f16(h)));
        vst1q_f32(data + i + 4, vcvt_f32_f16(vget_high_f16(h)));
    }
    // Scalar tail
    for (; i < count; i++) {
        data[i] = (float)src[i];
    }

    IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, NULL);
}

void orion_tensor_read_f32_direct(IOSurfaceRef surface, float* data, int count) {
    if (!surface || !data || count <= 0) return;
    IOSurfaceLock(surface, kIOSurfaceLockReadOnly, NULL);
    memcpy(data, IOSurfaceGetBaseAddress(surface), (size_t)count * sizeof(float));
    IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, NULL);
}

#pragma mark - Region Copy

void orion_tensor_copy_into(IOSurfaceRef dst, int dst_ch_offset,
                             IOSurfaceRef src, int channels, int seq_len) {
    if (!dst || !src || channels <= 0 || seq_len <= 0) return;

    size_t elem_count = (size_t)channels * seq_len;
    size_t byte_count = elem_count * sizeof(_Float16);
    size_t dst_offset = (size_t)dst_ch_offset * seq_len * sizeof(_Float16);

    IOSurfaceLock(src, kIOSurfaceLockReadOnly, NULL);
    IOSurfaceLock(dst, 0, NULL);

    const void *src_ptr = IOSurfaceGetBaseAddress(src);
    void *dst_ptr = (uint8_t *)IOSurfaceGetBaseAddress(dst) + dst_offset;
    memcpy(dst_ptr, src_ptr, byte_count);

    IOSurfaceUnlock(dst, 0, NULL);
    IOSurfaceUnlock(src, kIOSurfaceLockReadOnly, NULL);
}

void orion_tensor_write_f32_at(IOSurfaceRef dst, int dst_ch_offset,
                                const float* data, int channels, int seq_len) {
    if (!dst || !data || channels <= 0 || seq_len <= 0) return;

    int count = channels * seq_len;
    size_t dst_offset = (size_t)dst_ch_offset * seq_len * sizeof(_Float16);

    IOSurfaceLock(dst, 0, NULL);
    _Float16 *dst_ptr = (_Float16 *)((uint8_t *)IOSurfaceGetBaseAddress(dst) + dst_offset);

    // NEON f32 → f16 conversion
    int i = 0;
    for (; i + 7 < count; i += 8) {
        float16x8_t h = vcombine_f16(
            vcvt_f16_f32(vld1q_f32(data + i)),
            vcvt_f16_f32(vld1q_f32(data + i + 4)));
        vst1q_f16((__fp16 *)(dst_ptr + i), h);
    }
    for (; i < count; i++) {
        dst_ptr[i] = (_Float16)data[i];
    }

    IOSurfaceUnlock(dst, 0, NULL);
}

#pragma mark - Release

void orion_tensor_release(IOSurfaceRef surface) {
    if (surface) {
        CFRelease(surface);
    }
}
