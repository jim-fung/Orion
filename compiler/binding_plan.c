// compiler/binding_plan.c — Generated ABI contract for ANE programs.
#include "binding_plan.h"

#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <CommonCrypto/CommonDigest.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

static void set_error(char* error, size_t error_size, const char* fmt, ...) {
    if (!error || error_size == 0) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(error, error_size, fmt, ap);
    va_end(ap);
}

static size_t dtype_size(OrionDtype dtype) {
    switch (dtype) {
        case ORION_DTYPE_FP16: return 2;
        case ORION_DTYPE_FP32: return 4;
        case ORION_DTYPE_INT32: return 4;
        case ORION_DTYPE_BOOL: return 1;
        default: return 0;
    }
}

static bool shape_bytes(const int shape[4], OrionDtype dtype, size_t* out) {
    size_t value = dtype_size(dtype);
    if (value == 0) return false;
    for (int i = 0; i < 4; i++) {
        if (shape[i] <= 0 || value > SIZE_MAX / (size_t)shape[i]) return false;
        value *= (size_t)shape[i];
    }
    *out = value;
    return true;
}

static bool bounded_nonempty(const char* value, size_t capacity) {
    return value && memchr(value, '\0', capacity) != NULL && value[0] != '\0';
}

static bool bounded_compare(const char* left, const char* right, size_t capacity, int* result) {
    if (!bounded_nonempty(left, capacity) || !bounded_nonempty(right, capacity)) return false;
    *result = strcmp(left, right);
    return true;
}

static bool valid_sha256(const char* value, size_t capacity) {
    if (!value || !memchr(value, '\0', capacity) || strnlen(value, capacity) != ORION_SHA256_HEX_LENGTH) return false;
    for (int i = 0; i < ORION_SHA256_HEX_LENGTH; i++) {
        if (!((value[i] >= '0' && value[i] <= '9') ||
              (value[i] >= 'a' && value[i] <= 'f') ||
              (value[i] >= 'A' && value[i] <= 'F'))) return false;
    }
    return true;
}

static int entry_compare(const void* a, const void* b) {
    const OrionBindingPlanEntry* left = a;
    const OrionBindingPlanEntry* right = b;
    return strcmp(left->name, right->name);
}

static int weight_compare(const void* a, const void* b) {
    const OrionBindingPlanWeight* left = a;
    const OrionBindingPlanWeight* right = b;
    return strcmp(left->name, right->name);
}

static void hex_digest(const unsigned char* digest, char out[65]) {
    for (int i = 0; i < CC_SHA256_DIGEST_LENGTH; i++)
        snprintf(out + i * 2, 3, "%02x", digest[i]);
}

static void hash_u32(CC_SHA256_CTX *context, uint32_t value) {
    unsigned char bytes[4] = {
        (unsigned char)value, (unsigned char)(value >> 8),
        (unsigned char)(value >> 16), (unsigned char)(value >> 24),
    };
    CC_SHA256_Update(context, bytes, sizeof(bytes));
}

static void hash_u64(CC_SHA256_CTX *context, uint64_t value) {
    for (int shift = 0; shift < 64; shift += 8) {
        unsigned char byte = (unsigned char)(value >> shift);
        CC_SHA256_Update(context, &byte, 1);
    }
}

static void hash_string(CC_SHA256_CTX *context, const char *value) {
    size_t length = strlen(value);
    hash_u32(context, (uint32_t)length);
    CC_SHA256_Update(context, value, (CC_LONG)length);
}

static void graph_identity(const OrionGraph* graph, char out[65]) {
    CC_SHA256_CTX context;
    CC_SHA256_Init(&context);
    hash_u32(&context, (uint32_t)graph->n_nodes);
    for (int i = 0; i < graph->n_nodes; i++) {
        const OrionNode* node = &graph->nodes[i];
        if (!node->is_live) continue;
        hash_u32(&context, (uint32_t)i);
        hash_u32(&context, (uint32_t)node->op);
        hash_string(&context, node->name);
        hash_u32(&context, (uint32_t)node->dtype);
        for (int j = 0; j < 4; j++) hash_u32(&context, (uint32_t)node->shape[j]);
        hash_u32(&context, (uint32_t)node->n_inputs);
        for (int j = 0; j < node->n_inputs; j++)
            hash_u32(&context, (uint32_t)node->inputs[j]);
        for (int j = 0; j < 4; j++) hash_u32(&context, (uint32_t)node->attrs.perm[j]);
        hash_u32(&context, (uint32_t)node->attrs.axis);
        hash_u32(&context, node->attrs.keep_dims ? 1u : 0u);
        hash_u32(&context, node->attrs.transpose_x ? 1u : 0u);
        hash_u32(&context, node->attrs.transpose_y ? 1u : 0u);
        hash_string(&context, node->attrs.blob_path);
        hash_u64(&context, node->attrs.blob_offset);
        uint32_t scalar_bits = 0;
        memcpy(&scalar_bits, &node->attrs.scalar_val, sizeof(scalar_bits));
        hash_u32(&context, scalar_bits);
        hash_u32(&context, (uint32_t)node->attrs.cast_dtype);
        hash_u32(&context, (uint32_t)node->attrs.bias_input);
        for (int j = 0; j < 4; j++) {
            hash_u32(&context, (uint32_t)node->attrs.slice_begin[j]);
            hash_u32(&context, (uint32_t)node->attrs.slice_end[j]);
        }
    }
    const OrionGraphIO *groups[] = {graph->inputs, graph->outputs};
    int counts[] = {graph->n_inputs, graph->n_outputs};
    for (int group = 0; group < 2; group++) {
        hash_u32(&context, (uint32_t)counts[group]);
        for (int i = 0; i < counts[group]; i++) {
            hash_string(&context, groups[group][i].name);
            hash_u32(&context, (uint32_t)groups[group][i].node_idx);
        }
    }
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256_Final(digest, &context);
    hex_digest(digest, out);
}

// Open a weight file confined under weight_root using openat to prevent
// TOCTOU on intermediate directory components. Each path segment is opened
// with O_NOFOLLOW; ".." and absolute components are rejected.
static int open_confined_fd(const char* weight_root, const char* relative_path,
                            char* error, size_t error_size) {
    int root_fd = open(weight_root, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    if (root_fd < 0) {
        set_error(error, error_size, "cannot open weight root '%s'", weight_root);
        return -1;
    }
    char path_copy[512];
    if (strlen(relative_path) >= sizeof(path_copy)) {
        set_error(error, error_size, "weight relative path is too long");
        close(root_fd);
        return -1;
    }
    snprintf(path_copy, sizeof(path_copy), "%s", relative_path);
    int current_fd = root_fd;
    bool root_owned = true;
    char* save = NULL;
    for (char* tok = strtok_r(path_copy, "/", &save); tok; tok = strtok_r(NULL, "/", &save)) {
        if (tok[0] == '\0' || strcmp(tok, ".") == 0) continue;
        if (strcmp(tok, "..") == 0 || tok[0] == '/') {
            if (root_owned) close(current_fd);
            set_error(error, error_size, "weight path contains an escaping component");
            return -1;
        }
        bool is_last = (save == NULL || save[0] == '\0');
        int next_fd = openat(current_fd, tok,
                             O_RDONLY | O_NOFOLLOW | O_CLOEXEC | (is_last ? 0 : O_DIRECTORY),
                             0);
        if (root_owned) { close(current_fd); root_owned = false; }
        if (next_fd < 0) {
            set_error(error, error_size, "cannot open weight path component '%s'", tok);
            return -1;
        }
        current_fd = next_fd;
    }
    if (root_owned) {
        close(current_fd);
        set_error(error, error_size, "weight relative path is empty");
        return -1;
    }
    return current_fd;
}

static uint32_t read_le32(const unsigned char *bytes) {
    return (uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8) |
           ((uint32_t)bytes[2] << 16) | ((uint32_t)bytes[3] << 24);
}

// Validate the individual 64-byte BLOBFILE chunk named by a plan weight.
// The payload declaration belongs to that chunk header, not to the first chunk
// or to the aggregate file as a whole.
static bool validate_blobfile_for_weight(int fd, uint64_t offset, size_t logical_bytes,
                                         uint64_t* out_payload_offset) {
    unsigned char preamble[5];
    unsigned char header[64];
    struct stat statbuf;
    if (fstat(fd, &statbuf) != 0 || !S_ISREG(statbuf.st_mode) ||
        statbuf.st_size < (off_t)sizeof(preamble) ||
        pread(fd, preamble, sizeof(preamble), 0) != (ssize_t)sizeof(preamble) ||
        preamble[0] != 0x01 || preamble[4] != 0x02 ||
        offset > UINT64_MAX - sizeof(header) ||
        offset > (uint64_t)statbuf.st_size ||
        sizeof(header) > (uint64_t)statbuf.st_size - offset ||
        pread(fd, header, sizeof(header), (off_t)offset) != (ssize_t)sizeof(header)) return false;
    uint64_t payload_size = read_le32(header + 8);
    uint64_t payload_offset = read_le32(header + 16);
    if (memcmp(header, (unsigned char[]){0xef, 0xbe, 0xad, 0xde}, 4) != 0 ||
        header[4] != 0x01 || payload_size != logical_bytes ||
        payload_offset < offset + sizeof(header) ||
        payload_offset > (uint64_t)statbuf.st_size ||
        payload_size > (uint64_t)statbuf.st_size - payload_offset) return false;
    if (out_payload_offset) *out_payload_offset = payload_offset;
    return true;
}

static bool ranges_overlap(uint64_t left_offset, size_t left_size,
                           uint64_t right_offset, size_t right_size) {
    if (left_offset > UINT64_MAX - left_size || right_offset > UINT64_MAX - right_size)
        return true;
    return left_offset < right_offset + right_size && right_offset < left_offset + left_size;
}

// Hash a byte range [offset, offset+length) within an open file descriptor.
static bool fd_range_sha256(int fd, uint64_t offset, size_t length, char out[65]) {
    CC_SHA256_CTX context;
    CC_SHA256_Init(&context);
    unsigned char buffer[16384];
    size_t remaining = length;
    uint64_t pos = offset;
    while (remaining > 0) {
        size_t to_read = remaining < sizeof(buffer) ? remaining : sizeof(buffer);
        ssize_t count = pread(fd, buffer, to_read, (off_t)pos);
        if (count <= 0) return false;
        CC_SHA256_Update(&context, buffer, (CC_LONG)count);
        remaining -= (size_t)count;
        pos += (uint64_t)count;
    }
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256_Final(digest, &context);
    hex_digest(digest, out);
    return true;
}

static bool copy_string(char* dst, size_t dst_size, const char* src,
                        const char* field, char* error, size_t error_size) {
    if (!src || !src[0] || strlen(src) >= dst_size) {
        set_error(error, error_size, "%s is missing or too long", field);
        return false;
    }
    memcpy(dst, src, strlen(src) + 1);
    return true;
}

static bool fill_entries(const OrionGraph* graph, const OrionGraphIO* ios, int count,
                         bool is_inputs, OrionBindingPlanEntry* entries,
                         char* error, size_t error_size) {
    size_t allocation = ORION_ANE_MIN_ALLOCATION_BYTES;
    for (int i = 0; i < count; i++) {
        if (ios[i].node_idx < 0 || ios[i].node_idx >= graph->n_nodes) {
            set_error(error, error_size, "binding '%s' references invalid node", ios[i].name);
            return false;
        }
        const OrionNode* node = &graph->nodes[ios[i].node_idx];
        if (node->shape[0] != 1 || node->shape[2] != 1) {
            set_error(error, error_size, "binding '%s' is not [1,C,1,S]", ios[i].name);
            return false;
        }
        size_t bytes = 0;
        if (!shape_bytes(node->shape, node->dtype, &bytes)) {
            set_error(error, error_size, "binding '%s' has invalid dtype or shape", ios[i].name);
            return false;
        }
        if (bytes > INT_MAX) {
            set_error(error, error_size, "binding '%s' allocation exceeds plan shape range",
                      ios[i].name);
            return false;
        }
        if (bytes > allocation) allocation = bytes;
        memset(&entries[i], 0, sizeof(entries[i]));
        // Inputs are named by their function argument. MIL return values are
        // named by the live node expression, not the GraphIO display label.
        const char* binding_name = is_inputs ? ios[i].name : node->name;
        snprintf(entries[i].name, sizeof(entries[i].name), "%s", binding_name);
        entries[i].dtype = node->dtype;
        memcpy(entries[i].declared_shape, node->shape, sizeof(node->shape));
        entries[i].logical_bytes = bytes;
        entries[i].packed_prefix_bytes = bytes;
        snprintf(entries[i].layout, sizeof(entries[i].layout), "packed-NCHW-prefix");
        snprintf(entries[i].allocation_layout, sizeof(entries[i].allocation_layout),
                 "flat-byte-buffer");
    }
    qsort(entries, (size_t)count, sizeof(*entries), entry_compare);
    for (int i = 0; i < count; i++) {
        entries[i].allocation_bytes = allocation;
    }
    return true;
}

bool orion_binding_plan_generate_with_identity(
    const OrionGraph* graph, const char* program_name,
    const OrionBindingPlanIdentity* identity, OrionBindingPlan* out_plan,
    char* error, size_t error_size) {
    if (!graph || !out_plan || !program_name || !program_name[0]) {
        set_error(error, error_size, "graph, plan, and program name are required");
        return false;
    }
    if (identity && (!identity->orion_revision || !identity->orion_revision[0] ||
                     !identity->compiler_build || !identity->compiler_build[0])) {
        set_error(error, error_size,
                  "qualified plans require Orion revision and compiler build identity");
        return false;
    }
    if (graph->n_inputs < 1 || graph->n_inputs > ORION_MAX_GRAPH_IO ||
        graph->n_outputs < 1 || graph->n_outputs > ORION_MAX_GRAPH_IO) {
        set_error(error, error_size, "graph has invalid I/O count");
        return false;
    }
    memset(out_plan, 0, sizeof(*out_plan));
    uint64_t payload_offsets[ORION_MAX_PLAN_WEIGHTS] = {0};
    out_plan->version = ORION_BINDING_PLAN_VERSION;
    if (!copy_string(out_plan->program_name, sizeof(out_plan->program_name), program_name,
                     "program name", error, error_size)) return false;
    graph_identity(graph, out_plan->graph_identity);
    if (!copy_string(out_plan->orion_revision, sizeof(out_plan->orion_revision),
                     identity ? identity->orion_revision : "unspecified",
                     "Orion revision", error, error_size) ||
        !copy_string(out_plan->compiler_build, sizeof(out_plan->compiler_build),
                     identity ? identity->compiler_build : "unspecified",
                     "compiler build", error, error_size)) return false;
    snprintf(out_plan->position_semantics, sizeof(out_plan->position_semantics),
             "frontend-declared-static-positions");
    snprintf(out_plan->mask_semantics, sizeof(out_plan->mask_semantics),
             "explicit-additive-mask-no-sdpa-mask");
    snprintf(out_plan->failure_semantics, sizeof(out_plan->failure_semantics),
             "reject-before-compile; no implicit binding or allocation fallback");
    out_plan->n_inputs = graph->n_inputs;
    out_plan->n_outputs = graph->n_outputs;
    if (!fill_entries(graph, graph->inputs, graph->n_inputs, true, out_plan->inputs, error, error_size) ||
        !fill_entries(graph, graph->outputs, graph->n_outputs, false, out_plan->outputs, error, error_size)) {
        return false;
    }
    for (int i = 0; i < graph->n_nodes; i++) {
        const OrionNode* node = &graph->nodes[i];
        if (!node->is_live || node->op != ORION_OP_CONST || !node->attrs.blob_path[0]) continue;
        if (out_plan->n_weights >= ORION_MAX_PLAN_WEIGHTS) {
            set_error(error, error_size, "graph exceeds binding-plan weight limit");
            return false;
        }
        OrionBindingPlanWeight* weight = &out_plan->weights[out_plan->n_weights++];
        if (!copy_string(weight->name, sizeof(weight->name), node->name,
                         "weight name", error, error_size) ||
            !copy_string(weight->path, sizeof(weight->path), node->attrs.blob_path,
                         "weight path", error, error_size)) return false;
        weight->offset = node->attrs.blob_offset;
        weight->dtype = node->dtype;
        memcpy(weight->shape, node->shape, sizeof(weight->shape));
        if (!shape_bytes(weight->shape, weight->dtype, &weight->logical_bytes)) {
            set_error(error, error_size, "weight '%s' has invalid dtype or shape", weight->name);
            return false;
        }
        if (!identity || !identity->weight_root ||
            strncmp(weight->path, "@model_path/", 12) != 0) {
            set_error(error, error_size, "weight '%s' requires a weight root for hashing",
                      weight->name);
            return false;
        }
        int fd = open_confined_fd(identity->weight_root, weight->path + 12,
                                  error, error_size);
        if (fd < 0) {
            set_error(error, error_size, "cannot open weight '%s'", weight->name);
            return false;
        }
        uint64_t payload_offset = 0;
        bool valid_file = validate_blobfile_for_weight(fd, weight->offset,
                                                       weight->logical_bytes,
                                                       &payload_offset);
        if (valid_file) {
            for (int prior = 0; prior < out_plan->n_weights - 1; prior++) {
                const OrionBindingPlanWeight* other = &out_plan->weights[prior];
                if (strcmp(other->path, weight->path) == 0 &&
                    ranges_overlap(payload_offsets[prior], other->logical_bytes,
                                   payload_offset, weight->logical_bytes)) {
                    valid_file = false;
                    break;
                }
            }
        }
        if (valid_file)
            valid_file = fd_range_sha256(fd, payload_offset, weight->logical_bytes,
                                         weight->sha256);
        if (fd >= 0) close(fd);
        if (!valid_file) {
            set_error(error, error_size, "cannot hash weight '%s'", weight->name);
            return false;
        }
        payload_offsets[out_plan->n_weights - 1] = payload_offset;
    }
    qsort(out_plan->weights, (size_t)out_plan->n_weights,
          sizeof(out_plan->weights[0]), weight_compare);
    return orion_binding_plan_validate(out_plan, error, error_size);
}

bool orion_binding_plan_generate(const OrionGraph* graph, const char* program_name,
                                 OrionBindingPlan* out_plan, char* error,
                                 size_t error_size) {
    return orion_binding_plan_generate_with_identity(
        graph, program_name, NULL, out_plan, error, error_size);
}

static bool validate_entries(const OrionBindingPlanEntry* entries, int count,
                             char* error, size_t error_size) {
    size_t allocation = count ? entries[0].allocation_bytes : 0;
    for (int i = 0; i < count; i++) {
        const OrionBindingPlanEntry* entry = &entries[i];
        int order = 0;
        if (!bounded_nonempty(entry->name, sizeof(entry->name)) ||
            !bounded_nonempty(entry->layout, sizeof(entry->layout)) ||
            !bounded_nonempty(entry->allocation_layout, sizeof(entry->allocation_layout)) ||
            (i && (!bounded_compare(entries[i - 1].name, entry->name,
                                    sizeof(entry->name), &order) || order >= 0))) {
            set_error(error, error_size, "binding names must be terminated, unique, and alphabetical");
            return false;
        }
        size_t expected = 0;
        if (entry->declared_shape[0] != 1 || entry->declared_shape[2] != 1 ||
            !shape_bytes(entry->declared_shape, entry->dtype, &expected) ||
            entry->logical_bytes != expected || entry->packed_prefix_bytes != expected ||
            entry->allocation_bytes != allocation || allocation < expected ||
            allocation < ORION_ANE_MIN_ALLOCATION_BYTES ||
            strcmp(entry->layout, "packed-NCHW-prefix") != 0 ||
            strcmp(entry->allocation_layout, "flat-byte-buffer") != 0) {
            set_error(error, error_size, "binding '%s' has invalid allocation metadata", entry->name);
            return false;
        }
    }
    return true;
}

bool orion_binding_plan_validate(const OrionBindingPlan* plan, char* error,
                                  size_t error_size) {
    if (!plan || plan->version != ORION_BINDING_PLAN_VERSION ||
        !bounded_nonempty(plan->program_name, sizeof(plan->program_name)) ||
        !valid_sha256(plan->graph_identity, sizeof(plan->graph_identity)) ||
        !bounded_nonempty(plan->orion_revision, sizeof(plan->orion_revision)) ||
        !bounded_nonempty(plan->compiler_build, sizeof(plan->compiler_build)) ||
        !bounded_nonempty(plan->position_semantics, sizeof(plan->position_semantics)) ||
        !bounded_nonempty(plan->mask_semantics, sizeof(plan->mask_semantics)) ||
        !bounded_nonempty(plan->failure_semantics, sizeof(plan->failure_semantics)) ||
        plan->n_inputs < 1 || plan->n_outputs < 1 ||
        plan->n_inputs > ORION_MAX_GRAPH_IO || plan->n_outputs > ORION_MAX_GRAPH_IO) {
        set_error(error, error_size, "invalid binding plan header");
        return false;
    }
    if (!validate_entries(plan->inputs, plan->n_inputs, error, error_size) ||
        !validate_entries(plan->outputs, plan->n_outputs, error, error_size)) return false;
    if (plan->n_weights < 0 || plan->n_weights > ORION_MAX_PLAN_WEIGHTS) {
        set_error(error, error_size, "invalid weight count");
        return false;
    }
    for (int i = 0; i < plan->n_weights; i++) {
        const OrionBindingPlanWeight* weight = &plan->weights[i];
        size_t expected = 0;
        int order = 0;
        if (!bounded_nonempty(weight->name, sizeof(weight->name)) ||
            !bounded_nonempty(weight->path, sizeof(weight->path)) ||
            !valid_sha256(weight->sha256, sizeof(weight->sha256)) || weight->offset < 64 ||
            (i && (!bounded_compare(plan->weights[i - 1].name, weight->name,
                                    sizeof(weight->name), &order) || order >= 0)) ||
            !shape_bytes(weight->shape, weight->dtype, &expected) ||
            weight->logical_bytes != expected) {
            set_error(error, error_size, "binding-plan weight has invalid metadata");
            return false;
        }
    }
    return true;
}

bool orion_binding_plan_validate_finalized(const OrionBindingPlan* plan, char* error,
                                            size_t error_size) {
    if (!orion_binding_plan_validate(plan, error, error_size)) return false;
    if (!valid_sha256(plan->mil_identity, sizeof(plan->mil_identity))) {
        set_error(error, error_size, "finalized plan requires a valid MIL identity");
        return false;
    }
    return true;
}

bool orion_binding_plan_set_mil_identity(OrionBindingPlan* plan, const char* mil_text) {
    if (!plan || !mil_text || !mil_text[0]) return false;
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(mil_text, (CC_LONG)strlen(mil_text), digest);
    hex_digest(digest, plan->mil_identity);
    return true;
}

static void json_string(FILE* file, const char* value) {
    fputc('"', file);
    for (const unsigned char* p = (const unsigned char*)value; *p; p++) {
        if (*p == '"' || *p == '\\') fprintf(file, "\\%c", *p);
        else if (*p < 0x20) fprintf(file, "\\u%04x", *p);
        else fputc(*p, file);
    }
    fputc('"', file);
}

static void write_json_entries(FILE* file, const OrionBindingPlanEntry* entries, int count) {
    fputs("[", file);
    for (int i = 0; i < count; i++) {
        const OrionBindingPlanEntry* e = &entries[i];
        if (i) fputs(",", file);
        fputs("{\"name\":", file); json_string(file, e->name);
        fprintf(file, ",\"dtype\":\"%s\",\"declared_shape\":[%d,%d,%d,%d],"
                "\"logical_bytes\":%zu,\"allocation_bytes\":%zu,"
                "\"packed_prefix_bytes\":%zu,\"layout\":",
                orion_dtype_name(e->dtype), e->declared_shape[0], e->declared_shape[1],
                e->declared_shape[2], e->declared_shape[3], e->logical_bytes,
                e->allocation_bytes, e->packed_prefix_bytes);
        json_string(file, e->layout); fputs(",\"allocation_layout\":", file);
        json_string(file, e->allocation_layout); fputs("}", file);
    }
    fputs("]", file);
}

static bool open_output(const OrionBindingPlan* plan, const char* path, FILE** out,
                        char temp_out[PATH_MAX],
                        char* error, size_t error_size) {
    if (!path || !path[0] ||
        !orion_binding_plan_validate_finalized(plan, error, error_size)) return false;
    if (snprintf(temp_out, PATH_MAX, "%s.tmp.XXXXXX", path) >= (int)PATH_MAX) {
        set_error(error, error_size, "output path is too long");
        return false;
    }
    int fd = mkstemp(temp_out);
    *out = fd >= 0 ? fdopen(fd, "w") : NULL;
    if (!*out) {
        if (fd >= 0) close(fd);
        temp_out[0] = '\0';
        set_error(error, error_size, "cannot open output '%s'", path);
    }
    return *out != NULL;
}

static bool finish_output(FILE* file, const char* path, const char temp_path[PATH_MAX],
                          char* error, size_t error_size) {
    if (fclose(file) != 0 || rename(temp_path, path) != 0) {
        if (temp_path[0]) unlink(temp_path);
        set_error(error, error_size, "failed publishing output '%s'", path);
        return false;
    }
    return true;
}

bool orion_binding_plan_write_json(const OrionBindingPlan* plan, const char* path,
                                   char* error, size_t error_size) {
    FILE* file = NULL;
    char temp_path[PATH_MAX];
    if (!open_output(plan, path, &file, temp_path, error, error_size)) return false;
    fprintf(file, "{\"version\":%u,\"program_name\":", plan->version);
    json_string(file, plan->program_name); fputs(",\"graph_identity\":", file);
    json_string(file, plan->graph_identity); fputs(",\"mil_identity\":", file);
    json_string(file, plan->mil_identity); fputs(",\"orion_revision\":", file);
    json_string(file, plan->orion_revision); fputs(",\"compiler_build\":", file);
    json_string(file, plan->compiler_build); fputs(",\"position_semantics\":", file);
    json_string(file, plan->position_semantics); fputs(",\"mask_semantics\":", file);
    json_string(file, plan->mask_semantics); fputs(",\"failure_semantics\":", file);
    json_string(file, plan->failure_semantics); fputs(",\"inputs\":", file);
    write_json_entries(file, plan->inputs, plan->n_inputs); fputs(",\"outputs\":", file);
    write_json_entries(file, plan->outputs, plan->n_outputs); fputs(",\"weights\":[", file);
    for (int i = 0; i < plan->n_weights; i++) {
        const OrionBindingPlanWeight* w = &plan->weights[i];
        if (i) fputs(",", file);
        fputs("{\"name\":", file); json_string(file, w->name);
        fputs(",\"path\":", file); json_string(file, w->path);
        fputs(",\"sha256\":", file); json_string(file, w->sha256);
        fprintf(file, ",\"offset\":%llu,\"dtype\":\"%s\",\"shape\":[%d,%d,%d,%d],\"logical_bytes\":%zu}",
                (unsigned long long)w->offset, orion_dtype_name(w->dtype), w->shape[0],
                w->shape[1], w->shape[2], w->shape[3], w->logical_bytes);
    }
    fputs("]}\n", file);
    return finish_output(file, path, temp_path, error, error_size);
}

bool orion_binding_plan_write_markdown(const OrionBindingPlan* plan, const char* path,
                                       char* error, size_t error_size) {
    FILE* file = NULL;
    char temp_path[PATH_MAX];
    if (!open_output(plan, path, &file, temp_path, error, error_size)) return false;
    fprintf(file, "# ABI: %s\n\n- Plan version: %u\n- Graph identity: `%s`\n- MIL identity: `%s`\n"
            "- Orion revision: `%s`\n- Compiler build: `%s`\n"
            "- Position semantics: %s\n- Mask semantics: %s\n- Failure semantics: %s\n\n",
             plan->program_name, plan->version, plan->graph_identity, plan->mil_identity, plan->orion_revision,
            plan->compiler_build, plan->position_semantics, plan->mask_semantics,
            plan->failure_semantics);
    fputs("| Direction | Name | Dtype | Logical shape | Logical bytes | Allocation bytes | Allocation layout | Packed prefix |\n"
          "|---|---|---:|---|---:|---:|---|---:|\n", file);
    for (int group = 0; group < 2; group++) {
        const OrionBindingPlanEntry* entries = group ? plan->outputs : plan->inputs;
        int count = group ? plan->n_outputs : plan->n_inputs;
        for (int i = 0; i < count; i++) fprintf(file,
            "| %s | `%s` | %s | [%d,%d,%d,%d] | %zu | %zu | %s | %zu |\n",
            group ? "output" : "input", entries[i].name, orion_dtype_name(entries[i].dtype),
            entries[i].declared_shape[0], entries[i].declared_shape[1],
            entries[i].declared_shape[2], entries[i].declared_shape[3],
            entries[i].logical_bytes, entries[i].allocation_bytes,
            entries[i].allocation_layout, entries[i].packed_prefix_bytes);
    }
    fputs("\n| Weight | Path | SHA-256 | Offset | Dtype | Shape | Bytes |\n"
          "|---|---|---|---:|---|---|---:|\n", file);
    for (int i = 0; i < plan->n_weights; i++) {
        const OrionBindingPlanWeight* w = &plan->weights[i];
        fprintf(file, "| `%s` | `%s` | `%s` | %llu | %s | [%d,%d,%d,%d] | %zu |\n",
                w->name, w->path, w->sha256, (unsigned long long)w->offset,
                orion_dtype_name(w->dtype), w->shape[0], w->shape[1],
                w->shape[2], w->shape[3], w->logical_bytes);
    }
    return finish_output(file, path, temp_path, error, error_size);
}
