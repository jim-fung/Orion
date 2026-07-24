// compiler/pass_ane_validate.c — T132: ANE constraint validation

#include "pass_ane_validate.h"
#include "pass_sram.h"
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

static void add_error(OrionANEValidationResult* r, int constraint_id, int node_idx,
                      const char* fmt, ...) {
    if (r->n_errors >= 32) return;
    OrionANEConstraintError* e = &r->errors[r->n_errors++];
    e->constraint_id = constraint_id;
    e->node_idx = node_idx;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(e->message, sizeof(e->message), fmt, ap);
    va_end(ap);
    r->valid = false;
}

// Minimum IOSurface size in bytes (~49KB)
#define ANE_MIN_TENSOR_BYTES 49152

static bool codegen_supports(OrionOp op) {
    switch (op) {
        case ORION_OP_INPUT: case ORION_OP_CONST: case ORION_OP_CONV1X1:
        case ORION_OP_ADD: case ORION_OP_SUB: case ORION_OP_MUL:
        case ORION_OP_MATMUL: case ORION_OP_RESHAPE: case ORION_OP_TRANSPOSE:
        case ORION_OP_CAST: case ORION_OP_RELU: case ORION_OP_TANH:
        case ORION_OP_SIGMOID: case ORION_OP_SOFTMAX: case ORION_OP_EXP:
        case ORION_OP_POW: case ORION_OP_REDUCE_SUM: case ORION_OP_REDUCE_MEAN:
        case ORION_OP_REDUCE_MAX: case ORION_OP_NEG: case ORION_OP_SQRT:
        case ORION_OP_RSQRT: case ORION_OP_SLICE: case ORION_OP_IDENTITY:
            return true;
        default:
            return false;
    }
}

OrionANEValidationResult orion_pass_ane_validate(const OrionGraph* graph) {
    OrionANEValidationResult r;
    memset(&r, 0, sizeof(r));
    r.valid = true;

    if (!graph) {
        add_error(&r, 0, -1, "NULL graph");
        return r;
    }

    for (int i = 0; i < graph->n_nodes; i++) {
        const OrionNode* n = &graph->nodes[i];
        if (!n->is_live) continue;

        if (n->op < ORION_OP_INPUT || n->op >= ORION_OP_COUNT) {
            add_error(&r, 15, i, "node '%s': unknown operation", n->name);
            continue;
        }

        // Constraint 1: No concat op
        if (n->op == ORION_OP_CONCAT_BANNED) {
            add_error(&r, 1, i, "node '%s': concat op rejected by ANE compiler", n->name);
            continue;
        }

        if (!codegen_supports(n->op)) {
            add_error(&r, 15, i, "node '%s': operation '%s' has no MIL codegen",
                      n->name, orion_op_name(n->op));
            continue;
        }

        if (n->op == ORION_OP_CONV1X1 && (n->attrs.bias_input >= 0 || n->n_inputs != 2)) {
            add_error(&r, 10, i,
                      "node '%s': conv bias is unsupported; use a separate add", n->name);
        }

        if (n->op == ORION_OP_CAST &&
            n->attrs.cast_dtype != ORION_DTYPE_FP16 &&
            n->attrs.cast_dtype != ORION_DTYPE_FP32) {
            add_error(&r, 15, i, "node '%s': unsupported cast target '%s'", n->name,
                      orion_dtype_name(n->attrs.cast_dtype));
        }

        if ((n->op == ORION_OP_INPUT || n->is_output) &&
            (n->shape[0] != 1 || n->shape[1] <= 0 || n->shape[2] != 1 || n->shape[3] <= 0)) {
            add_error(&r, 4, i, "node '%s': external tensor must have static [1,C,1,S] shape",
                      n->name);
        }

        if (n->op == ORION_OP_CONST && n->attrs.blob_path[0]) {
            if (strncmp(n->attrs.blob_path, "@model_path/", 12) != 0 ||
                strstr(n->attrs.blob_path, "..")) {
                add_error(&r, 8, i, "node '%s': unsafe BLOBFILE path '%s'", n->name,
                          n->attrs.blob_path);
            }
            if (n->attrs.blob_offset != 64) {
                add_error(&r, 8, i, "node '%s': BLOBFILE offset must be 64", n->name);
            }
        }

        // Constraint 10: No gelu op (must be decomposed to tanh approximation)
        // Check for any node that might represent a monolithic gelu
        // (Our builder always decomposes, but validate external graphs)
        // We don't have ORION_OP_GELU — this is by design. Nothing to check.

        // Internal tensors do not have IOSurfaces. The binding plan pads
        // external I/O to the minimum allocation, so do not reject valid
        // token-sized intermediate tensors here.
    }

    for (int i = 0; i < graph->n_outputs; i++) {
        int node_idx = graph->outputs[i].node_idx;
        if (node_idx < 0 || node_idx >= graph->n_nodes) {
            add_error(&r, 4, -1, "output '%s' references an invalid node",
                      graph->outputs[i].name);
            continue;
        }
        const OrionNode* output = &graph->nodes[node_idx];
        if (output->shape[0] != 1 || output->shape[1] <= 0 ||
            output->shape[2] != 1 || output->shape[3] <= 0) {
            add_error(&r, 4, node_idx,
                      "output '%s' must have static [1,C,1,S] shape",
                      graph->outputs[i].name);
        }
    }

    // Binding order and uniform allocation are enforced by generated plans,
    // which sort I/O independently of graph declaration/return order.

    return r;
}
