// compiler/validate.c — T121: Graph validation

#include "validate.h"
#include "topo.h"
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <stdio.h>

static OrionValidationResult ok_result(void) {
    OrionValidationResult r;
    r.valid = true;
    r.error_node = -1;
    r.message[0] = '\0';
    return r;
}

static OrionValidationResult err(int node, const char* fmt, ...) {
    OrionValidationResult r;
    r.valid = false;
    r.error_node = node;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(r.message, sizeof(r.message), fmt, ap);
    va_end(ap);
    return r;
}

static bool scalar_shape(const int shape[4]) {
    return shape[0] == 0 && shape[1] == 0 && shape[2] == 0 && shape[3] == 0;
}

static bool positive_shape(const int shape[4]) {
    return shape[0] > 0 && shape[1] > 0 && shape[2] > 0 && shape[3] > 0;
}

static int64_t element_count(const int shape[4]) {
    if (!positive_shape(shape)) return -1;
    int64_t count = 1;
    for (int i = 0; i < 4; i++) count *= shape[i];
    return count;
}

static bool broadcast_shape(const int a[4], const int b[4], int output[4]) {
    if (scalar_shape(a)) {
        memcpy(output, b, sizeof(int) * 4);
        return positive_shape(b);
    }
    if (scalar_shape(b)) {
        memcpy(output, a, sizeof(int) * 4);
        return positive_shape(a);
    }
    for (int i = 0; i < 4; i++) {
        if (a[i] != b[i] && a[i] != 1 && b[i] != 1) return false;
        output[i] = a[i] > b[i] ? a[i] : b[i];
    }
    return true;
}

static bool shape_equal(const int a[4], const int b[4]) {
    return memcmp(a, b, sizeof(int) * 4) == 0;
}

OrionValidationResult orion_graph_validate(const OrionGraph* graph) {
    if (!graph) return err(-1, "NULL graph");
    if (graph->n_nodes == 0) return err(-1, "empty graph");

    // Check: at least one output
    if (graph->n_outputs == 0)
        return err(-1, "no outputs marked");

    // Check each node
    for (int i = 0; i < graph->n_nodes; i++) {
        const OrionNode* n = &graph->nodes[i];
        if (!n->is_live) continue;

        // Name required
        if (n->name[0] == '\0')
            return err(i, "node %d has no name", i);

        // No banned ops
        if (n->op == ORION_OP_CONCAT_BANNED)
            return err(i, "node '%s': concat op is banned on ANE (use multi-output instead)", n->name);

        // Input references valid
        for (int j = 0; j < n->n_inputs; j++) {
            int inp = n->inputs[j];
            if (inp < 0 || inp >= graph->n_nodes)
                return err(i, "node '%s': input[%d]=%d out of range [0,%d)",
                          n->name, j, inp, graph->n_nodes);
            if (!graph->nodes[inp].is_live)
                return err(i, "node '%s': input[%d] references dead node %d",
                          n->name, j, inp);
        }


        if (n->op == ORION_OP_RESHAPE) {
            if (n->n_inputs != 2 ||
                element_count(graph->nodes[n->inputs[1]].shape) != element_count(n->shape)) {
                return err(i, "node '%s': reshape changes element count", n->name);
            }
        } else if (n->op == ORION_OP_TRANSPOSE) {
            if (n->n_inputs != 2) return err(i, "node '%s': transpose requires two inputs", n->name);
            const OrionNode* input = &graph->nodes[n->inputs[1]];
            bool seen[4] = {false, false, false, false};
            int expected[4];
            for (int axis = 0; axis < 4; axis++) {
                int source_axis = n->attrs.perm[axis];
                if (source_axis < 0 || source_axis >= 4 || seen[source_axis])
                    return err(i, "node '%s': transpose permutation is invalid", n->name);
                seen[source_axis] = true;
                expected[axis] = input->shape[source_axis];
            }
            if (!shape_equal(n->shape, expected))
                return err(i, "node '%s': transpose output shape is invalid", n->name);
        } else if (n->op == ORION_OP_MATMUL) {
            if (n->n_inputs != 2) return err(i, "node '%s': matmul requires two inputs", n->name);
            const int* a = graph->nodes[n->inputs[0]].shape;
            const int* b = graph->nodes[n->inputs[1]].shape;
            if (!positive_shape(a) || !positive_shape(b))
                return err(i, "node '%s': matmul requires static rank-4 inputs", n->name);
            int a_rows = n->attrs.transpose_x ? a[3] : a[2];
            int a_inner = n->attrs.transpose_x ? a[2] : a[3];
            int b_inner = n->attrs.transpose_y ? b[3] : b[2];
            int b_cols = n->attrs.transpose_y ? b[2] : b[3];
            if (a_inner != b_inner)
                return err(i, "node '%s': matmul inner dimensions do not match", n->name);
            int expected[4];
            int a_batch[4] = {a[0], a[1], 1, 1};
            int b_batch[4] = {b[0], b[1], 1, 1};
            if (!broadcast_shape(a_batch, b_batch, expected))
                return err(i, "node '%s': matmul batch dimensions do not broadcast", n->name);
            expected[2] = a_rows;
            expected[3] = b_cols;
            if (!shape_equal(n->shape, expected))
                return err(i, "node '%s': matmul output shape is invalid", n->name);
        } else if (n->op == ORION_OP_ADD || n->op == ORION_OP_SUB ||
                   n->op == ORION_OP_MUL || n->op == ORION_OP_POW) {
            if (n->n_inputs != 2) return err(i, "node '%s': binary op requires two inputs", n->name);
            int expected[4];
            if (!broadcast_shape(graph->nodes[n->inputs[0]].shape,
                                 graph->nodes[n->inputs[1]].shape, expected))
                return err(i, "node '%s': binary input shapes do not broadcast", n->name);
            if (!shape_equal(n->shape, expected))
                return err(i, "node '%s': binary output shape is invalid", n->name);
        } else if (n->op == ORION_OP_CONV1X1) {
            if (n->n_inputs < 2) return err(i, "node '%s': convolution lacks weights", n->name);
            const int* input = graph->nodes[n->inputs[0]].shape;
            const int* weight = graph->nodes[n->inputs[1]].shape;
            int expected[4] = {input[0], weight[0], input[2], input[3]};
            if (!positive_shape(input) || !positive_shape(weight) ||
                weight[1] != input[1] || weight[2] != 1 || weight[3] != 1 ||
                !shape_equal(n->shape, expected))
                return err(i, "node '%s': convolution shape contract is invalid", n->name);
        }
    }

    // Check output nodes exist and are live
    for (int i = 0; i < graph->n_outputs; i++) {
        int idx = graph->outputs[i].node_idx;
        if (idx < 0 || idx >= graph->n_nodes)
            return err(-1, "output '%s' references invalid node %d",
                      graph->outputs[i].name, idx);
        if (!graph->nodes[idx].is_live)
            return err(idx, "output '%s' references dead node", graph->outputs[i].name);
    }

    // Cycle detection via topological sort
    int count = 0;
    int* order = orion_topo_sort(graph, &count);
    if (!order)
        return err(-1, "graph contains a cycle");
    free(order);

    return ok_result();
}
