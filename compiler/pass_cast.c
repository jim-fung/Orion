// compiler/pass_cast.c — T127: Cast hoisting/elimination

#include "pass_cast.h"

// Replace all references to old_idx with new_idx
static void rewrite_uses(OrionGraph* g, int old_idx, int new_idx) {
    for (int i = 0; i < g->n_nodes; i++) {
        if (!g->nodes[i].is_live) continue;
        for (int j = 0; j < g->nodes[i].n_inputs; j++) {
            if (g->nodes[i].inputs[j] == old_idx) {
                g->nodes[i].inputs[j] = new_idx;
            }
        }
    }
    for (int i = 0; i < g->n_outputs; i++) {
        if (g->outputs[i].node_idx == old_idx) {
            g->outputs[i].node_idx = new_idx;
        }
    }
}

// Bit-width model for the floating-point dtypes that casts may cross.
// A cast chain A -> B -> A is only observationally lossless when the
// intermediate B is at least as wide as A (no narrowing round trip).
// Integers are treated as exact-width; a chain through a narrower int
// is also lossy. Returns -1 for dtypes outside the numeric model.
static int dtype_width_bits(OrionDtype dt) {
    switch (dt) {
        case ORION_DTYPE_FP16: return 16;
        case ORION_DTYPE_FP32: return 32;
        case ORION_DTYPE_INT32: return 32;
        default: return -1;  // BOOL/STRING are not cast operands
    }
}

// A cast pair cast(X, A->B) followed by cast(inp, B->A) is a round trip.
// It is only safe to eliminate when the intermediate B is not narrower
// than the endpoints A, i.e. the round trip does not lose precision.
// Examples:
//   fp16 -> fp32 -> fp16 : safe (fp32 intermediate is wider)
//   fp32 -> fp16 -> fp32 : UNSAFE (fp16 narrows) — must be preserved
//   int32 -> int32 chain : safe (same width, identity)
static bool cast_roundtrip_is_lossless(OrionDtype endpoint, OrionDtype intermediate) {
    int wa = dtype_width_bits(endpoint);
    int wb = dtype_width_bits(intermediate);
    if (wa < 0 || wb < 0) return false;        // unknown dtype → never fold
    return wb >= wa;                            // intermediate must not narrow
}

bool orion_pass_cast(OrionGraph* graph) {
    if (!graph) return false;
    bool changed = false;

    for (int i = 0; i < graph->n_nodes; i++) {
        OrionNode* n = &graph->nodes[i];
        if (!n->is_live || n->op != ORION_OP_CAST) continue;
        if (n->n_inputs < 1) continue;

        int inp_idx = n->inputs[0];
        OrionNode* inp = &graph->nodes[inp_idx];
        if (!inp->is_live || inp->op != ORION_OP_CAST) continue;

        // We have: inp=cast(X, A->B), n=cast(inp, B->C)
        // A round trip requires A == C. Even then it is only foldable
        // when the intermediate B is not narrower than the endpoints.
        OrionDtype dtype_a = graph->nodes[inp->inputs[0]].dtype;
        OrionDtype dtype_b = inp->attrs.cast_dtype;  // intermediate
        OrionDtype dtype_c = n->attrs.cast_dtype;

        if (dtype_a == dtype_c &&
            cast_roundtrip_is_lossless(dtype_a, dtype_b)) {
            // Lossless round-trip cast: bypass both to the original input
            int original = inp->inputs[0];
            rewrite_uses(graph, i, original);
            n->is_live = false;
            // Don't kill inp yet — it might have other users
            changed = true;
        }
    }

    return changed;
}
