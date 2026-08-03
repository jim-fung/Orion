// compiler/pass_cast.h — T127: Cast hoisting/elimination pass
#ifndef ORION_PASS_CAST_H
#define ORION_PASS_CAST_H

#include "graph.h"

// Eliminate redundant round-trip casts:
// - fp16->fp32 followed by fp32->fp16 → eliminate both (fp32 intermediate is wider)
// - fp32->fp16 followed by fp16->fp32 → PRESERVED (fp16 intermediate narrows/loses data)
// Only round trips whose intermediate dtype is not narrower than the endpoints
// are folded; a narrowing intermediate preserves a lossy conversion on purpose.
// Casts at graph boundaries (inputs/outputs) are preserved.
// Returns true if any casts were eliminated.
bool orion_pass_cast(OrionGraph* graph);

#endif // ORION_PASS_CAST_H
