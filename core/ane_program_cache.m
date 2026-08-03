#import "ane_program_cache.h"
#import <Foundation/Foundation.h>

// T084: Program cache for weight swapping (M4).
//
// Store/lookup cache keyed by (kernel_name, layer_idx, weights_id, bucket).
// Cache owns all stored programs — callers must NOT release them.
// Thread-safe via @synchronized on the cache dictionary.
//
// Usage pattern:
//   OrionProgram *prog = orion_cache_lookup("prefill_attn", layer, &wb);
//   if (!prog) {
//       prog = orion_compile_mil(mil, wdict, tag);
//       orion_cache_store("prefill_attn", layer, &wb, prog);
//   }
//   orion_eval(prog, ...);  // do NOT release prog

/// Cache entry wrapping an OrionProgram* in an ObjC object.
/// Needed because NSMutableDictionary values must be ObjC objects,
/// and OrionProgram is an opaque C struct (not ObjC-managed).
@interface _OrionCacheEntry : NSObject
@property (nonatomic, assign) OrionProgram *program;
@property (nonatomic, copy)   NSString *weightsId;
@end

@implementation _OrionCacheEntry
- (void)dealloc {
    if (_program) {
        orion_release_program(_program);
        _program = NULL;
    }
}
@end

static NSMutableDictionary<NSString*, _OrionCacheEntry*> *_programCache = nil;

static void _ensureCache(void) {
    if (!_programCache) {
        _programCache = [NSMutableDictionary dictionary];
    }
}

static NSString* _cacheKey(const char* kernel_name, int layer_idx,
                           const OrionWeightsBinding* wb) {
    return [NSString stringWithFormat:@"%s:%d:%s:%d",
            kernel_name, layer_idx, wb->weights_id, wb->bucket];
}

OrionProgram* orion_cache_lookup(
    const char* kernel_name,
    int layer_idx,
    const OrionWeightsBinding* wb
) {
    @synchronized ([_OrionCacheEntry class]) {
        _ensureCache();
        NSString *key = _cacheKey(kernel_name, layer_idx, wb);
        _OrionCacheEntry *entry = _programCache[key];
        return entry ? entry.program : NULL;
    }
}

OrionProgram* orion_cache_lookup_retained(
    const char* kernel_name,
    int layer_idx,
    const OrionWeightsBinding* wb,
    void** out_lease
) {
    if (!out_lease) return NULL;
    *out_lease = NULL;
    @synchronized ([_OrionCacheEntry class]) {
        _ensureCache();
        NSString *key = _cacheKey(kernel_name, layer_idx, wb);
        _OrionCacheEntry *entry = _programCache[key];
        if (!entry) return NULL;
        // Hand the caller a retained reference to the entry itself. While the
        // caller holds this lease, the entry (and therefore its program) cannot
        // be deallocated even if orion_cache_evict removes it from the
        // dictionary concurrently. The program pointer stays valid until
        // orion_cache_release drops the lease.
        *out_lease = (__bridge_retained void *)entry;
        return entry.program;
    }
}

void orion_cache_release(void* lease) {
    if (!lease) return;
    // Transfer ownership back to ARC, which immediately releases it. If the
    // entry is still in the dictionary, the dictionary's strong reference keeps
    // it alive; if evicted, this was the last reference and dealloc frees the
    // program now that evaluation is complete.
    _OrionCacheEntry *entry = (__bridge_transfer _OrionCacheEntry *)lease;
    (void)entry;
}

void orion_cache_store(
    const char* kernel_name,
    int layer_idx,
    const OrionWeightsBinding* wb,
    OrionProgram* program
) {
    if (!program) return;

    @synchronized ([_OrionCacheEntry class]) {
        _ensureCache();
        NSString *key = _cacheKey(kernel_name, layer_idx, wb);

        // If entry exists for this key, dealloc will release the old program
        _OrionCacheEntry *entry = [[_OrionCacheEntry alloc] init];
        entry.program = program;
        entry.weightsId = @(wb->weights_id);
        _programCache[key] = entry;
    }
}

void orion_cache_evict(const char* weights_id) {
    @synchronized ([_OrionCacheEntry class]) {
        if (!_programCache) return;

        NSString *wid = @(weights_id);
        NSMutableArray *keysToRemove = [NSMutableArray array];

        for (NSString *key in _programCache) {
            _OrionCacheEntry *entry = _programCache[key];
            if ([entry.weightsId isEqualToString:wid]) {
                [keysToRemove addObject:key];
            }
        }

        // Remove outside enumeration (dealloc releases programs)
        [_programCache removeObjectsForKeys:keysToRemove];
    }
}

void orion_cache_clear(void) {
    @synchronized ([_OrionCacheEntry class]) {
        // Removing all entries triggers dealloc → orion_release_program
        [_programCache removeAllObjects];
    }
}

int orion_cache_size(void) {
    @synchronized ([_OrionCacheEntry class]) {
        return _programCache ? (int)[_programCache count] : 0;
    }
}

#pragma mark - T086: Weights ID Abstraction

static int _stepCounter = 0;
static char _stepIdBuf[32]; // "step_00000001" = 14 chars + null

const char* orion_weights_id_next_step(void) {
    @synchronized ([_OrionCacheEntry class]) {
        _stepCounter++;
        snprintf(_stepIdBuf, sizeof(_stepIdBuf), "step_%08d", _stepCounter);
        return _stepIdBuf;
    }
}

void orion_weights_id_reset(int start_step) {
    @synchronized ([_OrionCacheEntry class]) {
        _stepCounter = start_step;
    }
}

int orion_weights_id_current_step(void) {
    @synchronized ([_OrionCacheEntry class]) {
        return _stepCounter;
    }
}
