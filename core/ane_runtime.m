#import "ane_runtime.h"
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <CommonCrypto/CommonDigest.h>

// T015: orion_compile_mil — compile MIL text to ANE program
// T016: orion_eval — evaluate a compiled program
// T017: orion_release_program — unload and release

#pragma mark - Private ANE Class References

static Class g_Desc;  // _ANEInMemoryModelDescriptor
static Class g_IMM;   // _ANEInMemoryModel
static Class g_AR;    // _ANERequest
static Class g_AIO;   // _ANEIOSurfaceObject
static bool  g_init;
static int   g_compile_count;

static uint32_t read_le32(const uint8_t* bytes) {
    return (uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8) |
           ((uint32_t)bytes[2] << 16) | ((uint32_t)bytes[3] << 24);
}

static NSString* sha256_hex(NSData* data) {
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(data.bytes, (CC_LONG)data.length, digest);
    NSMutableString* text = [NSMutableString stringWithCapacity:CC_SHA256_DIGEST_LENGTH * 2];
    for (int i = 0; i < CC_SHA256_DIGEST_LENGTH; i++) [text appendFormat:@"%02x", digest[i]];
    return text;
}

// A plan weight offset identifies its 64-byte chunk header.  Each chunk carries
// its own declared payload range; the header at byte 64 is not a file-wide
// aggregate header.
static bool blobfile_chunk_payload_range(NSData* data, uint64_t chunk_offset,
                                         size_t logical_bytes, NSRange* out_range) {
    static const uint64_t chunk_header_size = 64;
    if (![data isKindOfClass:[NSData class]] || data.length < 5 ||
        ((const uint8_t*)data.bytes)[0] != 0x01 ||
        ((const uint8_t*)data.bytes)[4] != 0x02 ||
        chunk_offset > UINT64_MAX - chunk_header_size ||
        chunk_offset > (uint64_t)data.length ||
        chunk_header_size > (uint64_t)data.length - chunk_offset) return false;
    const uint8_t* header = (const uint8_t*)data.bytes + (NSUInteger)chunk_offset;
    uint64_t payload_size = read_le32(header + 8);
    uint64_t payload_offset = read_le32(header + 16);
    if (header[0] != 0xef || header[1] != 0xbe || header[2] != 0xad ||
        header[3] != 0xde || header[4] != 0x01 ||
        payload_size != logical_bytes ||
        payload_offset > (uint64_t)data.length ||
        payload_size > (uint64_t)data.length - payload_offset ||
        payload_offset < chunk_offset + chunk_header_size ||
        payload_offset > NSUIntegerMax || payload_size > NSUIntegerMax) return false;
    if (out_range) *out_range = NSMakeRange((NSUInteger)payload_offset,
                                             (NSUInteger)payload_size);
    return true;
}

bool orion_validate_weight_dictionary_for_plan(
    const OrionBindingPlan* plan, NSDictionary* weight_dict
) {
    char error[256] = {0};
    NSDictionary* dictionary = weight_dict ?: @{};
    if (!plan || !orion_binding_plan_validate_finalized(plan, error, sizeof(error))) return false;

    // Dictionary keys correspond to physical BLOBFILEs, so aggregate chunks
    // share one entry while each is independently range- and hash-verified.
    NSMutableDictionary<NSString*, NSMutableArray<NSValue*>*>* payload_ranges =
        [NSMutableDictionary dictionary];
    for (int i = 0; i < plan->n_weights; i++) {
        @autoreleasepool {
            const OrionBindingPlanWeight* weight = &plan->weights[i];
            NSString* path = [NSString stringWithUTF8String:weight->path];
            NSDictionary* entry = dictionary[path];
            if (![entry isKindOfClass:[NSDictionary class]] ||
                ![entry[@"offset"] isKindOfClass:[NSNumber class]] ||
                [entry[@"offset"] unsignedLongLongValue] != 0 ||
                ![entry[@"data"] isKindOfClass:[NSData class]]) return false;
            NSData* data = entry[@"data"];
            NSRange range;
            if (!blobfile_chunk_payload_range(data, weight->offset, weight->logical_bytes, &range))
                return false;
            NSData* payload = [data subdataWithRange:range];
            if (![[sha256_hex(payload) lowercaseString] isEqualToString:
                  [[NSString stringWithUTF8String:weight->sha256] lowercaseString]])
                return false;
            NSMutableArray<NSValue*>* ranges = payload_ranges[path];
            if (!ranges) {
                ranges = [NSMutableArray array];
                payload_ranges[path] = ranges;
            }
            for (NSValue* existing in ranges) {
                if (NSIntersectionRange(existing.rangeValue, range).length != 0) return false;
            }
            [ranges addObject:[NSValue valueWithRange:range]];
        }
    }
    // Verify every dictionary entry was consumed by at least one weight.
    return dictionary.count == payload_ranges.count;
}

#pragma mark - OrionProgram

struct OrionProgram {
    void *model;       // _ANEInMemoryModel (retained via CFBridgingRetain)
    void *tmpDir;      // NSString* tmp directory (retained)
    char tag[256];
    bool loaded;
    OrionBindingPlan plan;  // Cached when compiled via orion_compile_mil_with_plan
    bool has_plan;
    char primary_weight_path[512]; // BLOBFILE materialized as compiled data
};

static bool rebound_plan_for_weights(const OrionProgram* donor,
                                     NSDictionary* weight_dict,
                                     const char* patched_mil,
                                     OrionBindingPlan* out_plan) {
    if (!donor || !donor->has_plan || !patched_mil || !out_plan) return false;
    *out_plan = donor->plan;
    for (int i = 0; i < out_plan->n_weights; i++) {
        NSString* path = @(out_plan->weights[i].path);
        NSDictionary* entry = weight_dict[path];
        NSData* data = [entry isKindOfClass:[NSDictionary class]] ? entry[@"data"] : nil;
        if (![data isKindOfClass:[NSData class]]) return false;
        NSRange range;
        if (!blobfile_chunk_payload_range(data, out_plan->weights[i].offset,
                                    out_plan->weights[i].logical_bytes, &range)) return false;
        NSData* payload = [data subdataWithRange:range];
        NSString* digest = sha256_hex(payload);
        strlcpy(out_plan->weights[i].sha256, digest.UTF8String,
                sizeof(out_plan->weights[i].sha256));
    }
    if (!orion_binding_plan_set_mil_identity(out_plan, patched_mil)) return false;
    return orion_validate_weight_dictionary_for_plan(out_plan, weight_dict);
}

static NSString* patched_mil_text(const char* mil_text, NSDictionary* weight_dict) {
    CC_SHA256_CTX context;
    CC_SHA256_Init(&context);
    NSArray<NSString*>* paths = [[weight_dict allKeys]
        sortedArrayUsingSelector:@selector(compare:)];
    for (NSString* path in paths) {
        NSData* path_data = [path dataUsingEncoding:NSUTF8StringEncoding];
        NSData* blob = weight_dict[path][@"data"];
        CC_SHA256_Update(&context, path_data.bytes, (CC_LONG)path_data.length);
        CC_SHA256_Update(&context, blob.bytes, (CC_LONG)blob.length);
    }
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256_Final(digest, &context);
    NSMutableString* identity = [NSMutableString stringWithCapacity:64];
    for (int i = 0; i < CC_SHA256_DIGEST_LENGTH; i++)
        [identity appendFormat:@"%02x", digest[i]];
    return [NSString stringWithFormat:@"%s\n// orion-weight-patch:%@\n",
                                      mil_text, identity];
}

#pragma mark - Init

bool orion_ane_init(void) {
    if (g_init) return true;

    void *h = dlopen("/System/Library/PrivateFrameworks/"
                     "AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    if (!h) return false;

    g_Desc = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    g_IMM  = NSClassFromString(@"_ANEInMemoryModel");
    g_AR   = NSClassFromString(@"_ANERequest");
    g_AIO  = NSClassFromString(@"_ANEIOSurfaceObject");

    if (!g_Desc || !g_IMM || !g_AR || !g_AIO) return false;

    g_init = true;
    g_compile_count = 0;
    return true;
}

#pragma mark - T015: Compile

OrionProgram* orion_compile_mil(
    const char* mil_text,
    NSDictionary* weight_dict,
    const char* program_tag
) {
    if (!g_init || !mil_text) return NULL;

    @autoreleasepool {
        NSData *milData = [NSData dataWithBytes:mil_text length:strlen(mil_text)];
        NSDictionary *wdict = weight_dict ?: @{};

        // Step 1: Create descriptor
        id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
            g_Desc, @selector(modelWithMILText:weights:optionsPlist:),
            milData, wdict, nil);
        if (!desc) return NULL;

        // Step 2: Create in-memory model
        id model = ((id(*)(Class,SEL,id))objc_msgSend)(
            g_IMM, @selector(inMemoryModelWithDescriptor:), desc);
        if (!model) return NULL;

        // Step 3: Pre-populate temp directory (ANE compiler reads from filesystem)
        id hexId = ((id(*)(id,SEL))objc_msgSend)(model, @selector(hexStringIdentifier));
        NSString *tmpDir = [NSTemporaryDirectory() stringByAppendingPathComponent:hexId];
        NSFileManager *fm = [NSFileManager defaultManager];
        [fm createDirectoryAtPath:[tmpDir stringByAppendingPathComponent:@"weights"]
            withIntermediateDirectories:YES attributes:nil error:nil];
        [milData writeToFile:[tmpDir stringByAppendingPathComponent:@"model.mil"]
                  atomically:YES];

        // Write weight blobs to temp directory
        for (NSString *path in wdict) {
            NSDictionary *entry = wdict[path];
            NSData *data = entry[@"data"];
            if (data) {
                // path is like "@model_path/weights/weight.bin"
                NSString *relPath = [path stringByReplacingOccurrencesOfString:@"@model_path/"
                                                                    withString:@""];
                NSString *fullPath = [tmpDir stringByAppendingPathComponent:relPath];
                NSString *dir = [fullPath stringByDeletingLastPathComponent];
                [fm createDirectoryAtPath:dir withIntermediateDirectories:YES attributes:nil error:nil];
                [data writeToFile:fullPath atomically:YES];
            }
        }

        // Step 4: Compile
        NSError *e = nil;
        BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
            model, @selector(compileWithQoS:options:error:), 21, @{}, &e);
        if (!ok) {
            if (e) NSLog(@"ANE compile error [%s, attempt %d]: %@",
                         program_tag ? program_tag : "untagged",
                         g_compile_count + 1, e);
            [fm removeItemAtPath:tmpDir error:nil];
            return NULL;
        }

        // Step 5: Load into ANE
        ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
            model, @selector(loadWithQoS:options:error:), 21, @{}, &e);
        if (!ok) {
            [fm removeItemAtPath:tmpDir error:nil];
            return NULL;
        }

        g_compile_count++;

        // Record which declared BLOBFILE the compiler materialized as data.
        // Byte equality is unambiguous for replacement: equal candidates are
        // interchangeable payloads, and a missing match disables patching.
        NSString *dataPath = [tmpDir stringByAppendingPathComponent:@"data"];
        NSData *compiledData = [NSData dataWithContentsOfFile:dataPath];

        // Wrap in OrionProgram (manual retain for C struct)
        OrionProgram *prog = (OrionProgram *)calloc(1, sizeof(OrionProgram));
        if (!prog) {
            // calloc failure is not deterministically injectable here. The model
            // has loaded but has not crossed a CF bridge, so ARC owns its release.
            ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
                model, @selector(unloadWithQoS:error:), 21, &e);
            [fm removeItemAtPath:tmpDir error:nil];
            model = nil;
            return NULL;
        }
        prog->model = (void *)CFBridgingRetain(model);
        prog->tmpDir = (void *)CFBridgingRetain(tmpDir);
        prog->loaded = true;
        for (NSString *path in wdict) {
            NSData *data = wdict[path][@"data"];
            if (compiledData && [data isEqualToData:compiledData] &&
                (!prog->primary_weight_path[0] ||
                 strcmp(path.UTF8String, prog->primary_weight_path) < 0)) {
                strlcpy(prog->primary_weight_path, path.UTF8String,
                        sizeof(prog->primary_weight_path));
            }
        }
        if (program_tag) {
            strlcpy(prog->tag, program_tag, sizeof(prog->tag));
        }

        return prog;
    }
}

OrionProgram* orion_compile_mil_with_plan(
    const char* mil_text, const OrionBindingPlan* plan,
    NSDictionary* weight_dict, const char* program_tag,
    char* error, size_t error_size
) {
    if (error && error_size) error[0] = '\0';
    if (!mil_text || !plan || !plan->mil_identity[0]) {
        if (error && error_size) snprintf(error, error_size, "MIL text and MIL-bound plan are required");
        return NULL;
    }
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    char mil_identity[ORION_SHA256_HEX_LENGTH + 1];
    CC_SHA256(mil_text, (CC_LONG)strlen(mil_text), digest);
    for (int i = 0; i < CC_SHA256_DIGEST_LENGTH; i++) snprintf(mil_identity + i * 2, 3, "%02x", digest[i]);
    if (strcmp(mil_identity, plan->mil_identity) != 0) {
        if (error && error_size) snprintf(error, error_size, "MIL text does not match the binding plan");
        return NULL;
    }
    if (!orion_validate_weight_dictionary_for_plan(plan, weight_dict)) {
        if (error && error_size) snprintf(error, error_size, "weight dictionary does not match the binding plan");
        return NULL;
    }
    OrionProgram* prog = orion_compile_mil(mil_text, weight_dict, program_tag);
    if (prog && plan) {
        prog->plan = *plan;
        prog->has_plan = true;
    }
    return prog;
}

const OrionBindingPlan* orion_program_plan(const OrionProgram* prog) {
    return (prog && prog->has_plan) ? &prog->plan : NULL;
}

const char* orion_program_model_dir(const OrionProgram* prog) {
    if (!prog || !prog->tmpDir) return NULL;
    static _Thread_local char dir_buffer[4096];
    NSString* dir = (__bridge NSString*)prog->tmpDir;
    if (![dir getFileSystemRepresentation:dir_buffer
                                maxLength:sizeof(dir_buffer)]) return NULL;
    return dir_buffer;
}

#pragma mark - T016: Eval

#import "iosurface_tensor.h"

OrionEvalResult orion_eval_report(
    OrionProgram* prog,
    IOSurfaceRef* inputs, int num_inputs,
    IOSurfaceRef* outputs, int num_outputs
) {
    if (!prog || !prog->loaded || !inputs || !outputs) return ORION_EVAL_NOT_INVOKED;
    if (num_inputs <= 0 || num_outputs <= 0) return ORION_EVAL_NOT_INVOKED;

    @autoreleasepool {
        @try {
            // Wrap inputs as _ANEIOSurfaceObject
            NSMutableArray *inArr = [NSMutableArray arrayWithCapacity:num_inputs];
            NSMutableArray *inIdx = [NSMutableArray arrayWithCapacity:num_inputs];
            for (int i = 0; i < num_inputs; i++) {
                id wrapped = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                    g_AIO, @selector(objectWithIOSurface:), inputs[i]);
                if (!wrapped) return ORION_EVAL_NOT_INVOKED;
                [inArr addObject:wrapped];
                [inIdx addObject:@(i)];
            }

            // Wrap outputs
            NSMutableArray *outArr = [NSMutableArray arrayWithCapacity:num_outputs];
            NSMutableArray *outIdx = [NSMutableArray arrayWithCapacity:num_outputs];
            for (int i = 0; i < num_outputs; i++) {
                id wrapped = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                    g_AIO, @selector(objectWithIOSurface:), outputs[i]);
                if (!wrapped) return ORION_EVAL_NOT_INVOKED;
                [outArr addObject:wrapped];
                [outIdx addObject:@(i)];
            }

            // Build request. Failures before the private evaluation selector are
            // reported as not-invoked so callers can issue a chained receipt.
            id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(
                g_AR,
                @selector(requestWithInputs:inputIndices:outputs:outputIndices:
                          weightsBuffer:perfStats:procedureIndex:),
                inArr, inIdx, outArr, outIdx, nil, nil, @0);
            if (!req) return ORION_EVAL_NOT_INVOKED;

            // This is the invocation boundary. Once control enters the selector,
            // the private call is counted as invoked even if it returns NO.
            id model = (__bridge id)prog->model;
            NSError *e = nil;
            @try {
                BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                    model, @selector(evaluateWithQoS:options:request:error:),
                    21, @{}, req, &e);
                if (!ok && e) NSLog(@"orion_eval ERROR [%s]: %@", prog->tag, e);
                return ok ? ORION_EVAL_PRIVATE_SUCCEEDED : ORION_EVAL_PRIVATE_FAILED;
            } @catch (NSException *exception) {
                NSLog(@"orion_eval EXCEPTION [%s]: %@", prog->tag, exception);
                return ORION_EVAL_PRIVATE_FAILED;
            }
        } @catch (NSException *exception) {
            NSLog(@"orion_eval preparation EXCEPTION [%s]: %@", prog->tag, exception);
            return ORION_EVAL_NOT_INVOKED;
        }
    }
}

bool orion_eval(
    OrionProgram* prog,
    IOSurfaceRef* inputs, int num_inputs,
    IOSurfaceRef* outputs, int num_outputs
) {
    return orion_eval_report(prog, inputs, num_inputs, outputs, num_outputs) ==
        ORION_EVAL_PRIVATE_SUCCEEDED;
}

static bool binding_entry_equal(const OrionBindingPlanEntry* left,
                                const OrionBindingPlanEntry* right) {
    return strcmp(left->name, right->name) == 0 && left->dtype == right->dtype &&
        memcmp(left->declared_shape, right->declared_shape, sizeof(left->declared_shape)) == 0 &&
        left->logical_bytes == right->logical_bytes &&
        left->allocation_bytes == right->allocation_bytes &&
        left->packed_prefix_bytes == right->packed_prefix_bytes &&
        strcmp(left->layout, right->layout) == 0 &&
        strcmp(left->allocation_layout, right->allocation_layout) == 0;
}

static bool plan_has_same_eval_abi(const OrionBindingPlan* supplied,
                                   const OrionBindingPlan* cached) {
    if (!supplied || !cached || supplied->version != cached->version ||
        strcmp(supplied->program_name, cached->program_name) != 0 ||
        strcmp(supplied->graph_identity, cached->graph_identity) != 0 ||
        strcmp(supplied->orion_revision, cached->orion_revision) != 0 ||
        strcmp(supplied->compiler_build, cached->compiler_build) != 0 ||
        strcmp(supplied->position_semantics, cached->position_semantics) != 0 ||
        strcmp(supplied->mask_semantics, cached->mask_semantics) != 0 ||
        strcmp(supplied->failure_semantics, cached->failure_semantics) != 0 ||
        supplied->n_inputs != cached->n_inputs ||
        supplied->n_outputs != cached->n_outputs) return false;
    for (int i = 0; i < supplied->n_inputs; i++)
        if (!binding_entry_equal(&supplied->inputs[i], &cached->inputs[i])) return false;
    for (int i = 0; i < supplied->n_outputs; i++)
        if (!binding_entry_equal(&supplied->outputs[i], &cached->outputs[i])) return false;
    return true;
}

// Weight hashes and MIL identity are provenance, not surface-evaluation ABI.
// They must nevertheless exactly identify the cached program: a patched model
// is evaluated only with its rebound plan from orion_program_plan().
static bool plan_has_same_provenance(const OrionBindingPlan* supplied,
                                     const OrionBindingPlan* cached) {
    if (!supplied || !cached || strcmp(supplied->mil_identity, cached->mil_identity) != 0 ||
        supplied->n_weights != cached->n_weights) return false;
    for (int i = 0; i < supplied->n_weights; i++) {
        const OrionBindingPlanWeight* left = &supplied->weights[i];
        const OrionBindingPlanWeight* right = &cached->weights[i];
        if (strcmp(left->name, right->name) != 0 || strcmp(left->path, right->path) != 0 ||
            strcmp(left->sha256, right->sha256) != 0 || left->offset != right->offset ||
            left->dtype != right->dtype ||
            memcmp(left->shape, right->shape, sizeof(left->shape)) != 0 ||
            left->logical_bytes != right->logical_bytes) return false;
    }
    return true;
}

OrionEvalResult orion_eval_with_plan_report(
    OrionProgram* prog, const OrionBindingPlan* plan,
    IOSurfaceRef* inputs, int num_inputs,
    IOSurfaceRef* outputs, int num_outputs
) {
    char error[256] = {0};
    if (!prog || !prog->has_plan || !plan || !inputs || !outputs ||
        !orion_binding_plan_validate_finalized(plan, error, sizeof(error)) ||
        !plan_has_same_eval_abi(plan, &prog->plan) ||
        !plan_has_same_provenance(plan, &prog->plan) ||
        num_inputs != plan->n_inputs || num_outputs != plan->n_outputs)
        return ORION_EVAL_NOT_INVOKED;
    for (int i = 0; i < num_inputs; i++) {
        if (orion_tensor_validate_for_binding(inputs[i], &plan->inputs[i]) !=
            ORION_SURFACE_OK) return ORION_EVAL_NOT_INVOKED;
    }
    for (int i = 0; i < num_outputs; i++) {
        if (orion_tensor_validate_for_binding(outputs[i], &plan->outputs[i]) !=
            ORION_SURFACE_OK) return ORION_EVAL_NOT_INVOKED;
    }
    return orion_eval_report(prog, inputs, num_inputs, outputs, num_outputs);
}

bool orion_eval_with_plan(
    OrionProgram* prog, const OrionBindingPlan* plan,
    IOSurfaceRef* inputs, int num_inputs,
    IOSurfaceRef* outputs, int num_outputs
) {
    return orion_eval_with_plan_report(prog, plan, inputs, num_inputs, outputs,
                                       num_outputs) == ORION_EVAL_PRIVATE_SUCCEEDED;
}

#pragma mark - T017: Release

void orion_release_program(OrionProgram* prog) {
    if (!prog) return;

    @autoreleasepool {
        // Unload from ANE
        if (prog->loaded && prog->model) {
            id model = (__bridge id)prog->model;
            NSError *e = nil;
            ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
                model, @selector(unloadWithQoS:error:), 21, &e);
        }

        // Clean up temp directory
        if (prog->tmpDir) {
            NSString *td = (__bridge id)prog->tmpDir;
            [[NSFileManager defaultManager] removeItemAtPath:td error:nil];
            CFRelease(prog->tmpDir);
            prog->tmpDir = NULL;
        }

        // Release model ref
        if (prog->model) {
            CFRelease(prog->model);
            prog->model = NULL;
        }

        free(prog);
    }
}

int orion_compile_count(void) {
    return g_compile_count;
}

#pragma mark - T151: Delta weight patching

OrionProgram* orion_program_patch_weights(
    OrionProgram* donor,
    const char* mil_text,
    NSDictionary* weight_dict,
    const char* program_tag
) {
    if (!g_init || !donor || !donor->tmpDir || !mil_text ||
        !donor->primary_weight_path[0]) return NULL;

    @autoreleasepool {
        if (donor->has_plan) {
            OrionBindingPlan caller_plan = donor->plan;
            if (!orion_binding_plan_set_mil_identity(&caller_plan, mil_text) ||
                strcmp(caller_plan.mil_identity, donor->plan.mil_identity) != 0)
                return NULL;
        }
        NSDictionary *wdict = weight_dict ?: @{};
        NSString* patchedMIL = patched_mil_text(mil_text, wdict);
        NSData *milData = [patchedMIL dataUsingEncoding:NSUTF8StringEncoding];
        NSFileManager *fm = [NSFileManager defaultManager];
        OrionBindingPlan rebound_plan;
        bool has_rebound_plan = donor->has_plan;
        if (has_rebound_plan &&
            !rebound_plan_for_weights(donor, wdict, patchedMIL.UTF8String,
                                      &rebound_plan)) return NULL;

        // Create descriptor + model with new weights
        id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
            g_Desc, @selector(modelWithMILText:weights:optionsPlist:),
            milData, wdict, nil);
        if (!desc) return NULL;

        id model = ((id(*)(Class,SEL,id))objc_msgSend)(
            g_IMM, @selector(inMemoryModelWithDescriptor:), desc);
        if (!model) return NULL;

        // Get the new model's expected temp directory
        id hexId = ((id(*)(id,SEL))objc_msgSend)(model, @selector(hexStringIdentifier));
        NSString *newDir = [NSTemporaryDirectory() stringByAppendingPathComponent:hexId];
        NSString *donorDir = (__bridge NSString *)donor->tmpDir;
        bool sameDir = [newDir isEqualToString:donorDir];
        if (sameDir) {
            // Patching in place mutates the donor and cannot be transactional.
            // Constant-only programs are reused directly by callers instead.
            return NULL;
        } else {
            // Different hexId — set up new directory with donor's net.plist
            [fm removeItemAtPath:newDir error:nil];
            NSError* mkdirErr = nil;
            if (![fm createDirectoryAtPath:newDir withIntermediateDirectories:YES
                                attributes:nil error:&mkdirErr]) {
                NSLog(@"patch_weights[%s]: model directory creation failed: %@",
                      program_tag ?: "?", mkdirErr);
                return NULL;
            }

            NSString *srcPlist = [donorDir stringByAppendingPathComponent:@"net.plist"];
            NSString *dstPlist = [newDir stringByAppendingPathComponent:@"net.plist"];
            NSError *copyErr = nil;
            if (![fm copyItemAtPath:srcPlist toPath:dstPlist error:&copyErr]) {
                NSLog(@"patch_weights[%s]: net.plist copy failed: %@ -> %@ err=%@",
                      program_tag ?: "?", srcPlist, dstPlist, copyErr);
                [fm removeItemAtPath:newDir error:nil];
                return NULL;
            }
        }

        // Write new BLOBFILE(s) as the "data" file.
        // The compiled ANE "data" file IS the BLOBFILE — no transformation.
        // For single-weight programs, the data file = the BLOBFILE.
        // For multi-weight programs, we need to find which blob maps to "data".
        //
        // Strategy: write the BLOBFILE as the data file. If there are multiple
        // weight blobs, write the primary one (first entry) as data and others
        // to their weight paths. Also write model.mil for completeness.

        if (![milData writeToFile:[newDir stringByAppendingPathComponent:@"model.mil"]
                       atomically:YES]) {
            [fm removeItemAtPath:newDir error:nil];
            return NULL;
        }

        // Write weight blobs to weights/ and also as the data file
        NSError* weightsDirErr = nil;
        if (![fm createDirectoryAtPath:[newDir stringByAppendingPathComponent:@"weights"]
             withIntermediateDirectories:YES attributes:nil error:&weightsDirErr]) {
            [fm removeItemAtPath:newDir error:nil];
            return NULL;
        }

        // Use the primary BLOBFILE observed during donor compilation. Never
        // infer it from size or NSDictionary iteration order.
        NSData* baked = wdict[@(donor->primary_weight_path)][@"data"];
        if (!baked) {
            [fm removeItemAtPath:newDir error:nil];
            return NULL;
        }
        for (NSString *path in wdict) {
            NSDictionary *entry = wdict[path];
            NSData *data = entry[@"data"];
            if (data) {
                // Write to weights/ directory
                NSString *relPath = [path stringByReplacingOccurrencesOfString:@"@model_path/"
                                                                    withString:@""];
                NSString *fullPath = [newDir stringByAppendingPathComponent:relPath];
                NSString *dir = [fullPath stringByDeletingLastPathComponent];
                NSError* dirErr = nil;
                if (![fm createDirectoryAtPath:dir withIntermediateDirectories:YES
                                     attributes:nil error:&dirErr] ||
                    ![data writeToFile:fullPath atomically:YES]) {
                    [fm removeItemAtPath:newDir error:nil];
                    return NULL;
                }

            }
        }
        bool dataWritten = [baked writeToFile:
            [newDir stringByAppendingPathComponent:@"data"] atomically:YES];

        // If no weights, copy the donor's data file
        if (!dataWritten) {
            NSString *srcData = [donorDir stringByAppendingPathComponent:@"data"];
            NSError* dataErr = nil;
            if (![fm copyItemAtPath:srcData
                             toPath:[newDir stringByAppendingPathComponent:@"data"]
                              error:&dataErr]) {
                [fm removeItemAtPath:newDir error:nil];
                return NULL;
            }
        }

        // Load WITHOUT compiling — the key delta patching optimization
        NSError *e = nil;
        BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
            model, @selector(loadWithQoS:options:error:), 21, @{}, &e);
        if (!ok) {
            if (e) NSLog(@"orion_program_patch_weights LOAD FAILED: %@", e);
            [fm removeItemAtPath:newDir error:nil];
            return NULL;
        }

        // Wrap in OrionProgram
        OrionProgram *prog = (OrionProgram *)calloc(1, sizeof(OrionProgram));
        if (!prog) {
            // calloc failure is not deterministically injectable here. The model
            // has loaded but has not crossed a CF bridge, so ARC owns its release.
            ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
                model, @selector(unloadWithQoS:error:), 21, &e);
            [fm removeItemAtPath:newDir error:nil];
            model = nil;
            return NULL;
        }
        prog->model = (void *)CFBridgingRetain(model);
        prog->tmpDir = (void *)CFBridgingRetain(newDir);
        prog->loaded = true;
        if (has_rebound_plan) prog->plan = rebound_plan;
        prog->has_plan = has_rebound_plan;
        strlcpy(prog->primary_weight_path, donor->primary_weight_path,
                sizeof(prog->primary_weight_path));
        if (program_tag) {
            strlcpy(prog->tag, program_tag, sizeof(prog->tag));
        }

        // Note: does NOT increment g_compile_count — no compile happened
        return prog;
    }
}

bool orion_program_reload_weights(
    OrionProgram* prog,
    NSDictionary* weight_dict
) {
    // Private runtime reload is not transactional and has no proven mapping
    // from declared BLOBFILE paths to compiled data. Callers must use
    // orion_program_patch_weights(), which creates an isolated model.
    (void)prog;
    (void)weight_dict;
    return false;
}
