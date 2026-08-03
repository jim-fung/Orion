# Orion — Makefile
# Build the Orion CLI, test suite, and benchmarks.

CC       = xcrun clang
CFLAGS   = -O2 -fobjc-arc -DACCELERATE_NEW_LAPACK -Wall -Wextra -I . -I core -I compiler
FRAMEWORKS = -framework Foundation -framework IOSurface -framework Accelerate
LDFLAGS  = -ldl $(FRAMEWORKS)

BUILDDIR = build

# ---------------------------------------------------------------------------
# Source files (38 total: 24 runtime + 14 compiler)
# ---------------------------------------------------------------------------

CORE_SRC = \
	core/ane_runtime.m \
	core/ane_program_cache.m \
	core/mil_builder.m \
	core/iosurface_tensor.m \
	core/profiler.m \
	core/bucket.m \
	core/checkpoint.m \
	core/model_registry.m \
	core/kernel.m \
	core/runtime.m \
	core/lora_adapter.m

INFERENCE_SRC = \
	kernels/inference/prefill_ane.m \
	kernels/inference/decode_ane.m \
	kernels/inference/decode_cpu.m \
	kernels/inference/kv_cache.m

TRAINING_SRC = \
	kernels/training/stories_train.m \
	kernels/training/stories_cpu_ops.m \
	kernels/training/data_loader.m

MODEL_SRC = model/weight_loader.m

TOKENIZER_SRC = \
	tokenizer/gpt2_bpe.m \
	tokenizer/sentencepiece_wrap.m

CLI_SRC = \
	apps/cli/commands/infer.m \
	apps/cli/commands/train.m \
	apps/cli/commands/bench.m \
	apps/cli/commands/depth_canonical.m \
	apps/cli/commands/depth_eval.m \
	apps/cli/commands/depth_serve.m \
	apps/cli/commands/depth_serve_support.m

# Stage 2: Compiler sources (pure C + ObjC for codegen/adapter)
COMPILER_C_SRC = \
	compiler/graph.c \
	compiler/builder.c \
	compiler/topo.c \
	compiler/patterns.c \
	compiler/validate.c \
	compiler/pass_dce.c \
	compiler/pass_identity.c \
	compiler/pass_conv_bias.c \
	compiler/pass_cast.c \
	compiler/pass_sram.c \
	compiler/pass_uniform_outputs.c \
	compiler/pass_ane_validate.c \
	core/blobfile.c \
	compiler/binding_plan.c \
	compiler/pipeline.c \
	compiler/frontends/gpt2_prefill.c \
	compiler/frontends/gpt2_decode.c \
	compiler/frontends/gpt2_final.c \
	compiler/frontends/classifier_softmax.c \
	compiler/frontends/stories_train.c \
	compiler/frontends/voxtream_depth.c \
	compiler/frontends/voxtream_depth_chain.c \
	compiler/frontends/lora.c

COMPILER_M_SRC = \
	compiler/codegen.m \
	compiler/kernel_adapter.m \
	compiler/mil_diff.m

MAIN_SRC = apps/cli/main.m

# All library sources (everything except main.m)
RUNTIME_C_SRC = core/orion_compile_ledger.c
LIB_SRC = $(CORE_SRC) $(INFERENCE_SRC) $(TRAINING_SRC) $(MODEL_SRC) $(TOKENIZER_SRC) $(CLI_SRC)

# Object files for library
LIB_OBJ  = $(patsubst %.m,$(BUILDDIR)/%.o,$(LIB_SRC)) \
           $(patsubst %.c,$(BUILDDIR)/%.o,$(RUNTIME_C_SRC))
MAIN_OBJ = $(patsubst %.m,$(BUILDDIR)/%.o,$(MAIN_SRC))
ALL_OBJ  = $(LIB_OBJ) $(MAIN_OBJ)

# Compiler object files (C and ObjC)
COMPILER_C_OBJ = $(patsubst %.c,$(BUILDDIR)/%.o,$(COMPILER_C_SRC))
COMPILER_M_OBJ = $(patsubst %.m,$(BUILDDIR)/%.o,$(COMPILER_M_SRC))
COMPILER_OBJ = $(COMPILER_C_OBJ) $(COMPILER_M_OBJ)

# ---------------------------------------------------------------------------
# Test binaries (41 total: 35 runtime + 6 compiler)
# ---------------------------------------------------------------------------

TEST_NAMES = \
	test_ane_runtime \
	test_mil_builder \
	test_cpu_forward \
	test_tokenizer \
	test_decode \
	test_infer_golden \
	test_ane_prefill \
	test_cpu_training_ops \
	test_sp_tokenizer \
	test_data_loader \
	test_train_kernels \
	test_train_smoke \
	test_program_cache \
	test_decode_ane \
	test_decode_ane_step \
	test_infer_golden_ane \
	test_bench_decode \
	test_delta_compile \
	test_lora \
	test_iosurface_tensor_checked \
	test_plan_weight_dict \
	test_voxtream_depth_ane \
	test_voxtream_depth_chain_ane \
	test_voxtream_multichunk_blob_ane \
	test_voxtream_aggregate_layer_ane \
	test_voxtream_aggregate_plan_verified \
	test_voxtream_g008_failure_injection \
	test_voxtream_depth_loop_ane \
	test_voxtream_probe_channels \
	test_voxtream_seqpack_plan_verified \
	test_m5_compile_limit \
	test_voxtream_seqpack_timing \
	test_voxtream_seqpack_depth_loop_ane \
	test_depth_serve_support \
	test_aggregate_plan_generation

# Compiler tests (link with compiler objects, not full library)
COMPILER_TEST_NAMES = \
	test_graph_ir \
	test_passes \
	test_ane_passes \
	test_compiler_equiv \
	test_depth_serve_protocol_v2 \
	test_blobfile \
	test_orion_error \
	test_orion_ledger

TEST_BINS = $(patsubst %,$(BUILDDIR)/tests/%,$(TEST_NAMES))
COMPILER_TEST_BINS = $(patsubst %,$(BUILDDIR)/tests/%,$(COMPILER_TEST_NAMES))

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

.PHONY: all clean test test-compiler test-depth-serve-lifecycle-static test-depth-serve-protocol-static bench

all: orion

# Build the CLI binary (runtime + compiler objects)
orion: $(ALL_OBJ) $(COMPILER_OBJ)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^

# Compile .m → .o (auto-create directories)
$(BUILDDIR)/%.o: %.m
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c -o $@ $<

# Compile .c → .o (for pure C compiler sources)
$(BUILDDIR)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) -O2 -Wall -Wextra -I . -I core -I compiler -c -o $@ $<

# Build and run all test binaries plus the static lifecycle and protocol tests.
test: test-depth-serve-lifecycle-static test-depth-serve-protocol-static $(TEST_BINS) $(COMPILER_TEST_BINS)
	@passed=0; failed=0; total=0; \
	for t in $(TEST_BINS) $(COMPILER_TEST_BINS); do \
		total=$$((total + 1)); \
		name=$$(basename $$t); \
		printf "%-30s " "$$name"; \
		if $$t > /dev/null 2>&1; then \
			printf "PASS\n"; \
			passed=$$((passed + 1)); \
		else \
			printf "FAIL\n"; \
			failed=$$((failed + 1)); \
		fi; \
	done; \
	echo ""; \
	echo "$$passed/$$total passed, $$failed failed"; \
	if [ $$failed -gt 0 ]; then exit 1; fi

test-depth-serve-lifecycle-static:
	@python3 tests/test_depth_serve_lifecycle_static.py

test-depth-serve-protocol-static:
	@python3 tests/test_depth_serve_protocol_static.py

# Run only compiler tests
test-compiler: $(COMPILER_TEST_BINS)
	@passed=0; failed=0; total=0; \
	for t in $(COMPILER_TEST_BINS); do \
		total=$$((total + 1)); \
		name=$$(basename $$t); \
		printf "%-30s " "$$name"; \
		if $$t > /dev/null 2>&1; then \
			printf "PASS\n"; \
			passed=$$((passed + 1)); \
		else \
			printf "FAIL\n"; \
			failed=$$((failed + 1)); \
		fi; \
	done; \
	echo ""; \
	echo "$$passed/$$total passed, $$failed failed"; \
	if [ $$failed -gt 0 ]; then exit 1; fi

# Build each test binary: compile test .m + link with all library + compiler .o files
$(BUILDDIR)/tests/%: tests/%.m $(LIB_OBJ) $(COMPILER_OBJ)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $< $(LIB_OBJ) $(COMPILER_OBJ)

# Build compiler test binaries: link with compiler objects only (+ Foundation)
$(BUILDDIR)/tests/test_graph_ir: tests/test_graph_ir.m $(COMPILER_OBJ)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -framework Foundation -o $@ $< $(COMPILER_OBJ)

$(BUILDDIR)/tests/test_passes: tests/test_passes.m $(COMPILER_OBJ)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -framework Foundation -o $@ $< $(COMPILER_OBJ)

$(BUILDDIR)/tests/test_ane_passes: tests/test_ane_passes.m $(COMPILER_OBJ)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -framework Foundation -o $@ $< $(COMPILER_OBJ)

$(BUILDDIR)/tests/test_compiler_equiv: tests/test_compiler_equiv.m $(COMPILER_OBJ)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -framework Foundation -o $@ $< $(COMPILER_OBJ)

$(BUILDDIR)/tests/test_depth_serve_protocol_v2: tests/test_depth_serve_protocol_v2.m apps/cli/commands/depth_serve_support.m
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -framework Foundation -o $@ $^

$(BUILDDIR)/tests/test_blobfile: tests/test_blobfile.c core/blobfile.c
	@mkdir -p $(dir $@)
	$(CC) -O2 -Wall -Wextra -I . -I core -o $@ $^ -framework Foundation

# Standalone pure-C compiler-only tests (ORX-400 / ORX-408): no runtime or
# compiler object dependencies, link only Foundation.
$(BUILDDIR)/tests/test_orion_error: tests/test_orion_error.c core/orion_error.c
	@mkdir -p $(dir $@)
	$(CC) -O2 -Wall -Wextra -I . -I core -o $@ tests/test_orion_error.c core/orion_error.c -framework Foundation

$(BUILDDIR)/tests/test_orion_ledger: tests/test_orion_ledger.c core/orion_compile_ledger.c
	@mkdir -p $(dir $@)
	$(CC) -O2 -Wall -Wextra -I . -I core -o $@ tests/test_orion_ledger.c core/orion_compile_ledger.c -framework Foundation

# Run benchmarks
bench: orion
	./orion bench kernels --iters 10

# Clean all build artifacts
clean:
	rm -rf $(BUILDDIR)
	rm -f orion
