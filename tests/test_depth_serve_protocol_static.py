#!/usr/bin/env python3
"""Static regressions for strict fail-closed depth-serve protocol v2."""

from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "apps/cli/commands/depth_serve.m"
RUNTIME = ROOT / "core/ane_runtime.m"
SUPPORT = Path(__file__).resolve().parents[1] / "apps/cli/commands/depth_serve_support.m"


def function_body(source: str, name: str, return_type: str = "bool") -> str:
    match = re.search(
        r"static\s+" + re.escape(return_type) + r"\s*" + re.escape(name) + r"\([^)]*\)\s*\{",
        source,
    )
    if not match:
        raise AssertionError(f"missing static {return_type} {name}")
    opening = source.find("{", match.start(), match.end())
    depth = 0
    for index in range(opening, len(source)):
        if source[index] == "{": depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0: return source[opening + 1:index]
    raise AssertionError(f"unterminated {name}")


def require_in_order(body: str, *fragments: str) -> None:
    cursor = 0
    for fragment in fragments:
        found = body.find(fragment, cursor)
        if found < 0: raise AssertionError(f"missing or out-of-order fragment: {fragment!r}")
        cursor = found + len(fragment)


def test_protocol_version_launch_challenge_and_readiness_are_v2() -> None:
    source = SOURCE.read_text()
    assert '#define DEPTH_PROTOCOL_VERSION 2' in source
    assert '#define DEPTH_RECEIPT_SCHEMA @"voxtream.orion.depth-operational-receipt/2"' in source
    assert 'strcmp(argv[i], "--launch-challenge")' in source
    assert 'orion_depth_serve_is_lower_hex_sha256(launch_challenge)' in source
    readiness = source[source.index("// Signal strict v2 readiness"):]
    for field in (
        'receipt_schema', 'worker_session_id', 'launch_challenge', 'pid',
        'start_monotonic_ns', 'mode', 'startup_compile_count',
        'expected_private_evals_per_frame', 'final_norm_placement',
        'audio_head_placement', 'cfg_placement', 'sampler_placement',
        'program_set_digest',
    ):
        assert f'@"{field}"' in readiness


def test_request_lines_are_bounded_and_exact_schemas_include_receipt_identity() -> None:
    source = SOURCE.read_text()
    reader = function_body(source, "read_bounded_request_line", "int")
    require_in_order(reader, "if (len == MAX_REQUEST_BYTES + 1u)",
                     "discard_stdin_line_remainder();", "return 2;")
    command = source[source.index('bool has_input_dir = req[@"input_dir"]'):]
    assert 'request_has_exact_keys(req, frame_keys)' in command
    for field in (
        'stream_session_id', 'logical_frame_id', 'request_id', 'request_sequence',
        'challenge_nonce', 'hidden_cond_sha256', 'hidden_uncond_sha256',
        'semantic_request_sha256', 'program_set_digest',
    ):
        assert command.count(f'@"{field}"') >= 2
    assert '@"frame_id"' not in command


def test_native_hash_validation_precedes_private_evaluation() -> None:
    source = SOURCE.read_text()
    loop = source[source.index('NSString *cond_sha = orion_depth_serve_sha256_data(in_cond);'):]
    require_in_order(
        loop,
        'hidden-cond-sha256-mismatch',
        'hidden-uncond-sha256-mismatch',
        'orion_depth_serve_semantic_request_sha256(',
        'semantic-request-sha256-mismatch',
        'depth_worker_run_frame(&worker, in_cond, in_uncond,',
    )
    assert 'program-set-digest-mismatch' in source[:source.index('depth_worker_run_frame(&worker')]


def test_actual_private_calls_are_counted_and_expected_counts_are_exact() -> None:
    source = SOURCE.read_text()
    runtime = RUNTIME.read_text()
    init = function_body(source, "depth_eval_counters_init", "void")
    assert 'seqpack ? DEPTH_STEPS : DEPTH_STEPS * CFG_BRANCHES' in init
    assert 'counters->expected[LAYERS] = seqpack ? 0 : DEPTH_STEPS * CFG_BRANCHES;' in init
    counted = function_body(source, "depth_eval_counted")
    # Both seqpack and separate-CFG layers are plan-verified: every layer and
    # the final-norm eval go through orion_eval_with_plan_report so the cached
    # ABI/provenance is authenticated. The raw orion_eval_report must NOT be
    # used for any layer eval.
    require_in_order(counted, 'orion_eval_with_plan_report(w->layer_programs[program_index],',
                     'orion_eval_with_plan_report(w->final_norm_program',
                     'if (result == ORION_EVAL_NOT_INVOKED)',
                     'counters->actual[program_index]++;',
                     'counters->successful[program_index]++;',
                     'counters->failed[program_index]++;')
    assert 'orion_eval_report(' not in counted, \
        "seqpack layers must use plan-aware eval, not raw orion_eval_report"
    report = runtime[runtime.index('OrionEvalResult orion_eval_report('):]
    require_in_order(report, 'if (!wrapped) return ORION_EVAL_NOT_INVOKED;',
                     'if (!req) return ORION_EVAL_NOT_INVOKED;',
                     '@selector(evaluateWithQoS:options:request:error:)',
                     'ORION_EVAL_PRIVATE_SUCCEEDED', 'ORION_EVAL_PRIVATE_FAILED')
    run = function_body(source, "depth_worker_run_frame")
    assert run.count('depth_eval_counted(') >= 3
    assert 'native_fallback_used' in source and '@NO' in source


def test_receipt_has_compile_timing_token_output_and_chain_fields() -> None:
    source = SOURCE.read_text()
    receipt = source[source.index('NSMutableDictionary *receipt = [@{'):]
    for field in (
        'private_invocation_known', 'private_invocations_expected',
        'private_invocations_actual', 'private_invocations_successful',
        'private_invocations_failed', 'private_invocations_by_program', 'compile_count_before',
        'compile_count_after', 'compile_count_delta', 'native_fallback_used',
        'frame_start_monotonic_ns', 'frame_end_monotonic_ns',
        'frame_duration_ns', 'tokens', 'token_sha256', 'tokens_bin_sha256',
        'output_placement', 'previous_receipt_sha256', 'receipt_sha256',
    ):
        assert f'@"{field}"' in receipt
    require_in_order(receipt, 'orion_depth_serve_receipt_sha256(receipt)',
                     'receipt[@"receipt_sha256"] = receipt_sha;',
                     'serialize_depth_response(receipt)',
                     'emit_serialized_depth_response(serialized_receipt)',
                     'previous_receipt_sha256 = receipt_sha;')



def test_directory_tensor_files_are_opened_once_and_read_exactly() -> None:
    source = SOURCE.read_text()
    loader = function_body(source, "read_exact_regular_file")
    require_in_order(loader, "open(fs_path, flags)", "fstat(fd, &st)",
                     "while (offset < expected_size)",
                     "read(fd, bytes + offset", "read(fd, &trailing, 1)",
                     "if (count != 0) goto done;")
    assert "O_NOFOLLOW" in loader and "stat(fs_path" not in loader
    directory_path = source[source.index('if (has_input_dir) {'):]
    assert "read_exact_regular_file(cond_path, TENSOR_BYTES, &in_cond)" in directory_path
    assert "dataWithContentsOfFile:cond_path" not in directory_path


def test_receipt_emission_is_serialized_before_write_and_chain_advance() -> None:
    source = SOURCE.read_text()
    serializer = function_body(source, "serialize_depth_response", "NSData *")
    assert "NSJSONSerialization dataWithJSONObject" in serializer
    emitter = function_body(source, "emit_serialized_depth_response")
    require_in_order(emitter, "fwrite(", "fflush(stdout)", "ferror(stdout)")
    receipt = source[source.index('NSString *receipt_sha = orion_depth_serve_receipt_sha256(receipt);'):]
    require_in_order(receipt, "serialize_depth_response(receipt)",
                     "emit_serialized_depth_response(serialized_receipt)",
                     "previous_receipt_sha256 = receipt_sha;")
    assert "_Exit(EXIT_FAILURE)" in receipt


def test_final_norm_digest_derivations_are_documented_exactly() -> None:
    docs = (ROOT / "docs/depth_serve_protocol.md").read_text()
    assert "final-norm finalized binding plan's `mil_identity`" in docs
    assert "exact 22 UTF-8 bytes of the literal `cpu-checked-rmsnorm-v1`" in docs
    assert "no NUL, newline, or normalization" in docs


def test_phase_timings_are_opt_in_stderr_diagnostics_outside_v2_receipts() -> None:
    source = SOURCE.read_text()
    support = SUPPORT.read_text()
    assert 'strcmp(argv[i], "--phase-timings")' in source
    assert '"  --phase-timings' in source
    assert '#define DEPTH_PHASE_TIMINGS_SCHEMA @"voxtream.orion.depth-phase-timings/1"' in source
    assert 'DepthPhaseTimings *phase_timings_ptr =' in source
    assert 'phase_timings_enabled ? &phase_timings : NULL' in source
    assert 'static void emit_phase_timing_diagnostic(' in source
    assert 'write(STDERR_FILENO, bytes, line.length)' in source
    assert 'F_SETFL, flags | O_NONBLOCK' in source
    assert '(void)fcntl(STDERR_FILENO, F_SETFL, flags);' in source
    assert 'F_SETNOSIGPIPE' in source
    assert 'configure_phase_timing_stderr()' in source
    assert 'best-effort and must never affect protocol lifetime' in source
    assert '@"phase_duration_ns": phase_timings_dictionary(timings)' in source
    assert 'phase_timings' not in support

    receipt = source[source.index('NSString *receipt_sha = orion_depth_serve_receipt_sha256(receipt);'):]
    require_in_order(
        receipt,
        'orion_depth_serve_receipt_sha256(receipt)',
        'receipt[@"receipt_sha256"] = receipt_sha;',
        'serialize_depth_response(receipt)',
        'emit_serialized_depth_response(serialized_receipt)',
        'previous_receipt_sha256 = receipt_sha;',
        'emit_phase_timing_diagnostic(',
    )


def test_phase_timings_cover_both_execution_modes_and_private_programs() -> None:
    source = SOURCE.read_text()
    run = function_body(source, "depth_worker_run_frame")
    counted = function_body(source, "depth_eval_counted")
    assert run.count('phase_start(timings)') >= 13
    for field in (
        'surface_write_ns', 'surface_read_ns', 'host_rmsnorm_ns',
        'audio_head_ns', 'cfg_combine_ns', 'argmax_ns',
        'embedding_update_ns',
    ):
        assert f'&timings->{field}' in run
    assert 'timings->private_evaluation_ns += elapsed;' in counted
    assert 'timings->private_evaluation_by_program_ns[program_index] += elapsed;' in counted
    assert run.count('counters, timings,') >= 2
    assert run.count('timings, failure') >= 1


def test_canonical_hash_domains_and_no_json_hashing() -> None:
    support = SUPPORT.read_text()
    for domain in (
        'voxtream.orion.depth-semantic-request/2',
        'voxtream.orion.depth-plan-weights/2',
        'voxtream.orion.depth-program-set/2',
        'voxtream.orion.depth-operational-receipt/2',
    ):
        assert domain in support
    assert 'typed_u32' in support and 'typed_u64' in support
    assert 'NSJSONSerialization' not in support


def main() -> int:
    tests = [name for name, value in globals().items() if name.startswith("test_") and callable(value)]
    failures = 0
    print("test_depth_serve_protocol_static:")
    for name in tests:
        try: globals()[name]()
        except (AssertionError, OSError) as error:
            failures += 1; print(f"  {name}: FAIL ({error})")
        else: print(f"  {name}: PASS")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__": sys.exit(main())
