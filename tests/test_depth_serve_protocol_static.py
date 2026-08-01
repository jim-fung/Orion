#!/usr/bin/env python3
"""Static source regressions for the strict depth-serve v1 request protocol.

This deliberately inspects the owning Objective-C source's control flow.  It
checks that validation and failure gates remain in place, but does not claim to
exercise the ANE worker or prove runtime behavior.
"""

from pathlib import Path
import re
import sys


SOURCE = Path(__file__).resolve().parents[1] / "apps/cli/commands/depth_serve.m"


def function_body(source: str, name: str, return_type: str = "bool") -> str:
    match = re.search(
        r"static " + re.escape(return_type) + r" " + re.escape(name) + r"\([^)]*\)\s*\{",
        source,
    )
    if not match:
        raise AssertionError(f"missing static {return_type} {name}")
    opening = source.find("{", match.start(), match.end())
    depth = 0
    for index in range(opening, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[opening + 1:index]
    raise AssertionError(f"unterminated {name}")


def require_in_order(body: str, *fragments: str) -> None:
    cursor = 0
    for fragment in fragments:
        found = body.find(fragment, cursor)
        if found < 0:
            raise AssertionError(f"missing or out-of-order fragment: {fragment!r}")
        cursor = found + len(fragment)


def test_protocol_version_is_exactly_v1_and_every_response_is_versioned() -> None:
    source = SOURCE.read_text()
    emit = function_body(source, "emit_depth_response", "void")
    assert 'versioned[@"protocol_version"] = @(DEPTH_PROTOCOL_VERSION);' in emit
    assert '#define DEPTH_PROTOCOL_VERSION 1' in source
    assert 'json_exact_integer(req[@"protocol_version"], DEPTH_PROTOCOL_VERSION,' in source
    assert 'DEPTH_PROTOCOL_VERSION, &protocol_version)' in source
    assert 'emit_depth_error(nil, @"unsupported-protocol-version");' in source


def test_request_lines_are_bounded_and_overlong_input_is_discarded() -> None:
    source = SOURCE.read_text()
    reader = function_body(source, "read_bounded_request_line", "int")
    assert '#define MAX_REQUEST_BYTES (1024u * 1024u)' in source
    require_in_order(
        reader,
        "if (len == MAX_REQUEST_BYTES + 1u)",
        "discard_stdin_line_remainder();",
        "return 2;",
    )
    command = source[source.index("while ((line_status = read_bounded_request_line"):]
    require_in_order(
        command,
        'if (line_status == 2)',
        'emit_depth_error(nil, @"request-too-large");',
        "continue;",
        "if (len > MAX_REQUEST_BYTES)",
        'emit_depth_error(nil, @"request-too-large");',
    )


def test_shutdown_and_frame_schemas_are_exact_and_input_modes_exclusive() -> None:
    source = SOURCE.read_text()
    command = source[source.index("id cmd = req[@\"cmd\"];"):]
    require_in_order(
        command,
        '[(NSString *)cmd isEqualToString:@"shutdown"]',
        'request_has_exact_keys(req, @[@"protocol_version", @"cmd"])',
        'emit_depth_response(@{ @"ok": @YES, @"shutdown": @YES });',
        "bool has_input_dir",
        "bool has_b64_pair = has_cond_b64 && has_uncond_b64;",
        "if ((has_input_dir && (has_cond_b64 || has_uncond_b64)) ||",
        'emit_depth_error(nil, @"invalid-hidden-input-method");',
        '@[@"protocol_version", @"input_dir", @"output_dir", @"c0", @"gamma", @"frame_id"]',
        '@[@"protocol_version", @"hidden_cond_b64", @"hidden_uncond_b64",',
        "if (!request_has_exact_keys(req, frame_keys))",
        'emit_depth_error(nil, @"invalid-request-schema");',
    )


def test_inline_tensors_require_canonical_base64_at_exact_size() -> None:
    source = SOURCE.read_text()
    decoder = function_body(source, "strict_base64_tensor")
    require_in_order(
        decoder,
        "initWithBase64EncodedString:encoded options:0",
        "if (!decoded || decoded.length != TENSOR_BYTES) return false;",
        "[decoded base64EncodedStringWithOptions:0]",
        "isEqualToString:encoded",
        "return false;",
    )
    assert 'strict_base64_tensor(cond_b64_value, &in_cond)' in source
    assert 'strict_base64_tensor(uncond_b64_value, &in_uncond)' in source
    assert 'emit_depth_error(frame_id, @"invalid-hidden-tensor-base64");' in source


def test_directory_tensors_must_be_regular_files_of_exact_size() -> None:
    source = SOURCE.read_text()
    checker = function_body(source, "regular_file_with_exact_size")
    require_in_order(
        checker,
        "stat(fs_path, &st) != 0",
        "!S_ISREG(st.st_mode)",
        "st.st_size < 0",
        "(uintmax_t)st.st_size != (uintmax_t)expected_size",
        "return false;",
    )
    require_in_order(
        source,
        "regular_file_with_exact_size(cond_path, TENSOR_BYTES)",
        "regular_file_with_exact_size(uncond_path, TENSOR_BYTES)",
        'emit_depth_error(frame_id, @"invalid-hidden-tensor-files");',
        "in_cond = [NSData dataWithContentsOfFile:cond_path];",
        "in_uncond = [NSData dataWithContentsOfFile:uncond_path];",
        "if (in_cond.length != TENSOR_BYTES || in_uncond.length != TENSOR_BYTES)",
    )


def test_each_request_is_isolated_and_success_follows_checked_output_write() -> None:
    source = SOURCE.read_text()
    loop = source[source.index("while ((line_status = read_bounded_request_line"):]
    require_in_order(
        loop,
        "while ((line_status = read_bounded_request_line(line, &len)) != 0)",
        "@autoreleasepool",
        "@try",
        "if (!write_tokens_bin_checked(output_dir, tokens))",
        'emit_depth_error(frame_id, @"output-write-failed");',
        "continue;",
        "emit_depth_response(@{",
        "} @catch (NSException *exception)",
        'emit_depth_error(nil, @"request-handler-failed");',
    )
    writer = function_body(source, "write_tokens_bin_checked")
    assert "existing_directory(output_dir)" in writer
    assert "NSDataWritingAtomic" in writer
    assert "writeToFile:path" in writer


def main() -> int:
    tests = [
        test_protocol_version_is_exactly_v1_and_every_response_is_versioned,
        test_request_lines_are_bounded_and_overlong_input_is_discarded,
        test_shutdown_and_frame_schemas_are_exact_and_input_modes_exclusive,
        test_inline_tensors_require_canonical_base64_at_exact_size,
        test_directory_tensors_must_be_regular_files_of_exact_size,
        test_each_request_is_isolated_and_success_follows_checked_output_write,
    ]
    failures = 0
    print("test_depth_serve_protocol_static:")
    for test in tests:
        try:
            test()
        except (AssertionError, OSError) as error:
            failures += 1
            print(f"  {test.__name__}: FAIL ({error})")
        else:
            print(f"  {test.__name__}: PASS")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
