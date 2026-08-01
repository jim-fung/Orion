#!/usr/bin/env python3
"""Portable source regressions for depth-serve ORION-0074 and ORION-0075.

`DepthWorker` and its compile path are private to depth_serve.m and require ANE
runtime/assets to exercise.  This test deliberately checks the small,
load-bearing control-flow contract at the owning source rather than duplicating
that private implementation.
"""

from pathlib import Path
import re
import sys


SOURCE = Path(__file__).resolve().parents[1] / "apps/cli/commands/depth_serve.m"


def function_body(source: str, name: str) -> str:
    match = re.search(r"static bool " + re.escape(name) + r"\([^)]*\)\s*\{", source)
    if not match:
        raise AssertionError(f"missing static bool {name}")
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


def test_final_norm_is_validated_before_any_compile() -> None:
    body = function_body(SOURCE.read_text(), "depth_worker_compile")
    require_in_order(
        body,
        "w->final_norm_data = [NSData dataWithContentsOfFile:final_norm_path];",
        "orion_depth_serve_final_norm_scale(",
        "if (norm_status != ORION_DEPTH_SERVE_SUPPORT_OK)",
        "return false;",
        "if (!isfinite((float)w->final_norm_scale[c]))",
        "return false;",
        "if (w->seqpack)",
        "orion_compile_mil",
    )


def test_seqpack_compile_schedule_is_exactly_four_layers() -> None:
    body = function_body(SOURCE.read_text(), "depth_worker_compile")
    assert "const int planned_compiles = w->seqpack ? LAYERS : LAYERS + 1;" in body
    seqpack = body[body.index("if (w->seqpack)"):body.index("} else {", body.index("if (w->seqpack)"))]
    assert "for (int n = 0; n < LAYERS; n++)" in seqpack
    assert "compile_seqpack_layer(root, n, mil_n," in seqpack
    assert "final_norm_program" not in seqpack
    assert "depth-serve-final-norm" not in seqpack
    assert "orion_compile_mil(" not in seqpack


def test_compile_preflight_rejects_an_insufficient_budget() -> None:
    body = function_body(SOURCE.read_text(), "depth_worker_compile")
    require_in_order(
        body,
        "w->compile_ledger = orion_ledger_init(planned_compiles, max_compile);",
        "if (max_compile <= 0 || planned_compiles > max_compile)",
        "depth-serve: compile preflight planned=%d exceeds max=%d",
        "return false;",
        "w->final_norm_data = [NSData dataWithContentsOfFile:final_norm_path];",
    )


def test_readiness_requires_the_planned_ledger_count_and_allocations() -> None:
    body = function_body(SOURCE.read_text(), "depth_worker_compile")
    require_in_order(
        body,
        "if (w->compile_count != planned_compiles ||",
        "return false;",
        "if (!w->output || !w->prefix[0] || !w->prefix[1] ||",
        "return false;",
        "w->ready = true;",
    )


def test_seqpack_runtime_fails_closed_when_checked_norm_fails() -> None:
    body = function_body(SOURCE.read_text(), "depth_worker_run_frame")
    require_in_order(
        body,
        "orion_depth_serve_host_rmsnorm_checked(",
        "orion_depth_serve_host_rmsnorm_checked(",
        "if (cond_norm_status != ORION_DEPTH_SERVE_SUPPORT_OK ||",
        "return false;",
        "audio_head_gemv(w->hidden_cond_f32, head_step",
    )
    assert "host_rmsnorm(w->output" not in body
    assert "host_rmsnorm_custom(w->output" not in body


def main() -> int:
    tests = [
        test_final_norm_is_validated_before_any_compile,
        test_seqpack_compile_schedule_is_exactly_four_layers,
        test_compile_preflight_rejects_an_insufficient_budget,
        test_readiness_requires_the_planned_ledger_count_and_allocations,
        test_seqpack_runtime_fails_closed_when_checked_norm_fails,
    ]
    failures = 0
    print("test_depth_serve_lifecycle_static:")
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
