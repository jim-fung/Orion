# `depth-serve` Transport Protocol v1

## Status and scope

This document is the normative transport contract for `orion depth-serve` protocol version 1 as implemented by `/Users/dekal/tooling/Orion/apps/cli/commands/depth_serve.m`.

The worker communicates using newline-delimited JSON (JSONL): exactly one JSON value per input line and exactly one JSON object per output line. Standard output is reserved for protocol responses; diagnostics are written to standard error.

This contract describes the worker's input/output transport behavior. It is **not** evidence that an Apple Neural Engine (ANE) execution path is qualified, validated, certified, or otherwise suitable for any particular workload.

## Versioning and envelopes

Every request, including a shutdown request, MUST contain:

```json
{"protocol_version": 1}
```

`protocol_version` MUST be a JSON number that is an exact integer equal to `1`. Missing, Boolean, fractional, non-finite, or any other version value is rejected.

Every response, including readiness and error responses, contains:

```json
{"protocol_version": 1}
```

A successful frame response has this envelope:

```json
{
  "protocol_version": 1,
  "ok": true,
  "frame_id": 42,
  "tokens": [0, 1, 2],
  "eval_ms": 12.3,
  "compile_count": 5
}
```

`tokens` contains exactly 15 integer tokens. `eval_ms` is the measured frame evaluation duration in milliseconds, and `compile_count` is the worker's startup compile count.

After successful startup and before it accepts requests, the worker emits one readiness response:

```json
{"protocol_version": 1, "ready": true, "compile_count": 5}
```

The compile count depends on worker mode: the ordinary path compiles five programs and the sequence-packed path compiles four.

## Framing and line limit

Input and output use UTF-8 JSON text terminated by `LF` (`\n`). `CRLF` request delimiters are accepted; the trailing `CR` is removed before JSON parsing.

The maximum JSON request payload is **1 MiB** (1,048,576 bytes), excluding the newline delimiter. A CRLF-delimited request may therefore have an additional trailing `CR`. An overlong line is discarded in full before processing continues with the next line, preventing stream desynchronization.

An empty line is not a request and produces an error response. A non-object JSON value is also invalid: request JSON MUST decode to an object.

## Frame request

A frame request is an object without `cmd`. It MUST include all of the following fields:

| Field | Required value |
| --- | --- |
| `protocol_version` | Exact JSON integer `1`. |
| `frame_id` | Exact, non-Boolean JSON integer in the signed 32-bit range `-2147483648` through `2147483647`. |
| `c0` | Exact, non-Boolean JSON integer from `0` through `2049`, inclusive. |
| `gamma` | Finite, non-Boolean JSON number from `0.0` through `100.0`, inclusive. |
| `output_dir` | Nonempty string naming an existing directory. |
| hidden input | Exactly one supported hidden-input method, defined below. |

The worker does not treat omitted fields as defaults. In particular, `gamma` is required; there is no protocol-level default gamma.

### Hidden input methods

The request MUST use exactly one of these mutually exclusive methods:

1. **Directory input:** `input_dir` is present, and neither base64 field is present.
2. **Inline input:** both `hidden_cond_b64` and `hidden_uncond_b64` are present, and `input_dir` is absent.

Providing one inline field without the other, mixing a directory with either inline field, or omitting both methods is invalid.

#### Directory input

`input_dir` MUST be a nonempty string naming an existing directory. That directory MUST contain both files:

- `hidden_cond.fp16.bin`
- `hidden_uncond.fp16.bin`

Each file MUST be exactly 65,536 bytes. The worker requires the files by these exact names and does not accept a partial directory input.

#### Inline base64 input

`hidden_cond_b64` and `hidden_uncond_b64` MUST each be strings containing canonical standard Base64 for exactly 65,536 decoded bytes.

Canonical means the supplied string MUST exactly equal the worker's standard Base64 re-encoding of its decoded bytes. Consequently, whitespace, ignored characters, alternate formatting, malformed padding, and noncanonical encodings are rejected. Both decoded tensors must have the required exact length.

## Output directory and `tokens.bin`

`output_dir` is mandatory for every frame request. It MUST already exist and be a directory; the worker does not create it.

After a successful evaluation, the worker writes `tokens.bin` in `output_dir` atomically. The file contains the 15 returned tokens as contiguous 32-bit little-endian integers (60 bytes total). A failure to write this file makes the request fail; a success response is emitted only after the checked write succeeds.

## Errors

An error response is a versioned object with `ok: false` and an `error` string:

```json
{"protocol_version": 1, "ok": false, "error": "invalid-gamma"}
```

When the worker has successfully validated `frame_id`, frame-related errors additionally include that `frame_id`. Errors occurring before `frame_id` validation do not include it.

The defined error strings are:

| Error | Meaning |
| --- | --- |
| `request-too-large` | Request payload exceeds the 1 MiB limit. |
| `empty-request` | Input line is empty. |
| `invalid-json-object` | Input is not valid JSON that decodes to an object. |
| `unsupported-protocol-version` | `protocol_version` is absent or is not exact integer version 1. |
| `invalid-command` | `cmd` is present but is not the string `"shutdown"`. |
| `invalid-frame-id` | Frame request `frame_id` is missing or is not an exact signed 32-bit integer. |
| `invalid-c0` | `c0` is missing or outside its required integer range. |
| `invalid-gamma` | `gamma` is missing, non-finite, nonnumeric, Boolean, or outside `[0.0, 100.0]`. |
| `invalid-output-dir` | `output_dir` is not a nonempty string naming an existing directory. |
| `invalid-hidden-input-method` | Hidden-input methods are absent, mixed, or incomplete. |
| `invalid-input-dir` | `input_dir` is not a nonempty string naming an existing directory. |
| `invalid-hidden-tensor-files` | Required directory tensor files are missing or are not exactly 65,536 bytes each. |
| `invalid-hidden-tensor-base64` | Either inline tensor is not a canonical Base64 string decoding to exactly 65,536 bytes. |
| `eval-failed` | Evaluation did not complete successfully. |
| `output-write-failed` | `tokens.bin` could not be atomically written to `output_dir`. |
| `response-encode-failed` | Fixed fallback error if a normal response cannot be JSON encoded. |

Clients MUST match error strings exactly and MUST NOT infer a successful evaluation from partial output or from stderr text.

## Shutdown

To stop a ready worker, send:

```json
{"protocol_version": 1, "cmd": "shutdown"}
```

On success, the worker writes:

```json
{"protocol_version": 1, "ok": true, "shutdown": true}
```

It then exits after cleanup. `shutdown` is the only recognized command. A request containing `cmd` is handled as a command request rather than as a frame request; no frame fields are required for the shutdown command.

## Compatibility

Protocol v1 is strict. There is no legacy request format, no implicit version negotiation, no default protocol version, and no compatibility fallback for legacy file or inline payload forms. Clients MUST send v1 envelopes and MUST conform to the field, range, framing, and input-method requirements in this document.
