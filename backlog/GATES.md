# Qualification Gates

## Local integrity gates

1. Canonical BLOBFILE ranges and hashes are consistent across generation, validation, patch, and eval.
2. A plan supplied at eval is cryptographically/canonically bound to the compiled program.
3. Operational depth paths use finalized plan-aware compile/eval.
4. Stories patch generations are globally transactional.
5. Operational tensor paths use checked allocations/copies.
6. PASS/FAIL/SKIP and required capability coverage are explicit.

## External gates

### G008 / VXANE-G008
Reported by `AGENTS.md` as reliability/production-path qualification. Local Orion work can provide failure-injection, cleanup, plan, and serving evidence, but the external owner decides closure.

### G009
Reported multi-chunk aggregate BLOBFILE qualification. Required local evidence includes canonical chunk parsing/hashing, first-chunk support, overlap safety, plan-aware compile/eval, tamper rejection, and numerical parity.

## Gate evidence minimum

Each gate record must name source revision, external revision where relevant, asset manifests, hardware/OS/compiler profile, commands, PASS/FAIL/SKIP, numerical tolerances, performance methodology, and surviving limitations. Historical results cannot silently stand in for current operational-path evidence.
