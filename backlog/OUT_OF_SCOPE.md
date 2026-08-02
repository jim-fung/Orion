# Out of Scope and Required Research Spikes

## Explicit non-goals unless the product boundary changes

- Cross-platform ANE execution or implied Linux/Windows support.
- iOS/iPadOS or App Store support without entitlement/sandbox/private-API evidence.
- Arbitrary Hugging Face architecture conversion.
- Committing large model weights, datasets, checkpoints, or raw benchmark logs to Git.
- Hostile multi-tenant serving, authentication, quotas, or public-network exposure.
- Claims of measured ANE utilization/TOPS without validated counters.
- Treating experiments/archive as supported API.
- Using historical M4/M5 numbers as current release gates.
- Building a full macOS app before a product/distribution decision.
- Automatic network downloads during ordinary builds/tests.

## Research spikes

1. Private-API signing, notarization, hardened-runtime, and sandbox viability.
2. Cross-macOS/SoC compatibility for selectors, MIL profiles, offsets, binding order, and patch semantics.
3. Current four-layer seqpack external parity with host RMSNorm.
4. Supervised subprocess isolation overhead and recovery behavior.
5. Minimal macapp viability before committing to a GUI roadmap.
6. Need for general SentencePiece `.model` support.
7. Intended Stories positional-encoding/model contract.
8. Release channel choice: source-only, tarball, Homebrew, installer, or app.

A spike produces a written decision, measurements, and follow-up tasks; it does not silently become a supported feature.
