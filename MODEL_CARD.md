# Atlas Nano model/system card

## System summary

Atlas Nano is a model-coupled activation-space safety sensing system. Full Atlas
aggregates seven learned projection gates. Sign-Check Atlas distills a safety
signal into one energy axis and threshold for use as a Tier 1 filter.

Atlas Nano is not itself a generative model or a standalone natural-language
judge. Profiles are only meaningful with the base model and extraction point for
which they were calibrated.

## Intended uses

- Research on activation-space safety signals.
- Evaluation of tiered routing architectures.
- Offline comparison of thresholds and gate aggregation strategies.
- Prototyping with local language models under human supervision.

## Out-of-scope uses

- Sole enforcement control for high-stakes or production systems.
- Claims that a model or output is universally safe.
- Surveillance, profiling, or decisions about people.
- Reuse of a profile with an unverified model, revision, architecture, or hidden
  dimension.

## Included profile

`qwen3-4b-signcheck-v1` targets `Qwen/Qwen3-4B`, residual component, layer 22,
hidden dimension 2560. The selected calibration threshold is
`-0.03469539650041742`.

On the 761 harmful and 419 benign examples used for calibration, the selected
threshold produced precision 0.9485, recall 0.8476, F1 0.8952, accuracy 0.8720,
and false-positive rate 0.0835. Because the same data informed threshold
selection, these values are not held-out generalization estimates.

## Known limitations

- Results depend on prompt formatting, tokenizer, model revision, quantization,
  extraction implementation, and threshold policy.
- Category balance does not necessarily represent real traffic.
- Coverage across languages, encodings, long conversations, tool calls, and
  adaptive attacks is incomplete.
- A binary training label cannot represent every application policy.
- False negatives can pass Tier 1 and false positives can impose unnecessary
  routing or refusal.
- Included cross-family result folders are research experiments, not supported
  release profiles.

## Deployment guidance

Validate profile compatibility at startup and fail closed on mismatches. Log the
profile ID and version with every decision. Calibrate thresholds on traffic that
resembles the intended deployment, maintain a boundary route, and monitor metrics
by category rather than relying on aggregate F1 alone.
