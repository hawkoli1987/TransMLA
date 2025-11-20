## KL Divergence Tracking During Conversion

This document explains how KL divergence between Q K dot-product distributions is computed during the Qwen3 conversion pipeline.

### Evaluation Flow
- **Dataset**: Both the original model and the converted checkpoints (Partial RoPE and LoraQKV) run on the *same* test dataloader produced by `prepare_test_dataloader()` (defaults: WikiText2 test split, batch size taken from `--ppl-eval-batch-size`, sequence length 2048). This guarantees identical samples.
- **Phase coverage**:
  - *Phase 0 (Original)*: `evaluate_ppl()` is called once on the unmodified model with `qk_monitor=QKDotProductMonitor("original")`. Samples captured here become the baseline distribution.
  - *Phase 1 (Partial RoPE)* and *Phase 2 (LoraQKV)*: their evaluation calls reuse the same test loader and enable their own monitors. Each phase prints its perplexity and the KL divergence vs. the baseline.

### Q K Sampling Logic
1. We intercept `torch.nn.functional.scaled_dot_product_attention` through `_ScaledDotProductAttentionCapture`.
2. For each attention invocation we receive `query` and `key` tensors shaped `[batch, heads, tokens, head_dim(+rope_dim)]`.
3. Each tensor is reshaped to `[batch * heads * tokens, head_dim_eff]`.
4. We randomly select subsets of queries/keys (roughly √N per side) and compute a reduced dot-product matrix `scores = Q_sel · K_selᵀ`, yielding an approximate `[n_q, n_k]` similarity matrix.
5. Scores are flattened and appended (on CPU) to the monitor. By default we keep up to 2048 samples per attention call; `monitor.num_values()` reports the aggregate count.

### Histogram-Based KL Estimate
- Let `P` denote the histogram of baseline scores and `Q` denote the histogram for a converted model.
- We build 512 bins spanning the combined min/max range with a small margin, normalize each histogram, then evaluate:

```
KL(P‖Q) = Σ_i  P_i * log((P_i + ε) / (Q_i + ε))    with ε = 1e-8
```

- The result is printed after each phase:
  - `Partial RoPE QK KL vs original: …`
  - `LoraQKV QK KL vs original: …`

### Interpreting KL Values
- **Good conversion**: KL near zero (≤ 0.1) indicates the converted attention scores closely match the original distribution—expected when perplexity regresses minimally.
- **Acceptable drift**: KL in the ~0.1–0.5 range usually corresponds to moderate attention shifts; check perplexity and downstream accuracy before accepting.
- **Poor conversion**: KL > 0.5 or “unavailable (insufficient samples)” flags significant divergence. Revisit calibration data, partial-rope settings, or low-rank parameters.

> KL is asymmetric, so `KL(original‖converted)` tells us how well the converted model covers the original distribution. Large values typically correlate with degraded perplexity.
