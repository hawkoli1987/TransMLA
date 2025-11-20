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

### QK Dot Product Tensor Shapes

| Phase | Query Shape | Key Shape | Dot Product Matrix Shape (after reshape) |
|-------|-------------|-----------|------------------------------------------|
| **Phase 0 (Original)** | `[B, H, L, d]`<br>`[B, 32, L, 128]` | `[B, H_kv, L, d]`<br>`[B, 8, L, 128]` | `[B×H×L, B×H_kv×L]`<br>`[B×32×L, B×8×L]` |
| **Phase 1 (Partial RoPE)** | `[B, H, L, D]`<br>`[B, 32, L, 1024]` | `[B, 1, L, D]`<br>`[B, 1, L, 1024]` | `[B×H×L, B×1×L]`<br>`[B×32×L, B×L]` |
| **Phase 2 (LoRA QKV)** | `[B, H, L, d + m]`<br>`[B, 32, L, 192]` | `[B, H, L, d + m]`<br>`[B, 32, L, 192]` | `[B×H×L, B×H×L]`<br>`[B×32×L, B×32×L]` |

**Symbols** (Qwen3-4B):
- `B` = batch size
- `H` = num_attention_heads = 32
- `H_kv` = num_key_value_heads = 8
- `L` = sequence length
- `d` = head_dim = 128
- `D` = latent_dim = H_kv × d = 1024
- `m` = qk_mqa_dim = 64

**Verification**: The dot product matrix is indeed 2D with shape `[n_tokens_q, n_tokens_k]` where:
- `n_tokens_q = B × H × L` (or `B × 1 × L` for Phase 1 keys)
- `n_tokens_k = B × H_kv × L` (Phase 0) or `B × L` (Phase 1) or `B × H × L` (Phase 2)

### Evaluation Data Comparison

All three phases use the **same test dataloader object**:

- **Created once**: `train_loader, test_loader = get_dataset_loader(...)` (line 70 in `converter.py`)
- **Reused**: Same `test_loader` passed to all three `evaluate_ppl()` calls:
  - Phase 0: `evaluate_ppl(model, ..., test_loader, ..., qk_monitor=original_qk_monitor)` (line 76)
  - Phase 1: `evaluate_ppl(model, ..., test_loader, ..., qk_monitor=qk_monitor)` (line 221 in `partial_rope.py`)
  - Phase 2: `evaluate_ppl(model, ..., test_loader, ..., qk_monitor=qk_monitor)` (line 478 in `lora_qkv.py`)

**Result**: Identical token sequences, batch boundaries, and attention positions across all phases. KL divergence reflects only architectural changes, not data differences.

### How KL Divergence Handles Different Matrix Shapes

**Key insight**: Despite different dot product matrix shapes across phases, KL divergence compares **scalar value distributions**, not matrix structures.

**Process** (`transmla/utils.py:285-348`):

1. **Reshape and sampling** (lines 269-284): 
   - Input Q: `[B, H, L, feature_dim]` → reshape to `[B×H×L, feature_dim]` (line 269)
   - Input K: `[B, H_kv, L, feature_dim]` → reshape to `[B×H_kv×L, feature_dim]` (line 270)
   - **No aggregation over seq_len**: Each (batch, head, token_position) is treated as an independent vector
   - Randomly sample `sample_q` query vectors from `B×H×L` total queries (line 281-283)
   - Randomly sample `sample_k` key vectors from `B×H_kv×L` (or `B×L`) total keys (line 282-284)
   - Example: From `B×H×L = 2×32×2048 = 131072` query vectors, randomly select `sample_q ≈ 45` vectors

2. **Dot product matrix** (line 285):
   ```python
   scores = torch.matmul(q_sel, k_sel.transpose(0, 1))  # [sample_q, sample_k]
   ```
   - `q_sel`: `[sample_q, feature_dim]` (e.g., `[45, 128]`)
   - `k_sel`: `[sample_k, feature_dim]` (e.g., `[45, 128]`)
   - `scores`: `[sample_q, sample_k]` (e.g., `[45, 45]`) - each element is a dot product between one query vector and one key vector

3. **Flattening** (line 286):
   ```python
   flat = scores.reshape(-1)  # [sample_q × sample_k]
   ```
   - Shape: `[sample_q × sample_k]` (e.g., `[45 × 45] = [2025]`)
   - **1D vector contains all pairwise dot products** between sampled query-key pairs
   - No aggregation over seq_len; each token position is sampled independently

4. **Aggregation** (line 297): All flattened samples from all attention calls are concatenated into a single 1D tensor:
   ```python
   return torch.cat(self._samples)  # 1D tensor: [total_samples] where total_samples = sum(sample_q × sample_k) across all calls
   ```
   - Each attention call contributes `sample_q × sample_k` scalar values
   - Final shape: `[total_samples]` where `total_samples` is the sum across all attention layers and batches

5. **Histogram computation** (lines 326-339): Both distributions are converted to histograms:
   - Find shared min/max across both distributions (line 326-327)
   - Build 512 bins with same range (line 335-336)
   - Compute histograms: `torch.histc(reference, bins=512, ...)` and `torch.histc(target, bins=512, ...)`
   - **Both histograms have the same shape**: `[512]` (probability mass over bins)

6. **KL divergence** (line 347): Compare the two 1D probability distributions:
   ```
   KL(P‖Q) = Σ_i  P_i * log((P_i + ε) / (Q_i + ε))    with ε = 1e-8
   ```

**Example** (Qwen3-4B, B=2, L=2048):
- Phase 0: 
  - Q: `[2, 32, 2048, 128]` → reshape → `[131072, 128]` → sample 45 → `[45, 128]`
  - K: `[2, 8, 2048, 128]` → reshape → `[32768, 128]` → sample 45 → `[45, 128]`
  - Dot product: `[45, 45]` → flatten → `[2025]` (1D vector)
- Phase 1:
  - Q: `[2, 32, 2048, 1024]` → reshape → `[131072, 1024]` → sample 45 → `[45, 1024]`
  - K: `[2, 1, 2048, 1024]` → reshape → `[4096, 1024]` → sample 45 → `[45, 1024]`
  - Dot product: `[45, 45]` → flatten → `[2025]` (1D vector)
- Both 1D vectors → histograms `[512]` → KL divergence compares the two `[512]` histograms

**Are Phase 0 and Phase 1 sampling the same points?**

**No, they are not the same**:

1. **Random sampling** (line 281-282): Each call to `observe()` generates new random indices:
   ```python
   q_idx = torch.randint(0, total_q, (sample_q,), device=q.device)  # Different random indices each time
   k_idx = torch.randint(0, total_k, (sample_k,), device=k.device)
   ```
   - Phase 0 and Phase 1 are evaluated in **separate passes** through the test loader
   - Each pass generates different random indices for sampling

2. **Different vector spaces**: Even if indices were the same, the vectors are different:
   - Phase 0: Query vectors are `[128]` dimensional (original head_dim)
   - Phase 1: Query vectors are `[1024]` dimensional (mapped to latent space via `k_up_proj`)
   - Same `(batch, head, token_position)` index → completely different vector content

3. **Same test data, different models**: 
   - Both phases process the **same test batches** (same token sequences)
   - But the **model architecture** is different, so the query/key vectors at the same positions have different values
   - Sampling happens **after** the model transformation, so the sampled vectors reflect the architectural differences

**Implication**: The comparison ignores matrix structure and focuses on whether the **distribution of attention score magnitudes** is preserved across phases, even though the specific sampled points and their values are different.

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
