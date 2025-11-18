# Qwen2.5 vs Qwen3 Conversion Workflow Differences

This document identifies all differences between the Qwen2.5 and Qwen3 conversion workflows to MLA models.

## Summary

| Aspect | Qwen2.5 | Qwen3 |
|--------|---------|-------|
| **Model Files** | Uses LlamaMLA model files | Uses dedicated Qwen3MLA model files |
| **Script Complexity** | Minimal (4 parameters) | Detailed (12+ parameters) |
| **Default Arguments** | Relies on converter.py defaults | Explicitly overrides most defaults |
| **Calibration Settings** | Default (128 samples, batch 8, seqlen 256) | Minimal (4 samples, batch 1, seqlen 128) |
| **Device** | Default "auto" | Explicitly "cpu" |
| **Python Interpreter** | `python` | `python3` |
| **Q LoRA Rank** | None (default) | 512 (explicit) |

---

## 1. Conversion Script Differences

### Qwen2.5 Script (`scripts/qwen2.5-7B-Instruct.sh`)

```bash
model_path=Qwen/Qwen2.5-7B-Instruct
save_path=outputs/qwen2_5-7B-Instruct-deepseek
eval_batch_size=8

python transmla/converter.py \
    --model-path $model_path \
    --save-path $save_path \
    --freqfold 4 \
    --ppl-eval-batch-size $eval_batch_size
```

**Characteristics**:
- **Minimal configuration**: Only 4 parameters explicitly set
- **Relies on defaults**: Uses all default values from `converter.py` for:
  - `dtype`: bf16 (default)
  - `device`: auto (default)
  - `cal-dataset`: wikitext2 (default)
  - `cal-nsamples`: 128 (default)
  - `cal-batch-size`: 8 (default)
  - `cal-max-seqlen`: 256 (default)
  - `collapse`: auto (default)
  - `qk-mqa-dim`: 64 (default)
  - `q-lora-rank`: None (default)
  - `kv-lora-rank`: 512 (default)
- **Python**: Uses `python` command

### Qwen3 Script (`scripts/qwen3-4B.sh`)

```bash
model_path=Qwen/Qwen3-4B
save_path=outputs/qwen3-4B-deepseek
cal_nsamples=4
cal_batch_size=1
cal_max_seqlen=128
freqfold=4
eval_batch_size=0

python3 transmla/converter.py \
    --model-path $model_path \
    --save-path $save_path \
    --dtype bf16 \
    --device cpu \
    --cal-nsamples $cal_nsamples \
    --cal-batch-size $cal_batch_size \
    --cal-max-seqlen $cal_max_seqlen \
    --ppl-eval-batch-size $eval_batch_size \
    --freqfold $freqfold \
    --collapse auto \
    --qk-mqa-dim 64 \
    --q-lora-rank 512 \
    --kv-lora-rank 512
```

**Characteristics**:
- **Explicit configuration**: 12+ parameters explicitly set
- **Minimal calibration**: Uses much smaller calibration dataset
  - `cal-nsamples`: 4 (vs 128 default)
  - `cal-batch-size`: 1 (vs 8 default)
  - `cal-max-seqlen`: 128 (vs 256 default)
- **CPU device**: Explicitly sets `--device cpu` (vs "auto" default)
- **Q LoRA enabled**: Sets `--q-lora-rank 512` (vs None default)
- **No evaluation**: Sets `--ppl-eval-batch-size 0` (disables PPL evaluation)
- **Python**: Uses `python3` command

---

## 2. Model File Differences

### Qwen2.5 Model Files

**Location**: `transmla/transformers/llama/`

**Configuration**: `transmla/modify_config.py` line 41
```python
settings["qwen2"] = settings["llama"]
```

**Files Used**:
- `transmla/transformers/llama/configuration_llamamla.py` → `LlamaMLAConfig`
- `transmla/transformers/llama/modeling_llamamla.py` → `LlamaMLAModel`, `LlamaMLAForCausalLM`

**AutoMap Settings** (from `modify_config.py`):
```python
"auto_map": {
    "AutoConfig": "configuration_llamamla.LlamaMLAConfig",
    "AutoModel": "modeling_llamamla.LlamaMLAModel",
    "AutoModelForCausalLM": "modeling_llamamla.LlamaMLAForCausalLM"
}
"architectures": ["LlamaMLAForCausalLM"]
```

**Base Class**: Inherits from `LlamaConfig` and `LlamaModel`

### Qwen3 Model Files

**Location**: `transmla/transformers/qwen3/`

**Configuration**: `transmla/modify_config.py` lines 42-49
```python
settings["qwen3"] = {
    "auto_map": {
        "AutoConfig": "configuration_qwen3mla.Qwen3MLAConfig",
        "AutoModel": "modeling_qwen3mla.Qwen3MLAModel",
        "AutoModelForCausalLM": "modeling_qwen3mla.Qwen3MLAForCausalLM"
    },
    "architectures": ["Qwen3MLAForCausalLM"],
}
```

**Files Used**:
- `transmla/transformers/qwen3/configuration_qwen3mla.py` → `Qwen3MLAConfig`
- `transmla/transformers/qwen3/modeling_qwen3mla.py` → `Qwen3MLAModel`, `Qwen3MLAForCausalLM`

**Base Class**: Inherits from `Qwen3Config` and `Qwen3Model`

**Key Difference**: Qwen3 has dedicated model files that properly inherit from Qwen3 base classes, while Qwen2.5 reuses Llama model files.

---

## 3. Configuration Parameter Differences

| Category | Parameter | Qwen2.5 | Qwen3 | Default |
|----------|-----------|---------|-------|---------|
| **Calibration** | `cal-nsamples` | 128 | **4** | 128 |
| | `cal-batch-size` | 8 | **1** | 8 |
| | `cal-max-seqlen` | 256 | **128** | 256 |
| | `cal-dataset` | wikitext2 | wikitext2 | wikitext2 |
| **Architecture** | `q-lora-rank` | **None** | **512** | None |
| | `kv-lora-rank` | 512 | 512 | 512 |
| | `qk-mqa-dim` | 64 | 64 | 64 |
| | `collapse` | auto | auto | auto |
| | `freqfold` | 4 | 4 | auto |
| **Device/Evaluation** | `device` | auto | **cpu** | auto |
| | `dtype` | bf16 | bf16 | bf16 |
| | `ppl-eval-batch-size` | 8 | **0** | 2 |

**Key Architectural Difference**: 
- **Qwen2.5**: `q_proj` is a full-rank projection (no LoRA)
- **Qwen3**: `q_proj` is decomposed into `q_a_proj` (hidden_size → 512) and `q_b_proj` (512 → num_heads * head_dim)

---

## 4. Code Path Differences

### Model Type Detection

**File**: `transmla/converter.py` line 27

Both models are validated:
```python
assert model.config.model_type in ["llama", "qwen2", "qwen3", "mistral", "mimo"] or not args.deepseek_style
```

### Configuration Modification

**File**: `transmla/modify_config.py` lines 41-49, 59-60

**Qwen2.5**:
```python
settings["qwen2"] = settings["llama"]  # Line 41
transformers_dirs["qwen2"] = transformers_dirs["llama"]  # Line 59
```

**Qwen3**:
```python
settings["qwen3"] = {
    "auto_map": {
        "AutoConfig": "configuration_qwen3mla.Qwen3MLAConfig",
        "AutoModel": "modeling_qwen3mla.Qwen3MLAModel",
        "AutoModelForCausalLM": "modeling_qwen3mla.Qwen3MLAForCausalLM"
    },
    "architectures": ["Qwen3MLAForCausalLM"],
}  # Lines 42-49
transformers_dirs["qwen3"] = "transmla/transformers/qwen3"  # Line 60
```

**Impact**: When `modify_config()` is called:
- Qwen2.5 copies files from `transmla/transformers/llama/`
- Qwen3 copies files from `transmla/transformers/qwen3/`

---

## 5. Conversion Pipeline Differences

### Phase 0: Model Loading

**No differences** - Both use the same `load_model_and_tokenizer()` function.

### Phase 1: Partial RoPE Transformation

**No differences** - Both use the same `partial_rope()` function with identical logic.

**Note**: The calibration dataset size difference affects:
- Number of samples used for PCA computation
- Time taken for calibration
- Memory usage during calibration

### Phase 2: Low-Rank QKV Decomposition

**Key Difference**: Query projection handling (see Section 6 for detailed code)
- **Qwen2.5**: Single `q_proj` layer (full-rank)
- **Qwen3**: Two-layer `q_a_proj` → `q_b_proj` (low-rank with rank 512)

### Phase 3: Model Saving and Configuration

**File Copying Difference**:

**Qwen2.5**:
- Copies: `transmla/transformers/llama/configuration_llamamla.py`
- Copies: `transmla/transformers/llama/modeling_llamamla.py`
- Copies: `transmla/transformers/mla.py`

**Qwen3**:
- Copies: `transmla/transformers/qwen3/configuration_qwen3mla.py`
- Copies: `transmla/transformers/qwen3/modeling_qwen3mla.py`
- Copies: `transmla/transformers/mla.py`

---

## 6. Architecture Differences in Saved Models

### Attention Structure Comparison

**Qwen2.5** (Full-rank query):
```
hidden_states → q_proj → [num_heads, qk_nope_head_dim + qk_rope_head_dim]
             → kv_a_proj_with_mqa → [kv_lora_rank + qk_rope_head_dim]
             → kv_a_layernorm (optional)
             → kv_b_proj → [num_heads, qk_nope_head_dim + v_head_dim]
```

**Qwen3** (Low-rank query):
```
hidden_states → q_a_proj → [q_lora_rank]
             → q_a_layernorm (optional)
             → q_b_proj → [num_heads, qk_nope_head_dim + qk_rope_head_dim]
             → kv_a_proj_with_mqa → [kv_lora_rank + qk_rope_head_dim]
             → kv_a_layernorm (optional)
             → kv_b_proj → [num_heads, qk_nope_head_dim + v_head_dim]
```

### Source Code Implementation

The key difference is controlled by the conditional `if q_lora_rank is not None:` in both conversion-time (`lora_qkv.py`) and final model (`mla.py`) code:

**Qwen2.5 Path** (`q_lora_rank=None`):
```python
# Initialization (lora_qkv.py:81-88, mla.py:47-48)
self.q_proj = nn.Linear(hidden_size, num_heads * qk_head_dim, bias=attention_bias)

# Forward (lora_qkv.py:239-240, mla.py:89-90)
query_states = self.q_proj(hidden_states)
```

**Qwen3 Path** (`q_lora_rank=512`):
```python
# Initialization (lora_qkv.py:64-80, mla.py:49-53)
self.q_a_proj = nn.Linear(hidden_size, q_lora_rank, bias=False)
if use_qkv_norm:
    self.q_a_layernorm = nn.RMSNorm(q_lora_rank, ...)
self.q_b_proj = nn.Linear(q_lora_rank, num_heads * qk_head_dim, bias=attention_bias)

# Forward (lora_qkv.py:234-238, mla.py:91-94)
query_states = self.q_a_proj(hidden_states)
if hasattr(self, "q_a_layernorm"):
    query_states = self.q_a_layernorm(query_states)
query_states = self.q_b_proj(query_states)
```

---

## 7. Memory and Performance Implications

### Calibration Phase

| Aspect | Qwen2.5 | Qwen3 |
|--------|---------|-------|
| Calibration samples | 128 | 4 |
| Batch size | 8 | 1 |
| Max sequence length | 256 | 128 |
| **Total tokens** | **~32,768** | **~512** |
| **Memory usage** | Higher | Lower |
| **Calibration time** | Longer | Shorter |

### Model Size

**Qwen2.5**:
- Query projection: `hidden_size × (num_heads × (qk_mqa_dim + head_dim))` parameters
- No query LoRA bottleneck

**Qwen3**:
- Query projection: `hidden_size × q_lora_rank + q_lora_rank × (num_heads × (qk_mqa_dim + head_dim))` parameters
- Query LoRA bottleneck reduces parameters if `q_lora_rank < hidden_size`

For Qwen3-4B (hidden_size=3584, num_heads=32, head_dim=128, qk_mqa_dim=64):
- **Qwen2.5 style**: 3,584 × (32 × 192) = 22,020,096 parameters
- **Qwen3 style**: 3,584 × 512 + 512 × (32 × 192) = 1,835,008 + 3,145,728 = 4,980,736 parameters
- **Reduction**: ~77% fewer query projection parameters

---

## 8. Summary of Key Differences

### Critical Differences

1. **Model Files**: Qwen2.5 uses Llama model files; Qwen3 has dedicated Qwen3 model files
2. **Query LoRA**: Qwen2.5 has no query LoRA (`q-lora-rank=None`); Qwen3 uses query LoRA with rank 512
3. **Calibration**: Qwen2.5 uses standard calibration (128 samples, batch 8); Qwen3 uses minimal calibration (4 samples, batch 1)
4. **Device**: Qwen2.5 uses auto device; Qwen3 explicitly uses CPU
5. **Evaluation**: Qwen2.5 evaluates perplexity; Qwen3 disables evaluation

### Non-Critical Differences

1. **Python interpreter**: `python` vs `python3` (environment-dependent)
2. **Script verbosity**: Qwen3 script explicitly sets more parameters

---

## 9. Recommendations

### When to Use Qwen2.5 Approach

- Standard conversion with full calibration
- When query compression is not needed
- When using GPU resources
- When perplexity evaluation is desired

### When to Use Qwen3 Approach

- Memory-constrained environments
- Faster conversion needed
- Query compression desired (reduced model size)
- CPU-only environments
- When evaluation is not needed

---

**Last Updated**: November 18, 2025

