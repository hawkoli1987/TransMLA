# Qwen3 4B to MLA Conversion - Complete Technical Guide

This document provides a comprehensive, step-by-step guide for converting the native Qwen3 4B model into a Multi-head Latent Attention (MLA) model, with specific references to all Python scripts, classes, and functions used in each phase.

## Table of Contents
- [Overview](#overview)
- [Conversion Pipeline](#conversion-pipeline)
- [Phase 0: Model Loading and Setup](#phase-0-model-loading-and-setup)
- [Phase 1: Partial RoPE Transformation](#phase-1-partial-rope-transformation)
- [Phase 2: Low-Rank QKV Decomposition](#phase-2-low-rank-qkv-decomposition)
- [Phase 3: Model Saving and Configuration](#phase-3-model-saving-and-configuration)
- [Final Architecture](#final-architecture)
- [Key Hyperparameters](#key-hyperparameters)

---

## Overview

The conversion process transforms a standard Qwen3 4B model with Grouped Query Attention (GQA) into a DeepSeek-style Multi-head Latent Attention (MLA) model. The conversion is performed using calibration data and involves two main transformations:

1. **Partial RoPE**: Decouple and compress rotary position embeddings (RoPE)
2. **Low-Rank QKV**: Apply low-rank decomposition to query, key, and value projections

**Entry Script**: `scripts/qwen3-4B.sh`  
**Main Converter**: `transmla/converter.py`

---

## Conversion Pipeline

The complete conversion pipeline follows this flow:

```
Original Qwen3 Model (GQA)
    ↓
[Load Model & Tokenizer]
    ↓
[Evaluate Original PPL (Optional)]
    ↓
[Phase 1: Partial RoPE Transformation]
    ↓
[Evaluate Partial RoPE PPL (Optional)]
    ↓
[Phase 2: Low-Rank QKV Decomposition]
    ↓
[Evaluate MLA PPL (Optional)]
    ↓
[Save Model & Tokenizer]
    ↓
[Modify Configuration]
    ↓
Qwen3 MLA Model (DeepSeek-compatible)
```

---

## Phase 0: Model Loading and Setup

### Entry Point
**Script**: `scripts/qwen3-4B.sh`
```bash
python3 transmla/converter.py \
    --model-path Qwen/Qwen3-4B \
    --save-path outputs/qwen3-4B-deepseek \
    --dtype bf16 \
    --device cpu \
    --ppl-eval-batch-size 0 \
    --freqfold 4 \
    --collapse auto \
    --qk-mqa-dim 64 \
    --q-lora-rank 512 \
    --kv-lora-rank 512
```

**Note**: Calibration settings (`cal-nsamples`, `cal-batch-size`, `cal-max-seqlen`) use default values (128, 8, 256 respectively).

### Main Function
**File**: `transmla/converter.py`  
**Function**: `main(args)` (lines 52-107)

### Step 0.1: Load Model and Tokenizer
**Function**: `load_model_and_tokenizer(args)` in `transmla/converter.py` (lines 12-29)

**Key Operations**:
- Loads Qwen3 model using `AutoModelForCausalLM.from_pretrained()`
- Loads tokenizer using `AutoTokenizer.from_pretrained()`
- Sets tokenizer pad token if missing
- Validates model type is compatible (qwen3 supported)

**Classes Used**:
- `transformers.AutoModelForCausalLM`
- `transformers.AutoTokenizer`

### Step 0.2: Prepare Calibration Dataset
**Function**: `get_dataset_loader(tokenizer, **kwargs)` in `transmla/converter.py` (lines 32-49)

**Helper Functions** (from `transmla/utils.py`):
- `get_dataset(name)` (lines 10-57): Loads dataset (wikitext2, ptb, c4, or alpaca)
- `prepare_dataloader()` (lines 103-183): Creates calibration dataloader with specified batch size and sequence length
- `prepare_test_dataloader()` (lines 59-101): Creates test dataloader for perplexity evaluation (optional)

**Key Operations**:
- Downloads and prepares calibration dataset (e.g., wikitext2)
- Tokenizes and batches samples with max sequence length
- Returns train_loader and test_loader (if eval enabled)

### Step 0.3: Evaluate Original Model (Optional)
**Function**: `evaluate_ppl()` in `transmla/utils.py` (lines 206-257)

**Key Operations**:
- Computes perplexity on test dataset
- Uses CrossEntropyLoss with autoregressive shifting
- Reports baseline performance before conversion

---

## Phase 1: Partial RoPE Transformation

### Main Function
**File**: `transmla/partial_rope.py`  
**Function**: `partial_rope(model, tokenizer, train_loader, test_loader, **kwargs)` (lines 196-250)

### Step 1.1: Calibrate Original QKV Outputs
**Function**: `get_qkv_calibrate_outputs()` in `transmla/utils.py` (lines 316-385)

**Key Operations**:
- Registers forward hooks on q_proj, k_proj, v_proj layers
- Runs model on calibration data to capture intermediate activations
- Collects query, key, value outputs from all layers
- Masks padded positions using attention_mask

**Helper Function**: `insert_qkv_hooks()` in `transmla/utils.py` (lines 259-313)

### Step 1.2: Joint Complex PCA on Key Outputs
**Method**: `PartialRope.joint_complex_pca()` in `transmla/partial_rope.py` (lines 86-106)

**Key Operations**:
- Computes covariance matrix H from key outputs
- Reshapes keys to handle complex RoPE structure (interleaved real/imaginary)
- Applies damping to diagonal for numerical stability
- Computes eigendecomposition and sorts by descending eigenvalues
- Returns rotation matrix U for RoPE compression

### Step 1.3: Create PartialRope Attention Module
**Class**: `PartialRope` in `transmla/partial_rope.py` (lines 47-192)

**Initialization** (lines 48-73):
- Inherits from `nn.Module`
- Copies q_proj, k_proj, v_proj, o_proj from original attention
- Inserts k_up_proj and v_up_proj for upsampling
- Applies PCA rotation to k_proj and k_up_proj

**Method**: `_insert_kv_up_proj()` (lines 74-83)
- Creates identity upsampling matrices for K and V
- Initialized as expanded identity to maintain equivalence

**Method**: `rotate_k_proj()` (lines 108-127)
- Rotates k_proj weights using PCA rotation matrix U
- Handles bias if present
- Permutes dimensions to align with DeepSeek's interleaved RoPE format

**Method**: `rotate_k_up_proj()` (lines 129-140)
- Rotates k_up_proj weights using same rotation matrix
- Maintains consistency between down-projection and up-projection

### Step 1.4: PartialRope Forward Pass
**Method**: `PartialRope.forward()` in `transmla/partial_rope.py` (lines 142-192)

**Key Operations**:
1. Projects queries: `q_proj(hidden_states)` → shape `[B, num_heads, seq_len, head_dim]`
2. Projects keys/values: `k_proj(hidden_states)`, `v_proj(hidden_states)` → shape `[B, num_kv_heads, seq_len, head_dim]`
3. Upsamples queries using k_up_proj: `einsum("bthd,hdc->bhtc", query_states, k_up_weight)` → latent space
4. Applies partial RoPE: `apply_rotary_pos_emb(query_states, key_states, cos[::collapse], sin[::collapse])`
5. Upsamples values using v_up_proj
6. Computes attention using SDPA
7. Projects output through o_proj

**Helper Function**: `apply_rotary_pos_emb()` in `transmla/partial_rope.py` (lines 17-45)
- Splits Q/K into rope and nope parts
- Applies rotary embeddings only to rope part
- Uses interleaved rotation format (DeepSeek-style)

### Step 1.5: Replace All Attention Layers
Replaces each `layer.self_attn` with `PartialRope` instance across all transformer layers.

#### Original Qwen3 Self-Attention Components (Pre-Phase 1)

| Component | Weight Shape (Symbolic) | Weight Shape (Numeric) | Parameters |
|-----------|------------------------|------------------------|------------|
| `q_proj` | `[hidden_size, num_attention_heads × head_dim]` | `[3,584, 4,096]` | 14,680,064 |
| `k_proj` | `[hidden_size, num_key_value_heads × head_dim]` | `[3,584, 1,024]` | 3,670,016 |
| `v_proj` | `[hidden_size, num_key_value_heads × head_dim]` | `[3,584, 1,024]` | 3,670,016 |
| `o_proj` | `[hidden_size, hidden_size]` | `[3,584, 3,584]` | 12,845,056 |
| **Total** | | | **34,865,152** |

#### New Qwen3 Self-Attention Components (Post-Phase 1)

| Component | Weight Shape (Symbolic) | Weight Shape (Numeric) | Parameters |
|-----------|------------------------|------------------------|------------|
| `q_proj` | `[hidden_size, num_heads × head_dim]` | `[3,584, 4,096]` | 14,680,064 |
| *`k_proj`| `[hidden_size, latent_dim]` | `[3,584, 1,024]` | 3,670,016 |
| `v_proj` | `[hidden_size, latent_dim]` | `[3,584, 1,024]` | 3,670,016 |
| *`k_up_proj` | `[latent_dim, hidden_size]` | `[1,024, 3,584]` | 3,670,016 |
| *`v_up_proj` | `[latent_dim, hidden_size]` | `[1,024, 3,584]` | 3,670,016 |
| `o_proj` | `[hidden_size, hidden_size]` | `[3,584, 3,584]` | 12,845,056 |
| **Total** | | | **42,205,184** |

#### Weight Preservation vs Modification After Phase 1

**Code Reference**: `transmla/partial_rope.py` lines 64-72

After Phase 1 transformation, the original projection layers are handled as follows:

**Note**:
- Only **`k_proj`** is modified: its weight tensor is rotated in-place using PCA to compress RoPE dimensions. The rotation preserves the mathematical equivalence while enabling partial RoPE application. i.e. `transmla/partial_rope.py` lines 108-127 (`rotate_k_proj()` method)
- The **`k_up_proj`** and **`v_up_proj`** are newly created modules (not modifications of existing ones), initialized as identity matrices expanded to match the GQA structure.


### Step 1.6: Auto-search Optimal Freqfold (Optional)
**Logic**: In `partial_rope()` function (lines 221-250)

If `freqfold == "auto"`:
- Starts with `freqfold = collapse`
- Doubles freqfold until perplexity stops improving
- Keeps best freqfold value

### Step 1.7: Evaluate Partial RoPE Model (Optional)
Evaluates perplexity after RoPE transformation using `evaluate_ppl()`.

---

## Phase 2: Low-Rank QKV Decomposition

### Main Function
**File**: `transmla/lora_qkv.py`  
**Function**: `low_rank_qkv(model, tokenizer, train_loader, test_loader, **kwargs)` (lines 278-312)

### Step 2.1: Calibrate Post-RoPE QKV Outputs
**Function**: `get_qkv_calibrate_outputs()` in `transmla/utils.py` (lines 316-385)

Re-runs calibration on the PartialRope model to capture updated activations.

### Step 2.2: Create LoraQKV Attention Module
**Class**: `LoraQKV` in `transmla/lora_qkv.py` (lines 20-275)

**Initialization** (lines 21-130):
- Validates qk_mqa_dim * collapse == head_dim
- Defines compressed dimensions for query/key/value
- Creates low-rank projection layers:
  - `q_a_proj`: hidden_size → q_lora_rank (if q_lora_rank is not None)
  - `q_b_proj`: q_lora_rank → num_heads * (qk_mqa_dim + head_dim)
  - `kv_a_proj_with_mqa`: hidden_size → (kv_lora_rank + qk_mqa_dim)
  - `kv_b_proj`: kv_lora_rank → num_heads * head_dim * 2
- Applies balance_kv_ratio if specified (line 110-119)
- Computes PCA on query and kv outputs (lines 122-127)
- Initializes weights (line 130)

**Helper Function**: `pca_calc()` in `transmla/utils.py` (lines 388-403)
- Computes covariance matrix from calibration outputs
- Applies damping for stability
- Returns eigenvectors sorted by eigenvalue magnitude

### Step 2.3: Initialize LoraQKV Weights
**Method**: `LoraQKV._init_weights()` in `transmla/lora_qkv.py` (lines 132-217)

**Key Operations**:

1. **Split K/V weights into RoPE/NoRoPE parts** (lines 134-146):
   - `k_a_rope_weight`, `k_a_nope_weight` from k_proj
   - `k_b_rope_weight`, `k_b_nope_weight` from k_up_proj
   - Handles bias if present

2. **Initialize Query Projections** (lines 149-188):
   - If q_lora_rank is set:
     - `q_a_proj.weight = (R_q.T @ q_weight)[:q_lora_rank]`
     - `q_b_weight = R_q[:, :q_lora_rank]`
     - Absorbs k_b_rope_weight: `einsum("hdq,hdk->hkq", q_b_weight, k_b_rope_weight)`
     - Concatenates nope and rope parts: `[q_b_weight, q_b_rope_weight]`
   - Else: directly absorbs rope into q_proj
   - **Scales weights** by `original_scaling / self.scaling` to adjust for new attention dimension

3. **Low-Rank Decompose K/V Projections** (lines 191-217):
   - Concatenates k_nope and v_nope weights vertically
   - Appends bias as extra column if present
   - Creates kv_b_nope_weight with block-diagonal structure
   - Applies PCA: `kv_a_nope_weight = (R_kv.T @ kv_a_nope_weight)[:kv_lora_rank]`
   - Applies PCA: `kv_b_nope_weight = (kv_b_nope_weight @ R_kv)[:, :kv_lora_rank]`
   - Concatenates with rope weights to form final kv_a_proj_with_mqa

### Step 2.4: LoraQKV Forward Pass
**Method**: `LoraQKV.forward()` in `transmla/lora_qkv.py` (lines 220-275)

**Key Operations**:

1. **Query Computation** (lines 233-243):
   ```python
   if q_lora_rank:
       query = q_a_proj(hidden_states)  # → [B, L, q_lora_rank]
       if q_a_layernorm: query = q_a_layernorm(query)
       query = q_b_proj(query)  # → [B, L, num_heads * (head_dim + qk_mqa_dim)]
   else:
       query = q_proj(hidden_states)
   q_nope, q_rope = query.split([head_dim, qk_mqa_dim], dim=-1)
   ```

2. **Key/Value Computation** (lines 246-259):
   ```python
   compressed_kv = kv_a_proj_with_mqa(hidden_states)  # → [B, L, kv_lora_rank + qk_mqa_dim]
   kv_nope, k_rope = compressed_kv.split([kv_lora_rank, qk_mqa_dim], dim=-1)
   
   if kv_a_layernorm: kv_nope = kv_a_layernorm(kv_nope)
   kv_nope = kv_b_proj(kv_nope)  # → [B, L, num_heads * head_dim * 2]
   k_nope, value_states = kv_nope.split([head_dim, head_dim], dim=-1)
   key_states = cat([k_nope, repeat_kv(k_rope, num_heads)], dim=-1)
   ```

3. **RoPE Application** (line 252):
   ```python
   q_rope, k_rope = apply_rotary_pos_emb_interleave(q_rope, k_rope, cos[::collapse], sin[::collapse])
   ```

4. **Attention Computation** (lines 261-273):
   - Concatenates q_nope and q_rope
   - Runs SDPA attention with scaling
   - Projects output through o_proj

**Helper Function**: `repeat_kv()` in `transmla/lora_qkv.py` (lines 11-18)
- Expands key_rope from 1 head to num_heads for broadcasting

### Step 2.5: Replace All Attention Layers
Replaces each `layer.self_attn` with `LoraQKV` instance across all transformer layers.

#### PartialRope Components (Pre-Phase 2)

Before Phase 2 transformation, each `self_attn` layer is a `PartialRope` module with the following components (from Phase 1):

| Component | Weight Shape (Symbolic) | Weight Shape (Numeric) | Parameters |
|-----------|------------------------|------------------------|------------|
| `q_proj` | `[hidden_size, num_heads × head_dim]` | `[3,584, 4,096]` | 14,680,064 |
| `k_proj` | `[hidden_size, latent_dim]` | `[3,584, 1,024]` | 3,670,016 |
| `v_proj` | `[hidden_size, latent_dim]` | `[3,584, 1,024]` | 3,670,016 |
| `k_up_proj` | `[latent_dim, hidden_size]` | `[1,024, 3,584]` | 3,670,016 |
| `v_up_proj` | `[latent_dim, hidden_size]` | `[1,024, 3,584]` | 3,670,016 |
| `o_proj` | `[hidden_size, hidden_size]` | `[3,584, 3,584]` | 12,845,056 |
| **Total** | | | **42,205,184** |

#### LoraQKV Components (Post-Phase 2)

After Phase 2 transformation, each `self_attn` layer is a `LoraQKV` module with the following components:

| Component | Weight Shape (Symbolic) | Weight Shape (Numeric) | Parameters |
|-----------|------------------------|------------------------|------------|
| `q_a_proj` | `[hidden_size, q_lora_rank]` | `[3,584, 512]` | 1,835,008 |
| `q_b_proj` | `[q_lora_rank, num_heads × (qk_mqa_dim + head_dim)]` | `[512, 6,144]` | 3,145,728 |
| `kv_a_proj_with_mqa` | `[hidden_size, kv_lora_rank + qk_mqa_dim]` | `[3,584, 576]` | 2,064,384 |
| `kv_b_proj` | `[kv_lora_rank, num_heads × head_dim × 2]` | `[512, 8,192]` | 4,194,304 |
| `o_proj` | `[hidden_size, hidden_size]` | `[3,584, 3,584]` | 12,845,056 |
| **Total** | | | **24,084,480** |

**Note**: For Qwen3-4B, `q_lora_rank=512` and `kv_lora_rank=512` are used. The `*` indicates components that are transformed from previous phase components.

#### Weight Changes Phase 2

**Phase 2 Transformation Overview**


**1. Query Projection (`q_proj`) `[3,584, 4,096]` → Decomposed into `q_a_proj` `[3,584, 512]` + `q_b_proj` `[512, 6,144]`**
  - **Step 1: Compute PCA matrix**
    - Compute `R_q` from calibration query outputs
  - **Step 2: Create `q_a_proj`**
    - Use top `q_lora_rank=512` PCA components
  - **Step 3: Create `q_b_proj`**
    - Combine PCA basis vectors with absorbed `k_b_rope_weight` from `k_up_proj` via einsum
  - **Step 4: Scale weights**
    - Adjust from `sqrt(head_dim)` to `sqrt(head_dim + qk_mqa_dim)`

**2. Key Projection (`k_proj`) `[3,584, 1,024]` → Absorbed into `kv_a_proj_with_mqa` `[3,584, 576]`**
  - **Step 1: Split into two parts**
    - Rope part: `k_a_rope_weight` (first `qk_mqa_dim=64` dimensions)
    - Nope part: `k_a_nope_weight` (remaining 960 dimensions)
  - **Step 2: Process nope part**
    - Concatenate `k_a_nope_weight` with `v_proj` weights
    - Decompose combined matrix using PCA matrix `R_kv`
  - **Step 3: Build new projection**
    - Preserve rope part (`k_a_rope_weight`)
    - Concatenate preserved rope part with PCA-decomposed nope part

**3. Value Projection (`v_proj`) `[3,584, 1,024]` → Absorbed into `kv_a_proj_with_mqa` `[3,584, 576]`**
  - **Step 1: Concatenate with key nope part**
    - Combine `v_proj` with `k_a_nope_weight` (from `k_proj`) to form combined weight matrix
  - **Step 2: Decompose via PCA**
    - Use `R_kv` computed from calibration key and value outputs
  - **Step 3: Combine with rope part**
    - Combine decomposed result with `k_a_rope_weight` to construct `kv_a_proj_with_mqa`

**4. Key Upsampling Projection (`k_up_proj`) `[1,024, 3,584]` → Absorbed into `q_b_proj` `[512, 6,144]` and `kv_b_proj` `[512, 8,192]`**
  - **Step 1: Split into two parts**
    - Rope part: `k_b_rope_weight` (first `qk_mqa_dim=64` dimensions)
    - Nope part: `k_b_nope_weight` (remaining 960 dimensions)
  - **Step 2: Handle rope part**
    - Absorbed into `q_b_proj` using einsum operation
    - Contributes to query's RoPE component
  - **Step 3: Handle nope part**
    - Combined with `v_b_nope_weight` (from `v_up_proj`) in block-diagonal structure
    - Decomposed via PCA to form `kv_b_proj`

**5. Value Upsampling Projection (`v_up_proj`) `[1,024, 3,584]` → Absorbed into `kv_b_proj` `[512, 8,192]`**
  - **Step 1: Combine with key nope part**
    - Combine `v_up_proj` weights with `k_b_nope_weight` from `k_up_proj`
  - **Step 2: Arrange in block-diagonal structure**
    - Key and value sections arranged diagonally
  - **Step 3: Decompose via PCA**
    - Apply PCA decomposition using matrix `R_kv` to transform combined structure

**6. Output Projection (`o_proj`) `[3,584, 3,584]` → Unchanged**


### Step 2.6: Calibrate for LayerNorm Statistics (Optional)
If `use_qkv_norm == True`:

**Function**: `statistics_qkv_rmsnorm()` in `transmla/utils.py` (lines 405-416)
- Recalibrates model to capture q_a_proj and kv_a_proj outputs
- Computes RMSNorm statistics from calibration data
- Initializes q_a_layernorm and kv_a_layernorm weights

### Step 2.7: Evaluate MLA Model (Optional)
Evaluates final perplexity using `evaluate_ppl()`.

---

## Phase 3: Model Saving and Configuration

### Step 3.1: Save Model and Tokenizer
**Code**: `transmla/converter.py` (lines 101-103)

```python
model.save_pretrained(os.path.join(args.save_path))
tokenizer.save_pretrained(os.path.join(args.save_path))
```

Saves model weights as SafeTensors format and tokenizer files to output directory.

### Step 3.2: Modify Configuration File
**Function**: `modify_config()` in `transmla/modify_config.py` (lines 66-98)

**Key Operations**:

1. **Load config.json** from save_path
2. **Update AutoModel mappings** (lines 42-49):
   ```python
   "auto_map": {
       "AutoConfig": "configuration_qwen3mla.Qwen3MLAConfig",
       "AutoModel": "modeling_qwen3mla.Qwen3MLAModel",
       "AutoModelForCausalLM": "modeling_qwen3mla.Qwen3MLAForCausalLM"
   }
   "architectures": ["Qwen3MLAForCausalLM"]
   ```
3. **Set MLA-specific config parameters** (lines 78-86):
   ```python
   config["model_type"] = "deepseek_v3"
   config["num_key_value_heads"] = config["num_attention_heads"]
   config["attention_bias"] = model.model.layers[0].self_attn.attention_bias
   config["qk_rope_head_dim"] = config["head_dim"] = args.qk_mqa_dim
   config["qk_nope_head_dim"] = config["v_head_dim"] = model.model.layers[0].self_attn.head_dim
   config["q_lora_rank"] = args.q_lora_rank
   config["kv_lora_rank"] = args.kv_lora_rank
   config["qk_latent_layernorm"] = hasattr(model.model.layers[0].self_attn, "kv_a_layernorm")
   ```
4. **Copy custom model files** to save_path (lines 91-98):
   - `configuration_qwen3mla.py`
   - `modeling_qwen3mla.py`
   - `mla.py`

### Step 3.3: Custom Model Files

#### Configuration Class
**File**: `transmla/transformers/qwen3/configuration_qwen3mla.py`  
**Class**: `Qwen3MLAConfig` (lines 4-26)

**Inherits**: `Qwen3Config`  
**Additional Parameters**:
- `kv_lora_rank`: Rank for KV low-rank decomposition
- `q_lora_rank`: Rank for Q low-rank decomposition (optional)
- `qk_rope_head_dim`: Dimension of RoPE part (qk_mqa_dim)
- `qk_nope_head_dim`: Dimension of non-RoPE part (original head_dim)
- `v_head_dim`: Value head dimension (original head_dim)
- `qk_latent_layernorm`: Whether to use RMSNorm on latent projections

#### MLA Attention Class
**File**: `transmla/transformers/mla.py`  
**Class**: `MLAAttention` (lines 22-149)

Implements the final MLA attention module with full DeepSeek V3 compatibility:
- Supports both q_lora_rank and direct q_proj modes
- Implements kv_a_proj_with_mqa and kv_b_proj structure
- Applies optional RMSNorm layers
- Uses interleaved RoPE format
- Compatible with Flash Attention 2, SDPA, and eager attention

#### Model Classes
**File**: `transmla/transformers/qwen3/modeling_qwen3mla.py`

**Classes**:
- `Qwen3MLADecoderLayer` (lines 22-26): Decoder layer with MLAAttention
- `Qwen3MLAPreTrainedModel` (lines 29-32): Base pretrained model class
- `Qwen3MLAModel` (lines 35-42): Main model with MLA layers
- `Qwen3MLAForCausalLM` (lines 45-49): Causal LM wrapper

---

## Final Architecture

### Original Qwen3 Attention
```
hidden_states → q_proj → [num_heads, seq_len, head_dim]
             → k_proj → [num_kv_heads, seq_len, head_dim]
             → v_proj → [num_kv_heads, seq_len, head_dim]
             → RoPE → attention → o_proj
```

### Converted MLA Attention
```
hidden_states → q_a_proj → [q_lora_rank]
             → q_a_layernorm (optional)
             → q_b_proj → [num_heads, qk_nope_head_dim + qk_rope_head_dim]
             → split into q_nope and q_rope

hidden_states → kv_a_proj_with_mqa → [kv_lora_rank + qk_rope_head_dim]
             → split into kv_nope and k_rope
             → kv_a_layernorm(kv_nope) (optional)
             → kv_b_proj → [num_heads, qk_nope_head_dim + v_head_dim]
             → split into k_nope and value_states

q_rope + k_rope → apply_rotary_pos_emb_interleave
q_states = concat(q_nope, q_rope)
k_states = concat(k_nope, repeat_kv(k_rope))
→ attention(q_states, k_states, value_states) → o_proj
```

### Memory Savings
**Original KV cache per token**: `2 * num_kv_heads * head_dim`  
**MLA KV cache per token**: `kv_lora_rank + qk_rope_head_dim`

For Qwen3-4B with default settings:
- Original: 2 * 8 * 128 = 2,048 floats
- MLA: 512 + 64 = 576 floats
- **Reduction**: 72% smaller KV cache

---

## Key Hyperparameters

### For Qwen3-4B (from `scripts/qwen3-4B.sh`)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `qk_mqa_dim` | 64 | Dimension of RoPE part (also called qk_rope_head_dim) |
| `q_lora_rank` | 512 | Rank for query low-rank decomposition |
| `kv_lora_rank` | 512 | Rank for key/value joint low-rank decomposition |
| `freqfold` | 4 | RoPE frequency folding factor (auto-searched) |
| `collapse` | auto | Collapse factor = head_dim / qk_mqa_dim = 128 / 64 = 2 |
| `cal_nsamples` | 128 (default) | Number of calibration samples |
| `cal_batch_size` | 8 (default) | Calibration batch size |
| `cal_max_seqlen` | 256 (default) | Maximum calibration sequence length |

### Architecture Dimensions (Qwen3-4B)
- `hidden_size`: 3,584
- `num_attention_heads`: 32
- `num_key_value_heads`: 8
- `head_dim`: 128 (original)
- `qk_nope_head_dim`: 128 (after conversion)
- `qk_rope_head_dim`: 64 (after conversion)
- `v_head_dim`: 128 (after conversion)
- `qk_head_dim`: 192 (qk_nope_head_dim + qk_rope_head_dim)

---

## Summary of Key Files

### Core Conversion Scripts
- `transmla/converter.py`: Main entry point, orchestrates the conversion
- `transmla/partial_rope.py`: Phase 1 - Partial RoPE transformation
- `transmla/lora_qkv.py`: Phase 2 - Low-rank QKV decomposition
- `transmla/modify_config.py`: Phase 3 - Configuration modification
- `transmla/utils.py`: Helper functions for datasets, hooks, PCA, evaluation

### Output Model Files
- `transmla/transformers/qwen3/configuration_qwen3mla.py`: MLA config class
- `transmla/transformers/qwen3/modeling_qwen3mla.py`: MLA model classes
- `transmla/transformers/mla.py`: Final MLA attention implementation

### Key Classes
- `PartialRope`: Intermediate attention module with compressed RoPE
- `LoraQKV`: Final MLA attention module with low-rank projections
- `MLAAttention`: Production-ready MLA attention for saved model
- `Qwen3MLAConfig`: Configuration class extending Qwen3Config
- `Qwen3MLAForCausalLM`: Complete causal language model with MLA

---

## Usage Example

### Converting the Model
```bash
bash scripts/qwen3-4B.sh
```

### Loading the Converted Model
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "outputs/qwen3-4B-deepseek",
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("outputs/qwen3-4B-deepseek")

# Generate text
inputs = tokenizer("Hello, how are you?", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

### Using with vLLM (for fast inference)
```python
import transmla.vllm_registry.deepseek  # Register MLA models
from vllm import LLM, SamplingParams

llm = LLM(model="outputs/qwen3-4B-deepseek", trust_remote_code=True)
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
outputs = llm.generate("Hello, how are you?", sampling_params)
print(outputs[0].outputs[0].text)
```

---

## References

- **Paper**: [TransMLA: Multi-head Latent Attention Is All You Need](https://arxiv.org/abs/2502.07864)
- **Repository**: [https://github.com/fxmeng/TransMLA](https://github.com/fxmeng/TransMLA)
- **DeepSeek Paper**: [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434)

---

**Last Updated**: November 18, 2025

