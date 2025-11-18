# 🚀 TransMLA: Migrating GQA Models to MLA with Full DeepSeek Compatibility and Speedup

Modern large-language models often face communication bottlenecks on current hardware rather than computational limitations. Multi-head latent attention (MLA) addresses this by compressing the key-value cache using low-rank matrices, while the Absorb operation prevents the KV cache from reverting to its original size, significantly boosting both training and inference speed. 

Despite the success of DeepSeek V2/V3/R1, most model vendors have heavily invested in optimizing GQA-based models and therefore lack strong incentives to retrain MLA-based models from scratch. In this paper, we introduce TransMLA, a framework that seamlessly converts any GQA-based pre-trained model (e.g., LLaMA, Qwen, Mixtral) into an MLA-based model. 


# 📰 News
- [2025.05.29] A new version of technical report is released: [https://arxiv.org/abs/2502.07864](https://arxiv.org/abs/2502.07864).
- [2025.04.28] Released TransMLA v3, successfully apply PCA across RoPE and reduce KV Cache.
- [2025.02.16] Released the second version of the TransMLA model and usage code, compatible with RoPE and supporting Absorb operation.
- [2025.02.13] The technical report of TransMLA is publicly available: [https://huggingface.co/papers/2502.07864](https://huggingface.co/papers/2502.07864)
- [2025.01.02] Released the first version of the TransMLA model code, providing usage code for converting Qwen2.5 and LLaMA-3’s GQA to MLA equivalence.

# 🛠 Installation
```
git clone https://github.com/fxmeng/TransMLA.git
cd TransMLA
conda create -n transmla python=3.12.8
conda activate transmla
pip install -r requirements.txt
```

# ⚡ Quick Start

1. Convert MHA / GQA models (e.g. Qwen2.5-7B-Instruct) into DeepSeek-MLA:
    ```bash
    bash scripts/convert/qwen2.5-7B-Instruct.sh
    ```
2. Have fun playing with the converted models!
    ```python
    # using `Transformers.AutoModelForCausalLM`
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("outputs/qwen2_5-7B-Instruct-deepseek", trust_remote_code=True)

    # using `vllm.LLM`
    # note that only Llama-type models(llama, qwen, mistral) are supported right now
    import transmla.vllm_registry.deepseek      # register mla models
    from vllm import LLM, SamplingParams
    llm = LLM(model="outputs/qwen2_5-7B-Instruct-deepseek", trust_remote_code=True)
    ```

## 🔧 Advanced Usage (`converter.py`)

The converter.py script allows you to perform fine-grained control over RoPE removal and low-rank QKV projection towards DeepSeek-MLA. It supports:
- Auto-search for optimal freqfold that minimizes PPL.
- Automatic computation of collapse based on head_dim / qk_mqa_dim.
- Evaluation of original, RoPE-removed, and final MLA models.


### ✅ Example Command:
```bash
python transmla/converter.py \
    --model-path meta-llama/Llama-2-7b-hf \
    --save-path ./outputs/llama2-7b-deepseek \
    --dtype bf16 \
    --device auto \
    --cal-dataset wikitext2 \
    --cal-nsamples 128 \
    --cal-max-seqlen 256 \
    --cal-batch-size 8 \
    --ppl-eval-batch-size 4 \
    --freqfold auto \
    --collapse auto \
    --qk-mqa-dim 64 \
    --q-lora-rank 512 \
    --kv-lora-rank 512
```

### 📘 Argument Details

| Argument | Description |
|----------|-------------|
| --model-path | Path to the base model (e.g., from HuggingFace hub). |
| --save-path | Output path for the converted model and tokenizer. |
| --cal-dataset | Calibration dataset: wikitext2, ptb, c4, or alpaca. |
| --cal-nsamples, --cal-max-seqlen, --cal-batch-size | Number, max sequence length, and batch size of samples used for calibration. |
| --freqfold | RoPE frequency folding factor, or `auto` to search for the best value. Note: Automatic freqfold search is only supported in single-GPU setups currently. Please set the device explicitly, for example: `cuda:0`. |
| --collapse | Collapse factor for RoPE. Use `auto` to compute as `head_dim // qk_mqa_dim`. Collapse factor reduces the dim of RoPEd KV cache from `head_dim` to `head_dim // collapse`. |
| --qk-mqa-dim | Target dimension for decoupled RoPE. |
| --q-lora-rank | The inner dimension for query low-rank decomposition, or `None` to disable low-rank decomposition for query. |
| --kv-lora-rank | The inner dimension for key/value joint low-rank decomposition. |
| --deepseek-style | Use deepseek style modeling / configuration files from transformers. Only support Llama-type models(llama, qwen, mistral)


### 🧠 Tips
- Set `--freqfold auto` and `--collapse auto` to simplify configuration. The script will automatically search for the best freqfold factor based on ppl results.
- We recommend setting `--qk-mqa-dim` to 64 and `--kv-lora-rank` to 512 to satisfy FlashMLA's requirements on H100.

## 🔍 Conversion Call Stack (Qwen2.5 Example)

This section details the complete call stack for converting a Qwen2.5 GQA model to MLA, step by step.

### Entry Point
```bash
python transmla/converter.py --model-path Qwen/Qwen2.5-7B-Instruct ...
```

### Phase 1: Initialization (`converter.py`)

1. **`main(args)`** - Main entry function
   - Parses command-line arguments
   - Orchestrates the three-phase conversion process

2. **`load_model_and_tokenizer(args)`** - Load original Qwen2.5 model
   - `AutoModelForCausalLM.from_pretrained()` - Loads Qwen2.5 with GQA attention
   - `AutoTokenizer.from_pretrained()` - Loads tokenizer
   - Validates `model_type == "qwen2"` for compatibility

3. **`get_dataset_loader(tokenizer, **kwargs)`** - Prepare calibration data
   - `get_dataset(kwargs["cal_dataset"])` - Loads dataset (e.g., wikitext2)
   - `prepare_dataloader()` - Creates training DataLoader for calibration
   - `prepare_test_dataloader()` - Creates test DataLoader for PPL evaluation

4. **`evaluate_ppl(model, tokenizer.pad_token_id, test_loader, message)`** - Baseline evaluation
   - Evaluates original GQA model perplexity

### Phase 2: Partial RoPE Conversion (`partial_rope.py`)

5. **`partial_rope(model, tokenizer, train_loader, test_loader, **kwargs)`** - Main conversion function
   - **Auto collapse calculation** (if `collapse == "auto"`):
     - Computes `head_dim = hidden_size // num_attention_heads`
     - Sets `collapse = head_dim // qk_mqa_dim`
   
   - **Calibration for original model**:
     - `get_qkv_calibrate_outputs(model, train_loader, message)` - Captures QKV activations
       - `insert_qkv_hooks(model)` - Registers forward hooks on `q_proj`, `k_proj`, `v_proj`
         - Creates hook functions: `query_hook_fn()`, `key_hook_fn()`, `value_hook_fn()`
         - Registers hooks via `register_forward_hook()` on each layer
       - Runs forward passes through calibration data
       - Collects outputs in dictionaries: `query_outputs`, `key_outputs`, `value_outputs`
   
   - **Freqfold search** (if `freqfold == "auto"`):
     - Iteratively tests `freqfold` values: `collapse, collapse*2, collapse*4, ...`
     - For each freqfold, calls `partial_rope_freqfold()` and evaluates PPL
     - Selects freqfold with minimum PPL
     
   - **How freqfold works**
     - `freqfold` controls how aggressively the RoPE (rotary position embedding) frequencies are *folded* (downsampled) before we rotate the key/value projections.
     - In `PartialRope.__init__`, after PCA, we reshape `k_proj` weights into blocks of `[num_kv_heads, head_dim // freqfold, freqfold // collapse, collapse, ...]` and apply `rotate_k_proj()` so that every `freqfold` contiguous rotary channels share the same learned frequency basis.
     - A larger `freqfold` → fewer distinct RoPE frequency bands → smaller effective RoPE dimension (since we keep just `head_dim // freqfold` unique bands). This reduces KV cache size but increases approximation error.
     - When `freqfold` is smaller (closer to `collapse`), more frequency bands are preserved, so the converted model stays closer to the original but retains more KV cache.
     - During auto-search, we start from `freqfold = collapse` (minimum folding) and keep doubling until perplexity stops improving. This balances KV compression with accuracy.
   
   - **Partial RoPE transformation**:
     - `partial_rope_freqfold(model, ori_qkv_outputs, test_loader, freqfold, collapse)`
       - For each layer: `setattr(layer, "self_attn", PartialRope(...))`

6. **`PartialRope.__init__(self_attn, key_outputs, freqfold, collapse)`** - Initialize PartialRoPE attention
   - Copies original attention config and projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`)
   - `_insert_kv_up_proj()` - Creates `k_up_proj` and `v_up_proj` for GQA expansion
   - **PCA-based RoPE compression**:
     - `joint_complex_pca(key_outputs, freqfold)` - Computes PCA on key outputs
       - For each batch: `pca_calc(X_batch, device)`
         - Builds covariance matrix: `H = Σ(X_batch^T @ X_batch)`
         - Adds diagonal damping for numerical stability
         - `torch.linalg.eigh(H)` - Eigendecomposition
         - Returns eigenvectors sorted by descending eigenvalues
       - Combines complex PCA results for RoPE dimension reduction
     - `rotate_k_proj(Rk, freqfold)` - Transforms `k_proj` weights using PCA results
     - `rotate_k_up_proj(U, freqfold)` - Transforms `k_up_proj` weights

7. **`PartialRope.forward()`** - Forward pass with partial RoPE
   - Projects QKV: `q_proj()`, `k_proj()`, `v_proj()`
   - Expands KV via `k_up_proj` and `v_up_proj` for multi-head attention
   - `apply_rotary_pos_emb()` - Applies RoPE with frequency folding (`freqfold`) and collapse
   - Standard attention computation with compressed KV cache

### Phase 3: Low-Rank QKV to MLA (`lora_qkv.py`)

8. **`low_rank_qkv(model, tokenizer, train_loader, test_loader, **kwargs)`** - Convert to MLA
   - **Calibration for PartialRoPE model**:
     - `get_qkv_calibrate_outputs(model, train_loader, message)` - Captures activations from PartialRoPE model
   
   - **Replace attention with LoraQKV**:
     - For each layer: `setattr(layer, "self_attn", LoraQKV(...))`

9. **`LoraQKV.__init__(self_attn, query_outputs, key_outputs, value_outputs, ...)`** - Initialize MLA attention
   - Extracts config from original attention module
   - **PCA for low-rank decomposition**:
     - `pca_calc(query_outputs[layer_idx], device)` - Computes PCA on query activations → `R_q`
     - `pca_calc(key_outputs[layer_idx] + value_outputs[layer_idx], device)` - Computes PCA on KV activations → `R_kv`
   
   - **Module creation**:
     - Creates `q_a_proj`, `q_b_proj` (if `q_lora_rank` specified) or `q_proj` (otherwise)
     - Creates `kv_a_proj_with_mqa` - Joint projection for compressed KV + MQA dimension
     - Creates `kv_b_proj` - Expands compressed KV back to full dimension
     - Creates `q_a_layernorm`, `kv_a_layernorm` (if `use_qkv_norm=True`)
   
   - **Weight initialization**:
     - `_init_weights(self_attn, R_q, R_kv)` - Transforms weights from PartialRoPE to MLA
       - **Splits K/V weights** into RoPE and non-RoPE parts
       - **Query projection**:
         - If `q_lora_rank` specified: `q_a_proj = R_q^T @ q_proj`, `q_b_proj = R_q`
         - Absorbs RoPE part of K into Q: `q_b_rope = einsum(q_b, k_b_rope)`
         - Concatenates: `q_b_with_mqa = [q_b, q_b_rope]`
       - **KV projection**:
         - Concatenates K and V non-RoPE parts: `kv_a_nope = [k_a_nope, v_a_nope]`
         - Low-rank decomposition: `kv_a_nope = R_kv^T @ kv_a_nope`, `kv_b_nope = kv_b_nope @ R_kv`
         - Combines with RoPE part: `kv_a_proj_with_mqa = [kv_a_nope, k_a_rope]`
       - Applies scaling adjustment for attention scores

10. **Optional: QKV Normalization** (if `use_qkv_norm=True`):
    - `get_qkv_calibrate_outputs(model, train_loader)` - Captures activations from LoraQKV
    - For each layer: `statistics_qkv_rmsnorm(layer.self_attn, q_a_outputs, kv_a_outputs)`
      - Computes RMS normalization statistics from calibration outputs
      - Sets `q_a_layernorm.weight` and `kv_a_layernorm.weight` to computed values

11. **`LoraQKV.forward()`** - Forward pass with MLA
    - Projects hidden states: `q_a_proj()` → (optional `q_a_layernorm()`) → `q_b_proj()`
    - Projects KV: `kv_a_proj_with_mqa()` → splits into `k_pass` and `k_rot`
    - Applies RoPE to `k_rot` and expands to all heads
    - (Optional) `kv_a_layernorm()` on `k_pass` → `kv_b_proj()`
    - Standard attention computation with compressed KV cache

### Phase 4: Model Saving (`converter.py` + `modify_config.py`)

12. **`model.save_pretrained(save_path)`** - Save converted model
    - Saves model weights and tokenizer

13. **`modify_config(model, config_path, args)`** - Update configuration
    - Updates `config.json` with MLA-specific settings:
      - Sets `model_type = "deepseek_v3"` (or keeps original if not deepseek-style)
      - Updates `auto_map` to point to MLA model classes
      - Sets `num_key_value_heads = num_attention_heads`
      - Sets `qk_rope_head_dim = qk_mqa_dim`
      - Sets `qk_nope_head_dim = v_head_dim = original_head_dim`
      - Sets `q_lora_rank` and `kv_lora_rank`
      - Sets `qk_latent_layernorm` flag
    - Copies MLA modeling files to save path:
      - `transmla/transformers/qwen3/*.py` → `save_path/`
      - `transmla/transformers/mla.py` → `save_path/`

### Summary Flow

```
Qwen2.5 GQA Model
    ↓
[Phase 1: Calibration] get_qkv_calibrate_outputs() → Capture QKV activations
    ↓
[Phase 2: Partial RoPE] PartialRope.__init__()
    ├─ joint_complex_pca() → PCA on key outputs
    ├─ rotate_k_proj() → Transform k_proj weights
    └─ rotate_k_up_proj() → Transform k_up_proj weights
    ↓
PartialRoPE Model (RoPE compressed, GQA structure maintained)
    ↓
[Phase 3: Low-Rank QKV] LoraQKV.__init__()
    ├─ pca_calc() on query → R_q
    ├─ pca_calc() on KV → R_kv
    ├─ _init_weights() → Transform weights to MLA structure
    │  ├─ Split K/V into RoPE/non-RoPE parts
    │  ├─ Low-rank decompose Q: q_a_proj, q_b_proj
    │  ├─ Absorb K RoPE into Q
    │  └─ Low-rank decompose KV: kv_a_proj_with_mqa, kv_b_proj
    └─ (Optional) statistics_qkv_rmsnorm() → Initialize layer norms
    ↓
MLA Model (DeepSeek-compatible)
    ↓
[Phase 4: Save] save_pretrained() + modify_config()
    ↓
Converted Qwen2.5 MLA Model
```

# 🐒 Model Zoo

| Model Family | Model | kv-lora-rank + qk-mqa-dim | freqfold | Original ppl | Partial RoPE ppl | MLA ppl |
| - | - | - | - | - | - | - |
| Llama2    | Llama-2-7B            | 512 + 64   | 8 | 5.4732 | 18.6373 | 41.6135 |
|           |                       | 448 + 128  | 8 |        | 8.9903  | 25.7731 |
| Llama3    | Llama-3-8B            | 512 + 64   | 4 | 6.1371 | 12.0550 | 25.8047 |
|           |                       | 448 + 128  | 4 |        | 8.3997  | 18.3500 |
|           | Llama-3.2-1B          | 512 + 64   | 4 | 9.7531 | 16.3391 | 16.1404 |
| Qwen2     | Qwen2.5-7B            | 512 + 64   | 4 | 6.8480 | 7.8448  | 8.4124  |
|           |                       | 448 + 128  | 4 |        | 7.3059  | 7.9812  |
|           | Qwen2.5-7B-Instruct   | 512 + 64   | 4 | 7.4570 | 8.8902  | 10.0082 |
|           |                       | 448 + 128  | 4 |        | 8.0734  | 9.1957  |
|           | Qwen2.5-72B-Instruct  | 512 + 64   | 4 | 4.2687 | 4.9650  | 7.3850  |
|           |                       | 448 + 128  | 4 |        | 4.6931  | 7.7172  |
| Gemma2    | gemma-2-9b-it         | 512 + 64   | 8 | 10.1612| 11.4207 | 21.6260 |
|           |                       | 448 + 128  | 8 |        | 10.9948 | 22.0038 |
|           |                       | 320 + 256  | 4 |        | 10.7075 | 32.4387 |
| Mistral   | Mistral-7B-v0.3       | 512 + 64   | 8 | 5.3178 | 7.4697  | 9.5830  |
|           |                       | 448 + 128  | 8 |        | 5.5915  | 7.0251  |
| Mixtral   | Mixtral-8x7B-v0.1     | 512 + 64   | 8 | 3.8422 | 5.6310  | 7.5179  |
|           |                       | 448 + 128  | 4 |        | 4.1407  | 5.8374  |
| MiMo      | MiMo-7B-Base          | 512 + 64   | 4 | 6.9108 | 7.9272  | 9.5810  |


# 📋 To-Do
- [x] Publish the technical report for the new version, detailing how TransMLA is compatible with RoPE, supports the Absorb operation.
- [x] Compress the dimensions of the KV cache to improve inference speed.
- [x] Add support for vLLM to improve inference speed.
- [x] Support FlashMLA.
- [x] Extend support to additional models (e.g., LLaMA, Mistral, Gemma2, etc.).
- [ ] Support GTA & GLA
- [ ] Release checkpoints.
- [ ] Fine-tune on R1 distillation datasets.


# 📚 Citation
```
@article{meng2025transmla,
  title={TransMLA: Multi-head Latent Attention Is All You Need},
  author={Meng, Fanxu and Tang, Pingzhi and Yao, Zengwei and Zhang, Muhan},
  journal={arXiv preprint arXiv:2502.07864},
  year={2025}
}
```

# ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=fxmeng/TransMLA&type=Date)](https://www.star-history.com/#fxmeng/TransMLA&Date)
