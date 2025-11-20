## Qwen3 Head-wise QK Norm Fix

- **Root cause**  
  Qwen3 applies per-head RMSNorm (`q_norm`/`k_norm`) after the `q_proj`/`k_proj` layers. The original TransMLA pipeline discarded those modules during partial RoPE and low-rank QKV stages, so the calibration data and the final MLA attention operated on unnormalized Q/K activations. This broke the expected scale of dot products, which explains the multi-order-of-magnitude perplexity regression observed after conversion.

- **What changed**  
  - Preserve and reuse the original `q_norm` / `k_norm` modules in `PartialRope`, `LoraQKV`, and the exported `MLAAttention`.  
  - Apply those norms to the head-wise (non-RoPE) components before rotary embedding so the converted model matches Qwen3’s behavior.  
  - Teach `get_qkv_calibrate_outputs` to tap post-norm activations when present and flatten them so PCA still works.  
  - Surface a `use_qk_head_norm` flag in `Qwen3MLAConfig`/`config.json` so downstream loads know to instantiate the extra norms.

- **How to verify locally**
  1. Install deps: `pip install -r requirements.txt`.
  2. Run conversion (example for 8B, requires a CUDA GPU ≥ 48 GB):  
     ```bash
     python3 transmla/converter.py \
       --model-path Qwen/Qwen3-8B \
       --save-path outputs/qwen3-8B-mla \
       --dtype bf16 --device cuda \
       --ppl-eval-batch-size 8 \
       --freqfold auto --collapse auto \
       --qk-mqa-dim 64 --q-lora-rank 512 \
       --kv-lora-rank 512 --use-qkv-norm \
       --use-original-norm-weights
     ```
  3. Compare the reported perplexities before/after conversion—they should now stay within the expected small delta (no more catastrophic blow-up).

- **Current limitation**  
  This environment has no CUDA devices (`torch.cuda.is_available() == False`), so the end-to-end Qwen3-8B conversion/eval couldn’t be executed here. The commands above can be run on a GPU machine to reproduce the fix.

- **PR checklist**  
  - [x] Code changes validated via `python3 -m compileall transmla`.  
  - [ ] Full Qwen3-8B conversion run (blocked by hardware; see limitation above).
