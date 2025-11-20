model_path=Qwen/Qwen3-0.6B
save_path=outputs/qwen3-0.6B-test
eval_batch_size=8

python3 transmla/converter.py \
    --model-path $model_path \
    --save-path $save_path \
    --dtype bf16 \
    --device cuda \
    --ppl-eval-batch-size $eval_batch_size \
    --freqfold 2 \
    --collapse 2 \
    --qk-mqa-dim 64 \
    --q-lora-rank 32 \
    --kv-lora-rank 32 \
    --use-qkv-norm \
    --use-original-norm-weights

