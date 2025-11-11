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
