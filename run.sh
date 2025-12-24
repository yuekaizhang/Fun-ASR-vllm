export CUDA_VISIBLE_DEVICES="0"

# dataset_name=yuekai/aishell
# subset_name=test
# split_name=test

dataset_name="yuekai/speechio"
subset_name="SPEECHIO_ASR_ZH00007"
split_name="test"

uv run python \
    infer.py \
    --model_dir ./Fun-ASR-Nano-2512 \
    --huggingface_dataset $dataset_name \
    --subset_name $subset_name \
    --split_name $split_name \
    --batch_size 1 \
    --log_dir ./logs_vllm_test2_$dataset_name_$subset_name \
    --vllm_model_dir ./new_llm
