export CUDA_VISIBLE_DEVICES="0"

# apt-get -y install ffmpeg

# dataset_name=yuekai/aishell
# subset_name=test
# split_name=test

dataset_name="yuekai/speechio"
subset_name="SPEECHIO_ASR_ZH00007"
split_name="test"


batch_size=16


uv run python \
    infer_batch.py \
    --model_dir ./Fun-ASR-Nano-2512 \
    --huggingface_dataset $dataset_name \
    --subset_name $subset_name \
    --split_name $split_name \
    --batch_size $batch_size \
    --encoder_trt_engine ./model.fp16.plan \
    --log_dir ./logs_rerun_encoder_fp16_trt_vllm_llm_cpu_fbank_dither_1.0_$batch_size \
    --vllm_model_dir ./new_llm
    # --enable_profiler --profiler_output_dir ./profiler_gpu_fbank --use_gpu_fbank


# uv run nsys profile -t cuda,nvtx -o profile_encoder_fp16_trt_vllm_gpu_fbank python \
#     infer_batch.py \
#     --model_dir ./Fun-ASR-Nano-2512 \
#     --huggingface_dataset $dataset_name \
#     --subset_name $subset_name \
#     --split_name $split_name \
#     --batch_size $batch_size \
#     --encoder_trt_engine ./model.fp16.plan \
#     --log_dir ./logs_nvtx_encoder_fp32_torch_vllm_llm_$batch_size \
#     --vllm_model_dir ./new_llm --enable_nvtx --use_gpu_fbank