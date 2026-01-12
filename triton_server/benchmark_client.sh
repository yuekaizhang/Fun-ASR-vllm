stage=4
stop_stage=4
if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    echo "Running Triton client"
    if [ ! -d "Triton-ASR-Client" ]; then
        git clone https://github.com/yuekaizhang/Triton-ASR-Client.git
    fi
    num_task=32
    dataset_name=yuekai/aishell
    subset_name=test
    split_name=test


    dataset_name="yuekai/speechio"
    subset_name="SPEECHIO_ASR_ZH00007"
    split_name="test"

    python3 Triton-ASR-Client/client.py \
        --server-addr localhost \
        --model-name funasr \
        --num-tasks $num_task \
        --log-dir ./log_fireredasr_$num_task \
        --huggingface_dataset $dataset_name \
        --subset_name $subset_name \
        --split_name $split_name \
        --enable_text_normalization \
        --compute-cer
fi