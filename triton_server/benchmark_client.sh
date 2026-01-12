
echo "Running Triton client"
if [ ! -d "Triton-ASR-Client" ]; then
    git clone https://github.com/yuekaizhang/Triton-ASR-Client.git
fi
num_task=16
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
    --log-dir ./log_funasr_$num_task \
    --huggingface_dataset $dataset_name \
    --subset_name $subset_name \
    --split_name $split_name \
    --enable_text_normalization \
    --compute-cer
