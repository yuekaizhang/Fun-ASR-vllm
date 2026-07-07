# Fun-ASR vLLM Acceleration

This repository provides an accelerated implementation of [Fun-ASR](https://github.com/FunAudioLLM/Fun-ASR) using [vLLM](https://github.com/vllm-project/vllm). By leveraging vLLM's efficient attention mechanisms and memory management, this project significantly boosts the inference performance of Fun-ASR models while maintaining accuracy.

## Environment Setup 🐍

To get started, clone the repository and install the required dependencies:

```shell
git clone https://github.com/yuekaizhang/Fun-ASR-vllm.git
cd Fun-ASR-vllm
apt-get install -y ffmpeg
uv pip install -r requirements.txt
```

<a name="usage-tutorial"></a>

## Features 📝
- Support vLLM
- Support batch > 1 Inference
- Support [FunAudioLLM/Fun-ASR-MLT-Nano-2512](https://huggingface.co/FunAudioLLM/Fun-ASR-MLT-Nano-2512) and [FunAudioLLM/Fun-ASR-Nano-2512](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512)
- Integration with [NVIDIA Triton Inference Server](./triton_server/)

## Usage 🛠️

### Python API Inference

You can run inference directly using the Python API:

```python
from model import FunASRNano
from vllm import LLM, SamplingParams

def main():
    model_dir = "FunAudioLLM/Fun-ASR-Nano-2512"
    # Load the base model
    m, kwargs = FunASRNano.from_pretrained(model=model_dir, device="cuda:0")
    m.eval()
    
    # Initialize vLLM
    vllm = LLM(model="yuekai/Fun-ASR-Nano-2512-vllm", enable_prompt_embeds=True, gpu_memory_utilization=0.4)
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=500,
        skip_special_tokens=True,
    )
    
    # Attach vLLM to the model
    m.vllm = vllm
    m.vllm_sampling_params = sampling_params

    # Run inference
    wav_path = f"{kwargs['model_path']}/example/zh.mp3"
    res = m.inference(data_in=[wav_path], **kwargs)
    print(res)
    text = res[0][0]["text"]
    print(text)


if __name__ == "__main__":
    main()
```


For multilingual [FunAudioLLM/Fun-ASR-MLT-Nano-2512](http://huggingface.co/FunAudioLLM/Fun-ASR-MLT-Nano-2512):

```python
from model import FunASRNano
from vllm import LLM, SamplingParams

def main():
    model_dir = "FunAudioLLM/Fun-ASR-MLT-Nano-2512"
    # Load the base model
    m, kwargs = FunASRNano.from_pretrained(model=model_dir, device="cuda:0")
    m.eval()
    
    # Initialize vLLM
    vllm = LLM(model="yuekai/Fun-ASR-MLT-Nano-2512-vllm", enable_prompt_embeds=True, gpu_memory_utilization=0.4)
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=500,
        skip_special_tokens=True,
    )
    
    # Attach vLLM to the model
    m.vllm = vllm
    m.vllm_sampling_params = sampling_params

    # Run inference
    wav_path = f"{kwargs['model_path']}/example/en.mp3"
    # 中文、英文、日文 for Fun-ASR-Nano-2512
    # 中文、英文、粤语、日文、韩文、越南语、印尼语、泰语、马来语、菲律宾语、阿拉伯语、
    # 印地语、保加利亚语、克罗地亚语、捷克语、丹麦语、荷兰语、爱沙尼亚语、芬兰语、希腊语、
    # 匈牙利语、爱尔兰语、拉脱维亚语、立陶宛语、马耳他语、波兰语、葡萄牙语、罗马尼亚语、
    # 斯洛伐克语、斯洛文尼亚语、瑞典语 for Fun-ASR-MLT-Nano-2512
    res = m.inference(data_in=[wav_path], language="英文", **kwargs)
    print(res)
    text = res[0][0]["text"]
    print(text)


if __name__ == "__main__":
    main()
```

### Running Benchmarks

To evaluate performance on a dataset (e.g., SpeechIO):

```bash
dataset_name="yuekai/speechio"
subset_name="SPEECHIO_ASR_ZH00007"
split_name="test"

uv run python \
    infer.py \
    --model_dir FunAudioLLM/Fun-ASR-Nano-2512 \
    --huggingface_dataset $dataset_name \
    --subset_name $subset_name \
    --split_name $split_name \
    --batch_size 16 \
    --log_dir ./logs_vllm_$dataset_name_$subset_name \
    --vllm_model_dir yuekai/Fun-ASR-Nano-2512-vllm
```

To evaluate multilingual performance of FunAudioLLM/Fun-ASR-MLT-Nano-2512:

```bash
dataset_name="google/fleurs"
subset_name="en_us"
split_name="test"

uv run python \
    infer.py \
    --model_dir FunAudioLLM/Fun-ASR-MLT-Nano-2512 \
    --huggingface_dataset $dataset_name \
    --subset_name $subset_name \
    --split_name $split_name \
    --batch_size 16 \
    --language "英文" \
    --log_dir ./logs_mlt_${batch_size}_${dataset_name}_${subset_name} \
    --vllm_model_dir yuekai/Fun-ASR-MLT-Nano-2512-vllm
```

## Performance 🚀

We compared the performance of the standard HuggingFace PyTorch implementation against our vLLM-accelerated version.

**Benchmark Details:**
- **Dataset:** [SPEECHIO_ASR_ZH00007](https://github.com/SpeechColab/Leaderboard) (770 utterances, 3536.26 s of audio)
- **Hardware:** Single NVIDIA H100 80GB GPU

| Mode | Decoding Time | RTF | RTFx | CER | Note |
|------|---------------|-----|------|-----|------|
| Huggingface PyTorch | 238.00 Secs | 0.0673 | 14.86 | 7.09% | batch_size=1 |
| Huggingface PyTorch | 44.04 Secs | 0.0125 | 80.30 | 8.40% | batch_size=16 |
| vLLM (Qwen3-0.6B) | 56.29 Secs | 0.0159 | 62.82 | 7.03% | batch_size=1 |
| **vLLM (Qwen3-0.6B)** | **10.93 Secs** | **0.0031** | **323.54** | **7.01%** | batch_size=16 |

*Note: RTF (Real Time Factor) - lower is better; RTFx (Speedup factor) - higher is better.*

### Bug-Fix Comparison (vLLM, batch_size=16, H100)

Numbers above reflect the speech-embedding length fix (each sample is now truncated to its exact per-sample token count instead of the adaptor's batch-padded length). Before vs. after the fix on the same hardware and config:

| Version | Decoding Time | RTF | RTFx | CER |
|---------|---------------|-----|------|-----|
| Before fix | 21.01 Secs | 0.0059 | 168.31 | 7.05% |
| **After fix**  | **10.93 Secs** | **0.0031** | **323.54** | **7.01%** |

Roughly 2× throughput at the same accuracy, since the padded portion of each prompt no longer hits prefill.

## Triton Inference Server Deployment 🚀

For production deployment with high concurrency, we provide integration with NVIDIA Triton Inference Server.

### Quick Start

```bash
cd triton_server

# Using Docker Compose (recommended)
docker compose up

# Or pull and run the pre-built image
docker pull soar97/triton-fun-asr:25.06
```

### Triton Performance
- **Dataset:** [SPEECHIO_ASR_ZH00007](https://github.com/SpeechColab/Leaderboard) (approx. 1 hour of audio)
- **Hardware:** Single NVIDIA H20 GPU

| Concurrency | CER | Processing Time | P50 Latency | RTF |
|-------------|-----|-----------------|-------------|-----|
| 8 | 7.04% | 44.56s | 450.99ms | 0.0126 |
| 16 | 7.00% | 27.96s | 533.36ms | 0.0079 |
| 32 | 7.07% | 24.51s | 952.93ms | 0.0069 |

For detailed setup instructions, see [triton_server/README.md](./triton_server/README.md).
