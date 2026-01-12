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
- Integration with [NVIDIA Triton Inference Server](./triton_server/)
- [ ] Support sensevoice encoder acceleration

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
        top_p=0.001,
        max_tokens=500,
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

## Performance 🚀

We compared the performance of the standard HuggingFace PyTorch implementation against our vLLM-accelerated version.

**Benchmark Details:**
- **Dataset:** [SPEECHIO_ASR_ZH00007](https://github.com/SpeechColab/Leaderboard) (approx. 1 hour of audio)
- **Hardware:** Single NVIDIA H20 GPU

| Mode | Decoding Time | RTF | RTFx | CER | Note |
|------|---------------|-----|------|-----|------|
| Huggingface PyTorch | 218.2 Secs | 0.06 | 16.5 | 7.02% | batch_size=1 |
| Huggingface PyTorch | 45.4 Secs | 0.013 | 79.3 | 8.53% | batch_size=16 |
| vLLM (Qwen3-0.6B) | 145.6 Secs | 0.04 | 24.7 | 6.99% | batch_size=1 |
| **vLLM (Qwen3-0.6B)** | **26.3 Secs** | **0.007** | **136.9** | **7.03%** | batch_size=16 |

*Note: RTF (Real Time Factor) - lower is better; RTFx (Speedup factor) - higher is better.*

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
