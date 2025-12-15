# Fun-ASR

「简体中文」|「[English](./README_en.md)」

Fun-ASR 是通义实验室推出的端到端语音识别大模型，是基于数千万小时真实语音数据训练而成，具备强大的上下文理解能力与行业适应性，支持低延迟实时听写，并且覆盖 31 个语种。在教育、金融等垂直领域表现出色，能准确识别专业术语与行业表达，有效应对"幻觉"生成和语种混淆等挑战，实现"听得清、懂其意、写得准"。

<div align="center">
<img src="image/funasr-v2.png">
</div>

<div align="center">
<h4>
<a href="https://funaudiollm.github.io/Fun-ASR/"> Homepage </a>
｜<a href="#核心特性"> 核心特性 </a>
｜<a href="#性能评测"> 性能评测 </a>
｜<a href="#环境安装"> 环境安装 </a>
｜<a href="#用法教程"> 用法教程 </a>

</h4>

模型仓库：[modelscope](https://www.modelscope.cn/models/FunAudioLLM/Fun-ASR-Nano-2512)，[huggingface(coming)](https://huggingface.co/FunAudioLLM/FunASR)

在线体验：
[魔搭社区创空间](https://modelscope.cn/studios/FunAudioLLM/Fun-ASR-Nano/)，[huggingface space(coming)](https://huggingface.co/spaces/FunAudioLLM/FunASR)

</div>

# 核心特性 🎯

**Fun-ASR** 专注于高精度语音识别、多语言支持和行业定制化能力

- **远场高噪声识别：** 针对远距离拾音及高噪声场景（如会议室、车载环境、工业现场等）进行深度优化，识别准确率提升至 **93%**。
- **中文方言与地方口音：**
  - 支持 **7 大方言**：吴语、粤语、闽语、客家话、赣语、湘语、晋语
  - 覆盖 **26 个地区口音**：包括河南、陕西、湖北、四川、重庆、云南、贵州、广东、广西等 20 多个地区
- **多语言自由说：** 支持 **31 种语言**识别，重点优化东亚与东南亚语种，支持语种自由切换和混合识别。
- **音乐背景歌词识别：** 强化在音乐背景干扰下的语音识别性能，支持对歌曲中歌词内容的精准识别。

# 环境安装 🐍

```shell
pip install -r requirements.txt
```

<a name="用法教程"></a>

# TODO

[ ] 支持返回时间戳

[ ] 支持区分说话人识别

[ ] 支持模型训练

# 用法 🛠️

## 推理

### 使用 funasr 推理

```python
from funasr import AutoModel


def main():
    model_dir = "FunAudioLLM/fun-asr-nano"
    model = AutoModel(
        model=model_dir,
        trust_remote_code=True,
        remote_code="./model.py",
        device="cuda:0",
    )

    wav_path = f"{model.model_path}/example/zh.mp3"
    res = model.generate(input=[wav_path], cache={}, batch_size=1)
    text = res[0]["text"]
    print(text)

    model = AutoModel(
        model=model_dir,
        trust_remote_code=True,
        vad_model="fsmn-vad",
        vad_kwargs={"max_single_segment_time": 30000},
        remote_code="./model.py",
        device="cuda:0",
    )
    res = model.generate(input=[wav_path], cache={}, batch_size=1)
    text = res[0]["text"]
    print(text)


if __name__ == "__main__":
    main()
```

### 直接推理

```python
from model import FunASRNano


def main():
    model_dir = "FunAudioLLM/fun-asr-nano"
    m, kwargs = FunASRNano.from_pretrained(model=model_dir, device="cuda:0")
    m.eval()

    wav_path = f"{kwargs['model_path']}/example/zh.mp3"
    res = m.inference(data_in=[wav_path], **kwargs)
    text = res[0][0]["text"]
    print(text)


if __name__ == "__main__":
    main()
```

<details><summary> 参数说明（点击展开）</summary>

- `model_dir`：模型名称，或本地磁盘中的模型路径。
- `trust_remote_code`：是否信任远程代码，用于加载自定义模型实现。
- `remote_code`：指定模型具体代码的位置（例如，当前目录下的 `model.py`），支持绝对路径与相对路径。
- `device`：指定使用的设备，如 "cuda:0" 或 "cpu"。

</details>

# 性能评测 📝

## 多语言语音识别

我们在开源基准数据集（包括 AISHELL-1、AISHELL-2、Wenetspeech、Librispeech 和 Common Voice）上比较了 Fun-ASR 与其他模型的多语言语音识别性能。在中文和粤语识别效果上，Fun-ASR 模型具有明显的效果优势。

<div align="center">
<img src="image/asr_results1.png" width="400" /><img src="image/asr_results2.png" width="400" />
</div>

## 方言识别性能

Fun-ASR 在中文方言识别任务上表现优异，覆盖 7 大方言和 26 个地区口音，识别准确率显著优于其他开源模型。

<div align="center">
<img src="image/dialect_results.png" width="800" />
</div>

## 热词定制效果

在工业应用场景中，Fun-ASR 的热词定制能力表现出色，在不损伤通用识别准确率的前提下，将定制热词数量从 1000 拓展到 10000。

<div align="center">
<img src="image/hotword_results.png" width="800" />
</div>
