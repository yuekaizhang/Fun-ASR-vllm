ARG BASE_IMAGE=nvcr.io/nvidia/tritonserver:25.06-trtllm-python-py3
FROM ${BASE_IMAGE}

# 设置工作目录
WORKDIR /workspace

# 1. 安装系统依赖 (添加 RUN)
RUN apt-get update && apt-get install -y ffmpeg

# 2. 安装 uv (假设基础镜像已有 pip，否则需先安装 pip)
RUN pip install uv

# 3. 创建虚拟环境
# 注意：这将使用 uv 创建名为 .venv 的环境，并指定 python 3.12
RUN uv venv .venv --python 3.12 --seed

# 4. 激活虚拟环境 (替代 source，通过环境变量持久化)
ENV VIRTUAL_ENV=/workspace/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# 5. 安装 Python 依赖
# 将原本散落的包名合并为一条安装命令，并处理版本号中的特殊字符
RUN uv pip install \
    vllm==0.13.0 \
    WeTextProcessing \
    kaldialign \
    torch \
    torchaudio \
    transformers \
    "funasr>=1.2.7"

RUN uv pip install tritonclient[all] soundfile datasets==3.1.0