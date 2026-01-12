"""
FunASR model wrapper for Triton Inference Server.
Handles model loading, feature extraction, and inference.
"""

import os
import sys
import logging
import torch
import numpy as np
from typing import List, Optional, Tuple

from torch.nn.utils.rnn import pad_sequence

# Add current directory to path for local imports
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if _CURRENT_DIR not in sys.path:
    sys.path.insert(0, _CURRENT_DIR)

class TrtAudioEncoderWrapper:
    """TensorRT wrapper for audio encoder."""

    def __init__(self, engine_path: str, device: str = "cuda:0"):
        import tensorrt as trt

        self.device = torch.device(device)
        self.logger = trt.Logger(trt.Logger.INFO)

        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream(device=self.device)

    def __call__(self, x: torch.Tensor, input_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run TensorRT encoder inference.

        Args:
            x: Input features [B, T, D]
            input_lengths: Input lengths [B]

        Returns:
            Tuple of (encoder_out, encoder_out_lengths)
        """
        import tensorrt as trt

        x = x.to(self.device).contiguous()
        input_lengths = input_lengths.to(self.device).contiguous()

        N, T, D = x.shape
        self.context.set_input_shape("x", (N, T, D))
        self.context.set_input_shape("input_lengths", (N,))

        outputs = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                if name == "x":
                    if self.engine.get_tensor_dtype(name) == trt.DataType.HALF and x.dtype != torch.float16:
                        x = x.half()
                    self.context.set_tensor_address(name, x.data_ptr())
                elif name == "input_lengths":
                    if self.engine.get_tensor_dtype(name) == trt.DataType.INT32 and input_lengths.dtype != torch.int32:
                        input_lengths = input_lengths.int()
                    self.context.set_tensor_address(name, input_lengths.data_ptr())
            else:
                out_shape = self.context.get_tensor_shape(name)
                dtype = self.engine.get_tensor_dtype(name)

                if dtype == trt.DataType.HALF:
                    torch_dtype = torch.float16
                elif dtype == trt.DataType.FLOAT:
                    torch_dtype = torch.float32
                elif dtype == trt.DataType.BF16:
                    torch_dtype = torch.bfloat16
                elif dtype == trt.DataType.INT32:
                    torch_dtype = torch.int32
                else:
                    torch_dtype = torch.float32

                out_tensor = torch.empty(tuple(out_shape), dtype=torch_dtype, device=self.device)
                outputs[name] = out_tensor
                self.context.set_tensor_address(name, out_tensor.data_ptr())

        self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        self.stream.synchronize()

        encoder_out = outputs["encoder_out"]
        encoder_out_lengths = outputs["encoder_out_lengths"]

        if encoder_out.dtype in (torch.float16, torch.bfloat16):
            encoder_out = encoder_out.float()

        return encoder_out, encoder_out_lengths


class FunASRTritonWrapper:
    """
    Wrapper class for FunASR model inference in Triton.
    """

    def __init__(
        self,
        model_dir: str,
        device: str = "cuda:0",
        vllm_model_dir: Optional[str] = None,
        encoder_trt_engine: Optional[str] = None,
        llm_dtype: str = "bfloat16",
        gpu_memory_utilization: float = 0.4,
    ):
        """
        Initialize the FunASR model wrapper.

        Args:
            model_dir: Path to FunASR model directory
            device: Device for inference
            vllm_model_dir: Path to vLLM model (optional)
            encoder_trt_engine: Path to TensorRT encoder engine (optional)
            llm_dtype: LLM data type (bfloat16, float16, float32)
            gpu_memory_utilization: GPU memory fraction for vLLM
        """
        self.device = device
        self.llm_dtype = llm_dtype

        logging.info(f"Loading FunASR model from {model_dir}")

        # Import model module to register FunASRNano class
        # This must happen before AutoModel.build_model is called
        import funasr_nano_model  # noqa: F401

        from funasr import AutoModel

        # Load model
        self.model, self.kwargs = AutoModel.build_model(
            model=model_dir,
            trust_remote_code=True,
            device=device
        )

        self.tokenizer = self.kwargs["tokenizer"]
        self.frontend = self.kwargs["frontend"]

        # Load TensorRT encoder if specified
        if encoder_trt_engine and os.path.exists(encoder_trt_engine):
            logging.info(f"Loading TensorRT encoder from {encoder_trt_engine}")
            del self.model.audio_encoder
            self.model.audio_encoder = TrtAudioEncoderWrapper(encoder_trt_engine, device)
            if hasattr(self.model, "feat_permute"):
                self.model.feat_permute = False

        # Load vLLM if specified
        self.vllm = None
        self.vllm_sampling_params = None
        if vllm_model_dir:
            logging.info(f"Loading vLLM model from {vllm_model_dir}")
            from vllm import LLM, SamplingParams

            self.vllm = LLM(
                model=vllm_model_dir,
                enable_prompt_embeds=True,
                gpu_memory_utilization=gpu_memory_utilization,
                dtype=llm_dtype
            )
            self.vllm_sampling_params = SamplingParams(
                top_p=0.001,
                max_tokens=500,
            )
            self.model.vllm = self.vllm
            self.model.vllm_sampling_params = self.vllm_sampling_params

        # Prepare prompt templates
        self.instruction = "语音转写："
        self.prompt_prefix = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{self.instruction}"
        self.prompt_suffix = "<|im_end|>\n<|im_start|>assistant\n"

        # Encode prompts
        self.prompt_prefix_ids = self.tokenizer.encode(self.prompt_prefix)
        self.prompt_suffix_ids = self.tokenizer.encode(self.prompt_suffix)
        self.prompt_prefix_ids = torch.tensor(self.prompt_prefix_ids, dtype=torch.int64).to(device)
        self.prompt_suffix_ids = torch.tensor(self.prompt_suffix_ids, dtype=torch.int64).to(device)

        # Get prompt embeddings
        self.prompt_prefix_embeddings = self.model.llm.model.get_input_embeddings()(self.prompt_prefix_ids)
        self.prompt_suffix_embeddings = self.model.llm.model.get_input_embeddings()(self.prompt_suffix_ids)

        logging.info("FunASR model loaded successfully")

    def transcribe(self, wavs: List[torch.Tensor]) -> List[str]:
        """
        Transcribe a batch of audio waveforms.

        Args:
            wavs: List of waveform tensors (1D, float32, 16kHz)

        Returns:
            List of transcription strings
        """
        from funasr.utils.load_utils import extract_fbank

        # Extract features
        speech, speech_lengths = extract_fbank(
            wavs,
            frontend=self.frontend,
            is_final=True,
        )

        speech = speech.to(self.device)
        speech_lengths = speech_lengths.to(self.device)

        # Run audio encoder
        encoder_out, encoder_out_lens = self.model.audio_encoder(
            speech, speech_lengths
        )

        # Run audio adaptor
        encoder_out, encoder_out_lens = self.model.audio_adaptor(
            encoder_out, encoder_out_lens
        )

        # Build input embeddings for each sample
        input_embeddings_list = []
        for i in range(len(wavs)):
            speech_embedding = encoder_out[i, :encoder_out_lens[i], :]
            input_embedding = torch.cat([
                self.prompt_prefix_embeddings,
                speech_embedding,
                self.prompt_suffix_embeddings
            ], dim=0)
            input_embeddings_list.append(input_embedding)

        # Generate transcriptions
        if self.vllm is not None:
            outputs = self.vllm.generate(
                [{"prompt_embeds": emb} for emb in input_embeddings_list],
                self.vllm_sampling_params,
                use_tqdm=False,
            )
            responses = [output.outputs[0].text for output in outputs]
        else:
            # Use HuggingFace generate
            input_embeddings = pad_sequence(input_embeddings_list, batch_first=True, padding_value=0.0)
            input_embeddings = input_embeddings.to(torch.bfloat16)

            attention_mask = torch.zeros(input_embeddings.shape[:2], dtype=torch.long, device=self.device)
            for i, embedding in enumerate(input_embeddings_list):
                attention_mask[i, :embedding.size(0)] = 1

            llm_kwargs = self.kwargs.get("llm_kwargs", {})
            generated_ids = self.model.llm.generate(
                inputs_embeds=input_embeddings,
                max_new_tokens=512,
                attention_mask=attention_mask,
                **llm_kwargs,
            )
            responses = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        return responses

    def transcribe_from_features(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor
    ) -> List[str]:
        """
        Transcribe from pre-extracted features.

        Args:
            speech: Features tensor [B, T, D]
            speech_lengths: Feature lengths [B]

        Returns:
            List of transcription strings
        """
        speech = speech.to(self.device)
        speech_lengths = speech_lengths.to(self.device)

        # Run audio encoder
        encoder_out, encoder_out_lens = self.model.audio_encoder(
            speech, speech_lengths
        )

        # Run audio adaptor
        encoder_out, encoder_out_lens = self.model.audio_adaptor(
            encoder_out, encoder_out_lens
        )

        batch_size = speech.shape[0]

        # Build input embeddings for each sample
        input_embeddings_list = []
        for i in range(batch_size):
            speech_embedding = encoder_out[i, :encoder_out_lens[i], :]
            input_embedding = torch.cat([
                self.prompt_prefix_embeddings,
                speech_embedding,
                self.prompt_suffix_embeddings
            ], dim=0)
            input_embeddings_list.append(input_embedding)

        # Generate transcriptions
        if self.vllm is not None:
            outputs = self.vllm.generate(
                [{"prompt_embeds": emb} for emb in input_embeddings_list],
                self.vllm_sampling_params,
                use_tqdm=False,
            )
            responses = [output.outputs[0].text for output in outputs]
        else:
            # Use HuggingFace generate
            input_embeddings = pad_sequence(input_embeddings_list, batch_first=True, padding_value=0.0)
            input_embeddings = input_embeddings.to(torch.bfloat16)

            attention_mask = torch.zeros(input_embeddings.shape[:2], dtype=torch.long, device=self.device)
            for i, embedding in enumerate(input_embeddings_list):
                attention_mask[i, :embedding.size(0)] = 1

            llm_kwargs = self.kwargs.get("llm_kwargs", {})
            generated_ids = self.model.llm.generate(
                inputs_embeds=input_embeddings,
                max_new_tokens=512,
                attention_mask=attention_mask,
                **llm_kwargs,
            )
            responses = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        return responses
