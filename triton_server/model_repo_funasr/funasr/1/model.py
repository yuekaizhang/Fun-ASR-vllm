"""
Triton Python Backend Model for FunASR.
This is the entry point for the Triton Inference Server.
"""

import json
import logging
import os
import sys

import numpy as np
import torch
import triton_python_backend_utils as pb_utils

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from funasr_wrapper import FunASRTritonWrapper
from feat_extractor import FunASRFeatExtractor


class TritonPythonModel:
    """
    Triton Python Backend Model for FunASR ASR inference.

    This model accepts raw audio waveforms and returns transcriptions.
    It supports batching via Triton's dynamic batching.
    """

    def initialize(self, args):
        """
        Initialize the model.

        Args:
            args: Dictionary containing model configuration
        """
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("FunASRTriton")

        # Parse model config
        self.model_config = json.loads(args['model_config'])
        params = self.model_config.get('parameters', {})

        # Get configuration parameters
        self.model_dir = params.get('model_dir', {}).get('string_value', './Fun-ASR-Nano-2512')
        self.vllm_model_dir = params.get('vllm_model_dir', {}).get('string_value', '')
        self.encoder_trt_engine = params.get('encoder_trt_engine', {}).get('string_value', '')
        self.llm_dtype = params.get('llm_dtype', {}).get('string_value', 'bfloat16')
        self.gpu_memory_utilization = float(params.get('gpu_memory_utilization', {}).get('string_value', '0.4'))

        # Get device from instance
        self.device_id = int(args.get("model_instance_device_id", 0))
        device_param = params.get('device', {}).get('string_value', '')
        if device_param:
            self.device = device_param
        else:
            self.device = f"cuda:{self.device_id}"

        self.logger.info(f"Initializing FunASR model on device {self.device}")
        self.logger.info(f"Model dir: {self.model_dir}")
        self.logger.info(f"vLLM model dir: {self.vllm_model_dir}")
        self.logger.info(f"Encoder TRT engine: {self.encoder_trt_engine}")

        # Initialize feature extractor
        self.feat_extractor = FunASRFeatExtractor(
            sample_rate=16000,
            device_id=self.device_id
        )

        # Initialize model wrapper
        vllm_dir = self.vllm_model_dir if self.vllm_model_dir else None
        trt_engine = self.encoder_trt_engine if self.encoder_trt_engine else None

        self.model = FunASRTritonWrapper(
            model_dir=self.model_dir,
            device=self.device,
            vllm_model_dir=vllm_dir,
            encoder_trt_engine=trt_engine,
            llm_dtype=self.llm_dtype,
            gpu_memory_utilization=self.gpu_memory_utilization,
        )

        self.logger.info("FunASR model initialized successfully")

    def execute(self, requests):
        """
        Execute inference on a batch of requests.

        Args:
            requests: List of pb_utils.InferenceRequest objects

        Returns:
            List of pb_utils.InferenceResponse objects
        """
        responses = []
        all_wavs = []
        request_counts = []

        # Collect all audio from requests
        for request in requests:
            wav_tensor = pb_utils.get_input_tensor_by_name(request, "WAV")
            wav_lens_tensor = pb_utils.get_input_tensor_by_name(request, "WAV_LENS")

            wav_data = wav_tensor.as_numpy()

            # Handle WAV_LENS if provided
            if wav_lens_tensor is not None:
                wav_len_data = wav_lens_tensor.as_numpy()
                valid_len = int(wav_len_data.flatten()[0])
            else:
                # Use full length if WAV_LENS not provided
                valid_len = wav_data.shape[-1]

            # Extract valid audio (assuming first dimension is batch=1 from client)
            if wav_data.ndim > 1:
                wav_seq = wav_data[0, :valid_len]
            else:
                wav_seq = wav_data[:valid_len]

            # Convert to tensor
            wav_tensor_torch = torch.from_numpy(wav_seq).float()
            all_wavs.append(wav_tensor_torch)
            request_counts.append(1)

        # Run inference
        try:
            transcripts = self.model.transcribe(all_wavs)
        except Exception as e:
            self.logger.error(f"Inference error: {e}")
            # Return error responses
            for _ in requests:
                error_msg = np.array([f"Error: {str(e)}"], dtype=object).reshape(-1, 1)
                out_tensor = pb_utils.Tensor("TRANSCRIPTS", error_msg)
                responses.append(pb_utils.InferenceResponse([out_tensor]))
            return responses

        # Build responses
        result_idx = 0
        for count in request_counts:
            req_texts = []
            for _ in range(count):
                if result_idx < len(transcripts):
                    req_texts.append(transcripts[result_idx])
                else:
                    req_texts.append("")
                result_idx += 1

            # Create output tensor
            out_np = np.array(req_texts, dtype=object).reshape(-1, 1)
            out_tensor = pb_utils.Tensor("TRANSCRIPTS", out_np)
            responses.append(pb_utils.InferenceResponse([out_tensor]))

        return responses

    def finalize(self):
        """
        Clean up resources.
        """
        self.logger.info("Cleaning up FunASR model...")
        if hasattr(self, 'model') and self.model is not None:
            if hasattr(self.model, 'vllm') and self.model.vllm is not None:
                del self.model.vllm
            del self.model
        torch.cuda.empty_cache()
        self.logger.info("FunASR model cleanup complete")
