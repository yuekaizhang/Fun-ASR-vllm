import torch
import tensorrt as trt
import os
import logging
from typing import Dict

def get_trt_kwargs_dynamic_batch(
    min_batch_size: int = 1,
    opt_batch_size: int = 2,
    max_batch_size: int = 4,
) -> Dict:
    """Get keyword arguments for TensorRT with dynamic batch size."""
    feat_dim = 560
    min_seq_len = 10
    opt_seq_len = 200
    max_seq_len = 3000
    
    # x: [N, T, 560]
    # input_lengths: [N]
    
    min_shape = [
        (min_batch_size, min_seq_len, feat_dim), # x
        (min_batch_size,),                       # input_lengths
    ]
    opt_shape = [
        (opt_batch_size, opt_seq_len, feat_dim),
        (opt_batch_size,),
    ]
    max_shape = [
        (max_batch_size, max_seq_len, feat_dim),
        (max_batch_size,),
    ]
    input_names = ["x", "input_lengths"]
    return {
        "min_shape": min_shape,
        "opt_shape": opt_shape,
        "max_shape": max_shape,
        "input_names": input_names,
    }


def convert_onnx_to_trt(
    trt_model: str, trt_kwargs: Dict, onnx_model: str, dtype: torch.dtype = torch.float32
):
    """
    Convert an ONNX model to a TensorRT engine.

    Args:
        trt_model (str): The path to save the TensorRT engine.
        trt_kwargs (Dict): Keyword arguments for TensorRT.
        onnx_model (str): The path to the ONNX model.
        dtype (torch.dtype, optional): The data type to use. Defaults to torch.float16.
    """
    logging.info("Converting onnx to trt...")
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)
    config = builder.create_builder_config()
    # config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 32)  # 4GB
    if dtype == torch.float16:
        config.set_flag(trt.BuilderFlag.FP16)
    elif dtype == torch.bfloat16:
        config.set_flag(trt.BuilderFlag.BF16)

    profile = builder.create_optimization_profile()
    # load onnx model
    with open(onnx_model, "rb") as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise ValueError('failed to parse {}'.format(onnx_model))
    
    # set input shapes
    for i in range(len(trt_kwargs['input_names'])):
        name = trt_kwargs['input_names'][i]
        profile.set_shape(name, trt_kwargs['min_shape'][i], trt_kwargs['opt_shape'][i], trt_kwargs['max_shape'][i])
    
    if dtype == torch.float16:
        tensor_dtype = trt.DataType.HALF
    elif dtype == torch.bfloat16:
        tensor_dtype = trt.DataType.BF16
    elif dtype == torch.float32:
        tensor_dtype = trt.DataType.FLOAT
    else:
        raise ValueError('invalid dtype {}'.format(dtype))
        
    # set input and output data type
    for i in range(network.num_inputs):
        input_tensor = network.get_input(i)
        if input_tensor.name == "input_lengths":
             input_tensor.dtype = trt.DataType.INT32
        else:
             input_tensor.dtype = tensor_dtype
             
    for i in range(network.num_outputs):
        output_tensor = network.get_output(i)
        if output_tensor.name == "encoder_out_lengths":
             output_tensor.dtype = trt.DataType.INT32
        else:
             output_tensor.dtype = tensor_dtype
             
    config.add_optimization_profile(profile)
    engine_bytes = builder.build_serialized_network(network, config)
    # save trt engine
    with open(trt_model, "wb") as f:
        f.write(engine_bytes)
    logging.info("Succesfully convert onnx to trt...")


class TrtAudioEncoderWrapper:
    def __init__(self, engine_path, device="cuda:0"):
        self.device = torch.device(device)
        self.logger = trt.Logger(trt.Logger.INFO)
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream(device=self.device)

    def __call__(self, x, input_lengths):
        # x: (B, T, D)
        # input_lengths: (B,)
        # Ensure inputs are on the correct device and contiguous
        x = x.to(self.device).contiguous()
        input_lengths = input_lengths.to(self.device).contiguous()

        N, T, D = x.shape
        # Input names based on test_trt_nano.py: "x", "input_lengths"
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

        # Output names based on test_trt_nano.py: "encoder_out", "encoder_out_lengths"
        encoder_out = outputs["encoder_out"]
        encoder_out_lengths = outputs["encoder_out_lengths"]
        
        # If the output is float16, cast it to float32 to ensure compatibility with the rest of the pipeline
        if encoder_out.dtype == torch.float16 or encoder_out.dtype == torch.bfloat16:
            encoder_out = encoder_out.float()
            
        return encoder_out, encoder_out_lengths

def load_trt_audio_encoder(model, trt_path, onnx_path = "model.onnx", dtype = torch.float32):
    if not os.path.exists(trt_path):
        logging.info(f"TRT model not found at {trt_path}, converting onnx to trt...")
        if not os.path.exists(onnx_path):
            os.system(f"wget -nc https://huggingface.co/yuekai/Fun-ASR-Nano-2512-Encoder-ONNX-FP32/resolve/main/model.onnx -O {onnx_path}")

        trt_kwargs = get_trt_kwargs_dynamic_batch()
        convert_onnx_to_trt(trt_path, trt_kwargs, onnx_path, dtype=dtype)
    
    print(f"Loading TRT model from {trt_path}")
    del model.audio_encoder
    model.audio_encoder = TrtAudioEncoderWrapper(trt_path)
    # TRT engine typically expects (B, T, D), so disable permutation if it was enabled
    if hasattr(model, "feat_permute"):
        model.feat_permute = False
    print("Replaced audio_encoder with TRT engine.")