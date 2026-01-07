from funasr import AutoModel
import argparse
import datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import torch
import torchaudio
import numpy as np
from funasr.utils.load_utils import extract_fbank
from torch.nn.utils.rnn import pad_sequence

def get_args():
    parser = argparse.ArgumentParser(description="FunASR Inference with HuggingFace Dataset")
    parser.add_argument(
        "--model_dir", 
        type=str, 
        default="./Fun-ASR-Nano-2512",
        help="Model directory"
    )
    parser.add_argument(
        "--huggingface_dataset", 
        type=str,
        default="yuekai/speechio",
        help="Dataset name"
    )
    parser.add_argument(
        "--subset_name", 
        type=str, 
        default="SPEECHIO_ASR_ZH00007", 
        help="Dataset subset name"
    )
    parser.add_argument(
        "--split_name", 
        type=str, 
        default="test", 
        help="Dataset split name"
    )
    parser.add_argument(
        "--ref_column",
        type=str,
        default="text",
        help="Column name for reference text in dataset"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda:0", 
        help="Device to use for inference"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size (Note: FunASRNano may only support 1)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for dataloader"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=".",
        help="Directory to save the results and stats"
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        default="hypos.txt", 
        help="Output file for transcripts"
    )
    parser.add_argument(
        "--stats_file", 
        type=str, 
        default="wer.txt", 
        help="Output file for error statistics"
    )
    parser.add_argument(
        "--vllm_model_dir",
        type=str,
        default=None,
        help="Directory to the vllm model"
    )
    parser.add_argument(
        "--encoder_trt_engine",
        type=str,
        default=None,
        help="Path to the TensorRT engine for the audio encoder"
    )
    return parser.parse_args()

class DataCollator:
    def __init__(self, ref_column="text"):
        self.ref_column = ref_column

    def __call__(self, batch):
        ids = []
        wavs = []
        texts = []
        target_sr = 16000
        
        for item in batch:
            # Handle 'id' or 'audio_id' or key that identifies the utterance
            utt_id = item.get("id") or item.get("segment_id") or str(item.get("key", "unknown"))
            ids.append(utt_id)
            
            # Get reference text
            ref_text = item.get(self.ref_column, "")
            if not ref_text:
                # Fallback common keys
                if "text" in item:
                    ref_text = item["text"]
                elif "sentence" in item:
                    ref_text = item["sentence"]
            texts.append(ref_text)
            
            audio_info = item["audio"]
            audio = audio_info["array"]
            sr = audio_info["sampling_rate"]
            
            # Ensure audio is float32 and correct sampling rate
            if isinstance(audio, np.ndarray):
                audio_tensor = torch.from_numpy(audio).float()
            else:
                audio_tensor = audio.float()
            
            if sr != target_sr:
                 resampler = torchaudio.transforms.Resample(sr, target_sr)
                 audio_tensor = resampler(audio_tensor)
            
            wavs.append(audio_tensor)
                 
        return ids, wavs, texts

def main():
    args = get_args()

    model, kwargs = AutoModel.build_model(
        model=args.model_dir, trust_remote_code=True, device=args.device
    )
    tokenizer, frontend = kwargs["tokenizer"], kwargs["frontend"]

    instruction = "语音转写："
    prompt_prefix = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{instruction}"
    prompt_suffix = "<|im_end|>\n<|im_start|>assistant\n"
    prompt_prefix_ids = tokenizer.encode(prompt_prefix)
    prompt_suffix_ids = tokenizer.encode(prompt_suffix)
    prompt_prefix_ids = torch.tensor(prompt_prefix_ids, dtype=torch.int64).to(args.device)
    prompt_suffix_ids = torch.tensor(prompt_suffix_ids, dtype=torch.int64).to(args.device)

    # [T,D]
    prompt_prefix_embeddings = model.llm.model.get_input_embeddings()(prompt_prefix_ids)
    prompt_suffix_embeddings = model.llm.model.get_input_embeddings()(prompt_suffix_ids)

    dataset = datasets.load_dataset(
        args.huggingface_dataset,
        args.subset_name,
        split=args.split_name,
        trust_remote_code=True,
    )

    collator = DataCollator(ref_column=args.ref_column)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collator,
        shuffle=False
    )

    results = []

    print("Starting inference...")
    iterator = tqdm(dataloader)
    start_time = time.time()
    for batch_ids, batch_wavs, batch_refs in iterator:
        # [B, T, 560], [B]
        speech, speech_lengths = extract_fbank(
            batch_wavs,
            frontend=frontend,
            is_final=True,
        )
        speech = speech.to(args.device)
        speech_lengths = speech_lengths.to(args.device)
        print(speech.shape)
        print(speech_lengths)
        encoder_out, encoder_out_lens = model.audio_encoder(
            speech, speech_lengths
        )
        encoder_out, encoder_out_lens = model.audio_adaptor(
            encoder_out, encoder_out_lens
        )
        print(encoder_out.shape)
        print(encoder_out_lens)
        input_embeddings_list = []
        for i in range(args.batch_size):
            speech_embedding = encoder_out[i, :encoder_out_lens[i], :]
            input_embedding = torch.cat([prompt_prefix_embeddings, speech_embedding, prompt_suffix_embeddings], dim=0)
            input_embeddings_list.append(input_embedding)
        # list of T,D tensors with different lengths
        # padding to the B, T_max, D, and get the attention_mask

        input_embeddings = pad_sequence(input_embeddings_list, batch_first=True, padding_value=0.0)
        input_embeddings = input_embeddings.to(torch.bfloat16)
        
        attention_mask = torch.zeros(input_embeddings.shape[:2], dtype=torch.long, device=args.device)
        for i, embedding in enumerate(input_embeddings_list):
            attention_mask[i, :embedding.size(0)] = 1 
        llm_kwargs = kwargs.get("llm_kwargs", {})
        generated_ids = model.llm.generate(
            inputs_embeds=input_embeddings,
            max_new_tokens=512,
            attention_mask=attention_mask,
            **llm_kwargs,
        )
        response = tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True)

        print(response)
        breakpoint()

if __name__ == "__main__":
    main()