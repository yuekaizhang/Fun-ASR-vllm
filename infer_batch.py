from funasr import AutoModel
import argparse
import datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import torch
import torch.profiler
from torch.profiler import profile, record_function, ProfilerActivity, schedule
import torchaudio
import numpy as np
import kaldialign
import unicodedata
import re
import os
import logging
from typing import Iterable, Tuple, List, TextIO, Dict
from collections import defaultdict
from funasr.utils.load_utils import extract_fbank
from funasr.frontends.wav_frontend import WavFrontend
from torch.nn.utils.rnn import pad_sequence
from tn.chinese.normalizer import Normalizer as ZhNormalizer
from audio_encoder_tensorrt import load_trt_audio_encoder
import kaldifeat
def store_transcripts(
    filename: str, texts: Iterable[Tuple[str, str, str]]
) -> None:
    """Save predicted results and reference transcripts to a file.

    Args:
      filename:
        File to save the results to.
      texts:
        An iterable of tuples. The first element is the cur_id, the second is
        the reference transcript and the third element is the predicted result.
    Returns:
      Return None.
    """
    with open(filename, "w") as f:
        for cut_id, ref, hyp in texts:
            print(f"{cut_id}:\tref={ref}", file=f)
            print(f"{cut_id}:\thyp={hyp}", file=f)


def write_error_stats(
    f: TextIO,
    test_set_name: str,
    results: List[Tuple[str, str]],
    enable_log: bool = True,
) -> float:
    """Write statistics based on predicted results and reference transcripts.

    It will write the following to the given file:

        - WER
        - number of insertions, deletions, substitutions, corrects and total
          reference words. For example::

              Errors: 23 insertions, 57 deletions, 212 substitutions, over 2606
              reference words (2337 correct)

        - The difference between the reference transcript and predicted result.
          An instance is given below::

            THE ASSOCIATION OF (EDISON->ADDISON) ILLUMINATING COMPANIES

          The above example shows that the reference word is `EDISON`,
          but it is predicted to `ADDISON` (a substitution error).

          Another example is::

            FOR THE FIRST DAY (SIR->*) I THINK

          The reference word `SIR` is missing in the predicted
          results (a deletion error).
      results:
        An iterable of tuples. The first element is the cur_id, the second is
        the reference transcript and the third element is the predicted result.
      enable_log:
        If True, also print detailed WER to the console.
        Otherwise, it is written only to the given file.
    Returns:
      Return None.
    """
    subs: Dict[Tuple[str, str], int] = defaultdict(int)
    ins: Dict[str, int] = defaultdict(int)
    dels: Dict[str, int] = defaultdict(int)

    # `words` stores counts per word, as follows:
    #   corr, ref_sub, hyp_sub, ins, dels
    words: Dict[str, List[int]] = defaultdict(lambda: [0, 0, 0, 0, 0])
    num_corr = 0
    ERR = "*"
    for cut_id, ref, hyp in results:
        ali = kaldialign.align(ref, hyp, ERR)
        for ref_word, hyp_word in ali:
            if ref_word == ERR:
                ins[hyp_word] += 1
                words[hyp_word][3] += 1
            elif hyp_word == ERR:
                dels[ref_word] += 1
                words[ref_word][4] += 1
            elif hyp_word != ref_word:
                subs[(ref_word, hyp_word)] += 1
                words[ref_word][1] += 1
                words[hyp_word][2] += 1
            else:
                words[ref_word][0] += 1
                num_corr += 1
    ref_len = sum([len(r.split()) for _, r, _ in results]) # Corrected to split by space for word count
    # Note: original code used len(r) which counts characters if r is string, or list items if r is list.
    # If ref is a string of words space-separated, len(r) is char count. WER is usually word level.
    # However, for Chinese (which FunASR often handles), character level (CER) is standard.
    # If the input is Chinese chars space-separated, split() works. If no spaces, len() works for CER.
    # Assuming standard behavior. The original code had `ref_len = sum([len(r) for _, r, _ in results])`.
    # I will revert to original behavior to be safe, assuming user inputs are compatible with what this function expects.
    ref_len = sum([len(r) for _, r, _ in results]) 

    sub_errs = sum(subs.values())
    ins_errs = sum(ins.values())
    del_errs = sum(dels.values())
    tot_errs = sub_errs + ins_errs + del_errs
    if ref_len > 0:
        tot_err_rate = "%.2f" % (100.0 * tot_errs / ref_len)
    else:
        tot_err_rate = "0.00"

    if enable_log:
        logging.info(
            f"[{test_set_name}] %WER {tot_errs / ref_len:.2%} "
            f"[{tot_errs} / {ref_len}, {ins_errs} ins, "
            f"{del_errs} del, {sub_errs} sub ]"
        )

    print(f"%WER = {tot_err_rate}", file=f)
    print(
        f"Errors: {ins_errs} insertions, {del_errs} deletions, "
        f"{sub_errs} substitutions, over {ref_len} reference "
        f"words ({num_corr} correct)",
        file=f,
    )
    print(
        "Search below for sections starting with PER-UTT DETAILS:, "
        "SUBSTITUTIONS:, DELETIONS:, INSERTIONS:, PER-WORD STATS:",
        file=f,
    )

    print("", file=f)
    print("PER-UTT DETAILS: corr or (ref->hyp)  ", file=f)
    for cut_id, ref, hyp in results:
        ali = kaldialign.align(ref, hyp, ERR)
        combine_successive_errors = True
        if combine_successive_errors:
            ali = [[[x], [y]] for x, y in ali]
            for i in range(len(ali) - 1):
                if ali[i][0] != ali[i][1] and ali[i + 1][0] != ali[i + 1][1]:
                    ali[i + 1][0] = ali[i][0] + ali[i + 1][0]
                    ali[i + 1][1] = ali[i][1] + ali[i + 1][1]
                    ali[i] = [[], []]
            ali = [
                [
                    list(filter(lambda a: a != ERR, x)),
                    list(filter(lambda a: a != ERR, y)),
                ]
                for x, y in ali
            ]
            ali = list(filter(lambda x: x != [[], []], ali))
            ali = [
                [
                    ERR if x == [] else " ".join(x),
                    ERR if y == [] else " ".join(y),
                ]
                for x, y in ali
            ]

        print(
            f"{cut_id}:\t"
            + " ".join(
                (
                    ref_word if ref_word == hyp_word else f"({ref_word}->{hyp_word})"
                    for ref_word, hyp_word in ali
                )
            ),
            file=f,
        )

    print("", file=f)
    print("SUBSTITUTIONS: count ref -> hyp", file=f)

    for count, (ref, hyp) in sorted([(v, k) for k, v in subs.items()], reverse=True):
        print(f"{count}   {ref} -> {hyp}", file=f)

    print("", file=f)
    print("DELETIONS: count ref", file=f)
    for count, ref in sorted([(v, k) for k, v in dels.items()], reverse=True):
        print(f"{count}   {ref}", file=f)

    print("", file=f)
    print("INSERTIONS: count hyp", file=f)
    for count, hyp in sorted([(v, k) for k, v in ins.items()], reverse=True):
        print(f"{count}   {hyp}", file=f)

    print("", file=f)
    print("PER-WORD STATS: word  corr tot_errs count_in_ref count_in_hyp", file=f)
    for _, word, counts in sorted(
        [(sum(v[1:]), k, v) for k, v in words.items()], reverse=True
    ):
        (corr, ref_sub, hyp_sub, ins, dels) = counts
        tot_errs = ref_sub + hyp_sub + ins + dels
        ref_count = corr + ref_sub + dels
        hyp_count = corr + hyp_sub + ins

        print(f"{word}   {corr} {tot_errs} {ref_count} {hyp_count}", file=f)
    if ref_len > 0:
        return float(tot_errs) / float(ref_len) * 100.0
    else:
        return 0.0

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
        default=16,
        help="Batch size (Note: FunASRNano may only support 1)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of workers for dataloader"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./log_new_pipeline_encoder_trt_opt16_vllm_batch_size_16",
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
    parser.add_argument(
        "--enable_profiler",
        action="store_true",
        help="Enable torch profiler for performance analysis"
    )
    parser.add_argument(
        "--profiler_output_dir",
        type=str,
        default="./profiler_output",
        help="Directory to save profiler trace files (for perfetto)"
    )
    parser.add_argument(
        "--enable_mock_llm",
        action="store_true",
        help="Use mock LLM to test encoder speed only"
    )
    parser.add_argument(
        "--enable_nvtx",
        action="store_true",
        help="Enable NVTX markers for Nsight profiling"
    )
    parser.add_argument(
        "--nvtx_wait",
        type=int,
        default=1,
        help="Number of wait steps before NVTX tracing"
    )
    parser.add_argument(
        "--nvtx_warmup",
        type=int,
        default=2,
        help="Number of warmup steps before NVTX tracing"
    )
    parser.add_argument(
        "--nvtx_active",
        type=int,
        default=3,
        help="Number of steps to trace with NVTX"
    )
    parser.add_argument(
        "--use_gpu_fbank",
        action="store_true",
        help="Use KaldifeatExtractor (GPU-based fbank) instead of CPU extract_fbank"
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

def normalize_text_alimeeting(text: str) -> str:
    """
    Text normalization similar to M2MeT challenge baseline.
    See: https://github.com/yufan-aslp/AliMeeting/blob/main/asr/local/text_normalize.pl
    """
    # def remove_punctuation(text: str) -> str:
    #     for x in punctuation_all:
    #         if x == '\'':
    #             continue
    #         text = text.replace(x, '')
    #     return text
    text = text.replace('\u00A0', '') # test_hard
    text = text.replace(" ", "")
    text = text.replace("<sil>", "")
    text = text.replace("<%>", "")
    text = text.replace("<->", "")
    text = text.replace("<$>", "")
    text = text.replace("<#>", "")
    text = text.replace("<_>", "")
    text = text.replace("<space>", "")
    text = text.replace("`", "")
    text = text.replace("&", "")
    text = text.replace(",", "")
    if re.search("[a-zA-Z]", text):
        text = text.upper()
    text = text.replace("Ａ", "A")
    text = text.replace("ａ", "A")
    text = text.replace("ｂ", "B")
    text = text.replace("ｃ", "C")
    text = text.replace("ｋ", "K")
    text = text.replace("ｔ", "T")
    text = text.replace("，", "")
    text = text.replace("丶", "")
    text = text.replace("。", "")
    text = text.replace("、", "")
    text = text.replace("？", "")
    # text = remove_punctuation(text)
    return text


class KaldifeatExtractor:
    def __init__(self, sample_rate, device_id=0):
        self.opts = kaldifeat.FbankOptions()
        self.opts.device = torch.device('cuda', device_id)
        # self.opts.device = torch.device('cpu')
        self.opts.frame_opts.dither = 0.0
        self.opts.mel_opts.num_bins = 80
        self.opts.frame_opts.frame_shift_ms = 10
        self.opts.frame_opts.frame_length_ms = 25
        self.opts.frame_opts.samp_freq = sample_rate
        self.opts.frame_opts.snip_edges = False
        self.opts.frame_opts.window_type = "hamming"
        self.fbank = kaldifeat.Fbank(self.opts)

    @staticmethod
    def apply_lfr(features, lengths, window_size=7, window_shift=6, max_len=-1):
        """
        Apply Low Frame Rate (LFR) to a batch of features.
        features: [B, T, D]
        lengths: [B]
        Returns:
            features_lfr: [B, T_lfr, D * window_size]
            lengths_lfr: [B]
        """
        B, T, D = features.shape
        
        # Calculate expected output lengths
        # Formula: (T - window_size) // window_shift + 1
        # We clamp min to 0 to avoid negative lengths for very short inputs
        lengths_lfr = (lengths - window_size) // window_shift + 1
        lengths_lfr = lengths_lfr.clamp(min=0)
        
        # If T < window_size, we might need to pad time dimension to at least window_size
        # to ensure unfold works.
        if T < window_size:
            pad_amt = window_size - T
            features = torch.nn.functional.pad(features, (0, 0, 0, pad_amt))
            T = window_size

        # Unfold features to create windows
        # dimension 1 is time. size=window_size, step=window_shift
        # Result shape: [B, num_windows, D, window_size]
        features_unfolded = features.unfold(1, window_size, window_shift)
        
        # We need to flatten the window content.
        # The target layout is [Frame 0, Frame 1, ..., Frame window_size-1] per LFR step.
        # Unfold gives [D, window_size] in the last two dims.
        # We want to flatten to [window_size * D] such that we have D elements of frame 0, then D of frame 1...
        # So we permute to [B, num_windows, window_size, D]
        features_unfolded = features_unfolded.permute(0, 1, 3, 2)
        
        # Make contiguous before reshape to avoid potential issues on CUDA
        features_unfolded = features_unfolded.contiguous()
        
        # Flatten the last two dimensions
        features_lfr = features_unfolded.reshape(B, features_unfolded.size(1), -1)
        
        # Handle max_len if specified
        if max_len > 0:
            cur_len = features_lfr.size(1)
            if cur_len > max_len:
                features_lfr = features_lfr[:, :max_len, :]
                lengths_lfr = lengths_lfr.clamp(max=max_len)
            elif cur_len < max_len:
                pad_amt = max_len - cur_len
                features_lfr = torch.nn.functional.pad(features_lfr, (0, 0, 0, pad_amt))
        
        return features_lfr, lengths_lfr

    def __call__(self, samples, max_len=-1, window_size=7, window_shift=6):
        """
        samples: List[torch.Tensor] (waveforms) or single torch.Tensor/np.ndarray
        """
        input_is_list = isinstance(samples, list)

        if not input_is_list:
            if isinstance(samples, np.ndarray):
                samples_list = [torch.from_numpy(samples).float()]
            elif isinstance(samples, torch.Tensor):
                samples_list = [samples.float()]
            else:
                raise TypeError("samples must be list of tensors, tensor, or numpy array")
        else:
            samples_list = samples

        # Move to device and scale (assuming input is float [-1, 1], scale to short range)
        # kaldifeat.Fbank expects waveforms.
        # We explicitly cast to float32 to ensure compatibility and correct scaling
        samples_device = [s.to(self.opts.device, dtype=torch.float32) * 32768.0 for s in samples_list]

        # Extract features (List[Tensor])
        features_list = self.fbank(samples_device)
        
        # Create lengths tensor
        lengths = torch.tensor([f.size(0) for f in features_list], dtype=torch.long, device=self.opts.device)
        
        # Pad sequence to create batch [B, T_max, D]
        features_padded = pad_sequence(features_list, batch_first=True, padding_value=0.0)
        
        # print(f"features.shape (KaldifeatExtractor before lfr, padded): {features_padded.shape}")
        
        # Apply LFR
        features_lfr, lengths_lfr = self.apply_lfr(features_padded, lengths, window_size, window_shift, max_len)
        
        # print(f"features.shape (KaldifeatExtractor final): {features_lfr.shape}")
        
        # If input was single item, we might still want to return batch format as requested
        # "help me padding, finally output speech B,T,D, speech_lengths B"
        # So we always return batch tensors.
        
        return features_lfr, lengths_lfr


def main():
    args = get_args()

    model, kwargs = AutoModel.build_model(
        model=args.model_dir, trust_remote_code=True, device=args.device
    )

    if args.encoder_trt_engine:
        load_trt_audio_encoder(model, args.encoder_trt_engine, dtype=torch.float16)

    if args.vllm_model_dir is not None and not args.enable_mock_llm:
        from vllm import LLM, SamplingParams
        vllm = LLM(model=args.vllm_model_dir, enable_prompt_embeds=True, gpu_memory_utilization=0.4, dtype="bfloat16")
        sampling_params = SamplingParams(
            top_p=0.001,
            max_tokens=500,
        )
        model.vllm = vllm
        model.vllm_sampling_params = sampling_params


    tokenizer, frontend = kwargs["tokenizer"], kwargs["frontend"]
    frontend = WavFrontend(
        fs=16000,
        lfr_m=7,
        lfr_n=6,
        n_mels=80,
        window="hamming",
        frame_length=25,
        frame_shift=10,
        cmvn_file=None,
        dither=0.0, # to remove torch.randn overhead
    )
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
    # dataset = dataset.select(range(128))

    collator = DataCollator(ref_column=args.ref_column)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collator,
        shuffle=False
    )

    # Initialize GPU fbank extractor if enabled
    gpu_fbank_extractor = None
    if args.use_gpu_fbank:
        device_id = int(args.device.split(":")[-1]) if ":" in args.device else 0
        gpu_fbank_extractor = KaldifeatExtractor(sample_rate=16000, device_id=device_id)
        print(f"Using GPU fbank extractor (KaldifeatExtractor) on device {device_id}")

    zh_tn_model = ZhNormalizer(
        cache_dir="./cache",
        remove_erhua=False,
        remove_interjections=False,
        remove_puncts=True,
        overwrite_cache=False,
    )
    
    def normalize_text(text):
        # Normalize full-width characters to half-width
        text = unicodedata.normalize("NFKC", text)
        text = normalize_text_alimeeting(text)
        return zh_tn_model.normalize(text)

    results = []

    print("Starting inference...")
    start_time = time.time()
    results = []
    total_datacollator_time = 0.0
    total_inference_time = 0.0
    # Detailed timing
    total_extract_fbank_time = 0.0
    total_encoder_time = 0.0
    total_adaptor_time = 0.0
    total_embedding_time = 0.0
    total_llm_time = 0.0
    batch_count = 0

    def run_inference_step(batch_ids, batch_wavs, batch_refs, use_nvtx=False):
        """Run a single inference step with profiler annotations.
        Returns: (response, timing_dict)
        """
        # Helper for optional NVTX range
        from contextlib import nullcontext
        def nvtx_range(name):
            if use_nvtx:
                return torch.cuda.nvtx.range(name)
            return nullcontext()

        timing = {}

        # [B, T, 560], [B]
        t0 = time.time()
        with record_function("extract_fbank"), nvtx_range("extract_fbank"):
            if gpu_fbank_extractor is not None:
                # Use GPU-based kaldifeat extractor
                speech, speech_lengths = gpu_fbank_extractor(batch_wavs)
                speech = speech.to(device=args.device).contiguous()
                speech_lengths = speech_lengths.to(args.device).contiguous()
            else:
                # Use CPU-based extract_fbank
                speech, speech_lengths = extract_fbank(
                    batch_wavs,
                    frontend=frontend,
                    is_final=True,
                )
                speech = speech.to(device=args.device).contiguous()
                speech_lengths = speech_lengths.to(args.device).contiguous()

            # torch.cuda.synchronize()  # Ensure GPU operations complete for accurate timing
        timing['extract_fbank'] = time.time() - t0

        t0 = time.time()
        with record_function("audio_encoder"), nvtx_range("audio_encoder"):
            encoder_out, encoder_out_lens = model.audio_encoder(
                speech, speech_lengths
            )
            # torch.cuda.synchronize()  # Ensure GPU operations complete for accurate timing
        timing['encoder'] = time.time() - t0

        t0 = time.time()
        with record_function("audio_adaptor"), nvtx_range("audio_adaptor"):
            encoder_out, encoder_out_lens = model.audio_adaptor(
                encoder_out, encoder_out_lens
            )
            # torch.cuda.synchronize()  # Ensure GPU operations complete for accurate timing
        timing['adaptor'] = time.time() - t0

        t0 = time.time()
        with record_function("prepare_embeddings"), nvtx_range("prepare_embeddings"):
            input_embeddings_list = []
            for i in range(len(batch_wavs)):
                speech_embedding = encoder_out[i, :encoder_out_lens[i], :]
                input_embedding = torch.cat([prompt_prefix_embeddings, speech_embedding, prompt_suffix_embeddings], dim=0)
                input_embeddings_list.append(input_embedding)
        timing['embedding'] = time.time() - t0

        t0 = time.time()
        with record_function("llm_generate"), nvtx_range("llm_generate"):
            if args.enable_mock_llm:
                # Mock LLM: skip actual LLM inference, return dummy response
                response = ["mock_output"] * len(batch_wavs)
            elif hasattr(model, "vllm"):
                outputs = model.vllm.generate([{
                    "prompt_embeds": input_embeddings_list[i],
                } for i in range(len(input_embeddings_list))],
                    model.vllm_sampling_params,
                    use_tqdm=False,
                )
                response = [output.outputs[0].text for output in outputs]
            else:
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
        timing['llm'] = time.time() - t0

        return response, timing

    # Profiler schedule: wait=1, warmup=2, active=3, repeat=1
    # Total batches traced: 1 (wait) + 2 (warmup) + 3 (active) = 6 batches
    if args.enable_profiler:
        os.makedirs(args.profiler_output_dir, exist_ok=True)
        profiler_schedule = schedule(
            wait=1,      # First batch is wait step
            warmup=2,    # Two warmup steps
            active=3,    # Trace three steps
            repeat=1     # Run once
        )

        def trace_handler(prof):
            """Export trace for perfetto visualization."""
            # Export gzip compressed trace (perfetto supports .json.gz directly)
            output_path = os.path.join(
                args.profiler_output_dir,
                f"trace_{prof.step_num}.json.gz"
            )
            prof.export_chrome_trace(output_path)
            print(f"Profiler trace saved to {output_path}")

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=profiler_schedule,
            on_trace_ready=trace_handler,
            record_shapes=False,    # 关闭可减小文件
            profile_memory=False,   # 关闭可减小文件
            with_stack=False,       # 关闭可大幅减小文件
            with_flops=False,       # 关闭可减小文件
        ) as prof:
            iterator = tqdm(dataloader)
            data_start = time.time()
            for batch_ids, batch_wavs, batch_refs in iterator:
                data_end = time.time()
                total_datacollator_time += data_end - data_start

                infer_start = time.time()
                response, timing = run_inference_step(batch_ids, batch_wavs, batch_refs)
                infer_end = time.time()
                total_inference_time += infer_end - infer_start
                total_extract_fbank_time += timing['extract_fbank']
                total_encoder_time += timing['encoder']
                total_adaptor_time += timing['adaptor']
                total_embedding_time += timing['embedding']
                total_llm_time += timing['llm']

                for cut_id, ref, hyp in zip(batch_ids, batch_refs, response):
                    results.append((cut_id, ref, hyp))

                batch_count += 1
                print(response)
                prof.step()
                data_start = time.time()

        # Print profiler summary
        print("\n" + "="*80)
        print("Profiler Summary (sorted by CUDA time):")
        print("="*80)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

        # Save detailed summary to file
        summary_path = os.path.join(args.profiler_output_dir, "profiler_summary.txt")
        with open(summary_path, "w") as f:
            f.write("Profiler Summary (sorted by CUDA time):\n")
            f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=50))
            f.write("\n\nProfiler Summary (sorted by CPU time):\n")
            f.write(prof.key_averages().table(sort_by="cpu_time_total", row_limit=50))
        print(f"Profiler summary saved to {summary_path}")
    elif args.enable_nvtx:
        # NVTX profiling mode for Nsight Systems
        nvtx_start = args.nvtx_wait + args.nvtx_warmup
        nvtx_end = nvtx_start + args.nvtx_active
        print(f"NVTX: wait={args.nvtx_wait}, warmup={args.nvtx_warmup}, active={args.nvtx_active}")
        print(f"NVTX: tracing steps {nvtx_start} to {nvtx_end - 1}")

        iterator = tqdm(dataloader)
        data_start = time.time()
        for step_idx, (batch_ids, batch_wavs, batch_refs) in enumerate(iterator):
            data_end = time.time()
            total_datacollator_time += data_end - data_start

            use_nvtx = nvtx_start <= step_idx < nvtx_end

            if use_nvtx:
                torch.cuda.nvtx.range_push(f"step_{step_idx}")

            infer_start = time.time()
            response, timing = run_inference_step(batch_ids, batch_wavs, batch_refs, use_nvtx=use_nvtx)
            infer_end = time.time()
            total_inference_time += infer_end - infer_start
            total_extract_fbank_time += timing['extract_fbank']
            total_encoder_time += timing['encoder']
            total_adaptor_time += timing['adaptor']
            total_embedding_time += timing['embedding']
            total_llm_time += timing['llm']

            if use_nvtx:
                torch.cuda.nvtx.range_pop()

            for cut_id, ref, hyp in zip(batch_ids, batch_refs, response):
                results.append((cut_id, ref, hyp))

            batch_count += 1
            print(response)
            data_start = time.time()

            # Early exit after NVTX tracing is done
            if step_idx >= nvtx_end - 1:
                print(f"NVTX tracing complete after step {step_idx}, exiting early.")
                break
    else:
        # Original inference loop without profiler
        iterator = tqdm(dataloader)
        data_start = time.time()
        for batch_ids, batch_wavs, batch_refs in iterator:
            data_end = time.time()
            total_datacollator_time += data_end - data_start

            infer_start = time.time()
            response, timing = run_inference_step(batch_ids, batch_wavs, batch_refs)
            infer_end = time.time()
            total_inference_time += infer_end - infer_start
            total_extract_fbank_time += timing['extract_fbank']
            total_encoder_time += timing['encoder']
            total_adaptor_time += timing['adaptor']
            total_embedding_time += timing['embedding']
            total_llm_time += timing['llm']

            for cut_id, ref, hyp in zip(batch_ids, batch_refs, response):
                results.append((cut_id, ref, hyp))

            batch_count += 1
            print(response)
            data_start = time.time()

    end_time = time.time()
    total_time = end_time - start_time

    # Calculate inference overhead
    total_module_time = total_extract_fbank_time + total_encoder_time + total_adaptor_time + total_embedding_time + total_llm_time
    inference_overhead = total_inference_time - total_module_time

    print(f"\n{'='*70}")
    print(f"Timing Summary ({batch_count} batches):")
    print(f"{'='*70}")
    print(f"Total time:              {total_time:.3f} seconds")
    print(f"DataCollator time:       {total_datacollator_time:.3f} seconds ({100*total_datacollator_time/total_time:.1f}%)")
    print(f"Inference time:          {total_inference_time:.3f} seconds ({100*total_inference_time/total_time:.1f}%)")
    print(f"  ├─ extract_fbank:      {total_extract_fbank_time:.3f} seconds ({100*total_extract_fbank_time/total_time:.1f}%)")
    print(f"  ├─ encoder:            {total_encoder_time:.3f} seconds ({100*total_encoder_time/total_time:.1f}%)")
    print(f"  ├─ audio_adaptor:      {total_adaptor_time:.3f} seconds ({100*total_adaptor_time/total_time:.1f}%)")
    print(f"  ├─ embedding:          {total_embedding_time:.3f} seconds ({100*total_embedding_time/total_time:.1f}%)")
    print(f"  ├─ llm:                {total_llm_time:.3f} seconds ({100*total_llm_time/total_time:.1f}%)")
    print(f"  └─ inference_overhead: {inference_overhead:.3f} seconds ({100*inference_overhead/total_time:.1f}%)")
    print(f"Other overhead:          {total_time - total_datacollator_time - total_inference_time:.3f} seconds")
    print(f"{'='*70}")

    # Normalize results after timing (not counted in inference time)
    print("Normalizing results...")
    results = [(cut_id, normalize_text(ref).upper(), normalize_text(hyp).upper())
               for cut_id, ref, hyp in results]

    os.makedirs(args.log_dir, exist_ok=True)

    # write to file
    with open(os.path.join(args.log_dir, "inference_time.txt"), "w") as f:
        f.write(f"Timing Summary ({batch_count} batches):\n")
        f.write(f"{'='*70}\n")
        f.write(f"Total time:              {total_time:.3f} seconds\n")
        f.write(f"DataCollator time:       {total_datacollator_time:.3f} seconds ({100*total_datacollator_time/total_time:.1f}%)\n")
        f.write(f"Inference time:          {total_inference_time:.3f} seconds ({100*total_inference_time/total_time:.1f}%)\n")
        f.write(f"  ├─ extract_fbank:      {total_extract_fbank_time:.3f} seconds ({100*total_extract_fbank_time/total_time:.1f}%)\n")
        f.write(f"  ├─ encoder:            {total_encoder_time:.3f} seconds ({100*total_encoder_time/total_time:.1f}%)\n")
        f.write(f"  ├─ audio_adaptor:      {total_adaptor_time:.3f} seconds ({100*total_adaptor_time/total_time:.1f}%)\n")
        f.write(f"  ├─ embedding:          {total_embedding_time:.3f} seconds ({100*total_embedding_time/total_time:.1f}%)\n")
        f.write(f"  ├─ llm:                {total_llm_time:.3f} seconds ({100*total_llm_time/total_time:.1f}%)\n")
        f.write(f"  └─ inference_overhead: {inference_overhead:.3f} seconds ({100*inference_overhead/total_time:.1f}%)\n")
        f.write(f"Other overhead:          {total_time - total_datacollator_time - total_inference_time:.3f} seconds\n")
        f.write(f"{'='*70}\n")

    output_path = os.path.join(args.log_dir, args.output_file)
    stats_path = os.path.join(args.log_dir, args.stats_file)
    
    print(f"Saving transcripts to {output_path}...")
    store_transcripts(output_path, results)
    
    print(f"Saving error stats to {stats_path}...")
    with open(stats_path, "w") as f:
        write_error_stats(f, args.huggingface_dataset, results)
    
    print("Done.")
if __name__ == "__main__":
    main()