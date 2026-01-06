import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, Tuple, List, TextIO, Dict
from collections import defaultdict
import logging
import torch
import datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchaudio
import numpy as np
import kaldialign
import unicodedata
import re
import time
from tn.chinese.normalizer import Normalizer as ZhNormalizer

# Add current directory to sys.path to ensure model can be imported
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

from model import FunASRNano
from audio_encoder_tensorrt import load_trt_audio_encoder

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
        required=True,
        help="Dataset name"
    )
    parser.add_argument(
        "--subset_name", 
        type=str, 
        default=None, 
        help="Dataset subset name"
    )
    parser.add_argument(
        "--split_name", 
        type=str, 
        default="test", 
        help="Dataset split name"
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
        default=1,
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

def main():
    args = get_args()
    
    logging.basicConfig(level=logging.INFO)
    
    device = args.device
    print(f"Loading model from {args.model_dir} on {device}...")

    m, kwargs = FunASRNano.from_pretrained(model=args.model_dir, device=device)
    m.eval()
    if args.encoder_trt_engine:
        load_trt_audio_encoder(m, args.encoder_trt_engine)
    if args.vllm_model_dir is not None:
        from vllm import LLM, SamplingParams
        vllm = LLM(model=args.vllm_model_dir, enable_prompt_embeds=True, gpu_memory_utilization=0.4, dtype="bfloat16")
        sampling_params = SamplingParams(
            top_p=0.001,
            max_tokens=500,
        )
        m.vllm = vllm
        m.vllm_sampling_params = sampling_params

    print(f"Loading dataset: {args.huggingface_dataset} split: {args.split_name}")
    
    # Chinese text normalizer (cached globally)
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
        # batch_wavs is a list of tensors
        

        # inference expects a list of inputs (paths or tensors)
        # FunASRNano implementation only processes the first item in the batch
        # So we loop over the batch and process one by one
        for i, wav in enumerate(batch_wavs):
            # m.inference returns (results, meta_data) tuple

            res_list, _ = m.inference(data_in=[wav], **kwargs)
            # res_list contains results for the inputs we passed
            item = res_list[0]
            hyp = item["text"]
            hyp = normalize_text(hyp).upper()
            
            cut_id = batch_ids[i]
            ref = batch_refs[i]
            ref = normalize_text(ref).upper()

            results.append((cut_id, ref, hyp))
                


    # Gather results from all ranks
    end_time = time.time()
    print(f"Inference time: {end_time - start_time} seconds")
    
    os.makedirs(args.log_dir, exist_ok=True)
    
    # write to file
    with open(os.path.join(args.log_dir, "inference_time.txt"), "w") as f:
        f.write(f"Inference time: {end_time - start_time} seconds")

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
