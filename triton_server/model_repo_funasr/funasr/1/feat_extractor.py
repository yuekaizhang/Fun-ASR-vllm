"""
Feature extractor for FunASR using kaldifeat.
Extracts fbank features and applies Low Frame Rate (LFR) transformation.
"""

import numpy as np
import torch
import kaldifeat
from torch.nn.utils.rnn import pad_sequence


class FunASRFeatExtractor:
    """
    Feature extractor that produces fbank features compatible with FunASR models.
    Uses kaldifeat for efficient GPU-based feature extraction.
    """

    def __init__(self, sample_rate: int = 16000, device_id: int = 0):
        """
        Initialize the feature extractor.

        Args:
            sample_rate: Audio sample rate (default: 16000)
            device_id: GPU device ID for feature extraction
        """
        self.sample_rate = sample_rate
        self.device_id = device_id

        # Configure kaldifeat options
        self.opts = kaldifeat.FbankOptions()
        self.opts.device = torch.device('cuda', device_id)
        self.opts.frame_opts.dither = 0.0
        self.opts.mel_opts.num_bins = 80
        self.opts.frame_opts.frame_shift_ms = 10
        self.opts.frame_opts.frame_length_ms = 25
        self.opts.frame_opts.samp_freq = sample_rate
        self.opts.frame_opts.snip_edges = False
        self.opts.frame_opts.window_type = "hamming"

        self.fbank = kaldifeat.Fbank(self.opts)

    @staticmethod
    def apply_lfr(features: torch.Tensor, lengths: torch.Tensor,
                  window_size: int = 7, window_shift: int = 6,
                  max_len: int = -1) -> tuple:
        """
        Apply Low Frame Rate (LFR) transformation to features.

        Args:
            features: Input features [B, T, D]
            lengths: Input lengths [B]
            window_size: LFR window size (default: 7)
            window_shift: LFR window shift (default: 6)
            max_len: Maximum output length (-1 for no limit)

        Returns:
            Tuple of (features_lfr [B, T_lfr, D*window_size], lengths_lfr [B])
        """
        B, T, D = features.shape

        # Calculate output lengths
        lengths_lfr = (lengths - window_size) // window_shift + 1
        lengths_lfr = lengths_lfr.clamp(min=0)

        # Pad if needed
        if T < window_size:
            pad_amt = window_size - T
            features = torch.nn.functional.pad(features, (0, 0, 0, pad_amt))
            T = window_size

        # Unfold to create windows
        features_unfolded = features.unfold(1, window_size, window_shift)

        # Permute and flatten
        features_unfolded = features_unfolded.permute(0, 1, 3, 2)
        features_unfolded = features_unfolded.contiguous()
        features_lfr = features_unfolded.reshape(B, features_unfolded.size(1), -1)

        # Handle max_len
        if max_len > 0:
            cur_len = features_lfr.size(1)
            if cur_len > max_len:
                features_lfr = features_lfr[:, :max_len, :]
                lengths_lfr = lengths_lfr.clamp(max=max_len)
            elif cur_len < max_len:
                pad_amt = max_len - cur_len
                features_lfr = torch.nn.functional.pad(features_lfr, (0, 0, 0, pad_amt))

        return features_lfr, lengths_lfr

    def __call__(self, wavs: list, max_len: int = -1,
                 window_size: int = 7, window_shift: int = 6) -> tuple:
        """
        Extract features from a list of waveforms.

        Args:
            wavs: List of waveform tensors
            max_len: Maximum output sequence length
            window_size: LFR window size
            window_shift: LFR window shift

        Returns:
            Tuple of (features [B, T, D], lengths [B])
        """
        # Prepare waveforms
        samples_list = []
        for wav in wavs:
            if isinstance(wav, np.ndarray):
                wav_tensor = torch.from_numpy(wav).float()
            elif isinstance(wav, torch.Tensor):
                wav_tensor = wav.float()
            else:
                raise TypeError("wav must be tensor or numpy array")

            if wav_tensor.ndim > 1:
                wav_tensor = wav_tensor.squeeze()

            samples_list.append(wav_tensor)

        # Move to device and scale (kaldifeat expects int16 range)
        samples_device = [
            s.to(self.opts.device, dtype=torch.float32) * 32768.0
            for s in samples_list
        ]

        # Extract fbank features
        features_list = self.fbank(samples_device)

        # Get lengths
        lengths = torch.tensor(
            [f.size(0) for f in features_list],
            dtype=torch.long,
            device=self.opts.device
        )

        # Pad to create batch
        features_padded = pad_sequence(features_list, batch_first=True, padding_value=0.0)

        # Apply LFR
        features_lfr, lengths_lfr = self.apply_lfr(
            features_padded, lengths, window_size, window_shift, max_len
        )

        return features_lfr, lengths_lfr


class FunASRFeatExtractorFromFrontend:
    """
    Alternative feature extractor using FunASR's built-in frontend.
    Use this if kaldifeat is not available.
    """

    def __init__(self, frontend, device: str = "cuda:0"):
        """
        Initialize with FunASR frontend.

        Args:
            frontend: FunASR frontend object
            device: Device string
        """
        self.frontend = frontend
        self.device = device

    def __call__(self, wavs: list) -> tuple:
        """
        Extract features using FunASR frontend.

        Args:
            wavs: List of waveform tensors

        Returns:
            Tuple of (features [B, T, D], lengths [B])
        """
        from funasr.utils.load_utils import extract_fbank

        speech, speech_lengths = extract_fbank(
            wavs,
            frontend=self.frontend,
            is_final=True,
        )

        return speech.to(self.device), speech_lengths.to(self.device)
