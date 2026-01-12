#!/usr/bin/env python3
"""
gRPC Client for FunASR Triton Server.
"""

import argparse
import numpy as np
import soundfile as sf
import tritonclient.grpc as grpcclient
from tritonclient.utils import np_to_triton_dtype


def load_audio(audio_path: str, target_sr: int = 16000) -> np.ndarray:
    """
    Load audio file and resample if necessary.

    Args:
        audio_path: Path to audio file
        target_sr: Target sample rate

    Returns:
        Audio samples as float32 numpy array
    """
    import librosa
    audio, sr = librosa.load(audio_path, sr=target_sr)
    return audio.astype(np.float32)


def transcribe(
    audio_path: str,
    server_url: str = "localhost:8001",
    model_name: str = "funasr"
) -> str:
    """
    Transcribe audio using the Triton server via gRPC.

    Args:
        audio_path: Path to audio file
        server_url: Triton server URL (host:port)
        model_name: Name of the model

    Returns:
        Transcription text
    """
    # Load audio
    audio = load_audio(audio_path)

    # Create client
    client = grpcclient.InferenceServerClient(url=server_url)

    # Prepare inputs
    wav_input = grpcclient.InferInput("WAV", [1, len(audio)], "FP32")
    wav_input.set_data_from_numpy(audio.reshape(1, -1))

    wav_lens_input = grpcclient.InferInput("WAV_LENS", [1, 1], "INT32")
    wav_lens_input.set_data_from_numpy(np.array([[len(audio)]], dtype=np.int32))

    # Prepare outputs
    transcript_output = grpcclient.InferRequestedOutput("TRANSCRIPTS")

    # Run inference
    response = client.infer(
        model_name=model_name,
        inputs=[wav_input, wav_lens_input],
        outputs=[transcript_output]
    )

    # Get result
    transcript = response.as_numpy("TRANSCRIPTS")[0][0]
    if isinstance(transcript, bytes):
        transcript = transcript.decode("utf-8")

    return transcript


def main():
    parser = argparse.ArgumentParser(description="FunASR Triton gRPC Client")
    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="Path to audio file"
    )
    parser.add_argument(
        "--server",
        type=str,
        default="localhost:8001",
        help="Triton server URL (default: localhost:8001)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="funasr",
        help="Model name (default: funasr)"
    )

    args = parser.parse_args()

    print(f"Transcribing: {args.audio}")
    transcript = transcribe(args.audio, args.server, args.model)
    print(f"Transcript: {transcript}")


if __name__ == "__main__":
    main()
