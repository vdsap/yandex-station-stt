#!/usr/bin/env python3
"""Standalone CLI runner for Yandex Station Midi offline STT model.

Usage:
    python run_asr.py --audio path/to/speech.wav
    python run_asr.py --audio path/to/speech.wav --models-dir ./models
"""

import argparse
import os
import sys
import time
import wave
import numpy as np

# Ensure local package is importable
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from yandex_stt import StreamingRnntRecognizer


def load_wav(wav_path: str) -> tuple[np.ndarray, int]:
    """Loads a 16kHz mono PCM WAV file."""
    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"Audio file not found: {wav_path}")

    with wave.open(wav_path, "rb") as w:
        channels = w.getnchannels()
        sampwidth = w.getsampwidth()
        framerate = w.getframerate()
        n_frames = w.getnframes()
        raw = w.readframes(n_frames)

    if sampwidth != 2:
        raise ValueError(
            f"Unsupported sample width: {sampwidth * 8}-bit. Expected 16-bit PCM."
        )

    samples = np.frombuffer(raw, dtype=np.int16)

    # Convert stereo to mono if needed
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1).astype(np.int16)

    return samples, framerate


def main() -> None:
    default_models = os.path.join(current_dir, "models")
    if not os.path.exists(default_models):
        default_models = os.path.join(current_dir, "yandex_station_stt", "models")

    parser = argparse.ArgumentParser(
        description="Offline speech recognition using Yandex Station Midi Conformer RNN-T model."
    )
    parser.add_argument(
        "--audio",
        "-a",
        type=str,
        required=True,
        help="Path to input audio file (.wav 16kHz mono).",
    )
    parser.add_argument(
        "--models-dir",
        "-m",
        type=str,
        default=default_models,
        help="Path to directory containing .tflite models and subword_dict.txt.",
    )
    args = parser.parse_args()

    print(f"Loading models from: {args.models_dir}")
    t0 = time.perf_counter()
    recognizer = StreamingRnntRecognizer(models_dir=args.models_dir)
    load_time = time.perf_counter() - t0
    print(f"Models initialized in {load_time:.2f}s")

    print(f"Reading audio: {args.audio}")
    samples, sr = load_wav(args.audio)
    duration = len(samples) / sr
    print(f"Audio duration: {duration:.2f}s, sample rate: {sr} Hz")

    if sr != 16000:
        print(
            f"Warning: Audio sample rate is {sr} Hz. The model expects 16000 Hz. "
            "Please resample with ffmpeg: ffmpeg -i input.wav -ar 16000 -ac 1 output.wav"
        )

    print("Transcribing...")
    t1 = time.perf_counter()
    text = recognizer.transcribe_audio(samples)
    infer_time = time.perf_counter() - t1
    rtf = infer_time / duration if duration > 0 else 0.0

    print("\n" + "=" * 60)
    print(f"TRANSCRIPTION: \"{text}\"")
    print("=" * 60)
    print(f"Inference time: {infer_time:.2f}s (RTF: {rtf:.3f}x real-time)")


if __name__ == "__main__":
    main()
