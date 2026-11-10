"""Kaldi 80-bin Mel Filterbank feature extraction module for Yandex Conformer RNN-T."""

import numpy as np


class KaldiFbankExtractor:
    """Computes 80-bin log Mel filterbank features matching Yandex Quasar ASR specs.

    Parameters match features_config.json:
    - sample_rate: 16000 Hz
    - frame_length_ms: 20.0 ms (320 samples)
    - frame_shift_ms: 10.0 ms (160 samples)
    - num_mel_bins: 80
    - preemphasis: 0.97
    - window: Hamming
    - n_fft: 512
    - energy: power spectrum |FFT|^2
    - log_fbank: natural log
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_length_ms: float = 20.0,
        frame_shift_ms: float = 10.0,
        num_mel_bins: int = 80,
        preemphasis: float = 0.97,
        remove_dc_offset: bool = True,
        low_freq: float = 0.0,
        high_freq: float = 0.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_len = int(sample_rate * frame_length_ms / 1000.0)
        self.frame_shift = int(sample_rate * frame_shift_ms / 1000.0)
        self.num_mel_bins = num_mel_bins
        self.preemphasis = preemphasis
        self.remove_dc_offset = remove_dc_offset
        self.n_fft = 512

        # Hamming window: 0.54 - 0.46 * cos(2*pi*n/(N-1))
        self.window = 0.54 - 0.46 * np.cos(
            2.0 * np.pi * np.arange(self.frame_len) / (self.frame_len - 1)
        )

        # Build Mel filterbank matrix
        if high_freq <= 0.0:
            high_freq = sample_rate / 2.0

        def hz_to_mel(hz: float) -> float:
            return 1127.0 * np.log(1.0 + hz / 700.0)

        def mel_to_hz(mel: float) -> float:
            return 700.0 * (np.exp(mel / 1127.0) - 1.0)

        low_mel = hz_to_mel(low_freq)
        high_mel = hz_to_mel(high_freq)
        mel_points = np.linspace(low_mel, high_mel, num_mel_bins + 2)
        hz_points = mel_to_hz(mel_points)

        fft_freqs = np.linspace(0, sample_rate / 2.0, self.n_fft // 2 + 1)
        self.fbank = np.zeros(
            (num_mel_bins, self.n_fft // 2 + 1), dtype=np.float32
        )

        for m in range(num_mel_bins):
            left = hz_points[m]
            center = hz_points[m + 1]
            right = hz_points[m + 2]

            for k in range(self.n_fft // 2 + 1):
                f = fft_freqs[k]
                if left < f < center:
                    self.fbank[m, k] = (f - left) / (center - left)
                elif center <= f < right:
                    self.fbank[m, k] = (right - f) / (right - center)

    def extract(self, signal: np.ndarray) -> np.ndarray:
        """Extracts [num_frames, 80] log-mel filterbank from 16kHz mono audio.

        Args:
            signal: 1D numpy array of audio samples (int16 or float32).

        Returns:
            2D numpy array of shape [num_frames, 80] with float32 log-mel features.
        """
        if signal.dtype == np.int16:
            samples = signal.astype(np.float32)
        elif signal.max() <= 1.0 and signal.min() >= -1.0:
            samples = signal * 32768.0
        else:
            samples = signal.astype(np.float32)

        if len(samples) < self.frame_len:
            return np.zeros((0, self.num_mel_bins), dtype=np.float32)

        num_frames = 1 + (len(samples) - self.frame_len) // self.frame_shift
        features = np.empty((num_frames, self.num_mel_bins), dtype=np.float32)

        for i in range(num_frames):
            start = i * self.frame_shift
            frame = samples[start : start + self.frame_len].copy()

            if self.remove_dc_offset:
                frame -= np.mean(frame)

            if self.preemphasis != 0.0:
                frame[1:] -= self.preemphasis * frame[:-1]
                frame[0] -= self.preemphasis * frame[0]

            frame *= self.window

            # Power spectrum
            fft = np.fft.rfft(frame, n=self.n_fft)
            power = np.abs(fft) ** 2

            # Mel filterbank dot product
            mel_energy = np.dot(self.fbank, power)
            mel_energy = np.maximum(mel_energy, 1e-10)
            features[i] = np.log(mel_energy)

        return features
