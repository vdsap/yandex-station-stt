"""Conformer RNN-Transducer inference engine for Yandex Station Midi offline STT."""

import os
from typing import Generator, List, Optional
import numpy as np

try:
    import ai_edge_litert.interpreter as tflite
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
    except ImportError:
        raise ImportError(
            "Neither ai_edge_litert nor tflite_runtime is installed. "
            "Install with: pip install ai-edge-litert"
        )

from .features import KaldiFbankExtractor
from .vocab import SubwordVocab


class StreamingRnntRecognizer:
    """Streaming Conformer RNN-Transducer Speech Recognizer.

    Architecture:
    - Acoustic Encoder: 10-layer Streaming Conformer, 8 heads, dim=512, subsampling=4.
      Chunk input: [1, 52, 80] log-mel filterbanks.
      Output: [1, 10, 512] joint projections + updated KV-cache.
    - Prediction Network (Decoder): 2-layer LSTM (192 units) + projection (512 units).
      Input: Token ID [1], LSTM state [1, 4, 192].
      Output: Embedding [1, 512], updated state.
    - Joint Network: Linear(ReLU(h_enc + h_pred)) -> LogSoftmax(1004).
      Output: Log-probabilities over 1004 vocabulary tokens.
    """

    CHUNK_SIZE = 52
    STRIDE = 40
    LEFT_PAD = 12
    MAX_SYMBOLS_PER_FRAME = 4

    def __init__(self, models_dir: str) -> None:
        self.models_dir = models_dir
        self.encoder_path = os.path.join(models_dir, "encoder_fp32_opt_pc.tflite")
        self.decoder_path = os.path.join(models_dir, "decoder_fp32_opt_pc.tflite")
        self.joint_path = os.path.join(models_dir, "joint_fp32_opt_pc.tflite")
        self.dict_path = os.path.join(models_dir, "subword_dict.txt")

        # Load vocabulary
        self.vocab = SubwordVocab(self.dict_path)

        # Initialize feature extractor
        self.fbank_extractor = KaldiFbankExtractor()

        # Load TFLite models
        self.enc_interp = tflite.Interpreter(model_path=self.encoder_path)
        self.enc_interp.allocate_tensors()

        self.dec_interp = tflite.Interpreter(model_path=self.decoder_path)
        self.dec_interp.allocate_tensors()

        self.jnt_interp = tflite.Interpreter(model_path=self.joint_path)
        self.jnt_interp.allocate_tensors()

        # Cache input/output indices
        self.enc_in = {d["name"]: d["index"] for d in self.enc_interp.get_input_details()}
        self.enc_out = {d["name"]: d["index"] for d in self.enc_interp.get_output_details()}

        self.dec_in = {d["name"]: d["index"] for d in self.dec_interp.get_input_details()}
        self.dec_out = {d["name"]: d["index"] for d in self.dec_interp.get_output_details()}

        self.jnt_in = {d["name"]: d["index"] for d in self.jnt_interp.get_input_details()}
        self.jnt_out = {d["name"]: d["index"] for d in self.jnt_interp.get_output_details()}

        self.reset()

    def reset(self) -> None:
        """Resets recognizer state for a new speech utterance."""
        self.kv_cache = np.zeros((1, 10, 8, 80, 64), dtype=np.float32)
        self.dec_state = np.zeros((1, 4, 192), dtype=np.float32)
        self.chunk_pos = 0
        self.emitted_tokens: List[int] = []

        # Initialize decoder with blank/BOS token
        self._run_decoder_step(self.vocab.BLANK_ID)

    def _run_decoder_step(self, token_id: int) -> None:
        """Executes a single step of the RNN prediction network."""
        self.dec_interp.set_tensor(
            self.dec_in["serving_default_in_0_token:0"],
            np.array([token_id], dtype=np.int32),
        )
        self.dec_interp.set_tensor(
            self.dec_in["serving_default_in_1_state:0"],
            self.dec_state,
        )
        self.dec_interp.invoke()

        self.dec_proj = self.dec_interp.get_tensor(
            self.dec_out["StatefulPartitionedCall:0"]
        )
        self.dec_state = self.dec_interp.get_tensor(
            self.dec_out["StatefulPartitionedCall:1"]
        )

    def process_chunk(self, chunk_fbanks: np.ndarray) -> List[int]:
        """Processes a single 52-frame Mel Fbank chunk and returns newly emitted tokens.

        Args:
            chunk_fbanks: Array of shape [52, 80] or [1, 52, 80].

        Returns:
            List of newly emitted token IDs.
        """
        if chunk_fbanks.ndim == 2:
            chunk_fbanks = chunk_fbanks[np.newaxis, ...]

        self.enc_interp.set_tensor(
            self.enc_in["serving_default_in_0_fbanks:0"], chunk_fbanks
        )
        self.enc_interp.set_tensor(
            self.enc_in["serving_default_in_1_chunk_pos:0"],
            np.array([self.chunk_pos], dtype=np.int32),
        )
        self.enc_interp.set_tensor(
            self.enc_in["serving_default_in_2_kv_cache:0"], self.kv_cache
        )
        self.enc_interp.invoke()

        # Tensor 453 is out_2_joint_proj [1, 10, 512]
        enc_proj = self.enc_interp.get_tensor(
            self.enc_out["StatefulPartitionedCall:2"]
        )
        # Tensor 413 is updated kv_cache [1, 10, 8, 80, 64]
        self.kv_cache = self.enc_interp.get_tensor(
            self.enc_out["StatefulPartitionedCall:1"]
        )
        self.chunk_pos += 1

        new_tokens: List[int] = []

        # Greedy decoding over 10 output frames
        for t in range(10):
            enc_frame = enc_proj[:, t, :]

            for _ in range(self.MAX_SYMBOLS_PER_FRAME):
                self.jnt_interp.set_tensor(
                    self.jnt_in["serving_default_in_0_encoder_out:0"], enc_frame
                )
                self.jnt_interp.set_tensor(
                    self.jnt_in["serving_default_in_1_decoder_out:0"], self.dec_proj
                )
                self.jnt_interp.invoke()

                logits = self.jnt_interp.get_tensor(
                    self.jnt_out["StatefulPartitionedCall:0"]
                )[0]
                best_token = int(np.argmax(logits))

                if best_token == self.vocab.BLANK_ID or best_token >= 1000:
                    break

                new_tokens.append(best_token)
                self.emitted_tokens.append(best_token)
                self._run_decoder_step(best_token)

        return new_tokens

    def transcribe_audio(self, samples: np.ndarray) -> str:
        """Transcribes complete 16kHz mono audio into Russian text.

        Args:
            samples: Audio samples (int16 or float32, 16000 Hz, mono).

        Returns:
            Transcribed text.
        """
        self.reset()
        fbanks = self.fbank_extractor.extract(samples)

        if len(fbanks) == 0:
            return ""

        # Prepend receptive field left padding (12 frames)
        pad = np.repeat(fbanks[:1], self.LEFT_PAD, axis=0)
        fbanks = np.concatenate([pad, fbanks], axis=0)

        # Ensure fbanks has at least CHUNK_SIZE
        if len(fbanks) < self.CHUNK_SIZE:
            fbanks = np.pad(
                fbanks, ((0, self.CHUNK_SIZE - len(fbanks)), (0, 0))
            )

        # Pad right margin to complete whole stride chunks
        pad_len = (
            self.STRIDE - ((len(fbanks) - self.CHUNK_SIZE) % self.STRIDE)
        ) % self.STRIDE
        if pad_len > 0:
            fbanks = np.pad(fbanks, ((0, pad_len + self.STRIDE), (0, 0)))

        num_chunks = (len(fbanks) - self.CHUNK_SIZE) // self.STRIDE + 1

        for c in range(num_chunks):
            start = c * self.STRIDE
            chunk = fbanks[start : start + self.CHUNK_SIZE]
            self.process_chunk(chunk)

        return self.vocab.decode(self.emitted_tokens)
