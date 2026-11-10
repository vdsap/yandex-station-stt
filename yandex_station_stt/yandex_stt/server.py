"""Wyoming protocol server for Yandex Station Midi offline STT."""

import argparse
import asyncio
import logging
import os
import sys
from functools import partial
from typing import Optional
import numpy as np

from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStop
from wyoming.event import Event
from wyoming.info import AsrModel, AsrProgram, Attribution, Describe, Info
from wyoming.server import AsyncEventHandler, AsyncServer

# Ensure local package is importable
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from yandex_stt.engine import StreamingRnntRecognizer

_LOGGER = logging.getLogger(__name__)


class YandexWyomingEventHandler(AsyncEventHandler):
    """Handles incoming Wyoming protocol events for Speech-to-Text."""

    def __init__(
        self,
        recognizer: StreamingRnntRecognizer,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.recognizer = recognizer
        self._audio_buffer = bytearray()
        self._is_transcribing = False

    async def handle_event(self, event: Event) -> bool:
        """Processes Wyoming protocol events."""
        if Describe.is_type(event.type):
            info = Info(
                asr=[
                    AsrProgram(
                        name="yandex-station-stt",
                        description="Yandex Station Midi Conformer RNN-T offline ASR",
                        attribution=Attribution(
                            name="Yandex",
                            url="https://github.com/vdsap/yandex-station-stt",
                        ),
                        installed=True,
                        version="1.0.0",
                        models=[
                            AsrModel(
                                name="yandex-midi-conformer-rnnt",
                                description="Russian Conformer RNN-T (17 MB)",
                                attribution=Attribution(
                                    name="Yandex",
                                    url="https://github.com/vdsap/yandex-station-stt",
                                ),
                                installed=True,
                                languages=["ru"],
                                version="1.0.2",
                            )
                        ],
                    )
                ]
            )
            await self.write_event(info.event())
            _LOGGER.debug("Sent Describe response")
            return True

        if Transcribe.is_type(event.type):
            self._audio_buffer.clear()
            self._is_transcribing = True
            _LOGGER.debug("Transcribe started")
            return True

        if AudioChunk.is_type(event.type):
            if self._is_transcribing:
                chunk = AudioChunk.from_event(event)
                self._audio_buffer.extend(chunk.audio)
            return True

        if AudioStop.is_type(event.type):
            _LOGGER.debug("AudioStop received, audio bytes: %d", len(self._audio_buffer))
            text = ""
            if len(self._audio_buffer) > 0:
                # 16-bit PCM mono expected
                samples = np.frombuffer(bytes(self._audio_buffer), dtype=np.int16)
                # Run transcription in thread pool so asyncio event loop stays responsive
                loop = asyncio.get_running_loop()
                text = await loop.run_in_executor(
                    None, self.recognizer.transcribe_audio, samples
                )
                _LOGGER.info("Recognized: '%s'", text)

            await self.write_event(Transcript(text=text).event())
            self._audio_buffer = bytearray()
            self._is_transcribing = False
            return False  # End session

        return True


async def async_main() -> None:
    parser = argparse.ArgumentParser(
        description="Wyoming server for Yandex Station Midi STT"
    )
    parser.add_argument(
        "--uri",
        type=str,
        default="tcp://0.0.0.0:10300",
        help="Server URI (e.g. tcp://0.0.0.0:10300)",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=os.path.join(parent_dir, "models"),
        help="Path to directory with models and subword_dict.txt",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    _LOGGER.info("Initializing Yandex ASR engine from %s...", args.models_dir)
    recognizer = StreamingRnntRecognizer(models_dir=args.models_dir)
    _LOGGER.info("Engine initialized successfully.")

    _LOGGER.info("Starting Wyoming server on %s...", args.uri)
    server = AsyncServer.from_uri(args.uri)

    try:
        await server.run(partial(YandexWyomingEventHandler, recognizer))
    except asyncio.CancelledError:
        pass
    finally:
        _LOGGER.info("Server stopped.")


def main() -> None:
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
