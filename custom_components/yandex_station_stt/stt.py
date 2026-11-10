"""Speech-to-text platform for Yandex Station STT."""

from __future__ import annotations

from collections.abc import AsyncIterable
import logging

from homeassistant.components.stt import (
    AudioBitRates,
    AudioChannels,
    AudioCodecs,
    AudioFormats,
    AudioSampleRates,
    SpeechMetadata,
    SpeechResult,
    SpeechResultState,
    SpeechToTextEntity,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_HOST, CONF_PORT
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStop
from wyoming.client import AsyncTcpClient

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Yandex STT platform from config entry."""
    host = config_entry.data[CONF_HOST]
    port = config_entry.data[CONF_PORT]
    async_add_entities([YandexSpeechToTextEntity(hass, config_entry.entry_id, host, port)])


class YandexSpeechToTextEntity(SpeechToTextEntity):
    """Yandex Station STT Entity."""

    def __init__(
        self,
        hass: HomeAssistant,
        entry_id: str,
        host: str,
        port: int,
    ) -> None:
        """Initialize the entity."""
        self.hass = hass
        self._host = host
        self._port = port
        self._attr_name = "Yandex Station STT"
        self._attr_unique_id = f"yandex_station_stt_{entry_id}"

    @property
    def supported_languages(self) -> list[str]:
        """Return list of supported languages."""
        return ["ru"]

    @property
    def supported_formats(self) -> list[AudioFormats]:
        """Return list of supported audio formats."""
        return [AudioFormats.WAV, AudioFormats.OGG]

    @property
    def supported_codecs(self) -> list[AudioCodecs]:
        """Return list of supported codecs."""
        return [AudioCodecs.PCM, AudioCodecs.OPUS]

    @property
    def supported_bit_rates(self) -> list[AudioBitRates]:
        """Return list of supported bit rates."""
        return [AudioBitRates.BITRATE_16]

    @property
    def supported_sample_rates(self) -> list[AudioSampleRates]:
        """Return list of supported sample rates."""
        return [AudioSampleRates.SAMPLERATE_16000]

    @property
    def supported_channels(self) -> list[AudioChannels]:
        """Return list of supported channels."""
        return [AudioChannels.CHANNEL_MONO]

    async def async_process_audio_stream(
        self, metadata: SpeechMetadata, stream: AsyncIterable[bytes]
    ) -> SpeechResult:
        """Process an audio stream to speech."""
        client = AsyncTcpClient(self._host, self._port)
        try:
            await client.connect()
            await client.write_event(Transcribe().event())

            async for chunk in stream:
                if not chunk:
                    continue
                audio_chunk = AudioChunk(
                    rate=metadata.sample_rate,
                    width=2,
                    channels=metadata.channel,
                    audio=chunk,
                )
                await client.write_event(audio_chunk.event())

            await client.write_event(AudioStop().event())

            # Read transcript response
            event = await client.read_event()
            if event and Transcript.is_type(event.type):
                transcript = Transcript.from_event(event)
                _LOGGER.debug("Transcription received: '%s'", transcript.text)
                return SpeechResult(
                    text=transcript.text,
                    result=SpeechResultState.SUCCESS,
                )

            return SpeechResult(
                text=None,
                result=SpeechResultState.ERROR,
            )

        except Exception as err:
            _LOGGER.error("Error processing audio stream with Yandex STT: %s", err)
            return SpeechResult(
                text=None,
                result=SpeechResultState.ERROR,
            )
        finally:
            await client.disconnect()
