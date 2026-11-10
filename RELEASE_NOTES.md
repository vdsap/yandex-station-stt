# Release v1.0.0: Yandex Station Midi Offline STT

Первый официальный релиз автономной модели распознавания речи из Яндекс Станции Миди (Streaming Conformer RNN-T, 17.3 МБ).

### Что внутри
- **1-Click Home Assistant OS Add-on**: Автономный контейнер с Wyoming STT сервером (порт 10300) на Python 3.12 с аппаратным XNNPACK делегатом.
- **HACS Integration**: Кастомный компонент `custom_components/yandex_station_stt` с поддержкой Wyoming протокола и автоконфигурацией.
- **Standalone PC CLI Runner**: `run_asr.py` для автономного распознавания WAV аудиофайлов на ПК без интернета (RTF ~0.013x, ~77x быстрее реального времени на CPU).
- **Чистый Mel Fbank экстрактор**: 80-bin Kaldi-совместимый банк фильтров на чистом NumPy (16 кГц, 20 мс окно, 10 мс шаг, pre-emphasis 0.97).
- **Словарь русских BPE токенов**: 998 русских подслов + спецтокены (blank ID = 1003).

### Прикрепленные ассеты
- `yandex_station_stt.zip` — архив аддона и кастомного компонента для Home Assistant.
- `yandex_stt_models.zip` — оригинальные FP32 TFLite модели и словарь токенов для автономного запуска на ПК.
