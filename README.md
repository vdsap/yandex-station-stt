<div align="center">
  <h1>🎙️ Yandex Station STT (Midi)</h1>
  <p><strong>Полностью автономное, локальное распознавание русской речи для Home Assistant Assist и ПК</strong></p>

  <p align="center">
    <a href="https://github.com/hacs/default"><img src="https://img.shields.io/badge/HACS-Custom-orange.svg" alt="HACS"></a>
    <a href="https://github.com/vdsap/yandex-station-stt/releases"><img src="https://img.shields.io/github/v/release/vdsap/yandex-station-stt" alt="Release"></a>
    <img src="https://img.shields.io/badge/Model%20Size-17.3%20MB-blue" alt="Model Size">
    <img src="https://img.shields.io/badge/RTF-0.013x%20(77x%20RT)-brightgreen" alt="Speed">
    <img src="https://img.shields.io/badge/Protocol-Wyoming-purple" alt="Wyoming">
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
  </p>

  <p align="center">
    <em>Оффлайн нейросетевая модель Streaming Conformer RNN-T, извлеченная из Яндекс Станции Миди (прошивка Quasar), адаптированная для мгновенной интеграции в голосовой ассистент Home Assistant без интернета, облаков и подписок.</em>
  </p>
</div>

---

## 🚀 Установка в Home Assistant

Репозиторий поддерживает бесшовную установку **в один клик** как через **Home Assistant Add-on Store** (рекомендуется для HAOS — запуск изолированного Wyoming-сервиса в контейнере), так и через **HACS**:

### Вариант 1. Add-on в Home Assistant OS (Рекомендуется)
Устанавливает и запускает локальный Wyoming-движок внутри HAOS:

[![Open your Home Assistant instance and show the add add-on repository dialog with a specific repository URL pre-filled.](https://my.home-assistant.io/badges/supervisor_add_addon_repository.svg)](https://my.home-assistant.io/redirect/supervisor_add_addon_repository/?repository_url=https%3A%2F%2Fgithub.com%2Fvdsap%2Fyandex-station-stt)

1. Нажмите на кнопку выше, чтобы добавить репозиторий в ваш инстанс Home Assistant.
2. В **Add-on Store** найдите **Yandex Station STT (Midi)**, нажмите **Установить** и затем **Запустить**.
3. Home Assistant автоматически обнаружит голосовой сервис благодаря механизму `discovery: wyoming`. Нажмите **Настроить** в уведомлении — и голосовой ввод готов к работе!

### Вариант 2. Установка интеграции через HACS
Если вы хотите управлять интеграцией через каталог HACS:

[![Open your Home Assistant instance and open a repository inside the Home Assistant Community Store.](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?owner=vdsap&repository=yandex-station-stt&category=integration)

1. Нажмите на кнопку выше, чтобы открыть страницу интеграции в HACS.
2. Нажмите **Скачать** и перезагрузите Home Assistant.
3. В **Настройки → Устройства и службы → Добавить интеграцию** найдите **Yandex Station STT**.

---

## ✨ Ключевые возможности

- 🔒 **100% Локально & Приватно**: Обработка звука происходит исключительно на вашем сервере или ПК. Ни единого байта аудио не отправляется во внешние облака.
- ⚡ **Ультра-низкая задержка (RTF ~0.013x)**: Инференс аудио длительностью 13.2 секунды занимает всего **0.17 секунды** на обычном x86-64 процессоре (в 77 раз быстрее реального времени).
- 🪶 **Компактный размер**: Суммарный объем всех весов модели (энкодер, декодер, joint-сеть и словарь) — всего **17.3 МБ**.
- 🎙️ **Нативная совместимость с Assist**: Полная интеграция с голосовыми сателлитами ESPHome, микрофонами, Wyoming Satellite и мобильным приложением Home Assistant.
- 💻 **Standalone запуск**: Модель можно запускать автономно из консоли на Windows, Linux или macOS.

---

## 🧠 Архитектура и устройство «под капотом»

Модель представляет собой оптимизированный бортовой распознаватель речи (ASR) Станции Миди, спроектированный инженерами Яндекса для выполнения голосовых команд при обрыве интернет-соединения.

```text
  [ Микрофон 16 кГц PCM ] 
             │
             ▼
[ Kaldi 80-bin Mel Fbank ] (20 мс окно, 10 мс шаг, Хэмминг, преэмфазис 0.97)
             │  Чанки по 52 фрейма (40 новых + 12 overlap)
             ▼
   [ Conformer Encoder ]  (10 слоев, 8 голов внимания, dim=512, subsampling=4)
             │
             │ [1, 10, 512] вектор признаков
             ▼
      [ Joint Network ] ◄─── [ 2-layer LSTM Decoder ] ◄── [ Предыдущие токены ]
             │                     (192 скрытых юнита)
             ▼
    [ LogSoftmax 1004 ]
             │
             ▼
     [ BPE Токены ] ────► [ Русский текст: "выключи свет на кухне" ]
```

### 1. Акустический энкодер (`encoder_fp32_opt_pc.tflite`, 15.28 МБ)
- Потоковый 10-слойный **Streaming Conformer** с 8 головами внимания и размерностью эмбеддинга 512.
- Принимает логарифмические Mel-спектрограммы размером `[1, 52, 80]`.
- Сохраняет историю внимания в KV-кэше размерностью `[1, 10, 8, 80, 64]`, что позволяет непрерывно распознавать поток любой длины без накопления задержки.
- Входное прореживание во времени (time striding): **4x** (каждый чанк выдает 10 сжатых фреймов).

### 2. Языковой предиктор / декодер (`decoder_fp32_opt_pc.tflite`, 920 КБ)
- 2-слойная LSTM-сеть (размерность скрытого состояния `[1, 4, 192]`) с линейной проекцией в 512 признаков.
- Принимает предыдущий распознанный подсловный токен и предсказывает языковой контекст.

### 3. Совместная сеть (`joint_fp32_opt_pc.tflite`, 520 КБ)
- Нелинейное объединение векторов энкодера и предиктора: `Linear(ReLU(h_enc + h_pred))`.
- Выходной слой LogSoftmax на **1004 класса**:
  - `ID 0`: `<unk>`
  - `ID 1..999`: 998 BPE-подслов русского языка из `subword_dict.txt` (где `999` — `▁яндекс`, `844` — `▁свет`, `507` — `▁выключай`)
  - `ID 1000..1002`: контекстные маркеры
  - `ID 1003`: `<blank>` (символ пропуска)

---

## 💻 Автономный запуск на ПК (Standalone CLI)

Модель можно скачать и запускать на любом компьютере без Home Assistant.

### 1. Установка зависимостей
```bash
git clone https://github.com/vdsap/yandex-station-stt.git
cd yandex-station-stt
pip install ai-edge-litert numpy
```

### 2. Транскрипция аудиофайла
```bash
python run_asr.py --audio path/to/speech.wav
```

Пример вывода:
```text
Loading models from: ./models
Models initialized in 0.02s
Reading audio: test.wav
Audio duration: 13.21s, sample rate: 16000 Hz
Transcribing...

============================================================
TRANSCRIPTION: "настройка завершена скажите алиса выключи свет"
============================================================
Inference time: 0.17s (RTF: 0.013x real-time)
```

---

## 🛠️ Запуск Wyoming-сервера вручную

Если вы хотите запустить сервер вне контейнера HAOS (например, в Proxmox LXC, Docker на сервере или на ПК):

```bash
pip install wyoming ai-edge-litert numpy
python -m yandex_stt.server --uri tcp://0.0.0.0:10300 --models-dir ./models
```

После этого в Home Assistant перейдите в **Настройки → Устройства и службы → Добавить интеграцию → Wyoming Protocol**, укажите IP-адрес хоста и порт `10300`.

---

## 📄 Лицензия

Код обвязки, экстрактора признаков и интеграции распространяется под лицензией **Apache 2.0**. См. файл [LICENSE](LICENSE).
Оригинальные веса акустической модели принадлежат ООО «Яндекс».
