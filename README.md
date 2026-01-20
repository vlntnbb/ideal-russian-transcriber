# Ideal Russian Transcriber — локальный Telegram ASR бот

В этом репозитории есть:
- `dual_transcriber_gui.py` — GUI для Whisper + GigaAM
- `local_telegram_bot.py` — локальный Telegram-бот: принимает voice/audio и присылает две транскрипции подряд

## Требования

- Python 3.9+
- `ffmpeg` в `PATH`
- Telegram bot token (`TELEGRAM_BOT_TOKEN`)

## Установка

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -r requirements.txt
```

Создайте `.env` по примеру:

```bash
cp .env.example .env
```

## Запуск бота

```bash
source .venv/bin/activate
python3 local_telegram_bot.py
```

Отправьте боту voice/audio — он вернёт:
1) `Whisper: ...`
2) `GigaAM: ...`
3) `transcript_*.md` — файл с итогом (Gemini + обе транскрипции)

### Работа в групповом чате

Чтобы бот реагировал в групповом чате (reply + @mention), выдайте ему **админские права** в этом чате.

### Логи использования

Бот пишет отдельный usage-лог (JSONL) по каждой сессии распознавания: кто запросил, в каком чате, какие модели, длительность этапов и т.п.
По умолчанию файл: `usage_sessions.jsonl` (настраивается через `USAGE_LOG_PATH`, выключается через `USAGE_LOG_ENABLED=0`).

## Обработка транскрипций (Gemini)

Бот хранит последнюю транскрипцию (Whisper + GigaAM) для каждого пользователя в чате.
Дальше можно попросить Gemini обработать её по вашему промпту:

- `/process <промпт>`
- `обработать <промпт>` (сообщением в чат)
- `/obrabotat <промпт>` (alias)
- `/process model=<gemini-model> <промпт>` (переопределить модель на один запрос)

Для включения задайте в `.env`:
- `GEMINI_API_KEY`
- `GEMINI_MODEL` (по умолчанию `gemini-1.5-flash`)
Можно также задать системный промпт для итоговой обработки:
- `GEMINI_SYSTEM_PROMPT`

## Примечания про GigaAM long-form

Модели GigaAM в PyPI по умолчанию ограничены короткими аудио (примерно до 25 секунд).
Для длинных есть long-form режим, который использует `pyannote.audio` и требует `HF_TOKEN`.
Также обратите внимание: “v3_e2e_*” модели с пунктуацией/нормализацией есть в upstream-репозитории GigaAM, но могут отсутствовать в PyPI-пакете `gigaam`.

Включение long-form:

```bash
source .venv/bin/activate
python3 -m pip install -r requirements-longform.txt
```

В `.env` задайте `HF_TOKEN` и примите условия (gated) на страницах:
- `https://hf.co/pyannote/voice-activity-detection`
- `https://hf.co/pyannote/segmentation`
