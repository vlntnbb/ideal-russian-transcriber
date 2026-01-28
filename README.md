# Ideal Russian Transcriber — локальный Telegram ASR бот

В этом репозитории есть:
- `local_telegram_bot.py` — локальный Telegram-бот: принимает voice/audio/video, делает 2 транскрипции (Whisper + GigaAM) и формирует итог по шаблону через LLM
- `snapscript/` — обёртки для подготовки аудио и ASR

## Требования

- Python 3.9+
- `ffmpeg` в `PATH`
- Telegram bot token (`TELEGRAM_BOT_TOKEN`)
- (опционально) Gemini: `GEMINI_API_KEY` + SMTP для авторизации пользователей
- (опционально) локальная open-source LLM: Ollama (используется когда Gemini недоступен/не разрешён)

## Быстрый старт: macOS / Windows

### macOS (Homebrew)

```bash
# 1) (если нужно) инструменты сборки для macOS
xcode-select --install

# 2) зависимости
brew update
brew install python ffmpeg
brew install --cask ollama
# запустить Ollama (поднимет локальный API на 127.0.0.1:11434)
open -a Ollama

# 3) проект
git clone https://github.com/vlntnbb/ideal-russian-transcriber.git
cd ideal-russian-transcriber

# 4) окружение Python
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -r requirements.txt

# 5) конфиг
cp .env.example .env
# отредактируйте .env и задайте TELEGRAM_BOT_TOKEN=...

# 6) локальная LLM (Ollama)
ollama pull deepseek-r1:8b

# 7) запуск
python3 local_telegram_bot.py
```

### Windows 11/10 (PowerShell + winget)

Откройте PowerShell (желательно от имени администратора) и выполните:

```powershell
# 1) зависимости
winget install -e --id Git.Git
winget install -e --id Python.Python.3.11
winget install -e --id Gyan.FFmpeg
winget install -e --id Ollama.Ollama
# если команды python/ffmpeg/ollama не находятся — перезапустите терминал

# 2) проект
git clone https://github.com/vlntnbb/ideal-russian-transcriber.git
cd ideal-russian-transcriber

# 3) окружение Python
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt

# 4) конфиг
copy .env.example .env
# отредактируйте .env и задайте TELEGRAM_BOT_TOKEN=...

# 5) локальная LLM (Ollama)
# если будет connection refused — запустите в отдельном окне: ollama serve
ollama pull deepseek-r1:8b

# 6) запуск
python local_telegram_bot.py
```

Если PowerShell не даёт активировать venv, выполните один раз:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

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

Отправьте боту voice/audio/video — он вернёт:
1) `Whisper: ...`
2) `GigaAM: ...`
3) Итоговый вариант (только секция `### Итоговый текст:`; если тексты не слишком длинные)
4) `transcript_*.md` — файл (Markdown) с полным результатом:
   - Итоговый вариант по шаблону (Gemini или локальная LLM)
   - Вариант Whisper
   - Вариант GigaAM

## CLI транскрибации (как внешний скрипт)

Для интеграций (например, другого бота) есть простой CLI, который принимает локальный файл и пишет артефакты в папку.

Пример:

```bash
source .venv/bin/activate
python3 transcribe_cli.py --in ./Recording14.m4a --out ./out --norm
```

CLI создаёт в `--out`:
- `original.*` — копия исходного файла
- `asr_*.wav` — WAV (16kHz mono) и (если `--norm`) нормализованный вариант
- `whisper.txt` / `whisper.json`
- `gigaam.txt` / `gigaam.json`
- `transcript.md` — итог + обе транскрибации
- `result.json` — метаданные (тайминги, модели, пути)

### Работа в групповом чате

Как пользоваться в группе:

Ответьте на voice/audio/video сообщением и напишите текстом `@IdealRussianTranscribe_bot` — бот обработает именно то сообщение, на которое вы ответили.

Доп. режим (нормализация громкости + чистка щелчков): укажите ключ после упоминания бота, например `@IdealRussianTranscribe_bot norm` или `@IdealRussianTranscribe_bot норм`.

Чтобы бот реагировал в групповом чате (reply + @mention), выдайте ему **админские права** в этом чате.

### Логи использования

Бот пишет отдельный usage-лог (JSONL) по каждой сессии распознавания: кто запросил, в каком чате, какие модели, длительность этапов и т.п.
По умолчанию файл: `usage_sessions.jsonl` (настраивается через `USAGE_LOG_PATH`, выключается через `USAGE_LOG_ENABLED=0`).
В лог не пишутся транскрипции/промпты — только метаданные и тайминги.

### Локальный дашборд (аналитика)

Есть простая локальная веб‑страница с аналитикой по `usage_sessions.jsonl` (кол-во запусков, пользователи/чаты, пики нагрузки, тайминги и т.п.).

Запуск:

```bash
source .venv/bin/activate
python3 -m dashboard.server
```

Откройте в браузере `http://127.0.0.1:8765`.

Опции:
- другой путь к логу: `python3 -m dashboard.server --usage-log /path/to/usage_sessions.jsonl`
- другой порт: `python3 -m dashboard.server --port 9000`

Для live‑индикатора “Сейчас в работе” бот пишет файл `active_sessions.json` рядом с проектом (по умолчанию). Его можно отключить: `ACTIVE_SESSIONS_ENABLED=0`.

## Авторизация Gemini и белый список чатов

Авторизация включается параметром `AUTH_ENABLED=1`.
Если `AUTH_ENABLED=0`, Gemini доступен без авторизации/белого списка (при наличии `GEMINI_API_KEY`) и команда `/auth` не нужна.

Если `AUTH_ENABLED=1`, Gemini используется только если:
- задан `GEMINI_API_KEY`, и
- пользователь авторизован по email-домену `AUTH_DOMAIN` (по умолчанию `bbooster.io`) **или** чат добавлен в белый список.

Авторизация:
1) Отправьте боту (в личку или в чат) вашу почту вида `email@bbooster.io` — домен берётся из `AUTH_DOMAIN`
2) Бот отправит письмо с фразой из 6–10 слов (в стиле “заклинаний”)
3) Отправьте эту фразу в Telegram боту — авторизация сохранится локально (`auth_state.json`, не коммитится)

Белый список чатов:
- Если авторизованный пользователь запускает бота в групповом чате впервые, этот чат добавляется в белый список.
- После этого любой пользователь в этом чате сможет пользоваться Gemini (даже без личной авторизации), но только в этом чате.

Для работы авторизации через email настройте SMTP переменные в `.env`: `SMTP_HOST`, `SMTP_PORT`, `SMTP_USERNAME`, `SMTP_PASSWORD`, `SMTP_FROM`.

## Локальная open-source LLM (Ollama)

Если Gemini недоступен/не разрешён, бот использует локальную LLM через Ollama (по умолчанию `deepseek-r1:8b`).

Минимально:
- Запустите Ollama локально и убедитесь, что он доступен на `OLLAMA_BASE_URL` (по умолчанию `http://127.0.0.1:11434`)
- Скачайте модель: `ollama pull deepseek-r1:32b` (или любую другую)
- Рекомендуемо: `OLLAMA_MODEL=auto` — бот выберет самую “тяжёлую” локальную модель из `ollama list`

### Как выбрать и установить модель Ollama

Ollama не “подтягивает” модели автоматически (если вы не включили `OLLAMA_AUTO_PULL=1`), поэтому модель нужно поставить локально.

- Посмотреть установленные модели:
  - `ollama list`
- Найти доступные модели в каталоге Ollama:
  - `https://ollama.com/library`
- Установить модель:
  - `ollama pull <model>`
  - пример: `ollama pull deepseek-r1:32b`

Чтобы бот использовал конкретную модель, задайте в `.env`:
- `OLLAMA_MODEL=deepseek-r1:32b`

Чтобы бот сам выбрал лучшую из уже установленных локально:
- `OLLAMA_MODEL=auto`

### Как ускорить локальную LLM на мощном компьютере

На macOS Ollama обычно использует Metal (GPU), поэтому загрузка CPU может быть небольшой — это нормально.
Если хочется “нагрузить” машину сильнее и повысить throughput, можно включить более агрессивные настройки:

- Авто‑профиль:
  - `OLLAMA_PERF_PROFILE=auto` (по умолчанию) — лёгкий autotune на 12+ CPU cores
  - `OLLAMA_PERF_PROFILE=max` — более агрессивно (больше потоков/батч), может потреблять больше RAM
- Явные параметры (перекрывают профиль):
  - `OLLAMA_NUM_THREADS=14` (пример для 16‑ядерного CPU)
  - `OLLAMA_NUM_BATCH=512`

## Обработка транскрипций

Обработка через LLM встроена в основной процесс: после Whisper и GigaAM бот автоматически делает итоговый вариант и присылает результат.

Настройки:
- Gemini: `GEMINI_API_KEY`, `GEMINI_MODEL` (по умолчанию `gemini-3-pro-preview`), `GEMINI_SYSTEM_PROMPT`
- Ollama: `OLLAMA_BASE_URL`, `OLLAMA_MODEL`

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
