from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import time
import json
import threading
import io
import datetime
import sys
import warnings
import uuid
import html as _html
import re
from typing import Optional

from dotenv import load_dotenv
from telegram import Update
from telegram.constants import ChatAction, ChatType, ParseMode
from telegram.error import BadRequest, RetryAfter
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
import httpx

from snapscript.core.audio_processor import AudioProcessor, GigaAMTranscriptionService, TranscriptionService
from snapscript.utils.logging_utils import setup_logging

# Prevent multiple polling instances in the same repo (avoids Telegram getUpdates Conflict).
def _acquire_bot_lock(lock_path: str) -> None:
    import fcntl

    os.makedirs(os.path.dirname(lock_path) or ".", exist_ok=True)
    lock_file = open(lock_path, "a+", encoding="utf-8")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        raise SystemExit(
            f"Another bot instance seems to be running (lock: {lock_path}). Stop it before starting a new one."
        ) from exc
    # Keep FD alive for the whole process.
    globals()["_BOT_LOCK_FD"] = lock_file


TELEGRAM_TEXT_LIMIT = 4096
STATS_PATH = os.path.join(os.path.dirname(__file__), "asr_stats.json")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_USAGE_LOG_PATH = os.path.join(os.path.dirname(__file__), "usage_sessions.jsonl")

_USAGE_LOGGER: Optional[logging.Logger] = None

DEFAULT_GEMINI_SYSTEM_PROMPT = """\
вот две транскрибации одного аудио двумя разными моделями. они косячат в разных местах. твоя задача сделать итоговый текст, взяв из каждой модели пропущенные предложения и исправив явные ошибки и убрав повторы. когда какие-то слова будут отличаться выбирай тот вариант, который из контекста выглядит более правильным. если в обоих случаях текст выглядит как с ошибками - исправь исходя из контекста. ни в коем случае не выкидывай никакие смысловые фразы и предложения, тебе нужно соблюсти точность передачи текста. подсвети жирным те части текста, где тебе пришлось исправлять текст (где он отличается между моделями).

обращай внимание на указания для контент-менеджера, что надо найти, исправить и уточнить по этому тексту если они есть. в конце отчитайся какие действия с текстами ты сделал

вот такой формат выдачи (шаблон)

### Итоговый текст:

### Примечания для контент-менеджера:

### Отчет о проделанных действиях:
"""


def _fmt_dur(sec: float) -> str:
    sec = max(0, int(sec))
    if sec < 60:
        return f"{sec}s"
    m, s = divmod(sec, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


def _truthy_env(name: str, default: bool = True) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw not in {"0", "false", "no", "off", "disabled"}


def _setup_usage_logger() -> Optional[logging.Logger]:
    """
    Separate usage log for sessions (JSON Lines).
    Does not include transcripts/prompts; only metadata and timings.
    """
    global _USAGE_LOGGER
    if _USAGE_LOGGER is not None:
        return _USAGE_LOGGER

    if not _truthy_env("USAGE_LOG_ENABLED", True):
        _USAGE_LOGGER = None
        return None

    path = (os.environ.get("USAGE_LOG_PATH") or "").strip() or DEFAULT_USAGE_LOG_PATH
    try:
        logger = logging.getLogger("usage_sessions")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        _USAGE_LOGGER = logger
        logging.getLogger("local_telegram_bot").info("Usage log enabled: %s", path)
        return logger
    except Exception:
        logging.getLogger("local_telegram_bot").exception("Failed to set up usage logger")
        _USAGE_LOGGER = None
        return None


def _log_usage_session(payload: dict) -> None:
    logger = _USAGE_LOGGER or logging.getLogger("usage_sessions")
    if not logger.handlers:
        return
    try:
        logger.info(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    except Exception:
        # Usage logging must never break bot flow.
        pass


def _wav_duration_sec(wav_path: str) -> float:
    import wave

    try:
        with wave.open(wav_path, "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate() or 0
            if rate <= 0:
                return 0.0
            return float(frames) / float(rate)
    except Exception:
        return 0.0


def _load_stats() -> dict:
    try:
        with open(STATS_PATH, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _save_stats(data: dict) -> None:
    try:
        with open(STATS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _get_rtf_est(kind: str) -> float:
    stats = _load_stats()
    try:
        return float(((stats.get("rtf") or {}).get(kind) or 0.0) or 0.0)
    except Exception:
        return 0.0


def _update_rtf_est(kind: str, rtf: float) -> None:
    if not (rtf and rtf > 0):
        return
    stats = _load_stats()
    rtf_map = stats.setdefault("rtf", {})
    prev = float(rtf_map.get(kind) or 0.0) if isinstance(rtf_map.get(kind), (int, float)) else 0.0
    alpha = 0.3
    rtf_map[kind] = (prev * (1 - alpha) + rtf * alpha) if prev > 0 else rtf
    _save_stats(stats)


class _ProgressState:
    def __init__(self, *, audio_sec: float, est_rtf: float) -> None:
        self._lock = threading.Lock()
        self.audio_sec = float(audio_sec or 0.0)
        self.est_rtf = float(est_rtf or 0.0)
        self.processed_sec = 0.0
        self.est_total_sec = max(1.0, self.audio_sec * (self.est_rtf or 3.0))

    def set_processed_sec(self, sec: float) -> None:
        with self._lock:
            self.processed_sec = max(self.processed_sec, float(sec or 0.0))

    def snapshot(self) -> tuple[float, float, float]:
        with self._lock:
            return float(self.audio_sec), float(self.processed_sec), float(self.est_total_sec)

    def update_estimate(self, elapsed_sec: float) -> None:
        with self._lock:
            audio_sec = float(self.audio_sec)
            processed_sec = float(self.processed_sec)
            prev = float(self.est_total_sec)

            if audio_sec > 0 and processed_sec > 0:
                progress = min(processed_sec / audio_sec, 0.995)
                raw = max(elapsed_sec / max(1e-3, progress), elapsed_sec)
                # Smooth to avoid jumping ETA.
                self.est_total_sec = max(elapsed_sec, prev * 0.7 + raw * 0.3)
                return

            # No progress info yet (e.g., model warmup/download) — keep a conservative estimate.
            base = audio_sec * (self.est_rtf or 3.0)
            self.est_total_sec = max(elapsed_sec, prev, base)


def _chunks(text: str, *, limit: int = TELEGRAM_TEXT_LIMIT - 50):
    text = text or ""
    while len(text) > limit:
        cut = text.rfind("\n", 0, limit)
        if cut <= 0:
            cut = limit
        yield text[:cut].rstrip()
        text = text[cut:].lstrip()
    if text.strip():
        yield text


async def _reply_long(update: Update, text: str) -> None:
    for part in _chunks(text):
        await update.effective_message.reply_text(part)


async def _safe_edit(message, text: str) -> Optional[float]:
    try:
        if not message:
            return None
        await message.edit_text(text)
        return None
    except RetryAfter as exc:
        return float(getattr(exc, "retry_after", 0) or 0) or 1.0
    except Exception:
        # Message can be deleted/edited too often/etc. Don't fail ASR because of UI.
        return None


async def _safe_edit_formatted(message, text: str, *, parse_mode: Optional[str] = None) -> Optional[float]:
    """
    Like `_safe_edit`, but allows setting `parse_mode` (HTML/MarkdownV2).
    """
    try:
        if not message:
            return None
        await message.edit_text(text, parse_mode=parse_mode)
        return None
    except RetryAfter as exc:
        return float(getattr(exc, "retry_after", 0) or 0) or 1.0
    except Exception:
        return None


async def _safe_delete(message) -> None:
    try:
        if message:
            await message.delete()
    except Exception:
        pass


async def _ticker(
    *,
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    message,
    base_text: str,
    state: Optional[_ProgressState] = None,
    interval_sec: float = 5.0,
) -> None:
    start = time.monotonic()
    try:
        while True:
            elapsed_sec = time.monotonic() - start
            if state is not None:
                state.update_estimate(elapsed_sec)
                audio_sec, processed_sec, est_total_sec = state.snapshot()
                if audio_sec > 0 and processed_sec > 0:
                    pct = int(min(99, (processed_sec / audio_sec) * 100))
                    line = f"⏱ {_fmt_dur(elapsed_sec)}/{_fmt_dur(est_total_sec)} • {pct}%"
                else:
                    line = f"⏱ {_fmt_dur(elapsed_sec)}/{_fmt_dur(est_total_sec)}"
            else:
                line = f"⏱ {_fmt_dur(elapsed_sec)}"

            retry_after = await _safe_edit(message, f"{base_text}\n{line}")
            try:
                await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
            except RetryAfter as exc:
                retry_after = max(retry_after or 0.0, float(getattr(exc, "retry_after", 0) or 0) or 1.0)
            except Exception:
                pass

            sleep_for = interval_sec
            if retry_after is not None and retry_after > 0:
                sleep_for = max(sleep_for, retry_after + 0.5)
            await asyncio.sleep(sleep_for)
    except asyncio.CancelledError:
        return


def _get_env(name: str, default: str) -> str:
    v = (os.environ.get(name) or "").strip()
    return v if v else default


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    _ = context
    await update.effective_message.reply_text(
        "Отправьте voice/audio — верну транскрипцию двумя моделями: Whisper и GigaAM.\n"
        "Команды: /models, /process (или напишите: `обработать <промпт>`)"
    )


async def cmd_models(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    _ = context
    whisper_model = _get_env("WHISPER_MODEL", "medium")
    gigaam_model = _get_env("GIGAAM_MODEL", "v3_e2e_rnnt")
    device = _get_env("DEVICE", "cpu")
    language = _get_env("LANGUAGE", "ru")
    gemini_model = _get_env("GEMINI_MODEL", "gemini-3-pro-preview")
    gemini_key = bool((_get_env("GEMINI_API_KEY", "") or "").strip())
    gemini_temperature = _get_env("GEMINI_TEMPERATURE", "1")
    gemini_top_p = _get_env("GEMINI_TOP_P", "0.95")
    gemini_max_output = _get_env("GEMINI_MAX_OUTPUT_TOKENS", "65536")
    gemini_media_res = _get_env("GEMINI_MEDIA_RESOLUTION", "default")
    gemini_thinking = _get_env("GEMINI_THINKING_LEVEL", "high")
    await update.effective_message.reply_text(
        "ASR:\n"
        f"WHISPER_MODEL={whisper_model}\nGIGAAM_MODEL={gigaam_model}\nDEVICE={device}\nLANGUAGE={language}\n\n"
        "LLM:\n"
        f"GEMINI_MODEL={gemini_model}\n"
        f"GEMINI_API_KEY={'set' if gemini_key else 'not set'}\n"
        f"GEMINI_TEMPERATURE={gemini_temperature}\n"
        f"GEMINI_TOP_P={gemini_top_p}\n"
        f"GEMINI_MAX_OUTPUT_TOKENS={gemini_max_output}\n"
        f"GEMINI_MEDIA_RESOLUTION={gemini_media_res}\n"
        f"GEMINI_THINKING_LEVEL={gemini_thinking}"
    )


def _pick_telegram_file(update: Update):
    msg = update.effective_message
    if msg is None:
        return None, None, None
    return _pick_telegram_file_from_message(msg)


def _pick_telegram_file_from_message(msg):
    if msg.voice:
        return msg.voice.file_id, "voice.ogg", msg.voice.file_size
    if msg.audio:
        name = msg.audio.file_name or "audio"
        return msg.audio.file_id, name, msg.audio.file_size
    if msg.document and (msg.document.mime_type or "").startswith("audio/"):
        name = msg.document.file_name or "audio.bin"
        return msg.document.file_id, name, msg.document.file_size

    return None, None, None


def _telegram_max_get_file_bytes() -> int:
    # Telegram Bot API has file download limits; keep a conservative default.
    # Set TELEGRAM_MAX_GET_FILE_MB=0 to disable the precheck.
    mb = int((os.environ.get("TELEGRAM_MAX_GET_FILE_MB") or "").strip() or "20")
    if mb <= 0:
        return 0
    return mb * 1024 * 1024


def _telegram_local_mode_enabled() -> bool:
    return _truthy_env("TELEGRAM_LOCAL_MODE", False)


async def _send_transcript_file(update: Update, *, whisper_text: str, gigaam_text: str) -> None:
    msg = update.effective_message
    if msg is None:
        return

    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    chat_id = update.effective_chat.id if update.effective_chat else "chat"
    filename = f"transcript_{chat_id}_{now}.txt"

    content = (
        "Whisper:\n"
        f"{(whisper_text or '').strip()}\n\n"
        "GigaAM:\n"
        f"{(gigaam_text or '').strip()}\n"
    )
    data = content.encode("utf-8")
    bio = io.BytesIO(data)
    bio.name = filename
    await msg.reply_document(document=bio, filename=filename)


async def _send_markdown_file(
    update: Update,
    *,
    gemini_text: str,
    gemini_error: str,
    whisper_text: str,
    gigaam_text: str,
) -> None:
    msg = update.effective_message
    if msg is None:
        return

    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    chat_id = update.effective_chat.id if update.effective_chat else "chat"
    filename = f"transcript_{chat_id}_{now}.md"

    if gemini_error and not gemini_text:
        gemini_block = (
            "### Итоговый текст:\n\n"
            "_(Gemini не смог сформировать итог)_\n\n"
            "### Примечания для контент-менеджера:\n\n"
            "_(нет)_\n\n"
            "### Отчет о проделанных действиях:\n\n"
            f"- Ошибка Gemini: `{gemini_error.strip()}`\n"
        )
    else:
        gemini_block = (gemini_text or "").strip()

    content = (
        "## 1) Итоговый вариант по шаблону (Gemini)\n\n"
        f"{gemini_block}\n\n"
        "---\n\n"
        "## 2) Вариант Whisper\n\n"
        f"{(whisper_text or '').strip()}\n\n"
        "---\n\n"
        "## 3) Вариант GigaAM\n\n"
        f"{(gigaam_text or '').strip()}\n"
    )
    bio = io.BytesIO(content.encode("utf-8"))
    bio.name = filename
    await msg.reply_document(document=bio, filename=filename)


def _get_last_transcripts(context: ContextTypes.DEFAULT_TYPE, *, chat_id: int, user_id: int) -> Optional[dict]:
    store = context.application.bot_data.get("last_transcripts") or {}
    return store.get(f"{chat_id}:{user_id}")


def _set_last_transcripts(
    context: ContextTypes.DEFAULT_TYPE,
    *,
    chat_id: int,
    user_id: int,
    whisper_text: str,
    gigaam_text: str,
) -> None:
    store = context.application.bot_data.setdefault("last_transcripts", {})
    store[f"{chat_id}:{user_id}"] = {
        "whisper": whisper_text or "",
        "gigaam": gigaam_text or "",
        "ts": time.time(),
    }


def _parse_process_args(text: str) -> tuple[Optional[str], str]:
    """
    Returns (model_override, prompt).
    Supported:
      /process <prompt>
      /process model=<name> <prompt>
      /process модель=<name> <prompt>
    """
    raw = (text or "").strip()
    if not raw:
        return None, ""
    parts = raw.split()
    if not parts:
        return None, ""

    first = parts[0].strip()
    for key in ("model=", "model:", "модель=", "модель:"):
        if first.lower().startswith(key):
            model = first.split("=", 1)[1] if "=" in first else first.split(":", 1)[1]
            model = (model or "").strip() or None
            prompt = " ".join(parts[1:]).strip()
            return model, prompt
    return None, raw


def _parse_float_env(value: str, default: float) -> float:
    try:
        return float((value or "").strip())
    except Exception:
        return float(default)


def _parse_int_env(value: str, default: int) -> int:
    try:
        return int((value or "").strip())
    except Exception:
        return int(default)


def _thinking_level_value(level: str) -> str:
    v = (level or "").strip().lower()
    # Gemini API expects plain strings ("high"/"low"/"THINKING_LEVEL_UNSPECIFIED") for v1beta.
    if v in {"high", "h"}:
        return "high"
    if v in {"medium", "mid", "m"}:
        return "medium"
    if v in {"low", "l"}:
        return "low"
    return "THINKING_LEVEL_UNSPECIFIED"


def _media_resolution_value(value: str) -> str:
    v = (value or "").strip().lower()
    # Gemini API accepts enum-like strings for mediaResolution; keep mapping for future multimodal use.
    if v in {"default", "unspecified", ""}:
        return "MEDIA_RESOLUTION_UNSPECIFIED"
    if v in {"low"}:
        return "MEDIA_RESOLUTION_LOW"
    if v in {"medium", "mid"}:
        return "MEDIA_RESOLUTION_MEDIUM"
    if v in {"high"}:
        return "MEDIA_RESOLUTION_HIGH"
    return "MEDIA_RESOLUTION_UNSPECIFIED"


async def _gemini_generate(
    *,
    api_key: str,
    model: str,
    prompt: str,
    system_prompt: Optional[str],
    generation_config: dict,
) -> str:
    url = f"{GEMINI_API_URL}/models/{model}:generateContent"
    params = {"key": api_key}
    payload = {
        "systemInstruction": {"role": "system", "parts": [{"text": system_prompt}]} if system_prompt else None,
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": generation_config,
    }
    payload = {k: v for k, v in payload.items() if v is not None}

    timeout = httpx.Timeout(connect=10.0, read=240.0, write=30.0, pool=30.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(url, params=params, json=payload)
        r.raise_for_status()
        data = r.json()

    candidates = data.get("candidates") or []
    if not candidates:
        raise RuntimeError(f"Gemini: empty response: {data!r}")
    content = (candidates[0] or {}).get("content") or {}
    parts = content.get("parts") or []
    texts = []
    for p in parts:
        t = (p or {}).get("text")
        if t:
            texts.append(str(t))
    out = "\n".join(texts).strip()
    return out


def _trim_for_telegram(text: str, *, limit: int = TELEGRAM_TEXT_LIMIT - 50) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    # Keep the tail (usually most relevant in a live "thoughts" stream).
    return "…\n" + text[-limit:].lstrip()


def _markdown_bold_lines_to_html(text: str) -> str:
    """
    Minimal Markdown -> HTML conversion for lines like "**Header**" so Telegram shows them bold.
    Everything else is HTML-escaped (no other markdown is interpreted).
    """
    out_lines = []
    for line in (text or "").splitlines():
        s = line.strip()
        if s.startswith("**") and s.endswith("**") and len(s) >= 4 and s.count("**") == 2:
            inner = s[2:-2].strip()
            out_lines.append(f"<b>{_html.escape(inner)}</b>")
        else:
            out_lines.append(_html.escape(line))
    return "\n".join(out_lines).strip()


def _markdown_to_telegram_html(text: str) -> str:
    """
    Minimal Markdown -> Telegram HTML:
    - Headings like '### Title' become bold lines.
    - Inline **bold** becomes <b>bold</b> (within a single line).
    Everything else is HTML-escaped.
    """

    def md_line_to_html(line: str) -> str:
        raw = line.rstrip("\n")
        stripped = raw.strip()

        # Headings: "#", "##", ... "######"
        m = re.match(r"^(#{1,6})\\s+(.*)$", stripped)
        if m:
            title = m.group(2).strip()
            return f"<b>{_html.escape(title)}</b>"

        # Inline bold: **...** (same line only)
        parts = raw.split("**")
        # Even length => unmatched ** markers; keep as plain text.
        if len(parts) >= 3 and (len(parts) % 2 == 1):
            out = []
            for i, part in enumerate(parts):
                if i % 2 == 1:
                    out.append(f"<b>{_html.escape(part)}</b>")
                else:
                    out.append(_html.escape(part))
            return "".join(out)

        return _html.escape(raw)

    lines = (text or "").splitlines()
    return "\n".join(md_line_to_html(line) for line in lines).strip()


def _split_html_for_telegram(html_text: str, *, limit: int = TELEGRAM_TEXT_LIMIT - 50):
    """
    Splits HTML text into chunks that Telegram can accept, trying to avoid
    cutting inside tags and keeping <b> tags balanced across chunks.
    """
    s = (html_text or "").strip()
    if not s:
        return

    open_bold = False
    while s:
        if len(s) <= limit:
            chunk = s
            s = ""
        else:
            # Prefer cutting at a newline or space before the limit.
            cut = max(s.rfind("\n", 0, limit), s.rfind(" ", 0, limit))
            if cut <= 0:
                cut = limit

            # Avoid cutting inside an HTML tag.
            sub = s[:cut]
            lt = sub.rfind("<")
            gt = sub.rfind(">")
            if lt > gt:
                # We're inside a tag; back up to before the tag.
                cut = lt
                if cut <= 0:
                    cut = limit

            chunk = s[:cut].rstrip()
            s = s[cut:].lstrip()

        # Track bold tag state and balance within the chunk.
        opens = chunk.count("<b>")
        closes = chunk.count("</b>")
        if open_bold:
            chunk = "<b>" + chunk
            open_bold = False
            opens += 1
        if opens > closes:
            open_bold = True
            chunk = chunk + "</b>"

        if chunk.strip():
            yield chunk


async def _reply_long_html(update: Update, html_text: str) -> None:
    for part in _split_html_for_telegram(html_text):
        await update.effective_message.reply_text(part, parse_mode=ParseMode.HTML)


def _gemini_chat_excerpt(text: str) -> str:
    """
    For chat output, show only the final text section (before notes/actions).
    Full output always goes to the markdown file.
    """
    s = (text or "").strip()
    if not s:
        return ""
    markers = [
        "### Примечания для контент-менеджера",
        "### Примечания",
    ]
    for m in markers:
        idx = s.find(m)
        if idx != -1:
            return s[:idx].rstrip()
    return s


async def _gemini_stream_generate(
    *,
    api_key: str,
    model: str,
    prompt: str,
    system_prompt: Optional[str],
    generation_config: dict,
    on_update,
) -> str:
    """
    Streams `:streamGenerateContent` and calls `on_update(thoughts, elapsed_sec)` periodically.
    Returns final (non-thought) text.
    """
    url = f"{GEMINI_API_URL}/models/{model}:streamGenerateContent"
    # Gemini supports SSE streaming via `alt=sse`.
    params = {"key": api_key, "alt": "sse"}
    payload = {
        "systemInstruction": {"role": "system", "parts": [{"text": system_prompt}]} if system_prompt else None,
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": generation_config,
    }
    payload = {k: v for k, v in payload.items() if v is not None}

    timeout = httpx.Timeout(connect=10.0, read=600.0, write=30.0, pool=30.0)
    thoughts = ""
    final = ""
    finish_reason: Optional[str] = None
    started = time.monotonic()
    last_ui = 0.0
    ui_interval = 3.0

    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream(
            "POST",
            url,
            params=params,
            json=payload,
            headers={"Accept": "text/event-stream"},
        ) as r:
            r.raise_for_status()
            async for raw_line in r.aiter_lines():
                line = (raw_line or "").strip()
                if not line:
                    continue
                if line.startswith("data:"):
                    line = line[len("data:") :].strip()
                if line == "[DONE]":
                    break

                try:
                    chunk = json.loads(line)
                except Exception:
                    continue

                candidates = chunk.get("candidates") or []
                if not candidates:
                    continue
                cand0 = candidates[0] or {}
                finish_reason = cand0.get("finishReason") or finish_reason
                content = (cand0 or {}).get("content") or {}
                parts = content.get("parts") or []
                for p in parts:
                    t = (p or {}).get("text")
                    if not t:
                        continue
                    if (p or {}).get("thought") is True:
                        thoughts += str(t)
                    else:
                        final += str(t)

                now = time.monotonic()
                if (now - last_ui) >= ui_interval:
                    last_ui = now
                    try:
                        await on_update(thoughts, now - started)
                    except Exception:
                        pass

    try:
        await on_update(thoughts, time.monotonic() - started)
    except Exception:
        pass
    out = (final or "").strip()
    if not out:
        if finish_reason == "MAX_TOKENS":
            raise RuntimeError(
                "Gemini вернул только thinking-часть и упёрся в лимит токенов (MAX_TOKENS), поэтому финального ответа нет.\n"
                "Увеличьте `GEMINI_MAX_OUTPUT_TOKENS` или уменьшите промпт/сложность обработки."
            )
        raise RuntimeError("Gemini не вернул финальный текст (пусто).")
    return out


async def cmd_process(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.effective_message
    if msg is None:
        return

    chat_id = int(update.effective_chat.id) if update.effective_chat else None
    user_id = int(update.effective_user.id) if update.effective_user else None
    if chat_id is None or user_id is None:
        await msg.reply_text("Не удалось определить chat/user.")
        return

    api_key = (os.environ.get("GEMINI_API_KEY") or "").strip()
    if not api_key:
        await msg.reply_text("Не задан `GEMINI_API_KEY` в `.env`.")
        return

    model_override, user_prompt = _parse_process_args(" ".join(context.args or []))
    if not user_prompt:
        await msg.reply_text(
            "Использование:\n"
            "`/process <промпт>`\n"
            "или\n"
            "`/process model=<gemini-model> <промпт>`"
        )
        return

    last = _get_last_transcripts(context, chat_id=chat_id, user_id=user_id)
    if not last:
        await msg.reply_text("Нет сохранённой транскрипции. Сначала отправьте voice/audio.")
        return

    model = (model_override or os.environ.get("GEMINI_MODEL") or "gemini-3-pro-preview").strip()
    generation_config = {
        # Match the UI defaults from your screenshot:
        "temperature": _parse_float_env(os.environ.get("GEMINI_TEMPERATURE") or "1", 1.0),
        "topP": _parse_float_env(os.environ.get("GEMINI_TOP_P") or "0.95", 0.95),
        "maxOutputTokens": _parse_int_env(os.environ.get("GEMINI_MAX_OUTPUT_TOKENS") or "65536", 65536),
        "mediaResolution": _media_resolution_value(os.environ.get("GEMINI_MEDIA_RESOLUTION") or "default"),
        "thinkingConfig": {
            "thinkingLevel": _thinking_level_value(os.environ.get("GEMINI_THINKING_LEVEL") or "high"),
            "includeThoughts": True,
        },
    }
    system_prompt = (os.environ.get("GEMINI_SYSTEM_PROMPT") or "").strip() or DEFAULT_GEMINI_SYSTEM_PROMPT
    full_prompt = (
        "У тебя есть две транскрипции одного и того же аудио.\n\n"
        "Whisper:\n"
        f"{(last.get('whisper') or '').strip()}\n\n"
        "GigaAM:\n"
        f"{(last.get('gigaam') or '').strip()}\n\n"
        "Задание пользователя:\n"
        f"{user_prompt.strip()}\n"
    )

    llm_sem: asyncio.Semaphore = context.application.bot_data.setdefault("llm_semaphore", asyncio.Semaphore(1))
    async with llm_sem:
        status = await msg.reply_text(f"🧠 Gemini ({model}) — обрабатываю…")
        try:
            async def on_update(thoughts: str, elapsed_sec: float) -> None:
                if not update.effective_chat:
                    return
                # Best-effort "live thoughts" UI; will be deleted when done.
                body = _trim_for_telegram(thoughts or "(пока без мыслей)")
                body_html = _markdown_bold_lines_to_html(body)
                text_html = (
                    f"🧠 Gemini ({_html.escape(model)}) — думаю…\n"
                    f"⏱ {_html.escape(_fmt_dur(elapsed_sec))}\n\n"
                    f"{body_html}"
                )
                await _safe_edit_formatted(status, text_html, parse_mode=ParseMode.HTML)
                try:
                    await context.bot.send_chat_action(chat_id=int(update.effective_chat.id), action=ChatAction.TYPING)
                except Exception:
                    pass

            out = await _gemini_stream_generate(
                api_key=api_key,
                model=model,
                prompt=full_prompt,
                system_prompt=system_prompt,
                generation_config=generation_config,
                on_update=on_update,
            )
            await _safe_delete(status)
            out = out or "(пусто)"
            await _reply_long_html(update, _markdown_to_telegram_html(out))
        except Exception as exc:
            logging.getLogger("local_telegram_bot").exception("Gemini failed")
            await _safe_delete(status)
            await msg.reply_text(f"Ошибка Gemini: {exc}")


async def handle_process_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.effective_message
    if msg is None or not msg.text:
        return
    text = msg.text.strip()
    low = text.lower()
    if not (low == "обработать" or low.startswith("обработать ")):
        return
    prompt = text[len("обработать") :].strip()
    # Emulate /process <prompt>
    context.args = prompt.split() if prompt else []
    await cmd_process(update, context)


def _is_group_chat(update: Update) -> bool:
    chat = update.effective_chat
    if not chat or not chat.type:
        return False
    try:
        return chat.type in {ChatType.GROUP, ChatType.SUPERGROUP}
    except Exception:
        # Fallback for older/newer PTB types.
        return str(chat.type).lower() in {"group", "supergroup"}


def _text_mentions_bot(text: str, *, bot_username: Optional[str]) -> bool:
    if not text or not bot_username:
        return False
    return f"@{bot_username.lower()}" in text.lower()


async def handle_group_tag(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Group flow: reply to a voice/audio message and mention the bot (e.g. "@botname").
    """
    msg = update.effective_message
    if msg is None or not msg.text:
        return
    if not _is_group_chat(update):
        return

    replied = msg.reply_to_message
    if replied is None:
        return

    bot_username = context.application.bot_data.get("bot_username")
    if not bot_username:
        try:
            me = await context.bot.get_me()
            bot_username = me.username
            if bot_username:
                context.application.bot_data["bot_username"] = bot_username
        except Exception:
            bot_username = None
    if not _text_mentions_bot(msg.text, bot_username=bot_username):
        return

    try:
        ent_types = ",".join(sorted({(e.type or "") for e in (msg.entities or []) if e})) or "-"
        logging.getLogger("local_telegram_bot").info(
            "Group tag triggered chat=%s msg=%s reply_to=%s entities=%s",
            getattr(update.effective_chat, "id", None),
            getattr(msg, "message_id", None),
            getattr(replied, "message_id", None),
            ent_types,
        )
    except Exception:
        pass

    file_id, filename, file_size = _pick_telegram_file_from_message(replied)
    if not file_id:
        await msg.reply_text("Ответьте на voice/audio сообщение и упомяните меня (@...).")
        return

    await _process_audio(
        update,
        context,
        file_id=file_id,
        filename=filename or "audio",
        reply_target=msg,
        source_message_id=getattr(replied, "message_id", None),
        source_file_size=file_size,
    )


async def _process_audio(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    *,
    file_id: str,
    filename: str,
    reply_target,
    source_message_id: Optional[int] = None,
    source_file_size: Optional[int] = None,
) -> None:
    """
    Core pipeline used by both private chats (direct voice/audio)
    and group chats (reply+mention).
    """
    sem: asyncio.Semaphore = context.application.bot_data.setdefault("asr_semaphore", asyncio.Semaphore(1))
    async with sem:
        session_id = uuid.uuid4().hex
        started_at = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc).isoformat()
        t0 = time.monotonic()
        user = update.effective_user
        chat = update.effective_chat
        req_msg = update.effective_message
        session: dict = {
            "session_id": session_id,
            "started_at": started_at,
            "trigger": "group_mention" if _is_group_chat(update) else "private_audio",
            "telegram": {
                "user": {
                    "id": getattr(user, "id", None),
                    "username": getattr(user, "username", None),
                    "first_name": getattr(user, "first_name", None),
                    "last_name": getattr(user, "last_name", None),
                    "language_code": getattr(user, "language_code", None),
                },
                "chat": {
                    "id": getattr(chat, "id", None),
                    "type": str(getattr(chat, "type", "") or ""),
                    "title": getattr(chat, "title", None),
                },
                "request_message_id": getattr(reply_target, "message_id", None) or getattr(req_msg, "message_id", None),
                "audio_message_id": source_message_id
                or getattr(reply_target, "message_id", None)
                or getattr(req_msg, "message_id", None),
            },
            "audio": {"file_id": file_id, "filename": filename, "file_size": source_file_size},
            "status": "started",
        }
        try:
            whisper_model = _get_env("WHISPER_MODEL", "medium")
            gigaam_model = _get_env("GIGAAM_MODEL", "v3_e2e_rnnt")
            device = _get_env("DEVICE", "cpu")
            language = _get_env("LANGUAGE", "ru")
            hf_token: Optional[str] = (os.environ.get("HF_TOKEN") or "").strip() or None
            gemini_api_key = (os.environ.get("GEMINI_API_KEY") or "").strip()
            gemini_model = (os.environ.get("GEMINI_MODEL") or "gemini-3-pro-preview").strip()
            gemini_system_prompt = (os.environ.get("GEMINI_SYSTEM_PROMPT") or "").strip() or DEFAULT_GEMINI_SYSTEM_PROMPT
            session["models"] = {
                "whisper": whisper_model,
                "gigaam": gigaam_model,
                "gemini": gemini_model if gemini_api_key else None,
                "device": device,
                "language": language,
                "hf_token": bool(hf_token),
            }

            # Single "status/progress" message. It is recreated between stages so it's always below previously sent text.
            progress = await reply_target.reply_text("Старт…")
            chat_id = int(update.effective_chat.id) if update.effective_chat else None

            await _safe_edit(progress, "📥 Скачиваю аудио…")
            max_bytes = _telegram_max_get_file_bytes()
            if max_bytes and source_file_size and source_file_size > max_bytes:
                size_mb = source_file_size / (1024 * 1024)
                limit_mb = max_bytes / (1024 * 1024)
                session["status"] = "error"
                session["error"] = f"incoming_file_too_big({size_mb:.1f}MB>{limit_mb:.0f}MB)"
                await _safe_delete(progress)
                await reply_target.reply_text(
                    f"Ошибка: файл слишком большой для скачивания ботом через Telegram API "
                    f"({size_mb:.1f}MB, лимит ~{limit_mb:.0f}MB)."
                )
                return

            try:
                tg_file = await context.bot.get_file(file_id)
            except BadRequest as exc:
                if "File is too big" in str(exc):
                    session["status"] = "error"
                    session["error"] = "telegram_get_file_too_big"
                    await _safe_delete(progress)
                    await reply_target.reply_text(
                        "Ошибка: Telegram API не даёт скачать этот файл (File is too big). "
                        "Отправьте более короткое аудио или сожмите/обрежьте файл."
                    )
                    return
                raise

            with tempfile.TemporaryDirectory(prefix="tg_asr_") as td:
                src_path = os.path.join(td, filename)
                wav_dir = td

                dl0 = time.monotonic()
                await tg_file.download_to_drive(custom_path=src_path)
                session["timings"] = {"download_sec": round(time.monotonic() - dl0, 3)}

                ap = AudioProcessor()
                loop = asyncio.get_running_loop()

                await _safe_edit(progress, "✅ Аудио скачано\n🎛 Готовлю WAV (16kHz mono)…")
                ex0 = time.monotonic()
                wav_path = await loop.run_in_executor(None, lambda: ap.extract_audio(src_path, output_dir=wav_dir))
                session["audio"]["wav_path"] = os.path.basename(wav_path)
                session["audio"]["wav_sec"] = round(_wav_duration_sec(wav_path), 3)
                session["timings"]["extract_wav_sec"] = round(time.monotonic() - ex0, 3)

                await _safe_edit(
                    progress,
                    "✅ WAV готов\n"
                    f"🧠 Whisper ({whisper_model}) — распознаю…\n"
                    "(первый запуск может качать модель; для скорости beam_size=1)",
                )
                cpu_threads = int((os.environ.get("WHISPER_CPU_THREADS") or "").strip() or "0")
                whisper = TranscriptionService(
                    model_size=whisper_model,
                    device=device,
                    language=language,
                    cpu_threads=cpu_threads,
                )
                ticker_task = None
                whisper_state: Optional[_ProgressState] = None
                if chat_id is not None:
                    audio_sec = _wav_duration_sec(wav_path)
                    whisper_state = _ProgressState(audio_sec=audio_sec, est_rtf=_get_rtf_est("whisper"))
                    ticker_task = asyncio.create_task(
                        _ticker(
                            context=context,
                            chat_id=chat_id,
                            message=progress,
                            base_text=f"🧠 Whisper ({whisper_model}) — распознаю…",
                            state=whisper_state,
                        )
                    )
                try:
                    whisper_timeout = int((os.environ.get("WHISPER_TIMEOUT_SEC") or "").strip() or "240")
                    w0 = time.monotonic()
                    w_segments, _w_info = await asyncio.wait_for(
                        loop.run_in_executor(
                            None,
                            lambda: whisper.transcribe(
                                wav_path,
                                progress_cb=(whisper_state.set_processed_sec if whisper_state else None),
                            ),
                        ),
                        timeout=float(max(10, whisper_timeout)),
                    )
                    w_wall = time.monotonic() - w0
                    if whisper_state and whisper_state.audio_sec > 0:
                        _update_rtf_est("whisper", w_wall / whisper_state.audio_sec)
                    session["timings"]["whisper_sec"] = round(w_wall, 3)
                finally:
                    if ticker_task:
                        ticker_task.cancel()
                        try:
                            await ticker_task
                        except Exception:
                            pass
                w_text = " ".join((s.text or "").strip() for s in w_segments if (s.text or "").strip()).strip()
                session["results"] = {
                    "whisper_len": len(w_text),
                    "whisper_segments": len(w_segments or []),
                }
                await _safe_edit(
                    progress,
                    "✅ Whisper готов\n"
                    f"🧠 GigaAM ({gigaam_model}) — распознаю…\n"
                    "(первый запуск может качать веса)",
                )

                giga = GigaAMTranscriptionService(model_name=gigaam_model, device=device, hf_token=hf_token)
                ticker_task = None
                giga_state: Optional[_ProgressState] = None
                if chat_id is not None:
                    audio_sec = _wav_duration_sec(wav_path)
                    giga_state = _ProgressState(audio_sec=audio_sec, est_rtf=_get_rtf_est("gigaam"))
                    ticker_task = asyncio.create_task(
                        _ticker(
                            context=context,
                            chat_id=chat_id,
                            message=progress,
                            base_text=f"🧠 GigaAM ({gigaam_model}) — распознаю…",
                            state=giga_state,
                        )
                    )
                try:
                    g0 = time.monotonic()
                    g_segments, _g_info = await loop.run_in_executor(None, lambda: giga.transcribe(wav_path))
                    g_wall = time.monotonic() - g0
                    if giga_state and giga_state.audio_sec > 0:
                        _update_rtf_est("gigaam", g_wall / giga_state.audio_sec)
                    session["timings"]["gigaam_sec"] = round(g_wall, 3)
                    if isinstance(_g_info, dict):
                        session.setdefault("gigaam", {})["info"] = _g_info
                finally:
                    if ticker_task:
                        ticker_task.cancel()
                        try:
                            await ticker_task
                        except Exception:
                            pass
                g_text = " ".join((s.text or "").strip() for s in g_segments if (s.text or "").strip()).strip()
                session["results"].update(
                    {
                        "gigaam_len": len(g_text),
                        "gigaam_segments": len(g_segments or []),
                    }
                )

                transcripts_exceed_single_message = (
                    len(w_text) > TELEGRAM_TEXT_LIMIT or len(g_text) > TELEGRAM_TEXT_LIMIT
                )
                session["results"]["transcripts_exceed_single_message"] = transcripts_exceed_single_message
                if transcripts_exceed_single_message:
                    await _safe_edit(
                        progress,
                        "ℹ️ Текст распознавания слишком длинный для одного сообщения.\n"
                        "Итог и обе транскрибации пришлю только файлом…",
                    )
                else:
                    await _safe_delete(progress)
                    progress = None
                    await _reply_long(update, f"Whisper:\n{w_text or '(пусто)'}")
                    await _reply_long(update, f"GigaAM:\n{g_text or '(пусто)'}")

                if update.effective_chat and update.effective_user:
                    _set_last_transcripts(
                        context,
                        chat_id=int(update.effective_chat.id),
                        user_id=int(update.effective_user.id),
                        whisper_text=w_text,
                        gigaam_text=g_text,
                    )

                gemini_text = ""
                gemini_error = ""
                if gemini_api_key and chat_id is not None:
                    await _safe_delete(progress)
                    progress = await reply_target.reply_text(f"🧠 Gemini ({gemini_model}) — думаю над итогом…")
                    gmi0 = time.monotonic()

                    async def on_update(thoughts: str, elapsed_sec: float) -> None:
                        body = _trim_for_telegram(thoughts or "(пока без мыслей)")
                        body_html = _markdown_bold_lines_to_html(body)
                        text_html = (
                            f"🧠 Gemini ({_html.escape(gemini_model)}) — думаю над итогом…\n"
                            f"⏱ {_html.escape(_fmt_dur(elapsed_sec))}\n\n"
                            f"{body_html}"
                        )
                        await _safe_edit_formatted(progress, text_html, parse_mode=ParseMode.HTML)
                        try:
                            await context.bot.send_chat_action(chat_id=int(chat_id), action=ChatAction.TYPING)
                        except Exception:
                            pass

                    generation_config = {
                        "temperature": _parse_float_env(os.environ.get("GEMINI_TEMPERATURE") or "1", 1.0),
                        "topP": _parse_float_env(os.environ.get("GEMINI_TOP_P") or "0.95", 0.95),
                        "maxOutputTokens": _parse_int_env(os.environ.get("GEMINI_MAX_OUTPUT_TOKENS") or "65536", 65536),
                        "mediaResolution": _media_resolution_value(os.environ.get("GEMINI_MEDIA_RESOLUTION") or "default"),
                        "thinkingConfig": {
                            "thinkingLevel": _thinking_level_value(os.environ.get("GEMINI_THINKING_LEVEL") or "high"),
                            "includeThoughts": True,
                        },
                    }
                    user_prompt = (
                        "Whisper:\n"
                        f"{w_text.strip()}\n\n"
                        "GigaAM:\n"
                        f"{g_text.strip()}\n"
                    )
                    try:
                        gemini_text = await _gemini_stream_generate(
                            api_key=gemini_api_key,
                            model=gemini_model,
                            prompt=user_prompt,
                            system_prompt=gemini_system_prompt,
                            generation_config=generation_config,
                            on_update=on_update,
                        )
                    except Exception as exc:
                        logging.getLogger("local_telegram_bot").exception("Gemini default processing failed")
                        gemini_error = str(exc)
                    finally:
                        session["timings"]["gemini_sec"] = round(time.monotonic() - gmi0, 3)
                elif not gemini_api_key:
                    gemini_error = "GEMINI_API_KEY не задан — пропускаю обработку Gemini."

                if progress:
                    await _safe_edit(progress, "📄 Формирую итоговый файл…")

                if gemini_text and not transcripts_exceed_single_message:
                    chat_text = _gemini_chat_excerpt(gemini_text)
                    if chat_text.strip():
                        await _reply_long_html(update, _markdown_to_telegram_html(chat_text.strip()))
                elif gemini_error:
                    await reply_target.reply_text(f"Gemini: ошибка/пропуск: {gemini_error}")

                session["results"].update(
                    {
                        "gemini_used": bool(gemini_api_key),
                        "gemini_error": gemini_error or None,
                        "gemini_len": len(gemini_text or ""),
                    }
                )

                await _send_markdown_file(
                    update,
                    gemini_text=gemini_text,
                    gemini_error=gemini_error,
                    whisper_text=w_text,
                    gigaam_text=g_text,
                )
                session.setdefault("telegram", {}).update(
                    {
                        "sent_transcripts_to_chat": not transcripts_exceed_single_message,
                        "sent_gemini_to_chat": bool(gemini_text) and not transcripts_exceed_single_message,
                        "sent_markdown_file": True,
                    }
                )

                await _safe_delete(progress)
                await reply_target.reply_text("✅ Готово")
                session["status"] = "ok"
        except Exception as exc:
            logging.exception("ASR failed")
            await reply_target.reply_text(f"Ошибка: {exc}")
            session["status"] = "error"
            session["error"] = str(exc)
        finally:
            session["ended_at"] = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc).isoformat()
            session["timings"] = session.get("timings") or {}
            session["timings"]["total_sec"] = round(time.monotonic() - t0, 3)
            _log_usage_session(session)

async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # In groups, process only via reply+mention flow to avoid accidental triggers/spam.
    if _is_group_chat(update):
        return

    file_id, filename, file_size = _pick_telegram_file(update)
    if not file_id:
        await update.effective_message.reply_text("Пришлите voice/audio файл.")
        return

    # Use the shared pipeline (also used by reply+mention in group chats).
    await _process_audio(
        update,
        context,
        file_id=file_id,
        filename=filename or "audio",
        reply_target=update.effective_message,
        source_message_id=getattr(update.effective_message, "message_id", None),
        source_file_size=file_size,
    )
    return

    sem: asyncio.Semaphore = context.application.bot_data.setdefault("asr_semaphore", asyncio.Semaphore(1))
    async with sem:
        try:
            whisper_model = _get_env("WHISPER_MODEL", "medium")
            gigaam_model = _get_env("GIGAAM_MODEL", "v3_e2e_rnnt")
            device = _get_env("DEVICE", "cpu")
            language = _get_env("LANGUAGE", "ru")
            hf_token: Optional[str] = (os.environ.get("HF_TOKEN") or "").strip() or None
            gemini_api_key = (os.environ.get("GEMINI_API_KEY") or "").strip()
            gemini_model = (os.environ.get("GEMINI_MODEL") or "gemini-3-pro-preview").strip()
            gemini_system_prompt = (os.environ.get("GEMINI_SYSTEM_PROMPT") or "").strip() or DEFAULT_GEMINI_SYSTEM_PROMPT

            # Single "status/progress" message. It is recreated between stages so it's always below previously sent text.
            progress = await update.effective_message.reply_text("Старт…")
            chat_id = int(update.effective_chat.id) if update.effective_chat else None

            await _safe_edit(progress, "📥 Скачиваю аудио…")
            tg_file = await context.bot.get_file(file_id)

            with tempfile.TemporaryDirectory(prefix="tg_asr_") as td:
                src_path = os.path.join(td, filename)
                wav_dir = td

                await tg_file.download_to_drive(custom_path=src_path)

                ap = AudioProcessor()
                loop = asyncio.get_running_loop()

                await _safe_edit(progress, "✅ Аудио скачано\n🎛 Готовлю WAV (16kHz mono)…")
                wav_path = await loop.run_in_executor(None, lambda: ap.extract_audio(src_path, output_dir=wav_dir))

                await _safe_edit(
                    progress,
                    "✅ WAV готов\n"
                    f"🧠 Whisper ({whisper_model}) — распознаю…\n"
                    "(первый запуск может качать модель; для скорости beam_size=1)",
                )
                cpu_threads = int((os.environ.get("WHISPER_CPU_THREADS") or "").strip() or "0")
                whisper = TranscriptionService(
                    model_size=whisper_model,
                    device=device,
                    language=language,
                    cpu_threads=cpu_threads,
                )
                ticker_task = None
                whisper_state: Optional[_ProgressState] = None
                if chat_id is not None:
                    audio_sec = _wav_duration_sec(wav_path)
                    whisper_state = _ProgressState(audio_sec=audio_sec, est_rtf=_get_rtf_est("whisper"))
                    ticker_task = asyncio.create_task(
                        _ticker(
                            context=context,
                            chat_id=chat_id,
                            message=progress,
                            base_text=f"🧠 Whisper ({whisper_model}) — распознаю…",
                            state=whisper_state,
                        )
                    )
                try:
                    whisper_timeout = int((os.environ.get("WHISPER_TIMEOUT_SEC") or "").strip() or "240")
                    w0 = time.monotonic()
                    w_segments, _w_info = await asyncio.wait_for(
                        loop.run_in_executor(
                            None,
                            lambda: whisper.transcribe(
                                wav_path,
                                progress_cb=(whisper_state.set_processed_sec if whisper_state else None),
                            ),
                        ),
                        timeout=float(max(10, whisper_timeout)),
                    )
                    w_wall = time.monotonic() - w0
                    if whisper_state and whisper_state.audio_sec > 0:
                        _update_rtf_est("whisper", w_wall / whisper_state.audio_sec)
                finally:
                    if ticker_task:
                        ticker_task.cancel()
                        try:
                            await ticker_task
                        except Exception:
                            pass
                w_text = " ".join((s.text or "").strip() for s in w_segments if (s.text or "").strip()).strip()
                await _reply_long(update, f"Whisper:\n{w_text or '(пусто)'}")

                # Recreate status message so it stays below the Whisper output.
                await _safe_delete(progress)
                progress = await update.effective_message.reply_text(
                    f"🧠 GigaAM ({gigaam_model}) — распознаю…\n(первый запуск может качать веса)"
                )

                await _safe_edit(
                    progress,
                    f"🧠 GigaAM ({gigaam_model}) — распознаю…\n(первый запуск может качать веса)",
                )
                giga = GigaAMTranscriptionService(model_name=gigaam_model, device=device, hf_token=hf_token)
                ticker_task = None
                giga_state: Optional[_ProgressState] = None
                if chat_id is not None:
                    audio_sec = _wav_duration_sec(wav_path)
                    giga_state = _ProgressState(audio_sec=audio_sec, est_rtf=_get_rtf_est("gigaam"))
                    ticker_task = asyncio.create_task(
                        _ticker(
                            context=context,
                            chat_id=chat_id,
                            message=progress,
                            base_text=f"🧠 GigaAM ({gigaam_model}) — распознаю…",
                            state=giga_state,
                        )
                    )
                try:
                    g0 = time.monotonic()
                    g_segments, _g_info = await loop.run_in_executor(None, lambda: giga.transcribe(wav_path))
                    g_wall = time.monotonic() - g0
                    if giga_state and giga_state.audio_sec > 0:
                        _update_rtf_est("gigaam", g_wall / giga_state.audio_sec)
                finally:
                    if ticker_task:
                        ticker_task.cancel()
                        try:
                            await ticker_task
                        except Exception:
                            pass
                g_text = " ".join((s.text or "").strip() for s in g_segments if (s.text or "").strip()).strip()
                await _reply_long(update, f"GigaAM:\n{g_text or '(пусто)'}")

                # Store for /process and later use.
                if update.effective_chat and update.effective_user:
                    _set_last_transcripts(
                        context,
                        chat_id=int(update.effective_chat.id),
                        user_id=int(update.effective_user.id),
                        whisper_text=w_text,
                        gigaam_text=g_text,
                    )

                # Gemini post-processing (default pipeline) → markdown file.
                gemini_text = ""
                gemini_error = ""
                if gemini_api_key and chat_id is not None:
                    # Recreate status message so Gemini thoughts are always below the GigaAM output.
                    await _safe_delete(progress)
                    progress = await update.effective_message.reply_text(
                        f"🧠 Gemini ({gemini_model}) — думаю над итогом…"
                    )

                    async def on_update(thoughts: str, elapsed_sec: float) -> None:
                        body = _trim_for_telegram(thoughts or "(пока без мыслей)")
                        body_html = _markdown_bold_lines_to_html(body)
                        text_html = (
                            f"🧠 Gemini ({_html.escape(gemini_model)}) — думаю над итогом…\n"
                            f"⏱ {_html.escape(_fmt_dur(elapsed_sec))}\n\n"
                            f"{body_html}"
                        )
                        await _safe_edit_formatted(progress, text_html, parse_mode=ParseMode.HTML)
                        try:
                            await context.bot.send_chat_action(chat_id=int(chat_id), action=ChatAction.TYPING)
                        except Exception:
                            pass

                    system_prompt = gemini_system_prompt
                    user_prompt = (
                        "Whisper:\n"
                        f"{w_text.strip()}\n\n"
                        "GigaAM:\n"
                        f"{g_text.strip()}\n"
                    )
                    generation_config = {
                        "temperature": _parse_float_env(os.environ.get("GEMINI_TEMPERATURE") or "1", 1.0),
                        "topP": _parse_float_env(os.environ.get("GEMINI_TOP_P") or "0.95", 0.95),
                        "maxOutputTokens": _parse_int_env(os.environ.get("GEMINI_MAX_OUTPUT_TOKENS") or "65536", 65536),
                        "mediaResolution": _media_resolution_value(os.environ.get("GEMINI_MEDIA_RESOLUTION") or "default"),
                        "thinkingConfig": {
                            "thinkingLevel": _thinking_level_value(os.environ.get("GEMINI_THINKING_LEVEL") or "high"),
                            "includeThoughts": True,
                        },
                    }
                    try:
                        gemini_text = await _gemini_stream_generate(
                            api_key=gemini_api_key,
                            model=gemini_model,
                            prompt=user_prompt,
                            system_prompt=system_prompt,
                            generation_config=generation_config,
                            on_update=on_update,
                        )
                    except Exception as exc:
                        logging.getLogger("local_telegram_bot").exception("Gemini default processing failed")
                        gemini_error = str(exc)
                elif not gemini_api_key:
                    gemini_error = "GEMINI_API_KEY не задан — пропускаю обработку Gemini."

                # Send markdown file in requested order.
                if progress:
                    await _safe_edit(progress, "📄 Формирую итоговый файл…")

                # Also send the final Gemini block as a chat message (then the file).
                if gemini_text:
                    chat_text = _gemini_chat_excerpt(gemini_text)
                    if chat_text.strip():
                        await _reply_long_html(update, _markdown_to_telegram_html(chat_text.strip()))
                elif gemini_error:
                    await update.effective_message.reply_text(f"Gemini: ошибка/пропуск: {gemini_error}")

                await _send_markdown_file(
                    update,
                    gemini_text=gemini_text,
                    gemini_error=gemini_error,
                    whisper_text=w_text,
                    gigaam_text=g_text,
                )

                await _safe_delete(progress)
                await update.effective_message.reply_text("✅ Готово")

        except Exception as exc:
            logging.exception("ASR failed")
            await update.effective_message.reply_text(f"Ошибка: {exc}")


def main() -> None:
    load_dotenv()
    # Keep `bot.log` readable: disable third-party progress bars in console.
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TQDM_DISABLE", "1")
    # Reduce noisy third-party warnings in `bot.log`.
    try:
        from urllib3.exceptions import NotOpenSSLWarning

        warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
    except Exception:
        pass
    warnings.filterwarnings("ignore", message=r"Module 'speechbrain\\.pretrained' was deprecated.*")
    setup_logging()
    _setup_usage_logger()
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("snapscript").setLevel(logging.INFO)

    logging.getLogger("local_telegram_bot").info("Python exe=%s prefix=%s", sys.executable, sys.prefix)
    _acquire_bot_lock(os.path.join(os.path.dirname(__file__), ".bot.lock"))

    token = (os.environ.get("TELEGRAM_BOT_TOKEN") or "").strip()
    if not token:
        raise SystemExit("Set TELEGRAM_BOT_TOKEN (or create .env from .env.example).")

    async def post_init(app: Application) -> None:
        try:
            me = await app.bot.get_me()
            if me.username:
                app.bot_data["bot_username"] = me.username
                logging.getLogger("local_telegram_bot").info("Bot username=@%s", me.username)
            else:
                logging.getLogger("local_telegram_bot").warning("Bot has no username; group @mention flow won't work.")
        except Exception:
            logging.getLogger("local_telegram_bot").exception("Failed to fetch bot username via getMe()")

    builder = Application.builder().token(token).post_init(post_init)
    tg_base_url = (os.environ.get("TELEGRAM_API_BASE_URL") or "").strip()
    tg_file_base_url = (os.environ.get("TELEGRAM_FILE_BASE_URL") or "").strip()
    if tg_base_url:
        builder = builder.base_url(tg_base_url)
    if tg_file_base_url:
        builder = builder.base_file_url(tg_file_base_url)
    if _telegram_local_mode_enabled():
        builder = builder.local_mode(True)

    app = builder.build()

    async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        err = context.error
        logging.getLogger("local_telegram_bot").exception("Unhandled error: %r", err)

    app.add_error_handler(on_error)
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("models", cmd_models))
    app.add_handler(CommandHandler("process", cmd_process))
    app.add_handler(CommandHandler("obrabotat", cmd_process))
    app.add_handler(MessageHandler(filters.ChatType.GROUPS & filters.TEXT & filters.REPLY, handle_group_tag))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_process_text))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO | filters.Document.AUDIO, handle_audio))

    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
